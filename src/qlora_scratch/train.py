from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

from .data import load_json_splits
from .lora import LoRAConfig, count_trainable_parameters, prepare_model_for_kbit_training
from .paged_optim import PagedAdamW32bit


@dataclass
class ExperimentConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    max_seq_length: int = 256
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 1
    max_train_samples: int = 1024
    max_eval_samples: int = 256
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    quant_block_size: int = 64
    optimizer_page_size: int = 2**18
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "results/scratch_qlora"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    )
    return model, tokenizer


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer,
    *,
    max_seq_length: int,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> DatasetDict:
    if max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    if max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(max_eval_samples, len(dataset["validation"])))
        )
        dataset["test"] = dataset["test"].select(range(min(max_eval_samples, len(dataset["test"]))))

    def _tokenize(batch):
        output = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        output["labels"] = output["input_ids"].copy()
        return output

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


@torch.no_grad()
def evaluate(model, dataloader, device: torch.device) -> dict:
    model.eval()
    losses = []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        losses.append(outputs.loss.detach().float().cpu())

    mean_loss = torch.stack(losses).mean().item() if losses else float("nan")
    return {
        "eval_loss": mean_loss,
        "perplexity": math.exp(mean_loss) if math.isfinite(mean_loss) and mean_loss < 20 else float("inf"),
    }


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 80) -> str:
    model.eval()
    batch = tokenizer(prompt, return_tensors="pt").to(device)
    generated = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)


def run_experiment(
    data_dir: str | Path,
    config: ExperimentConfig | None = None,
    *,
    sample_prompt: str = "### Instruction:\nExplain what QLoRA is in two short paragraphs.\n\n### Response:\n",
) -> dict:
    config = config or ExperimentConfig()
    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_json_splits(data_dir)
    model, tokenizer = load_model_and_tokenizer(config.model_name)

    lora_config = LoRAConfig(
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        block_size=config.quant_block_size,
    )
    model = prepare_model_for_kbit_training(model, lora_config)
    device = torch.device(config.device)
    model.to(device)

    trainable, total = count_trainable_parameters(model)
    tokenized = tokenize_dataset(
        dataset,
        tokenizer,
        max_seq_length=config.max_seq_length,
        max_train_samples=config.max_train_samples,
        max_eval_samples=config.max_eval_samples,
    )
    tokenized.set_format(type="torch")

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=config.train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    optimizer = PagedAdamW32bit(
        (p for p in model.parameters() if p.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        page_size=config.optimizer_page_size,
    )

    pre_text = generate_text(model, tokenizer, sample_prompt, device)

    train_losses = []
    start_time = time.time()
    total_tokens = 0
    global_step = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    for epoch in range(config.epochs):
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()

            train_losses.append(loss.item() * config.gradient_accumulation_steps)
            total_tokens += int(batch["attention_mask"].sum().item())

            if step % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            progress.set_postfix(loss=f"{train_losses[-1]:.4f}")

    train_time = time.time() - start_time
    eval_metrics = evaluate(model, val_loader, device)
    post_text = generate_text(model, tokenizer, sample_prompt, device)

    metrics = {
        "config": asdict(config),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": 100.0 * trainable / total,
        "train_loss_last": train_losses[-1] if train_losses else float("nan"),
        "train_loss_curve": train_losses,
        "eval_loss": eval_metrics["eval_loss"],
        "perplexity": eval_metrics["perplexity"],
        "wall_time_s": train_time,
        "tokens_per_second": total_tokens / train_time if train_time > 0 else 0.0,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0
        ),
        "pre_sample": pre_text,
        "post_sample": post_text,
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
