from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


INSTALL_CELL = """import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(PROJECT_ROOT / "requirements.txt")])
"""


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
            "colab": {
                "name": path.name,
                "provenance": [],
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2))


def build_data_prep() -> list[dict]:
    return [
        md_cell("# 01 Data Preparation\n\nPrepare OpenAssistant instruction-response pairs and cache them locally."),
        code_cell(
            """from pathlib import Path
import sys

PROJECT_ROOT = Path("/content/qLoRA").resolve()
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
print(PROJECT_ROOT)
"""
        ),
        code_cell(INSTALL_CELL),
        code_cell(
            """from qlora_scratch.data import build_oasst1_splits

dataset = build_oasst1_splits(DATA_DIR)
dataset
"""
        ),
        code_cell(
            """import json
from pathlib import Path

print(json.loads((DATA_DIR / "metadata.json").read_text()))
"""
        ),
    ]


def build_experiments() -> list[dict]:
    return [
        md_cell(
            "# 02 Scratch QLoRA Experiments\n\nRun a readable from-scratch QLoRA experiment with NF4 quantization, FP16 LoRA adapters, and a paged optimizer."
        ),
        code_cell(
            """from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path("/content/qLoRA").resolve()
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""
        ),
        code_cell(INSTALL_CELL),
        code_cell(
            """from qlora_scratch.train import ExperimentConfig

config = ExperimentConfig(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_seq_length=256,
    train_batch_size=2,
    eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    epochs=1,
    max_train_samples=512,
    max_eval_samples=128,
    lora_rank=8,
    lora_alpha=16,
    quant_block_size=64,
    optimizer_page_size=2**18,
    output_dir=str(PROJECT_ROOT / "results" / "scratch_qlora"),
)
config
"""
        ),
        code_cell(
            """sample_prompts = [
    "### Instruction:\\nSummarize QLoRA in plain language for a student.\\n\\n### Response:\\n",
    "### Instruction:\\nWhat is the purpose of NF4 quantization in QLoRA?\\n\\n### Response:\\n",
    "### Instruction:\\nExplain what a paged optimizer is and why it helps memory usage.\\n\\n### Response:\\n",
]
sample_prompts
"""
        ),
        code_cell(
            """from qlora_scratch.train import run_experiment

metrics = run_experiment(PROJECT_ROOT / "data", config, sample_prompts=sample_prompts)
metrics
"""
        ),
        code_cell(
            """print("Eval loss:", metrics["eval_loss"])
print("Perplexity:", metrics["perplexity"])
print("Peak VRAM (MB):", metrics["peak_vram_mb"])
print("Tokens/sec:", metrics["tokens_per_second"])
"""
        ),
        code_cell(
            """for idx, (pre, post) in enumerate(zip(metrics["pre_samples"], metrics["post_samples"]), start=1):
    print(f"=== PROMPT {idx} ===")
    print(pre["prompt"])
    print("\\n--- PRE-FINETUNE ---")
    print(pre["response"])
    print("\\n--- POST-FINETUNE ---")
    print(post["response"])
    print("\\n")
"""
        ),
    ]


def build_analysis() -> list[dict]:
    return [
        md_cell("# 03 Analysis\n\nInspect metrics, visualize the training curve, and compare generated samples."),
        code_cell(
            """from pathlib import Path
import sys

PROJECT_ROOT = Path("/content/qLoRA").resolve()
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
"""
        ),
        code_cell(INSTALL_CELL),
        code_cell(
            """from qlora_scratch.analysis import (
    build_instruction_tuning_table,
    load_metrics,
    metrics_to_frame,
    plot_system_metrics,
    plot_training_curve,
)

metrics = load_metrics(PROJECT_ROOT / "results" / "scratch_qlora")
summary = metrics_to_frame(metrics)
summary
"""
        ),
        code_cell(
            """fig = plot_training_curve(metrics)
fig
"""
        ),
        code_cell(
            """fig = plot_system_metrics(metrics)
fig
"""
        ),
        code_cell(
            """instruction_table = build_instruction_tuning_table(metrics)
instruction_table
"""
        ),
        code_cell(
            """for row in instruction_table.to_dict(orient="records"):
    print(f"=== PROMPT {row['prompt_id']} ===")
    print(row["prompt"])
    print("\\n--- PRE-FINETUNE ---")
    print(row["pre_response"])
    print("\\n--- POST-FINETUNE ---")
    print(row["post_response"])
    print("\\nLatency (s):", row["pre_latency_s"], "->", row["post_latency_s"])
    print("Tokens/sec:", row["pre_tokens_per_second"], "->", row["post_tokens_per_second"])
    print("\\n")
"""
        ),
    ]


def main() -> None:
    write_notebook(NOTEBOOKS / "01_data_preparation.ipynb", build_data_prep())
    write_notebook(NOTEBOOKS / "02_scratch_qlora_experiments.ipynb", build_experiments())
    write_notebook(NOTEBOOKS / "03_analysis.ipynb", build_analysis())


if __name__ == "__main__":
    main()
