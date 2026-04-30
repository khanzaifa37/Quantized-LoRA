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


GIT_PULL_CELL = """import subprocess

subprocess.run(["git", "-C", str(PROJECT_ROOT), "pull", "origin", "main"], check=False)
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


def project_setup_cell() -> str:
    return """from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path("/content/qLoRA").resolve()
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Project root:", PROJECT_ROOT)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""


def build_data_prep() -> list[dict]:
    return [
        md_cell(
            "# 01 — Dataset Preparation\n\nPrepare the OpenAssistant instruction-tuning dataset and cache the train/validation/test splits."
        ),
        code_cell(project_setup_cell()),
        code_cell(GIT_PULL_CELL),
        code_cell(INSTALL_CELL),
        md_cell("## 1. Build OASST1 Splits"),
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
        md_cell("## 2. Quick Dataset Inspection"),
        code_cell(
            """import pandas as pd
import matplotlib.pyplot as plt

train_lengths = [len(text) for text in dataset["train"]["text"]]
stats = pd.Series(train_lengths).describe()
print(stats.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(train_lengths, bins=50, edgecolor="black", alpha=0.7)
axes[0].set_title("Text Length Distribution")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Count")

word_counts = [len(text.split()) for text in dataset["train"]["text"]]
axes[1].hist(word_counts, bins=50, edgecolor="black", alpha=0.7, color="orange")
axes[1].set_title("Word Count Distribution")
axes[1].set_xlabel("Words")
axes[1].set_ylabel("Count")
plt.tight_layout()
plt.show()
"""
        ),
    ]


def build_experiments() -> list[dict]:
    return [
        md_cell(
            "# 02 — Run Experiments\n\nRun several scratch QLoRA experiments using the custom NF4 quantization, FP16 LoRA adapters, and paged optimizer."
        ),
        code_cell(project_setup_cell()),
        code_cell(GIT_PULL_CELL),
        code_cell(INSTALL_CELL),
        md_cell("## 1. Define Experiment Grid"),
        code_cell(
            """from qlora_scratch.train import ExperimentConfig

sample_prompts = [
    "### Instruction:\\nSummarize QLoRA in plain language for a student.\\n\\n### Response:\\n",
    "### Instruction:\\nWhat is the purpose of NF4 quantization in QLoRA?\\n\\n### Response:\\n",
    "### Instruction:\\nExplain what a paged optimizer is and why it helps memory usage.\\n\\n### Response:\\n",
]

experiments = {
    "scratch_lora_r8": ExperimentConfig(
        method="lora",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_rank=8,
        lora_alpha=16,
        quant_block_size=64,
        max_train_samples=512,
        max_eval_samples=128,
        output_dir=str(RESULTS_DIR / "scratch_lora_r8"),
    ),
    "scratch_qlora_r8": ExperimentConfig(
        method="qlora",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_rank=8,
        lora_alpha=16,
        quant_block_size=64,
        max_train_samples=512,
        max_eval_samples=128,
        output_dir=str(RESULTS_DIR / "scratch_qlora_r8"),
    ),
    "scratch_lora_r16": ExperimentConfig(
        method="lora",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_rank=16,
        lora_alpha=32,
        quant_block_size=64,
        max_train_samples=512,
        max_eval_samples=128,
        output_dir=str(RESULTS_DIR / "scratch_lora_r16"),
    ),
    "scratch_qlora_r16": ExperimentConfig(
        method="qlora",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_rank=16,
        lora_alpha=32,
        quant_block_size=64,
        max_train_samples=512,
        max_eval_samples=128,
        output_dir=str(RESULTS_DIR / "scratch_qlora_r16"),
    ),
    "scratch_lora_r32": ExperimentConfig(
        method="lora",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_rank=32,
        lora_alpha=64,
        quant_block_size=64,
        max_train_samples=512,
        max_eval_samples=128,
        output_dir=str(RESULTS_DIR / "scratch_lora_r32"),
    ),
    "scratch_qlora_r32": ExperimentConfig(
        method="qlora",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_rank=32,
        lora_alpha=64,
        quant_block_size=64,
        max_train_samples=512,
        max_eval_samples=128,
        output_dir=str(RESULTS_DIR / "scratch_qlora_r32"),
    ),
}

print("Experiments to run:")
for name, cfg in experiments.items():
    print(name, "method=", cfg.method, "rank=", cfg.lora_rank, "output=", cfg.output_dir)
"""
        ),
        md_cell("## 2. Run Experiments"),
        code_cell(
            """import gc
import json
import torch
from qlora_scratch.train import run_experiment

all_metrics = []

for exp_name, config in experiments.items():
    print("\\n" + "#" * 70)
    print(f"Running {exp_name}")
    print("#" * 70)
    metrics = run_experiment(PROJECT_ROOT / "data", config, sample_prompts=sample_prompts)
    metrics["experiment"] = exp_name
    metrics_path = Path(config.output_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    all_metrics.append(metrics)
    print(f"Done: {exp_name} | eval_loss={metrics['eval_loss']:.4f} | peak_vram_mb={metrics['peak_vram_mb']:.1f}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

len(all_metrics)
"""
        ),
        code_cell(
            """from qlora_scratch.analysis import build_method_comparison_table, results_summary_table

summary = results_summary_table(all_metrics)
summary
"""
        ),
        code_cell(
            """comparison = build_method_comparison_table(summary)
comparison
"""
        ),
        code_cell(
            """latest = all_metrics[-1]
for idx, (pre, post) in enumerate(zip(latest["pre_samples"], latest["post_samples"]), start=1):
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
        md_cell(
            "# 03 — Analysis & Visualization\n\nLoad all scratch experiment results and compare loss, VRAM, throughput, latency, and instruction-following outputs."
        ),
        code_cell(project_setup_cell()),
        code_cell(GIT_PULL_CELL),
        code_cell(INSTALL_CELL),
        md_cell("## 1. Load Results"),
        code_cell(
            """from qlora_scratch.analysis import (
    build_instruction_tuning_table,
    build_method_comparison_table,
    load_all_metrics,
    metrics_to_frame,
    plot_lora_rank_ablation,
    plot_throughput_comparison,
    plot_system_metrics,
    plot_training_curve,
    plot_training_loss,
    plot_training_time_comparison,
    plot_vram_comparison,
    results_summary_table,
)

results = load_all_metrics(RESULTS_DIR)
summary = results_summary_table(results)
print(f"Loaded {len(results)} experiment results")
summary
"""
        ),
        md_cell("## 2. Summary Table"),
        code_cell(
            """display_cols = [
    "experiment",
    "model",
    "lora_rank",
    "eval_loss",
    "perplexity",
    "peak_vram_mb",
    "peak_reserved_vram_mb",
    "tokens_per_second",
    "avg_generation_latency_s",
    "avg_generation_tokens_per_second",
    "wall_time_s",
]
summary[display_cols]
"""
        ),
        md_cell("## 3. LoRA vs QLoRA Comparison"),
        code_cell(
            """comparison = build_method_comparison_table(summary)
comparison
"""
        ),
        md_cell("## 4. Training Loss Curves"),
        code_cell(
            """fig = plot_training_loss(results, title="Scratch QLoRA Training Loss — All Runs")
fig
"""
        ),
        md_cell("## 5. VRAM Usage Comparison"),
        code_cell(
            """fig = plot_vram_comparison(summary)
fig
"""
        ),
        md_cell("## 6. Throughput Comparison"),
        code_cell(
            """fig = plot_throughput_comparison(summary)
fig
"""
        ),
        md_cell("## 7. Training Time Comparison"),
        code_cell(
            """fig = plot_training_time_comparison(summary)
fig
"""
        ),
        md_cell("## 8. LoRA Rank Ablation"),
        code_cell(
            """fig = plot_lora_rank_ablation(results)
fig
"""
        ),
        md_cell("## 9. Detailed System Metrics For One Run"),
        code_cell(
            """metrics = results[0]
metrics_to_frame(metrics)
"""
        ),
        code_cell(
            """fig = plot_system_metrics(metrics)
fig
"""
        ),
        md_cell("## 10. Pre/Post Instruction-Tuning Comparison"),
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


def build_unified_workflow() -> list[dict]:
    cells: list[dict] = [
        md_cell(
            "# 04 — Unified Workflow\n\nRun the complete scratch QLoRA project in one notebook while keeping the separate notebooks available for modular use."
        ),
        code_cell(project_setup_cell()),
        code_cell(GIT_PULL_CELL),
        code_cell(INSTALL_CELL),
        md_cell("## 1. Prepare Dataset"),
    ]
    cells.extend(build_data_prep()[4:])
    cells.append(md_cell("## 2. Run Paired LoRA And QLoRA Experiments"))
    cells.extend(build_experiments()[4:])
    cells.append(md_cell("## 3. Analyze Saved Results"))
    cells.extend(build_analysis()[4:])
    return cells


def build_chunked_workflow() -> list[dict]:
    cells: list[dict] = [
        md_cell(
            "# 05 — Chunked QLoRA Workflow\n\n"
            "This notebook mirrors `04_unified_workflow.ipynb` but adds a third method, "
            "`qlora_chunked`, which streams the NF4 base-weight matmul one row-chunk at a time "
            "via a custom autograd Function. The full fp16 weight is never materialized in "
            "forward and is not retained for backward, so peak VRAM should drop below plain LoRA "
            "instead of exceeding it.\n\n"
            "We re-run `lora`, `qlora`, and `qlora_chunked` on the same OASST1 splits and "
            "compare the three on VRAM, throughput, training time, and eval loss."
        ),
        code_cell(project_setup_cell()),
        code_cell(GIT_PULL_CELL),
        code_cell(INSTALL_CELL),
        md_cell("## 1. Prepare Dataset"),
    ]
    cells.extend(build_data_prep()[4:])

    cells.append(md_cell("## 2. Sanity-check Chunked Forward"))
    cells.append(
        code_cell(
            """import torch
from qlora_scratch.lora import LoRAConfig, ChunkedQuantizedLoRALinear, QuantizedLoRALinear

torch.manual_seed(0)
ref_linear = torch.nn.Linear(256, 384, bias=True).half()
cfg = LoRAConfig(rank=4, alpha=8, dropout=0.0, block_size=64, chunk_size=64)

dense_qlora = QuantizedLoRALinear(ref_linear, cfg)
chunked_qlora = ChunkedQuantizedLoRALinear(ref_linear, cfg)
chunked_qlora.load_state_dict(dense_qlora.state_dict())

x = torch.randn(2, 16, 256, dtype=torch.float16)
with torch.no_grad():
    y_dense = dense_qlora(x)
    y_chunk = chunked_qlora(x)

max_abs_err = (y_dense - y_chunk).abs().max().item()
print(f"Max |dense - chunked| forward error: {max_abs_err:.3e}")
assert max_abs_err < 5e-3, "Chunked forward diverges from dense forward"
print("Chunked forward matches dense forward (within fp16 tolerance).")
"""
        )
    )

    cells.append(md_cell("## 3. Run LoRA / QLoRA / Chunked-QLoRA"))
    cells.append(
        code_cell(
            """from qlora_scratch.train import ExperimentConfig

sample_prompts = [
    "### Instruction:\\nSummarize QLoRA in plain language for a student.\\n\\n### Response:\\n",
    "### Instruction:\\nWhat is the purpose of NF4 quantization in QLoRA?\\n\\n### Response:\\n",
    "### Instruction:\\nExplain why streaming the weight matmul in chunks reduces VRAM.\\n\\n### Response:\\n",
]

base_kwargs = dict(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_rank=8,
    lora_alpha=16,
    quant_block_size=64,
    max_train_samples=512,
    max_eval_samples=128,
)

experiments = {
    "scratch_lora_r8": ExperimentConfig(
        method="lora",
        output_dir=str(RESULTS_DIR / "chunked_run" / "scratch_lora_r8"),
        **base_kwargs,
    ),
    "scratch_qlora_r8": ExperimentConfig(
        method="qlora",
        output_dir=str(RESULTS_DIR / "chunked_run" / "scratch_qlora_r8"),
        **base_kwargs,
    ),
    "scratch_qlora_chunked_r8": ExperimentConfig(
        method="qlora_chunked",
        quant_chunk_size=128,
        output_dir=str(RESULTS_DIR / "chunked_run" / "scratch_qlora_chunked_r8"),
        **base_kwargs,
    ),
}

print("Experiments to run:")
for name, cfg in experiments.items():
    print(name, "method=", cfg.method, "rank=", cfg.lora_rank, "output=", cfg.output_dir)
"""
        )
    )
    cells.append(
        code_cell(
            """import gc
import json
from pathlib import Path
import torch
from qlora_scratch.train import run_experiment

all_metrics = []

for exp_name, config in experiments.items():
    print("\\n" + "#" * 70)
    print(f"Running {exp_name} (method={config.method})")
    print("#" * 70)
    metrics = run_experiment(PROJECT_ROOT / "data", config, sample_prompts=sample_prompts)
    metrics["experiment"] = exp_name
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    all_metrics.append(metrics)
    print(
        f"Done: {exp_name} | eval_loss={metrics['eval_loss']:.4f} "
        f"| peak_vram_mb={metrics['peak_vram_mb']:.1f} "
        f"| tokens/s={metrics['tokens_per_second']:.1f}"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

len(all_metrics)
"""
        )
    )

    cells.append(md_cell("## 4. Three-Way Summary Table"))
    cells.append(
        code_cell(
            """from qlora_scratch.analysis import results_summary_table

summary = results_summary_table(all_metrics)
display_cols = [
    "experiment",
    "method",
    "lora_rank",
    "eval_loss",
    "perplexity",
    "peak_vram_mb",
    "peak_reserved_vram_mb",
    "tokens_per_second",
    "wall_time_s",
]
summary[display_cols]
"""
        )
    )

    cells.append(md_cell("## 5. VRAM, Throughput, Training Time"))
    cells.append(
        code_cell(
            """from qlora_scratch.analysis import (
    plot_throughput_comparison,
    plot_training_time_comparison,
    plot_vram_comparison,
)

plot_vram_comparison(summary)
"""
        )
    )
    cells.append(code_cell("plot_throughput_comparison(summary)\n"))
    cells.append(code_cell("plot_training_time_comparison(summary)\n"))

    cells.append(md_cell("## 6. Training Loss Curves"))
    cells.append(
        code_cell(
            """from qlora_scratch.analysis import plot_training_loss

plot_training_loss(all_metrics, title="LoRA vs QLoRA vs Chunked QLoRA")
"""
        )
    )

    cells.append(md_cell("## 7. Peak VRAM Deltas (Expect Chunked < LoRA < QLoRA)"))
    cells.append(
        code_cell(
            """import pandas as pd

vram_pivot = summary.set_index("method")[["peak_vram_mb", "peak_reserved_vram_mb"]]
vram_pivot["vs_lora_pct"] = (
    (vram_pivot["peak_vram_mb"] - vram_pivot.loc["lora", "peak_vram_mb"])
    / vram_pivot.loc["lora", "peak_vram_mb"]
    * 100.0
)
vram_pivot.round(2)
"""
        )
    )

    cells.append(md_cell("## 8. Pre/Post Instruction-Tuning Samples (Chunked QLoRA Run)"))
    cells.append(
        code_cell(
            """from qlora_scratch.analysis import build_instruction_tuning_table

chunked_metrics = next(m for m in all_metrics if m["method"] == "qlora_chunked")
instruction_table = build_instruction_tuning_table(chunked_metrics)
instruction_table
"""
        )
    )
    cells.append(
        code_cell(
            """for row in instruction_table.to_dict(orient="records"):
    print(f"=== PROMPT {row['prompt_id']} ===")
    print(row["prompt"])
    print("\\n--- PRE-FINETUNE ---")
    print(row["pre_response"])
    print("\\n--- POST-FINETUNE ---")
    print(row["post_response"])
    print("\\n")
"""
        )
    )

    return cells


def main() -> None:
    old_path = NOTEBOOKS / "01_end_to_end_qlora_workflow.ipynb"
    if old_path.exists():
        old_path.unlink()
    write_notebook(NOTEBOOKS / "01_data_preparation.ipynb", build_data_prep())
    write_notebook(NOTEBOOKS / "02_run_experiments.ipynb", build_experiments())
    write_notebook(NOTEBOOKS / "03_analysis.ipynb", build_analysis())
    write_notebook(NOTEBOOKS / "04_unified_workflow.ipynb", build_unified_workflow())
    write_notebook(NOTEBOOKS / "05_chunked_qlora_workflow.ipynb", build_chunked_workflow())


if __name__ == "__main__":
    main()
