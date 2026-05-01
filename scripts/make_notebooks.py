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


def build_triton_workflow() -> list[dict]:
    cells: list[dict] = [
        md_cell(
            "# 06 — CUDA-Optimized QLoRA (Triton-Fused Kernel)\n\n"
            "Phase 3 of the project. We add a custom Triton kernel that fuses NF4 "
            "dequantization with the matmul: each output tile is computed by streaming "
            "NF4 codes + per-block scales straight into the GPU's tensor cores, never "
            "materializing the full fp16 weight in HBM and never saving it for backward. "
            "This is conceptually what `bitsandbytes`' Linear4bit does in hand-tuned C++; "
            "Triton lets us write the same idea from Python.\n\n"
            "We compare four methods at LoRA ranks 8, 16, 32:\n"
            "* **`lora`** — fp16 base weight, fp16 LoRA adapters (memory-heavy baseline).\n"
            "* **`qlora`** — phase 1 scratch QLoRA (NF4 storage but full dequant per forward).\n"
            "* **`qlora_chunked`** — phase 2 chunked Python recompute (memory-efficient, slow).\n"
            "* **`qlora_triton`** — phase 3 fused CUDA kernel (memory-efficient *and* fast).\n\n"
            "Expected outcome: `qlora_triton` peak VRAM stays below `lora`, and "
            "throughput recovers most of the gap that the chunked path lost."
        ),
        code_cell(project_setup_cell()),
        code_cell(GIT_PULL_CELL),
        code_cell(INSTALL_CELL),
        code_cell(
            """import subprocess, sys
# Triton ships with PyTorch on Colab GPU runtimes, but install if missing.
try:
    import triton  # noqa: F401
    print(f"triton already available: {triton.__version__}")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "triton"])
    import triton
    print(f"installed triton: {triton.__version__}")
"""
        ),
        md_cell("## 1. Probe Triton Availability"),
        code_cell(
            """import torch
from qlora_scratch import TRITON_AVAILABLE

print(f"TRITON_AVAILABLE: {TRITON_AVAILABLE}")
print(f"CUDA available:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
assert TRITON_AVAILABLE and torch.cuda.is_available(), (
    "This notebook requires Triton + CUDA. Switch the Colab runtime to a GPU."
)
"""
        ),
        md_cell(
            "## 2. Sanity Check: Triton Forward + Backward vs Dense\n\n"
            "Before running the long experiments, verify the Triton kernels match the "
            "dense reference implementation to within fp16 tolerance, on both forward "
            "outputs and gradients. If this cell fails, the kernel is buggy and the "
            "experiments will produce garbage."
        ),
        code_cell(
            """import torch
from qlora_scratch.lora import (
    LoRAConfig,
    QuantizedLoRALinear,
    TritonQuantizedLoRALinear,
)

torch.manual_seed(0)
device = torch.device("cuda")

# Test on dimensions divisible by NF4 block_size = 64.
ref = torch.nn.Linear(512, 768, bias=True).to(device).half()
cfg = LoRAConfig(rank=8, alpha=16, dropout=0.0, block_size=64, chunk_size=128)

dense = QuantizedLoRALinear(ref, cfg).to(device)
triton_layer = TritonQuantizedLoRALinear(ref, cfg).to(device)
triton_layer.load_state_dict(dense.state_dict())

print(f"Using Triton kernel: {triton_layer.using_triton_kernel}")

x  = torch.randn(4, 32, 512, dtype=torch.float16, device=device, requires_grad=True)
x2 = x.detach().clone().requires_grad_(True)

y_dense  = dense(x)
y_triton = triton_layer(x2)
fwd_err = (y_dense - y_triton).abs().max().item()
print(f"forward max |err|: {fwd_err:.3e}")

g = torch.randn_like(y_dense)
y_dense.backward(g)
y_triton.backward(g)
grad_x_err = (x.grad - x2.grad).abs().max().item()
grad_A_err = (dense.lora_A.grad - triton_layer.lora_A.grad).abs().max().item()
grad_B_err = (dense.lora_B.grad - triton_layer.lora_B.grad).abs().max().item()

print(f"grad_x   max |err|: {grad_x_err:.3e}")
print(f"grad_A   max |err|: {grad_A_err:.3e}")
print(f"grad_B   max |err|: {grad_B_err:.3e}")

assert fwd_err     < 1e-2,  "Triton forward diverges from dense"
assert grad_x_err  < 1e-2,  "Triton backward grad_x diverges from dense"
assert grad_A_err  < 1e-2,  "Triton lora_A gradient mismatch"
assert grad_B_err  < 1e-2,  "Triton lora_B gradient mismatch"
print("Triton kernel matches the dense reference within fp16 tolerance.")
"""
        ),
        md_cell("## 3. Prepare Dataset"),
    ]
    cells.extend(build_data_prep()[4:])

    cells.append(md_cell("## 4. Build the 4-Method × 3-Rank Experiment Grid"))
    cells.append(
        code_cell(
            """from qlora_scratch.train import ExperimentConfig

sample_prompts = [
    "### Instruction:\\nSummarize QLoRA in plain language for a student.\\n\\n### Response:\\n",
    "### Instruction:\\nWhat is the purpose of NF4 quantization in QLoRA?\\n\\n### Response:\\n",
    "### Instruction:\\nExplain how a fused dequant-matmul kernel reduces VRAM.\\n\\n### Response:\\n",
]

ranks   = [8, 16, 32]
methods = ["lora", "qlora", "qlora_chunked", "qlora_triton"]

base_kwargs = dict(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    quant_block_size=64,
    max_train_samples=512,
    max_eval_samples=128,
)

experiments = {}
for rank in ranks:
    for method in methods:
        name = f"scratch_{method}_r{rank}"
        kwargs = dict(
            method=method,
            lora_rank=rank,
            lora_alpha=rank * 2,
            output_dir=str(RESULTS_DIR / "triton_run" / name),
            **base_kwargs,
        )
        if method in ("qlora_chunked", "qlora_triton"):
            kwargs["quant_chunk_size"] = 128
        experiments[name] = ExperimentConfig(**kwargs)

print(f"Total experiments: {len(experiments)}")
for name, cfg in experiments.items():
    print(f"  {name:36s} method={cfg.method:14s} rank={cfg.lora_rank}")
"""
        )
    )

    cells.append(md_cell("## 5. Run Experiments"))
    cells.append(
        code_cell(
            """import gc, json
from pathlib import Path
import torch
from qlora_scratch.train import run_experiment

all_metrics = []
for exp_name, config in experiments.items():
    print("\\n" + "#" * 78)
    print(f"# {exp_name}  (method={config.method}, rank={config.lora_rank})")
    print("#" * 78)
    metrics = run_experiment(PROJECT_ROOT / "data", config, sample_prompts=sample_prompts)
    metrics["experiment"] = exp_name
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    all_metrics.append(metrics)
    print(
        f"  -> eval_loss={metrics['eval_loss']:.4f}  "
        f"peak_vram={metrics['peak_vram_mb']:.1f} MB  "
        f"tokens/s={metrics['tokens_per_second']:.1f}  "
        f"wall={metrics['wall_time_s']:.1f}s"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\\nFinished {len(all_metrics)} experiments.")
"""
        )
    )

    cells.append(md_cell("## 6. Full Metrics Table"))
    cells.append(
        code_cell(
            """import pandas as pd
from qlora_scratch.analysis import results_summary_table

summary = results_summary_table(all_metrics)
display_cols = [
    "experiment", "method", "lora_rank",
    "eval_loss", "perplexity",
    "peak_vram_mb", "peak_reserved_vram_mb",
    "tokens_per_second", "avg_generation_tokens_per_second",
    "avg_optimizer_step_latency_s", "wall_time_s",
]
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
summary[display_cols].sort_values(["lora_rank", "method"]).reset_index(drop=True)
"""
        )
    )

    cells.append(md_cell("## 7. Plot Style"))
    cells.append(
        code_cell(
            """import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

METHOD_ORDER = ["lora", "qlora", "qlora_chunked", "qlora_triton"]
METHOD_LABELS = {
    "lora":          "LoRA",
    "qlora":         "QLoRA (dense dequant)",
    "qlora_chunked": "QLoRA (chunked)",
    "qlora_triton":  "QLoRA (Triton fused)",
}
METHOD_COLORS = {
    "lora":          "#33658A",
    "qlora":         "#BC4B51",
    "qlora_chunked": "#5B8E7D",
    "qlora_triton":  "#F4A259",
}
RANK_LINESTYLES = {8: "solid", 16: (0, (4, 2)), 32: (0, (1, 1.5))}

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelweight": "semibold",
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 110,
})

def grouped_bar(summary, metric, ylabel, title, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ranks = sorted(summary["lora_rank"].unique())
    width = 0.8 / len(METHOD_ORDER)
    x = np.arange(len(ranks))
    for i, method in enumerate(METHOD_ORDER):
        sub = summary[summary["method"] == method].set_index("lora_rank").reindex(ranks)
        values = sub[metric].values
        offset = (i - (len(METHOD_ORDER) - 1) / 2) * width
        bars = ax.bar(
            x + offset, values, width,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            edgecolor="white", linewidth=0.8,
        )
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.annotate(
                    fmt.format(val),
                    xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5,
                )
    ax.set_xticks(x)
    ax.set_xticklabels([f"r = {r}" for r in ranks])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Method", loc="best")
    ax.margins(y=0.18)
    fig.tight_layout()
    return fig

print("Style configured.")
"""
        )
    )

    cells.append(md_cell("## 8. Peak VRAM (Headline Plot)"))
    cells.append(
        code_cell(
            """grouped_bar(
    summary, "peak_vram_mb",
    ylabel="Peak allocated VRAM (MB)",
    title="Peak VRAM by Method × LoRA Rank",
)
"""
        )
    )

    cells.append(md_cell("## 9. Training Throughput"))
    cells.append(
        code_cell(
            """grouped_bar(
    summary, "tokens_per_second",
    ylabel="Tokens / second",
    title="Training Throughput by Method × LoRA Rank",
)
"""
        )
    )

    cells.append(md_cell("## 10. Wall-Clock Training Time"))
    cells.append(
        code_cell(
            """grouped_bar(
    summary, "wall_time_s",
    ylabel="Wall-clock time (s)",
    title="Total Training Time by Method × LoRA Rank",
)
"""
        )
    )

    cells.append(md_cell("## 11. Eval Loss & Perplexity vs Rank"))
    cells.append(
        code_cell(
            """fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
for metric, label, ax in [("eval_loss", "Eval loss", axes[0]),
                          ("perplexity", "Perplexity", axes[1])]:
    for method in METHOD_ORDER:
        sub = summary[summary["method"] == method].sort_values("lora_rank")
        ax.plot(
            sub["lora_rank"], sub[metric],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linewidth=2.2, marker="o", markersize=8,
            markerfacecolor="white", markeredgewidth=2,
        )
    ax.set_xticks(sorted(summary["lora_rank"].unique()))
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel(label)
    ax.set_title(f"{label} vs LoRA rank")
    ax.legend(title="Method", loc="best")
fig.tight_layout()
fig
"""
        )
    )

    cells.append(md_cell("## 12. Training Loss Curves (color = method, linestyle = rank)"))
    cells.append(
        code_cell(
            """def smooth(values, window=15):
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")

fig, ax = plt.subplots(figsize=(12, 5.8))
for metrics in all_metrics:
    method = metrics["config"]["method"]
    rank   = metrics["config"]["lora_rank"]
    curve  = smooth(metrics["train_loss_curve"], window=15)
    ax.plot(
        np.arange(len(curve)), curve,
        color=METHOD_COLORS[method],
        linestyle=RANK_LINESTYLES[rank],
        linewidth=2.0, alpha=0.9,
        label=f"{METHOD_LABELS[method]} \\u00b7 r={rank}",
    )
ax.set_xlabel("Mini-batch step (rolling mean, window = 15)")
ax.set_ylabel("Train loss")
ax.set_title("Training Loss Curves — All Methods × Ranks")
ax.legend(ncol=2, loc="upper right", fontsize=8.5,
          title="Method \\u00b7 rank", title_fontsize=9,
          columnspacing=1.0, handlelength=2.6)
fig.tight_layout()
fig
"""
        )
    )

    cells.append(md_cell("## 13. VRAM and Throughput Deltas vs LoRA"))
    cells.append(
        code_cell(
            """import pandas as pd

vram = summary.pivot(index="lora_rank", columns="method", values="peak_vram_mb")
tps  = summary.pivot(index="lora_rank", columns="method", values="tokens_per_second")

vram_delta = vram.sub(vram["lora"], axis=0).div(vram["lora"], axis=0).mul(100).round(2)
tps_delta  = tps.sub(tps["lora"], axis=0).div(tps["lora"], axis=0).mul(100).round(2)

vram_delta.columns = [f"{c} VRAM \\u0394 vs LoRA (%)" for c in vram_delta.columns]
tps_delta.columns  = [f"{c} TPS \\u0394 vs LoRA (%)"  for c in tps_delta.columns]

print("Peak VRAM delta vs LoRA (negative = lower memory):")
print(vram_delta.to_string())
print("\\nTraining throughput delta vs LoRA (negative = slower):")
print(tps_delta.to_string())
"""
        )
    )

    cells.append(md_cell("## 14. Pre/Post Instruction-Tuning Samples (Triton Run)"))
    cells.append(
        code_cell(
            """from qlora_scratch.analysis import build_instruction_tuning_table

triton_metrics = next(
    m for m in all_metrics
    if m["method"] == "qlora_triton" and m["config"]["lora_rank"] == 8
)
instruction_table = build_instruction_tuning_table(triton_metrics)
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
    print()
"""
        )
    )

    return cells


def build_autotuned_triton_workflow() -> list[dict]:
    cells: list[dict] = [
        md_cell(
            "# 07 — Autotuned Triton QLoRA (Rank 8)\n\n"
            "Phase 4 of the project. Same NF4-fused Triton kernels as notebook 06, "
            "but wrapped in `@triton.autotune` over a small grid of "
            "`(BLOCK_M, BLOCK_N, num_warps, num_stages)` configs. Triton benchmarks "
            "every config the first time it sees a new `(M, N, K)` shape and caches "
            "the winner — the same trick cuBLAS uses internally and the lowest-effort "
            "way to close the throughput gap to plain LoRA.\n\n"
            "**Scope:** rank 8 only, five methods. We compare against the static-tile "
            "Triton kernel from notebook 06 directly. Extend the experiment grid (rank "
            "16, 32) once the rank-8 numbers look right.\n\n"
            "**Important:** Triton autotune pays a one-time cost the first time each "
            "matmul shape appears. For TinyLlama there are roughly 7 distinct linear "
            "shapes (q/k/v/o, gate, up, down). The notebook runs a short *warmup* pass "
            "before the timed training so that all shapes are tuned and cached, "
            "otherwise the first epoch's tokens/sec is polluted by tuning overhead."
        ),
        code_cell(project_setup_cell()),
        code_cell(GIT_PULL_CELL),
        code_cell(INSTALL_CELL),
        code_cell(
            """import subprocess, sys
try:
    import triton  # noqa: F401
    print(f"triton already available: {triton.__version__}")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "triton"])
    import triton
    print(f"installed triton: {triton.__version__}")
"""
        ),
        md_cell("## 1. Probe Triton + CUDA"),
        code_cell(
            """import torch
from qlora_scratch import TRITON_AVAILABLE

print(f"TRITON_AVAILABLE: {TRITON_AVAILABLE}")
print(f"CUDA available:   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
assert TRITON_AVAILABLE and torch.cuda.is_available(), (
    "This notebook requires Triton + CUDA. Switch the Colab runtime to a GPU."
)
"""
        ),
        md_cell(
            "## 2. Sanity Check: Autotuned Triton vs Dense\n\n"
            "Verify the autotuned kernels produce the same outputs and gradients as "
            "the dense reference. The first call benchmarks all configs and may take "
            "a few seconds — that's the autotune doing its job."
        ),
        code_cell(
            """import torch
from qlora_scratch.lora import (
    LoRAConfig,
    QuantizedLoRALinear,
    AutotunedTritonQuantizedLoRALinear,
)

torch.manual_seed(0)
device = torch.device("cuda")

ref = torch.nn.Linear(512, 768, bias=True).to(device).half()
cfg = LoRAConfig(rank=8, alpha=16, dropout=0.0, block_size=64, chunk_size=128)

dense = QuantizedLoRALinear(ref, cfg).to(device)
auto  = AutotunedTritonQuantizedLoRALinear(ref, cfg).to(device)
auto.load_state_dict(dense.state_dict())

print(f"Using Triton kernel: {auto.using_triton_kernel}")

x  = torch.randn(4, 32, 512, dtype=torch.float16, device=device, requires_grad=True)
x2 = x.detach().clone().requires_grad_(True)

y_dense = dense(x)
y_auto  = auto(x2)
fwd_err = (y_dense - y_auto).abs().max().item()
print(f"forward max |err|: {fwd_err:.3e}")

g = torch.randn_like(y_dense)
y_dense.backward(g); y_auto.backward(g)
gx_err = (x.grad - x2.grad).abs().max().item()
gA_err = (dense.lora_A.grad - auto.lora_A.grad).abs().max().item()
gB_err = (dense.lora_B.grad - auto.lora_B.grad).abs().max().item()
print(f"grad_x max |err|: {gx_err:.3e}")
print(f"grad_A max |err|: {gA_err:.3e}")
print(f"grad_B max |err|: {gB_err:.3e}")
assert fwd_err < 1e-2 and gx_err < 1e-2 and gA_err < 1e-2 and gB_err < 1e-2
print("Autotuned Triton kernel matches the dense reference within fp16 tolerance.")
"""
        ),
        md_cell("## 3. Prepare Dataset"),
    ]
    cells.extend(build_data_prep()[4:])

    cells.append(md_cell(
        "## 4. Autotune Warmup\n\n"
        "Run a few forward + backward passes on the actual TinyLlama shapes so Triton "
        "can pick the best config for each linear layer **before** the timed training "
        "loop starts. Without this, the first epoch's wall time would include kernel "
        "benchmarking and you'd underestimate steady-state throughput."
    ))
    cells.append(
        code_cell(
            """import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qlora_scratch.lora import LoRAConfig, prepare_model_for_autotuned_triton_kbit_training

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading model for warmup...")
warmup_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

cfg = LoRAConfig(rank=8, alpha=16, dropout=0.05, block_size=64, chunk_size=128)
warmup_model = prepare_model_for_autotuned_triton_kbit_training(warmup_model, cfg)
warmup_model.to("cuda")

batch = tokenizer(
    ["Warmup pass for Triton autotune"] * 2,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=256,
).to("cuda")
batch["labels"] = batch["input_ids"]

warmup_model.train()
print("Running 3 warmup steps to trigger autotune for all matmul shapes...")
for i in range(3):
    out = warmup_model(**batch)
    out.loss.backward()
    warmup_model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    print(f"  warmup step {i+1}: loss={out.loss.item():.3f}")

del warmup_model
import gc; gc.collect(); torch.cuda.empty_cache()
print("Autotune cache populated.")
"""
        )
    )

    cells.append(md_cell("## 5. Build Rank-8 Experiment Grid (5 Methods)"))
    cells.append(
        code_cell(
            """from qlora_scratch.train import ExperimentConfig

sample_prompts = [
    "### Instruction:\\nSummarize QLoRA in plain language for a student.\\n\\n### Response:\\n",
    "### Instruction:\\nWhat is the purpose of NF4 quantization in QLoRA?\\n\\n### Response:\\n",
    "### Instruction:\\nExplain how Triton autotune picks tile sizes per matmul.\\n\\n### Response:\\n",
]

methods = ["lora", "qlora", "qlora_chunked", "qlora_triton", "qlora_triton_autotuned"]

base_kwargs = dict(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_rank=8,
    lora_alpha=16,
    quant_block_size=64,
    max_train_samples=512,
    max_eval_samples=128,
)

experiments = {}
for method in methods:
    name = f"scratch_{method}_r8"
    kwargs = dict(
        method=method,
        output_dir=str(RESULTS_DIR / "autotuned_run" / name),
        **base_kwargs,
    )
    if method in ("qlora_chunked", "qlora_triton", "qlora_triton_autotuned"):
        kwargs["quant_chunk_size"] = 128
    experiments[name] = ExperimentConfig(**kwargs)

print(f"Total experiments: {len(experiments)}")
for name, cfg in experiments.items():
    print(f"  {name:42s} method={cfg.method}")
"""
        )
    )

    cells.append(md_cell("## 6. Run Experiments"))
    cells.append(
        code_cell(
            """import gc, json
from pathlib import Path
import torch
from qlora_scratch.train import run_experiment

all_metrics = []
for exp_name, config in experiments.items():
    print("\\n" + "#" * 78)
    print(f"# {exp_name}  (method={config.method})")
    print("#" * 78)
    metrics = run_experiment(PROJECT_ROOT / "data", config, sample_prompts=sample_prompts)
    metrics["experiment"] = exp_name
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(config.output_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    all_metrics.append(metrics)
    print(
        f"  -> eval_loss={metrics['eval_loss']:.4f}  "
        f"peak_vram={metrics['peak_vram_mb']:.1f} MB  "
        f"tokens/s={metrics['tokens_per_second']:.1f}  "
        f"wall={metrics['wall_time_s']:.1f}s"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\\nFinished {len(all_metrics)} experiments.")
"""
        )
    )

    cells.append(md_cell("## 7. Full Metrics Table"))
    cells.append(
        code_cell(
            """import pandas as pd
from qlora_scratch.analysis import results_summary_table

summary = results_summary_table(all_metrics)
display_cols = [
    "experiment", "method",
    "eval_loss", "perplexity",
    "peak_vram_mb", "peak_reserved_vram_mb",
    "tokens_per_second", "avg_generation_tokens_per_second",
    "avg_optimizer_step_latency_s", "wall_time_s",
]
pd.set_option("display.float_format", lambda v: f"{v:.3f}")
summary[display_cols].sort_values("method").reset_index(drop=True)
"""
        )
    )

    cells.append(md_cell("## 8. Plot Style"))
    cells.append(
        code_cell(
            """import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

METHOD_ORDER = ["lora", "qlora", "qlora_chunked", "qlora_triton", "qlora_triton_autotuned"]
METHOD_LABELS = {
    "lora":                   "LoRA",
    "qlora":                  "QLoRA (dense dequant)",
    "qlora_chunked":          "QLoRA (chunked)",
    "qlora_triton":           "QLoRA (Triton fused)",
    "qlora_triton_autotuned": "QLoRA (Triton autotuned)",
}
METHOD_COLORS = {
    "lora":                   "#33658A",
    "qlora":                  "#BC4B51",
    "qlora_chunked":          "#5B8E7D",
    "qlora_triton":           "#F4A259",
    "qlora_triton_autotuned": "#7F5AA8",
}

sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelweight": "semibold",
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 110,
})

def single_rank_bar(summary, metric, ylabel, title, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    sub = summary.set_index("method").reindex(METHOD_ORDER)
    values = sub[metric].values
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    bars = ax.bar(
        np.arange(len(METHOD_ORDER)), values,
        color=colors, edgecolor="white", linewidth=0.8,
    )
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.annotate(
                fmt.format(val),
                xy=(bar.get_x() + bar.get_width() / 2, val),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )
    ax.set_xticks(np.arange(len(METHOD_ORDER)))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[m]) for m in METHOD_ORDER]
    ax.legend(handles, labels, title="Method", loc="best", fontsize=9)
    ax.margins(y=0.18)
    fig.tight_layout()
    return fig

print("Style configured.")
"""
        )
    )

    cells.append(md_cell("## 9. Peak VRAM"))
    cells.append(
        code_cell(
            """single_rank_bar(
    summary, "peak_vram_mb",
    ylabel="Peak allocated VRAM (MB)",
    title="Peak VRAM at LoRA rank 8",
)
"""
        )
    )

    cells.append(md_cell("## 10. Training Throughput"))
    cells.append(
        code_cell(
            """single_rank_bar(
    summary, "tokens_per_second",
    ylabel="Tokens / second",
    title="Training Throughput at LoRA rank 8",
)
"""
        )
    )

    cells.append(md_cell("## 11. Wall-Clock Training Time"))
    cells.append(
        code_cell(
            """single_rank_bar(
    summary, "wall_time_s",
    ylabel="Wall-clock time (s)",
    title="Total Training Time at LoRA rank 8",
)
"""
        )
    )

    cells.append(md_cell("## 12. Training Loss Curves (5 methods)"))
    cells.append(
        code_cell(
            """def smooth(values, window=15):
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")

fig, ax = plt.subplots(figsize=(11.5, 5.5))
for metrics in all_metrics:
    method = metrics["config"]["method"]
    curve  = smooth(metrics["train_loss_curve"], window=15)
    ax.plot(
        np.arange(len(curve)), curve,
        color=METHOD_COLORS[method],
        linewidth=2.0, alpha=0.92,
        label=METHOD_LABELS[method],
    )
ax.set_xlabel("Mini-batch step (rolling mean, window = 15)")
ax.set_ylabel("Train loss")
ax.set_title("Training Loss Curves at LoRA rank 8")
ax.legend(title="Method", loc="upper right", fontsize=9)
fig.tight_layout()
fig
"""
        )
    )

    cells.append(md_cell("## 13. Deltas vs Static-Tile Triton (Apples-to-Apples)"))
    cells.append(
        code_cell(
            """import pandas as pd

def normalize_total_tokens(metrics_list):
    rows = []
    for m in metrics_list:
        wall = m["wall_time_s"]
        tps  = m["tokens_per_second"]
        rows.append({
            "method": m["method"],
            "wall_time_s": wall,
            "tokens_per_second": tps,
            "implied_total_tokens": wall * tps,
            "peak_vram_mb": m["peak_vram_mb"],
            "eval_loss": m["eval_loss"],
        })
    return pd.DataFrame(rows).set_index("method").reindex(METHOD_ORDER)

table = normalize_total_tokens(all_metrics)
print("Per-method snapshot:")
print(table.round(2).to_string())

base_method = "qlora_triton"
if base_method in table.index:
    base_tps  = table.loc[base_method, "tokens_per_second"]
    base_vram = table.loc[base_method, "peak_vram_mb"]
    deltas = pd.DataFrame({
        "tps_delta_vs_static_triton_pct":  ((table["tokens_per_second"] - base_tps) / base_tps * 100).round(2),
        "vram_delta_vs_static_triton_pct": ((table["peak_vram_mb"] - base_vram) / base_vram * 100).round(2),
    })
    print(f"\\nDeltas vs {base_method!r} (autotune target):")
    print(deltas.to_string())
"""
        )
    )

    cells.append(md_cell("## 14. Pre/Post Instruction-Tuning Samples (Autotuned Run)"))
    cells.append(
        code_cell(
            """from qlora_scratch.analysis import build_instruction_tuning_table

auto_metrics = next(m for m in all_metrics if m["method"] == "qlora_triton_autotuned")
instruction_table = build_instruction_tuning_table(auto_metrics)
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
    print()
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
    write_notebook(NOTEBOOKS / "06_cuda_optimized_qlora.ipynb", build_triton_workflow())
    write_notebook(NOTEBOOKS / "07_autotuned_triton_qlora.ipynb", build_autotuned_triton_workflow())


if __name__ == "__main__":
    main()
