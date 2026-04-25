from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_metrics(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    return json.loads((results_dir / "metrics.json").read_text())


def metrics_to_frame(metrics: dict) -> pd.DataFrame:
    row = {
        "model": metrics["config"]["model_name"],
        "lora_rank": metrics["config"]["lora_rank"],
        "eval_loss": metrics["eval_loss"],
        "perplexity": metrics["perplexity"],
        "tokens_per_second": metrics["tokens_per_second"],
        "avg_generation_latency_s": metrics.get("avg_generation_latency_s", 0.0),
        "avg_generation_tokens_per_second": metrics.get("avg_generation_tokens_per_second", 0.0),
        "avg_optimizer_step_latency_s": metrics.get("avg_optimizer_step_latency_s", 0.0),
        "peak_vram_mb": metrics["peak_vram_mb"],
        "peak_reserved_vram_mb": metrics.get("peak_reserved_vram_mb", 0.0),
        "wall_time_s": metrics["wall_time_s"],
        "trainable_percent": metrics["trainable_percent"],
    }
    return pd.DataFrame([row])


def plot_training_curve(metrics: dict):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["train_loss_curve"], linewidth=2)
    ax.set_title("Scratch QLoRA Training Loss")
    ax.set_xlabel("Mini-batch step")
    ax.set_ylabel("Loss")
    return fig


def build_instruction_tuning_table(metrics: dict) -> pd.DataFrame:
    pre_samples = metrics.get("pre_samples", [])
    post_samples = metrics.get("post_samples", [])
    rows = []
    for idx, (pre, post) in enumerate(zip(pre_samples, post_samples), start=1):
        rows.append(
            {
                "prompt_id": idx,
                "prompt": post.get("prompt", pre.get("prompt", "")),
                "pre_response": pre.get("response", ""),
                "post_response": post.get("response", ""),
                "pre_latency_s": pre.get("latency_s", 0.0),
                "post_latency_s": post.get("latency_s", 0.0),
                "pre_tokens_per_second": pre.get("tokens_per_second", 0.0),
                "post_tokens_per_second": post.get("tokens_per_second", 0.0),
            }
        )
    return pd.DataFrame(rows)


def plot_system_metrics(metrics: dict):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    labels = ["Peak allocated VRAM", "Peak reserved VRAM"]
    values = [metrics["peak_vram_mb"], metrics.get("peak_reserved_vram_mb", 0.0)]
    axes[0].bar(labels, values, color=["#33658A", "#86BBD8"])
    axes[0].set_ylabel("MB")
    axes[0].set_title("GPU Memory Usage")
    axes[0].tick_params(axis="x", rotation=15)

    perf_labels = ["Train throughput", "Gen throughput", "Gen latency", "Step latency"]
    perf_values = [
        metrics["tokens_per_second"],
        metrics.get("avg_generation_tokens_per_second", 0.0),
        metrics.get("avg_generation_latency_s", 0.0),
        metrics.get("avg_optimizer_step_latency_s", 0.0),
    ]
    axes[1].bar(perf_labels, perf_values, color=["#758E4F", "#BC4B51", "#F4A259", "#5B8E7D"])
    axes[1].set_title("Throughput And Latency")
    axes[1].tick_params(axis="x", rotation=20)
    return fig
