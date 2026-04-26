from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_metrics(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    return json.loads((results_dir / "metrics.json").read_text())


def load_all_metrics(results_root: str | Path) -> list[dict]:
    results_root = Path(results_root)
    if (results_root / "metrics.json").exists():
        return [load_metrics(results_root)]

    metrics = []
    for metrics_file in sorted(results_root.glob("*/metrics.json")):
        payload = json.loads(metrics_file.read_text())
        payload.setdefault("experiment", metrics_file.parent.name)
        metrics.append(payload)
    return metrics


def metrics_to_frame(metrics: dict) -> pd.DataFrame:
    row = {
        "experiment": metrics.get("experiment", metrics["config"].get("output_dir", "scratch_qlora").split("/")[-1]),
        "method": metrics.get("method", metrics["config"].get("method", "qlora")),
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


def results_summary_table(results: list[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    frames = [metrics_to_frame(metrics) for metrics in results]
    summary = pd.concat(frames, ignore_index=True)
    return summary.sort_values(["model", "method", "lora_rank", "experiment"]).reset_index(drop=True)


def plot_training_curve(metrics: dict):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(metrics["train_loss_curve"], linewidth=2)
    ax.set_title("Scratch QLoRA Training Loss")
    ax.set_xlabel("Mini-batch step")
    ax.set_ylabel("Loss")
    return fig


def plot_training_loss(results: list[dict], title: str = "Training Loss Curves"):
    if not results:
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    for metrics in results:
        label = metrics.get("experiment", metrics["config"].get("output_dir", "run").split("/")[-1])
        ax.plot(metrics["train_loss_curve"], linewidth=2, label=label)
    ax.set_title(title)
    ax.set_xlabel("Mini-batch step")
    ax.set_ylabel("Loss")
    ax.legend()
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


def plot_vram_comparison(summary: pd.DataFrame):
    if summary.empty:
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="experiment", y="peak_vram_mb", hue="method", ax=ax)
    ax.set_title("Peak VRAM Usage")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("VRAM (MB)")
    ax.tick_params(axis="x", rotation=20)
    return fig


def plot_throughput_comparison(summary: pd.DataFrame):
    if summary.empty:
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="experiment", y="tokens_per_second", hue="method", ax=ax)
    ax.set_title("Training Throughput")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Tokens / second")
    ax.tick_params(axis="x", rotation=20)
    return fig


def plot_training_time_comparison(summary: pd.DataFrame):
    if summary.empty:
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=summary, x="experiment", y="wall_time_s", hue="method", ax=ax)
    ax.set_title("Training Time")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Seconds")
    ax.tick_params(axis="x", rotation=20)
    return fig


def plot_lora_rank_ablation(results: list[dict]):
    summary = results_summary_table(results)
    if summary.empty or summary["lora_rank"].nunique() <= 1:
        return None
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=summary, x="lora_rank", y="eval_loss", hue="method", style="model", marker="o", ax=ax)
    ax.set_title("LoRA Rank Ablation")
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("Eval loss")
    return fig


def build_method_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or summary["method"].nunique() < 2:
        return pd.DataFrame()

    index_cols = ["model", "lora_rank"]
    qlora = (
        summary[summary["method"] == "qlora"]
        .set_index(index_cols)[["eval_loss", "peak_vram_mb", "tokens_per_second", "wall_time_s"]]
        .add_prefix("qlora_")
    )
    lora = (
        summary[summary["method"] == "lora"]
        .set_index(index_cols)[["eval_loss", "peak_vram_mb", "tokens_per_second", "wall_time_s"]]
        .add_prefix("lora_")
    )
    comparison = qlora.join(lora, how="inner").reset_index()
    if comparison.empty:
        return comparison

    comparison["vram_savings_pct"] = (
        (comparison["lora_peak_vram_mb"] - comparison["qlora_peak_vram_mb"])
        / comparison["lora_peak_vram_mb"]
        * 100.0
    )
    comparison["throughput_delta_pct"] = (
        (comparison["qlora_tokens_per_second"] - comparison["lora_tokens_per_second"])
        / comparison["lora_tokens_per_second"]
        * 100.0
    )
    comparison["time_delta_pct"] = (
        (comparison["qlora_wall_time_s"] - comparison["lora_wall_time_s"])
        / comparison["lora_wall_time_s"]
        * 100.0
    )
    comparison["eval_loss_delta"] = comparison["qlora_eval_loss"] - comparison["lora_eval_loss"]
    return comparison
