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
        "peak_vram_mb": metrics["peak_vram_mb"],
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
