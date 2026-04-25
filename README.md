# QLoRA From Scratch for Colab

This repo is a Colab-first learning project for understanding QLoRA by rebuilding the key ideas with minimal abstractions.

The main scratch implementations are:

- `NF4` blockwise 4-bit quantization for frozen backbone linear weights
- `FP16` LoRA adapters trained on top of the quantized backbone
- a simple `PagedAdamW32bit` optimizer that keeps optimizer state on CPU and updates parameters page by page

The goal is not to reproduce every low-level optimization from `bitsandbytes`, but to make the mechanics readable and runnable in a Google Colab A100 environment.

## Repo Layout

- `src/qlora_scratch/`: scratch implementations and training helpers
- `notebooks/`: Colab-ready notebooks with discrete cells
- `scripts/make_notebooks.py`: regenerates the notebooks from Python cell templates
- `requirements.txt`: lightweight dependencies for Colab

## Suggested Colab Flow

1. Create a new Colab notebook session with GPU enabled.
2. Clone this repo into `/content`.
3. Open and run:
   - `notebooks/01_data_preparation.ipynb`
   - `notebooks/02_scratch_qlora_experiments.ipynb`
   - `notebooks/03_analysis.ipynb`

## Default Experiment Choices

- Dataset: `OpenAssistant/oasst1`
- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Metrics: train loss, eval loss, perplexity, tokens/sec, wall time, peak VRAM

These defaults are intentionally conservative for Colab while still being large enough to demonstrate QLoRA behavior clearly.

## Notes

- The scratch NF4 implementation uses a fixed 16-value NF4 codebook and blockwise absmax scaling.
- The paged optimizer is educational and intentionally transparent rather than highly optimized.
- The notebooks expose the config in plain dictionaries so you can edit experiments directly in Colab cells.
