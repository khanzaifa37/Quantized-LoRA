# QLoRA From Scratch for Colab

This repo is a Colab-first learning project for understanding QLoRA by rebuilding the key ideas with minimal abstractions.

The main scratch implementations are:

- `NF4` blockwise 4-bit quantization for frozen backbone linear weights
- `FP16` LoRA adapters trained on top of the quantized backbone
- a simple `PagedAdamW32bit` optimizer that keeps optimizer state on CPU and updates parameters page by page
- pre/post fine-tuning instruction-following samples for qualitative comparison
- analysis of VRAM, throughput, and latency from the scratch training run

The goal is not to reproduce every low-level optimization from `bitsandbytes`, but to make the mechanics readable and runnable in a Google Colab A100 environment.

## Repo Layout

- `src/qlora_scratch/`: scratch implementations and training helpers
- `notebooks/`: Colab-ready notebooks with discrete cells
- `scripts/make_notebooks.py`: regenerates the notebooks from Python cell templates
- `requirements.txt`: lightweight dependencies for Colab

## Suggested Colab Flow

1. Create a new Colab notebook session with GPU enabled.
2. Clone this repo into `/content`.
3. Choose either workflow:
   - modular notebooks:
     - `notebooks/01_data_preparation.ipynb`
     - `notebooks/02_run_experiments.ipynb`
     - `notebooks/03_analysis.ipynb`
   - single notebook:
     - `notebooks/04_unified_workflow.ipynb`

## Run In Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook.
3. In `Runtime -> Change runtime type`, select:
   - `T4 GPU` for lighter smoke tests, or
   - `A100 GPU` for the main experiments.
4. In the first cell, clone the repo:

```python
!git clone https://github.com/khanzaifa37/Quantized-LoRA.git /content/qLoRA
```

5. Change into the repo:

```python
%cd /content/qLoRA
```

6. Open the notebook(s) from the Colab file browser:
   - modular flow:
     - `/content/qLoRA/notebooks/01_data_preparation.ipynb`
     - `/content/qLoRA/notebooks/02_run_experiments.ipynb`
     - `/content/qLoRA/notebooks/03_analysis.ipynb`
   - unified flow:
     - `/content/qLoRA/notebooks/04_unified_workflow.ipynb`

7. Run either:
   - the three modular notebooks in order, or
   - the unified notebook from top to bottom

8. Both flows will:
   - prepare the `OpenAssistant/oasst1` dataset
   - run paired scratch LoRA and scratch QLoRA experiments
   - capture instruction-tuning generations before and after fine-tuning
   - summarize VRAM, throughput, latency, and qualitative outputs

9. If you want to rerun with different settings, edit the `ExperimentConfig` cell in `02_run_experiments.ipynb` or `04_unified_workflow.ipynb`. The most useful values to change first are:
   - `model_name`
   - `max_train_samples`
   - `epochs`
   - `lora_rank`
   - `quant_block_size`
   - `optimizer_page_size`

10. Your outputs will be written inside the cloned repo:
   - data cache: `/content/qLoRA/data`
   - experiment outputs: `/content/qLoRA/results`

11. If you want to save outputs permanently, either:
   - copy the `results/` folder to Google Drive, or
   - commit the result artifacts to a separate branch if they are small enough.

### Minimal Colab Bootstrap

If you want a single setup cell before opening the notebooks, use:

```python
!git clone https://github.com/khanzaifa37/Quantized-LoRA.git /content/qLoRA
%cd /content/qLoRA
!python scripts/make_notebooks.py
```

## Default Experiment Choices

- Dataset: `OpenAssistant/oasst1`
- Base model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Metrics: train loss, eval loss, perplexity, tokens/sec, wall time, peak VRAM

These defaults are intentionally conservative for Colab while still being large enough to demonstrate QLoRA behavior clearly.

## Notes

- The scratch NF4 implementation uses a fixed 16-value NF4 codebook and blockwise absmax scaling.
- The paged optimizer is educational and intentionally transparent rather than highly optimized.
- The notebooks expose the config in plain dictionaries so you can edit experiments directly in Colab cells.
