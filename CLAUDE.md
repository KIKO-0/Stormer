# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
conda env create -f env.yml   # creates "tnp" env (Python 3.11, CUDA 11.8)
conda activate tnp
pip install -e .
```

Key deps: PyTorch 2.1.0, PyTorch Lightning 2.2.1, xformers, flash-attn 2.3.3, timm 0.9.2, wandb.

## Training

```bash
# Pretrain (one-step loss)
python train.py --config configs/pretrain_one_step.yaml \
    --trainer.default_root_dir [EXP_DIR] \
    --model.net.patch_size 4 \
    --data.root_dir [H5DF_DIR] --data.steps 1 --data.batch_size 4

# Finetune (multi-step loss)
python train.py --config configs/finetune_multi_step.yaml \
    --trainer.default_root_dir [EXP_DIR] \
    --model.net.patch_size 4 \
    --model.pretrained_path [CKPT] \
    --data.root_dir [H5DF_DIR] --data.steps 4 --data.batch_size 4
```

Uses `LightningCLI` with omegaconf. Auto-resumes from `last.ckpt`. Logs to WandB.

## Data Preprocessing Pipeline

1. **Download ERA5** via `stormer/data_preprocessing/download_wb2.py` (WeatherBench2 zarr)
2. **Regrid** to 1.40625° via `stormer/data_preprocessing/regrid_wb2.py`
3. **Convert to HDF5** via `stormer/data_preprocessing/process_one_step_data.py` → `wb2_h5df/{train,val,test}/{year}_{idx}.h5`
4. **Compute normalization constants** via `stormer/data_preprocessing/compute_normalization.py` (pre-computed 1979–2018 stats included in `normalization_constants/`)

Each `.h5` file contains all 69 variables for one 6-hour timestep. The data directory must contain `normalize_mean.npz`, `normalize_std.npz`, and `normalize_diff_std_{6,12,18,24}.npz`.

## Architecture

### Stormer (`stormer/models/hub/stormer.py`)
Vision Transformer with **adaLN-Zero** conditioning on forecast interval:
- `WeatherEmbedding`: per-variable conv patch embedder + per-variable channel embeddings + 2D sinusoidal pos embeddings
- `TimestepEmbedder`: maps scalar interval (hours / 10.0) → conditioning vector `c`
- N× `Block`: transformer with adaLN-Zero shift/scale/gate applied to both attention and MLP
- `FinalLayer`: adaLN-modulated linear head → `patch_size² × num_variables` output, unpatchified to `(B, V, H, W)`
- Attention backend: xformers `memory_efficient_attention`

**Key design:** Model predicts **residuals** (differences), not absolute values. Rollout: `next = reverse_norm(current) + predicted_diff`, then re-normalize.

Default config: hidden_size=1024, depth=24, num_heads=16, patch_size=2 or 4, input 128×256 (1.40625° grid).

### Iterative Module (`stormer/models/iterative_module.py`)
`GlobalForecastIterativeModule` (LightningModule) handles autoregressive rollout:
- **Training** (`forward_train`): rolls out `n_steps`, computes latitude-weighted MSE on normalized diffs with variable weights from `WEIGHT_DICT`
- **Validation** (`forward_validation`): autoregressive rollout at a single base interval; ensembles predictions across all valid base intervals (6h, 12h, 24h) that divide the target lead time
- **Optimizer**: AdamW; channel/pos embeddings excluded from weight decay
- **LR schedule**: `LinearWarmupCosineAnnealingLR` (step-level)

### Data (`stormer/data/multi_step_datamodule.py`)
`MultiStepDataRandomizedModule` backed by:
- `ERA5MultiStepRandomizedDataset` (train): randomly samples one interval from `list_train_intervals` per sample, returns `steps` consecutive normalized diffs as targets
- `ERA5MultiLeadtimeDataset` (val/test): returns dict of absolute targets at each `val_lead_times` lead time

Input normalized by global mean/std; diff targets normalized by per-interval diff std (zero mean).

## Inference

See `inference.py` for a full standalone example. Two pretrained checkpoints on HuggingFace (`tungnd/stormer`):
- `stormer_1.40625_patch_size_2.ckpt`
- `stormer_1.40625_patch_size_4.ckpt`

Pattern: load model, call `model.set_transforms(inp_transform, out_transforms)`, then `model.forward_validation(inp, variables, interval, steps)` for each base interval, ensemble with `.mean(0)`.

## Key Config Parameters

| Area | Parameter | Notes |
|------|-----------|-------|
| `net` | `patch_size` | 2 (accurate) or 4 (fast) |
| `net` | `hidden_size`, `depth`, `num_heads` | 1024, 24, 16 |
| `model` | `lr` | 5e-4 pretrain, 5e-6 finetune |
| `model` | `pretrained_path` | required for finetune |
| `data` | `steps` | 1 pretrain, 4 finetune |
| `data` | `list_train_intervals` | [6, 12, 24] hours |
| `data` | `val_lead_times` | e.g. [6, 72, 120, 168] hours |
