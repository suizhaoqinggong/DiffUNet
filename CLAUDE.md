# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Diff-UNet: diffusion-embedded 3D medical image segmentation. Three dataset pipelines: AIIB2023 (airway), BraTS2023 (brain tumor), WORD (abdominal organs).

## Commands

All scripts run from repo root with `python <script>.py`. No build/lint/test commands exist.

### AIIB2023 (1-channel, 2-class airway)
```
python 1_preprocessing_aiib2023.py        # normalize + resample → ./data/fullres/train/
python 2_train_diffunet_aiib2023.py       # train → ./logs/diffunet/model/
python 3_predict_aiib2023.py              # inference → ./prediction_results/diffunet/
python 4_compute_metrics_aiib2023.py --pred=diffunet
```

### BraTS2023 (4-channel, 4-class brain tumor: TC/WT/ET)
```
python 1_rename_mri_data_brats2023.py     # rename MRI files
python 2_preprocessing_brats2023.py       # normalize + resample 4 modalities
python 3_train_diffunet_brats2023.py      # train (DDP, 2 GPUs, env="DDP")
python 4_predict_brats2023.py             # inference
python 5_compute_metrics_brats2023.py --pred=diffunet
tensorboard --logdir ./logs/diffunet/
```

### WORD (1-channel CT, 17-class abdominal)
```
python 1_preprocessing_word.py
python 2_train_diffunet_word.py           # checkpoints → ./logs/diffunet_word/
python 3_predict_word.py                  # sliding-window + optional mirror TTA
python 4_compute_metrics_word.py          # per-organ Dice
python 5_compute_metrics_word.py          # per-organ Dice + HD95
```

Validation splits: `test_list_aiib23.py` (24 cases), `test_list_brats2023.py` (~125 cases). WORD uses separate train/val directories.

`view_results.py` reads `./prediction_results/result_metrics/diffunet.npy` and prints mean/std Dice/HD95.

## Architecture

**Two-UNet design**: `DiffUNet` (in `diffunet/diffunet_model.py`) wraps an edge detection U-Net (`nnunet3d.py`) and a diffusion denoising U-Net (`nnunet3d_denoise.py`). Both use the same PlainConvUNet backbone, but the denoising version adds time-conditioned FiLM-style scale+shift via `temb_proj` in conv blocks (`block_denoise.py`).

- **Edge model** (`nnunet3d.py`): standard U-Net, produces `pred_edge` and encoder embeddings at each stage.
- **Denoise model** (`nnunet3d_denoise.py`): time-conditioned U-Net with DDIM sampling. During training (`ddim=False`), returns `(pred, uncertainty_map)`. During inference (`ddim=True`), returns refined `pred`.
- **MBA** (Multi-granularity Boundary Aggregation): `affinity_fusion()` in `encoder_denoise.py` — fuses boundary embeddings with encoder features at the bottleneck via an affinity matrix.
- **MC-Diff** (Monte Carlo Diffusion): `compute_uncer()` in `nnunet3d_denoise.py` — entropy-based uncertainty from multiple softmax forward passes during training.
- **PURE** (Progressive Uncertainty-driven Refinement): `ddim_sample()` in `nnunet3d_denoise.py` — iteratively refines segmentation during DDIM inference.
- **PR25 Boundary** (`pr25_boundary.py`, optional): `BoundaryExtractionModule` (BEM) with multi-scale dilated convs + Sobel refinement, `BoundaryDecoder`, and `BoundarySupervisionHead`. Enabled via `use_pr25_boundary=True` in DiffUNet constructor. When active, replaces `pred_edge` addition with boundary prediction output.

`diffunet/` is NOT a proper Python package (no `__init__.py`). Imports use `from diffunet.diffunet_model import DiffUNet` which works via implicit namespace packages in Python 3. Do not add `__init__.py` without verifying it doesn't break imports.

`light_training/` is the reusable infrastructure layer: trainers (AMP/FP32 + DDP), sliding-window inference, losses (Dice, CE, compound, deep supervision), dataloading, augmentation, preprocessing, and evaluation metrics (Dice, HD95, Jaccard, etc.).

## Path Configuration (CRITICAL)

Many scripts contain **hardcoded absolute paths**. Update these before running:

| Script(s) | Hardcoded path |
|-----------|---------------|
| All BraTS scripts (1–5) | `/data/chenjiahao/raw_data/...` |
| All WORD scripts (1–5) | `/data/chenjiahao/WORD-V0.1.0/...` |
| `3_predict_aiib2023.py` | `/data/xingzhaohu/aiib23/logs/...` (model checkpoint) |
| `4_predict_brats2023.py` | `/home/chenjiahao/DiffUNet/logs/...` (model checkpoint) |
| `5_compute_metrics_word.py` | `--gt_dir` defaults to `/data/chenjiahao/WORD-V0.1.0/labelsVal` |

AIIB2023 scripts use relative `./data/...` paths. Other pipelines do not.

## Environment

No `requirements.txt` or `setup.py`. Dependencies: torch 2.1.2, CUDA 12.2, monai, dynamic_network_architectures, batchgenerator, SimpleITK, medpy, nibabel, tqdm, numpy, scipy. AIIB2023 metrics require the custom `scoring_metrics` package.

## Validation

No pytest suite. Validate changes by running the affected pipeline step + its downstream metric script. When changing data splits, update the relevant `test_list_*.py` and document Dice/HD95 impact.

## Git

Do NOT commit: checkpoints (`.pt`), TensorBoard event files, prediction volumes (`.nii.gz`), `logs/`, `prediction_results/`, or other generated outputs. Use imperative commit messages.
