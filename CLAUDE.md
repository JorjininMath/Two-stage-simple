# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

- **Chinese** for explanations, summaries, and conceptual answers.
- **English** for all code, comments, docstrings, commit messages, file names, and shell commands.
- When both are needed: Chinese explanation first, then English code/commands in separate fenced blocks.

## Workflow

- Prefer small, reviewable diffs. Do not change unrelated files.
- Do not delete or rename files unless explicitly requested.
- This repository is **public**: never introduce secrets, credentials, tokens, or personal paths into tracked files.

## Project Overview

This is a **two-stage adaptive experimental design framework** for conditional distribution estimation and uncertainty quantification. The core method is CKME (Conditional Kernel Mean Embedding), which estimates conditional CDFs. Prediction intervals are constructed via split conformal prediction (CP) calibrated on adaptively collected Stage 2 data.

Benchmarks compare CKME against DCP-DR and hetGP (both implemented in R via `dcp_r.R`).

## Running Experiments

```bash
# Run exp3 (Branin-Hoo 2D) comparison with default LHS design
python exp3_test/run_exp3_compare.py

# Specify design method and macroreps
python exp3_test/run_exp3_compare.py --method sampling --n_macro 10 --output_dir my_output

# Run exp2 (1D nonlinear + skewed noise) comparison
python exp2_test/run_exp2_compare.py

# HPC: submit 50 parallel macroreps via SLURM
sbatch exp_stage2_impact_arc/run_all_stage2_arc.sh
```

## Using the Two-Stage API

```python
# Stage 1: Train CKME model
from Two_stage import run_stage1_train, run_stage2, save_stage1_train_result, load_stage1_train_result
from CKME.parameters import Params, ParamGrid

# Option A: Fixed params (fast)
params = Params(ell_x=0.5, lam=0.01, h=0.1)
result = run_stage1_train(n_0=100, r_0=10, simulator_func="exp1", params=params, random_state=42)

# Option B: CV hyperparameter tuning
param_grid = ParamGrid(ell_x_list=[0.5, 1.0], lam_list=[0.01, 0.1], h_list=[0.05, 0.1])
result = run_stage1_train(n_0=100, r_0=10, simulator_func="exp1", param_grid=param_grid, cv_folds=5)

save_stage1_train_result(result, "output/stage1_model")
result = load_stage1_train_result("output/stage1_model")

# Stage 2: Adaptive site selection, data collection, CP calibration
from Two_stage import run_stage2
stage2_result = run_stage2(stage1_result=result, n_1=200, r_1=10, method="mixed", alpha=0.1)
```

## Architecture

### Module Layout

- **`CKME/`** — Core model: `ckme.py` (CKMEModel), `parameters.py` (Params/ParamGrid), `kernels.py` (RBF), `indicators.py` (smooth step functions), `coefficients.py` (Cholesky solver), `tuning.py` (k-fold CV with CRPS), `cdf.py`, `loss_functions/crps.py`
- **`CP/`** — Conformal prediction: `cp.py` (CP class), `calibration.py`, `interval.py`, `scores.py`, `evaluation.py`
- **`Two_stage/`** — Pipeline orchestration: `stage1_train.py`, `stage2.py`, `s0_score.py`, `site_selection.py`, `data_collection.py`, `design.py`, `io.py`, `evaluation.py`, `sim_functions/`
- **`core/`** — Deprecated; use CKME/ and Two_stage/ instead
- **`exp2_test/`, `exp3_test/`, `exp_stage2_impact_arc/`** — Experiment scripts with their own `config.txt`, `config_utils.py`, and `output/`

### Two-Stage Pipeline Flow

1. **Stage 1** (`run_stage1_train`): Generate `D_0` (n_0 sites × r_0 reps) via simulator → train `CKMEModel` (optionally with CV tuning) → returns `Stage1TrainResult`
2. **S^0 Score** (`compute_s0`): For candidate sites, compute tail uncertainty = `q_{1-α/2}(x) - q_{α/2}(x)` (quantile interval width from Stage 1 CDF estimate); higher = more informative to sample
3. **Stage 2** (`run_stage2`): Select n_1 sites from candidates using S^0 scores (method: `lhs`, `sampling`, or `mixed`) → collect `D_1` (n_1 × r_1 reps) → calibrate split-CP on `D_1` → returns `Stage2Result` with calibrated `CP` object

### Key Data Structures

```python
Stage1TrainResult: model (CKMEModel), t_grid, X_0, X_all, Y_all, params, n_0, r_0, d
Stage2Result: model, t_grid, X_1, X_stage2, Y_stage2, cp (CP), n_1, r_1, selection_method, alpha
```

### Site Selection Methods (`Two_stage/site_selection.py`)

- `lhs` — Latin Hypercube Sampling (space-filling, ignores S^0)
- `sampling` — Sample proportional to S^0 scores (adaptive)
- `mixed` — γ × LHS + (1−γ) × sampling (balances space-filling and adaptive)

### Simulators (`Two_stage/sim_functions/`)

| Name | Description | Dimension | Noise |
|------|-------------|-----------|-------|
| `exp1` | MG1 queue: ζ(x)=1.5x²/(1−x) | 1D [0.1, 0.9] | Heteroscedastic Gaussian |
| `exp2` | ζ(x)=x+sin(πx) | 1D [0, 2π] | Heteroscedastic Gaussian |
| `exp2_test` | Same as exp2 | 1D [0, 2π] | Skewed bimodal mixture |
| `exp3` | exp3 true function | 2D [0, 2π]² | Student-t heavy tails |
| `exp_test` | Branin-Hoo function | 2D [0,1]² | Gaussian |

To add a new simulator, create a file in `Two_stage/sim_functions/` and register it in `simulator.py`.

### Hyperparameters (set in `config.txt` or via `Params`)

- `ell_x` — RBF kernel length scale for X
- `lam` — Tikhonov regularization in Cholesky solver
- `h` — Bandwidth of smooth indicator functions (logistic/gaussian_cdf)
- `alpha` — CP significance level (e.g., 0.1 → 90% coverage target)
- `t_grid_size` — Number of threshold points for CDF evaluation

### Evaluation Metrics (`Two_stage/evaluation.py`)

- **Coverage**: P(L ≤ Y ≤ U), target = 1 − α
- **Width**: E[U − L]
- **Interval Score**: (U − L) + (2/α)(L − Y)₊ + (2/α)(Y − U)₊

### R Integration

`dcp_r.R` implements DCP-DR (distributional conformal prediction with quantile regression) and hetGP benchmarks. Experiment scripts call it via subprocess. Requires R with `dcp`, `hetGP`, and `quantreg` packages installed.

### HPC (ARC/SLURM)

See `exp_stage2_impact_arc/README_ARC.md` for cluster setup. `run_all_stage2_arc.sh` submits a SLURM array of 50 macroreps; each macrorep runs `run_stage2_influence.py` with a unique seed. Per-macrorep outputs are Excel files in `output/macrorep_<k>/`.
