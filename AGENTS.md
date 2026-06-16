# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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
# Non-Gaussian noise comparison (Exp 2)
python exp_nongauss/pretrain_params.py
python exp_nongauss/run_nongauss_compare.py --n_macro 50 --method lhs

# Conditional coverage consistency (Exp 3)
python exp_conditional_coverage/run_consistency.py --n_macro 10

# Adaptive bandwidth h(x) — exp4 plug-in vs oracle vs fixed (Exp 6)
python exp_adaptive_h/run_exp4_plugin.py --n_macro 50
python exp_adaptive_h/summarize_exp4.py
python exp_adaptive_h/plot_exp4a.py
python exp_adaptive_h/plot_exp4b.py --simulator all

# HPC: submit a SLURM array
sbatch exp_gibbs_compare/run_all_gibbs_arc.sh
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

- **`CKME/`** — Core model: `ckme.py` (CKMEModel), `parameters.py` (Params/ParamGrid), `kernels.py` (RBF), `indicators.py` (smooth step functions), `coefficients.py` (Cholesky solver), `tuning.py` (k-fold CV with CRPS), `cdf.py`, `loss_functions/{crps,pinball}.py`
- **`CP/`** — Conformal prediction: `cp.py` (CP class), `calibration.py`, `interval.py`, `scores.py`, `evaluation.py`
- **`Two_stage/`** — Pipeline orchestration: `stage1_train.py`, `stage2.py`, `s0_score.py`, `site_selection.py`, `data_collection.py`, `design.py`, `io.py`, `evaluation.py`, `sim_functions/`
- **`exp_*/`** — Active/supporting experiments, each with `config.txt`, `pretrain_params.py`, `run_*.py`, `summarize_*.py`, `plot_*.py`, and `output_*/` (gitignored): `exp_adaptive_h`, `exp_conditional_coverage`, `exp_design`, `exp_gibbs_compare`, `exp_nongauss`, `exp_onesided`
- **`ckme_dcp_mm1/`** — Feasibility-only KME/CKME M/M/1 input-uncertainty module. It is not a conformal coverage experiment.
- **`manuscript/`** — Paper-level tex writeup: `manuscript/journal_scale_adaptive/` holds the active journal draft; `manuscript/reports/` holds per-experiment `.tex` reports.
- **`paper/`** — Dated shareable PDF snapshots, e.g. `CKME_CP_20260616_v0.1.pdf`.
- **`_archive/`** — Deprecated code and historical entrypoints. Only explicitly promoted public archive entries should be tracked; local history and generated outputs stay ignored.
- **`notes/`** — Local-only working notes (gitignored): `notes/planning/` (idea drafts), `notes/tex/` (working note drafts), reference MDs at top level
- **`dissertation_use/`** — Local-only dissertation working folder (gitignored)

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

| Name | Description | Dim | Noise |
|------|-------------|-----|-------|
| `exp1` | MG1 queue: ζ(x)=1.5x²/(1−x) | 1D [0.1, 0.9] | Heteroscedastic Gaussian |
| `exp2` | f(x)=x+sin(πx) | 1D [0, 2π] | Heteroscedastic Gaussian |
| `wsc_gauss` | f(x)=e^(x/10)sin(x), σ(x)=0.01+0.2(x−π)² | 1D [0, 2π] | Heteroscedastic Gaussian |
| `nongauss_A1S/L` | Same mean/scale as `wsc_gauss`, Student-t (ν=10 or 3) | 1D [0, 2π] | Student-t |
| `gibbs_s1` | Y=0.5x+σ(x)ε, σ(x)=\|sin(x)\| | 1D | Heteroscedastic Gaussian |
| `gibbs_s2` | Same form, σ(x)=2φ(x/1.5) | 1D | Heteroscedastic Gaussian |

To add a new simulator, create a file in `Two_stage/sim_functions/` and register it in `__init__.py`.

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

Per-experiment SLURM scripts (e.g. `exp_gibbs_compare/run_all_gibbs_arc.sh`) submit a 50-macrorep array; each macrorep runs the experiment with a unique seed. Per-macrorep outputs land in `output_*/macrorep_<k>/`. The top-level `submit_all.sh` batches multiple experiments.
