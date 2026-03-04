# Two-Stage Adaptive Experimental Design with CKME

A two-stage adaptive experimental design framework for **conditional distribution estimation** and **uncertainty quantification**. The core method is **CKME** (Conditional Kernel Mean Embedding), which estimates conditional CDFs from simulation data. Prediction intervals are constructed via **split conformal prediction (CP)** calibrated on adaptively collected Stage 2 data.

Benchmarks compare CKME against **DCP-DR** (Distributional Conformal Prediction with quantile regression) and **hetGP** (Heteroscedastic Gaussian Process), both implemented in R.

---

## How It Works

**Stage 1 — Initial model training**

Collect a small initial dataset $D_0$ (n_0 sites × r_0 replications) from a simulator, then train a CKME model to estimate the conditional CDF $\hat{F}(t \mid x)$.

**Stage 2 — Adaptive sampling + conformal calibration**

Use the Stage 1 model to compute $S^0$ scores (tail uncertainty = quantile interval width) at candidate sites. Sites with higher $S^0$ are more uncertain and more informative to sample. Three site selection strategies are supported:

| Method | Description |
|--------|-------------|
| `lhs` | Latin Hypercube Sampling (space-filling, ignores $S^0$) |
| `sampling` | Sample proportional to $S^0$ scores (adaptive) |
| `mixed` | Blend of LHS and sampling |

Collect $D_1$ (n_1 × r_1) at selected sites, then calibrate a split conformal predictor on $D_1$ to produce valid prediction intervals $[L(x), U(x)]$.

---

## Requirements

**Python** (≥ 3.8):
```
numpy, scipy, pandas, scikit-learn, openpyxl
```

**R** (for DCP-DR and hetGP benchmarks):
```r
install.packages(c("quantreg", "hetGP"))
```

---

## Quick Start

```python
from Two_stage import run_stage1_train, run_stage2, save_stage1_train_result, load_stage1_train_result
from CKME.parameters import Params, ParamGrid

# Stage 1: train with fixed hyperparameters
params = Params(ell_x=0.5, lam=0.01, h=0.1)
result = run_stage1_train(n_0=100, r_0=10, simulator_func="exp1", params=params, random_state=42)

# Or with cross-validated hyperparameter tuning
param_grid = ParamGrid(ell_x_list=[0.5, 1.0], lam_list=[0.01, 0.1], h_list=[0.05, 0.1])
result = run_stage1_train(n_0=100, r_0=10, simulator_func="exp1", param_grid=param_grid, cv_folds=5)

save_stage1_train_result(result, "output/stage1_model")

# Stage 2: adaptive site selection + CP calibration
stage2_result = run_stage2(stage1_result=result, n_1=200, r_1=10, method="mixed", alpha=0.1)
```

---

## Running Experiments

All experiment scripts are run from the **project root**.

```bash
# exp3: Branin-Hoo (2D, Gaussian noise) — compare CKME, DCP-DR, hetGP
python exp3_test/run_exp3_compare.py
python exp3_test/run_exp3_compare.py --method sampling --n_macro 10

# exp2: 1D nonlinear + skewed bimodal noise
python exp2_test/run_exp2_compare.py

# Stage 2 impact study (9 cases × 50 macroreps) on HPC via SLURM
sbatch exp_stage2_impact_arc/run_all_stage2_arc.sh
```

Configuration for each experiment lives in its own `config.txt`. Key parameters:

| Parameter | Description |
|-----------|-------------|
| `n_0`, `r_0` | Stage 1: sites × replications |
| `n_1`, `r_1` | Stage 2: sites × replications |
| `ell_x`, `lam`, `h` | CKME hyperparameters |
| `alpha` | CP significance level (e.g. 0.1 → 90% coverage) |
| `t_grid_size` | CDF threshold grid resolution |

---

## Simulators

| Name | Description | Dimension | Noise |
|------|-------------|-----------|-------|
| `exp1` | MG1 queue: $\zeta(x) = 1.5x^2/(1-x)$ | 1D $[0.1, 0.9]$ | Heteroscedastic Gaussian |
| `exp2` | $\zeta(x) = x + \sin(\pi x)$ | 1D $[0, 2\pi]$ | Heteroscedastic Gaussian |
| `exp2_test` | Same mean as exp2 | 1D $[0, 2\pi]$ | Skewed bimodal mixture |
| `exp3` | exp3 mean function | 2D $[0, 2\pi]^2$ | Student-t heavy tails |
| `exp_test` | Branin-Hoo function | 2D $[0,1]^2$ | Gaussian |

To add a new simulator, create a file in `Two_stage/sim_functions/` and register it in `simulator.py`.

---

## Evaluation Metrics

- **Coverage**: $P(L(x) \leq Y \leq U(x))$, target $= 1 - \alpha$
- **Width**: $E[U(x) - L(x)]$
- **Interval Score**: $(U - L) + \frac{2}{\alpha}(L - Y)_+ + \frac{2}{\alpha}(Y - U)_+$

---

## Project Structure

```
CKME/           # Core model: CKMEModel, RBF kernel, CRPS loss, CV tuning
CP/             # Conformal prediction: calibration, intervals, scores
Two_stage/      # Pipeline: stage1_train, stage2, site_selection, io, evaluation
exp2_test/      # Experiment: 1D nonlinear + skewed noise
exp3_test/      # Experiment: Branin-Hoo 2D
exp_stage2_impact_arc/  # Large-scale Stage 2 impact study (HPC/SLURM)
dcp_r.R         # R implementation of DCP-DR and hetGP benchmarks
core/           # Deprecated; superseded by CKME/ and Two_stage/
```
