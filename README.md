# Two-Stage Adaptive Design with CKME

> A two-stage adaptive experimental design framework for **conditional distribution estimation** and **uncertainty quantification** via Conditional Kernel Mean Embedding (CKME) with conformal prediction.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![R](https://img.shields.io/badge/R-4.4%2B-276DC3)

---

## Overview

Standard experimental designs collect data uniformly, but for heteroscedastic simulators, variance is concentrated in a small region of the input space. This repository implements a two-stage adaptive design that:

1. **Stage 1** — trains a nonparametric CDF model (CKME) on a small initial dataset to learn where uncertainty is high
2. **Stage 2** — allocates additional replication budget to high-uncertainty sites, then calibrates a split conformal predictor to produce valid prediction intervals

The resulting intervals achieve nominal marginal coverage with improved conditional (local) coverage and narrower widths in heteroscedastic regions compared to space-filling designs.

### Method Summary

| Component | Description |
|-----------|-------------|
| **CKME** | Estimates $\hat{F}(t \mid x)$ via kernel mean embedding with RBF kernel; tuned by CV on CRPS |
| **S⁰ score** | Tail uncertainty: $\hat{q}_{1-\alpha/2}(x) - \hat{q}_{\alpha/2}(x)$; drives Stage 2 allocation |
| **Adaptive h** | Bandwidth $h(x) = c \cdot \hat{\sigma}(x)$ (k-NN estimate) for uniform effective resolution |
| **Split CP** | Calibration on $D_1$ gives finite-sample marginal coverage guarantee |

Benchmarks: **DCP-DR** (distributional CP with quantile regression) and **hetGP** (heteroscedastic Gaussian process), both in R.

---

## Installation

```bash
git clone https://github.com/JorjininMath/Two-stage-simple.git
cd Two-stage-simple

# Option A: conda (recommended for Python dependencies)
conda env create -f environment.yml
conda activate ckme_env

# Option B: pip only (no R benchmarks)
pip install -r requirements.txt
```

**R dependencies** (for DCP-DR and hetGP benchmarks):
```r
install.packages(c("quantreg", "hetGP", "mvtnorm", "MASS"))
```

---

## Quick Start

```python
from Two_stage import run_stage1_train, run_stage2
from Two_stage.design import generate_space_filling_design
from Two_stage.sim_functions import get_experiment_config
from CKME.parameters import ParamGrid

# Stage 1: train CKME with cross-validated hyperparameters
param_grid = ParamGrid(
    ell_x_list=[0.3, 0.5, 1.0],
    lam_list=[0.001, 0.01, 0.1],
    h_list=[0.05, 0.1, 0.2],
)
result = run_stage1_train(
    n_0=100, r_0=10,
    simulator_func="exp2",   # 1D heteroscedastic Gaussian
    param_grid=param_grid,
    cv_folds=5,
    random_state=42,
)

# Candidate pool for Stage 2 site selection
cfg = get_experiment_config("exp2")
X_cand = generate_space_filling_design(
    n=1000,
    d=cfg["d"],
    bounds=cfg["bounds"],
    random_state=43,
)

# Stage 2: adaptive allocation + conformal calibration
stage2 = run_stage2(
    stage1_result=result,
    X_cand=X_cand,
    n_1=200, r_1=10,
    method="sampling",   # "lhs" | "sampling" | "mixed"
    alpha=0.1,      # 90% prediction intervals
)

# Access the calibrated conformal predictor
cp = stage2.cp
print(f"Calibrated quantile: {cp.q_hat:.4f}")
```

---

## Current Paper Workflow

The current active paper line is the target-aware, scale-adaptive CKME-DCP
workflow in [`exp_adaptive_h/`](exp_adaptive_h/). It keeps adaptive `h(x)` at
the experiment/evaluation layer while leaving the core CKME/CP API stable.

```bash
# Tune fixed CKME hyperparameters used by the adaptive-h experiments
python exp_adaptive_h/pretrain_params.py

# Main plug-in vs oracle vs fixed adaptive-h experiment
python exp_adaptive_h/run_exp4_plugin.py --n_macro 50
python exp_adaptive_h/summarize_exp4.py

# Figures for the journal-scale adaptive-h story
python exp_adaptive_h/plot_exp4a.py
python exp_adaptive_h/plot_exp4b.py --simulator all
```

The journal draft lives in
[`manuscript/journal_scale_adaptive/`](manuscript/journal_scale_adaptive/):

```bash
cd manuscript/journal_scale_adaptive
pdflatex target_aware_scale_adaptive_ckme_cp.tex
```

For the current active/supporting/archive experiment boundary, see
[`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md).

---

## Reproducing Experiments

All scripts are run from the **project root**. Each experiment folder contains a `config.txt` with hyperparameters and a `pretrained_params.json` with CV-tuned values (re-run `pretrain_params.py` to regenerate).

### Experiment 1 — Gibbs DGP Comparison (`exp_gibbs_compare/`)

Compares CKME-CP (fixed h and adaptive h) against RLCP ([Hore & Barber 2023](https://arxiv.org/abs/2210.14461)) on two heteroscedastic DGPs from Gibbs et al.

```bash
# Local: run a quick test (1 macrorep)
python exp_gibbs_compare/run_gibbs_compare.py --n_macro 1 --h_mode adaptive

# Full run (50 macroreps), local
python exp_gibbs_compare/run_gibbs_compare.py --n_macro 50 --h_mode adaptive \
    --output_dir exp_gibbs_compare/output_adaptive_c2.0

# HPC/SLURM (50 parallel array jobs)
sbatch exp_gibbs_compare/run_all_gibbs_arc.sh    # CKME-CP
sbatch exp_gibbs_compare/run_rlcp_arc.sh         # RLCP baseline

# Aggregate + plot
python exp_gibbs_compare/run_gibbs_compare.py --n_macro 50 --aggregate_only \
    --output_dir exp_gibbs_compare/output_adaptive_c2.0
python exp_gibbs_compare/plot_gibbs_compare.py \
    --output_dir exp_gibbs_compare/output_adaptive_c2.0
```

### Archived WSC 2026 Reproduction

The old WSC 2026 table-reproduction runner has been archived out of the active
workflow. The `wsc_gauss` DGP remains registered in `Two_stage/sim_functions/`
and is still used by `exp_adaptive_h/`, but current paper evidence should come
from the adaptive-h workflow above rather than partial WSC runner outputs.

### Experiment 2 — Non-Gaussian Noise (`exp_nongauss/`)

Supporting non-Gaussian benchmark evidence against DCP-DR and hetGP. This
folder still reflects the historical six-DGP plan, while the current simulator
registry keeps the active Student-t A1 variants. Refresh this folder before
treating it as a fully reproducible public workflow.

### Experiment 3 — Conditional Coverage Consistency (`exp_conditional_coverage/`)

Verifies that CKME-CP achieves asymptotic conditional coverage as $n \to \infty$.

```bash
python exp_conditional_coverage/pretrain_params.py
python exp_conditional_coverage/run_consistency.py --n_macro 10
python exp_conditional_coverage/plot_consistency.py
```

### Experiment 4 — Design Comparison (`exp_design/`)

Compares S⁰ variants (tail-width vs. epistemic) and adaptive vs. LHS across sample sizes. Identifies the "inverted-U gain curve" regime behavior.

```bash
python exp_design/pretrain_params.py
python exp_design/run_saturation_sweep.py --n_macro 20
python exp_design/plot_adaptive_gain_curve.py
```

### Experiment 5 — One-Sided Quantile Estimation (`exp_onesided/`)

Compares CKME (CDF-first) against quantile regression (QR) at the quantile estimation level, without conformal calibration.

```bash
python exp_onesided/exp2_quantile_error.py --n_macro 50
python exp_onesided/exp2_sup_vs_tau.py
```

### Experiment 6 — Adaptive Bandwidth $h(x)$ (`exp_adaptive_h/`)

Validates the **score-homogeneity** property of CKME-CP under adaptive bandwidth $h(x) = c \cdot \hat{\sigma}(x)$. Four sub-experiments:

- **exp1** — baseline coverage/width across simulators (fixed $h$ from CV)
- **exp2** — oracle $h(x)$ sweep, paired with fixed $h$
- **exp3** — sensitivity to the scaling constant $c$
- **exp4** — three-arm comparison: fixed / plug-in $\hat{\sigma}(x)$ / oracle $h(x)$, validating the **Gap Theorem** (decay of $|\mathrm{cov}_\text{plug} - \mathrm{cov}_\text{oracle}|$ with budget) on Gaussian DGPs and the score-homogeneity prediction on Student-t$_3$

```bash
python exp_adaptive_h/pretrain_params.py
python exp_adaptive_h/run_exp4_plugin.py --n_macro 50
python exp_adaptive_h/summarize_exp4.py
python exp_adaptive_h/plot_exp4a.py    # Gap Theorem decay (Gaussian DGPs)
python exp_adaptive_h/plot_exp4b.py --simulator all
```

See [`exp_adaptive_h/Exp_plan.md`](exp_adaptive_h/Exp_plan.md) for the full plan.

---

## Manuscript

Paper-level writeup assets live in [`manuscript/`](manuscript/). The active
journal draft is under
[`manuscript/journal_scale_adaptive/`](manuscript/journal_scale_adaptive/).
Auto-generated tables and figures stay in experiment output directories and are
referenced from the manuscript.

```bash
cd manuscript/journal_scale_adaptive
pdflatex target_aware_scale_adaptive_ckme_cp.tex
```

---

## Project Structure

```
Two-stage-simple/
│
├── CKME/                        # Core model
│   ├── ckme.py                  # CKMEModel: fit, predict_cdf, predict_quantile
│   ├── parameters.py            # Params, ParamGrid dataclasses
│   ├── kernels.py               # RBF kernel
│   ├── indicators.py            # Smooth step functions (logistic / Gaussian CDF)
│   ├── coefficients.py          # Cholesky linear solver
│   ├── tuning.py                # k-fold CV with CRPS
│   └── loss_functions/          # crps.py, pinball.py
│
├── CP/                          # Conformal prediction
│   ├── cp.py                    # CP class: calibrate, predict_interval
│   ├── calibration.py           # Nonconformity score calibration
│   ├── scores.py                # abs_median, abs_cdf scores
│   ├── interval.py              # Interval construction
│   └── evaluation.py           # Coverage, width, interval score
│
├── Two_stage/                   # Pipeline orchestration
│   ├── stage1_train.py          # run_stage1_train
│   ├── stage2.py                # run_stage2
│   ├── site_selection.py        # lhs / sampling / mixed strategies
│   ├── s0_score.py              # S⁰ tail-uncertainty score
│   ├── data_collection.py       # Simulator dispatch + data collection
│   ├── design.py                # LHS design generation
│   ├── evaluation.py            # Per-point and aggregate metrics
│   ├── io.py                    # Save/load stage results
│   ├── config_utils.py          # config.txt loader
│   └── sim_functions/           # Simulator implementations
│       ├── __init__.py          # Registry
│       ├── exp1.py              # MG1 queue (1D, Gaussian)
│       ├── exp2.py              # sin+x (1D, Gaussian)
│       ├── sim_exp2_gauss.py    # WSC-style Gaussian DGP variants
│       ├── sim_nongauss_A1.py   # Student-t noise (A1S / A1L)
│       ├── sim_gibbs_s1.py      # Gibbs Setting 1: σ(x) = |sin(x)|
│       └── sim_gibbs_s2.py      # Gibbs Setting 2: σ(x) = 2φ(x/1.5)
│
├── exp_gibbs_compare/           # Exp 1: CKME-CP vs RLCP
├── exp_nongauss/                # Exp 2: Non-Gaussian noise
├── exp_conditional_coverage/    # Exp 3: Coverage consistency
├── exp_design/                  # Exp 4: Design comparison
├── exp_onesided/                # Exp 5: One-sided quantile
├── exp_adaptive_h/              # Exp 6: Adaptive bandwidth h(x)
├── ckme_dcp_mm1/                # Feasibility-only KME/CKME M/M/1 input-uncertainty module
│
├── EXPERIMENT_INDEX.md          # Active/supporting/archive experiment boundary
├── paper/                       # Dated shareable PDF snapshots
├── manuscript/                  # Paper-level tex writeup
│   ├── journal_scale_adaptive/  # Current journal draft source
│   └── reports/                 # Per-experiment .tex reports
├── _archive/                    # Tracked public archive entries plus ignored local history
│
├── dcp_r.R                      # R: DCP-DR + hetGP benchmarks
├── run_benchmarks_one_case.R    # R: single-case benchmark runner
├── submit_all.sh                # HPC: submit all SLURM jobs
├── environment.yml              # Conda environment spec
└── requirements.txt             # Pip dependencies
```

---

## Simulators

| Name | Description | Dim | Noise type |
|------|-------------|-----|------------|
| `exp1` | MG1 queue: $\zeta(x)=1.5x^2/(1-x)$ | 1D $[0.1, 0.9]$ | Heteroscedastic Gaussian |
| `exp2` | $f(x)=x+\sin(\pi x)$ | 1D $[0, 2\pi]$ | Heteroscedastic Gaussian |
| `wsc_gauss` | $f(x)=e^{x/10}\sin x$, $\sigma(x)=0.01+0.2(x-\pi)^2$ | 1D $[0,2\pi]$ | Heteroscedastic Gaussian |
| `nongauss_A1S/L` | Same mean/scale as `wsc_gauss`, Student-t ($\nu=10$ or $3$) | 1D $[0,2\pi]$ | Student-t |
| `gibbs_s1` | $Y=0.5x+\sigma(x)\varepsilon$, $\sigma(x)=\lvert\sin x\rvert$ | 1D | Heteroscedastic Gaussian |
| `gibbs_s2` | Same form, $\sigma(x)=2\varphi(x/1.5)$ | 1D | Heteroscedastic Gaussian |

The archived non-Gaussian Gamma and mixture variants are not part of the current
active simulator registry.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ckme2025,
  author = {Jin Zhao},
  title  = {Two-Stage Adaptive Experimental Design with Conditional Kernel Mean Embedding},
  year   = {2025},
  url    = {https://github.com/JorjininMath/Two-stage-simple}
}
```

*(Will be updated with journal reference upon publication.)*

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
