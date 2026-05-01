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

# Option A: conda (recommended, includes R dependencies)
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

# Stage 2: adaptive allocation + conformal calibration
stage2 = run_stage2(
    stage1_result=result,
    n_1=200, r_1=10,
    method="lhs",   # "lhs" | "sampling" | "mixed"
    alpha=0.1,      # 90% prediction intervals
)

# Access the calibrated conformal predictor
cp = stage2.cp
print(f"Calibrated quantile: {cp.q_hat:.4f}")
```

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

### Experiment 1b — WSC 2026 Paper (`exp_wsc/`)

Reproduces **Tables 2–3** of the WSC 2026 paper. Two 1D DGPs (Gaussian and Student-t ν=3, same heteroscedastic σ(x)); Stage 2 budget sweep over (n₁,r₁) ∈ {(100,50),(200,25),(500,10)}; three site-selection methods; 50 macroreps.

```bash
# Pretune hyperparameters (run once)
python exp_wsc/pretrain_params.py

# Full run (50 macroreps, parallel)
python exp_wsc/run_wsc_compare.py --n_macro 50 --n_workers 8

# Print Tables 2-3
python exp_wsc/make_tables.py
```

See [`exp_wsc/README.md`](exp_wsc/README.md) for full details.

### Experiment 2 — Non-Gaussian Noise (`exp_nongauss/`)

Six simulators (Student-t / Gamma / Gaussian-mixture, small/large non-Gaussianity). Compares CKME-CP against DCP-DR and hetGP.

```bash
# Pretune hyperparameters (run once)
python exp_nongauss/pretrain_params.py

# Run comparison
python exp_nongauss/run_nongauss_compare.py --n_macro 50 --method lhs

# Plot
python exp_nongauss/plot_nongauss.py
python exp_nongauss/plot_nongauss_noise.py --mode hist
```

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
python exp_adaptive_h/plot_exp4b.py    # Coverage equivalence + h-ratio diagnostic (Student-t)
```

See [`exp_adaptive_h/Exp_plan.md`](exp_adaptive_h/Exp_plan.md) for the full plan.

---

## Manuscript

Paper-level writeup assets (per-experiment `.tex` reports, shared header, sections) live in [`manuscript/`](manuscript/). Auto-generated tables (e.g. `exp_adaptive_h/output_exp4/exp4_table.tex`) stay in their experiment output directories and are referenced via `\input{...}`; figures are pulled via `\graphicspath{{../../exp_*/output/}}`.

```bash
cd manuscript/reports && pdflatex nongauss_report.tex
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
│       ├── simulator.py         # Registry
│       ├── sim_exp1.py          # MG1 queue (1D, Gaussian)
│       ├── sim_exp2.py          # sin+x (1D, Gaussian)
│       ├── sim_nongauss_A1.py   # Student-t noise (A1S / A1L)
│       ├── sim_nongauss_B2.py   # Gamma noise (B2S / B2L)
│       ├── sim_nongauss_C1.py   # Gaussian mixture (C1S / C1L)
│       ├── sim_gibbs_s1.py      # Gibbs Setting 1: σ(x) = |sin(x)|
│       └── sim_gibbs_s2.py      # Gibbs Setting 2: σ(x) = 2φ(x/1.5)
│
├── exp_gibbs_compare/           # Exp 1: CKME-CP vs RLCP
├── exp_wsc/                     # Exp 1b: WSC 2026 paper reproduction
├── exp_nongauss/                # Exp 2: Non-Gaussian noise
├── exp_conditional_coverage/    # Exp 3: Coverage consistency
├── exp_design/                  # Exp 4: Design comparison
├── exp_onesided/                # Exp 5: One-sided quantile
├── exp_adaptive_h/              # Exp 6: Adaptive bandwidth h(x)
│
├── manuscript/                  # Paper-level tex writeup
│   └── reports/                 # Per-experiment .tex reports
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
| `nongauss_A1S/L` | $f(x)=e^{x/10}\sin x$, Student-t ($\nu=10/3$) | 1D $[0,2\pi]$ | Student-t |
| `nongauss_B2S/L` | Same mean, Gamma ($k=9/2$) | 1D $[0,2\pi]$ | Centered Gamma |
| `nongauss_C1S/L` | Same mean, Gaussian mixture ($\pi=0.02/0.10$) | 1D $[0,2\pi]$ | Gaussian mixture |
| `gibbs_s1` | $Y=0.5x+\sigma(x)\varepsilon$, $\sigma(x)=\lvert\sin x\rvert$ | 1D | Heteroscedastic Gaussian |
| `gibbs_s2` | Same form, $\sigma(x)=2\varphi(x/1.5)$ | 1D | Heteroscedastic Gaussian |

All noise variances are normalized to $\sigma_\text{tar}(x) = 0.01 + 0.2(x-\pi)^2$ for the nongauss family, so that S (small) and L (large) differ only in distributional shape.

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
