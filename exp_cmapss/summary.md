# Experiment Summary: CKME Two-Stage Framework

## Overview

This document summarizes two experiments evaluating the CKME (Conditional Kernel Mean
Embedding) method for conditional distribution estimation and one-sided quantile prediction.

---

## Experiment 1 — Non-Gaussian Noise (`exp_nongauss`)

### Research Question

Does CKME maintain valid prediction interval coverage under non-Gaussian conditional
distributions? How does it compare to DCP-DR (distributional conformal prediction with
density ratio) and hetGP (heteroscedastic Gaussian process)?

### Two-Stage Pipeline

1. **Stage 1**: Collect n_0 sites × r_0 reps via LHS → train CKMEModel (CV-tuned).
2. **S^0 Score**: Compute quantile interval width q_{1−α/2}(x) − q_{α/2}(x) as uncertainty
   proxy; higher = more informative.
3. **Stage 2**: Select n_1 sites from candidates using S^0 (LHS/sampling/mixed) → collect
   n_1 × r_1 reps → calibrate split conformal prediction on Stage 2 data.

### Simulators

True function shared by all six simulators:

```
f(x) = exp(x/10) * sin(x),    x ∈ [0, 2π]
```

Target variance (shared):

```
σ_tar(x) = 0.1 + 0.1 * (x − π)²
```

| Name | Noise Distribution | Shape Params | Non-Gaussianity |
|------|--------------------|--------------|-----------------|
| `nongauss_A1S` | Student-t | ν = 10, scale s(x) = σ_tar(x)·√(8/10) | Small |
| `nongauss_A1L` | Student-t | ν = 3, scale s(x) = σ_tar(x)·√(1/3) | Large |
| `nongauss_B2S` | Centered Gamma | k = 9, θ(x) = σ_tar(x)/3 | Small |
| `nongauss_B2L` | Centered Gamma | k = 2, θ(x) = σ_tar(x)/√2 | Large |
| `nongauss_C1S` | Gaussian mixture | π = 0.02, s1 = 0.3, s2 = 3.0, homoscedastic | Small |
| `nongauss_C1L` | Gaussian mixture | π = 0.10, s1 = 0.3, s2 = 3.0, homoscedastic | Large |

All noise is centered (mean zero) and scaled to match σ_tar(x) in variance.

### Current Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| n_0 | 250 | Stage 1 sites |
| r_0 | 20 | Stage 1 reps per site |
| Stage 1 total | 5,000 | |
| n_1 | 500 | Stage 2 sites |
| r_1 | 10 | Stage 2 reps per site |
| Stage 2 total | 5,000 | |
| n_test | 1,000 | Test points |
| n_cand | 1,000 | Candidate pool for Stage 2 |
| t_grid_size | 500 | CDF threshold grid |
| alpha | 0.1 | Target 90% CP coverage |
| method | lhs | Site selection |

### CV-Tuned Hyperparameters (from `pretrained_params.json`)

| Simulator | ell_x | lam | h |
|-----------|-------|-----|---|
| nongauss_A1S | — | — | — |
| nongauss_A1L | — | — | — |
| nongauss_B2S | — | — | — |
| nongauss_B2L | — | — | — |
| nongauss_C1S | — | — | — |
| nongauss_C1L | — | — | — |

*(Fill in after running `python exp_nongauss/pretrain_params.py`.)*

### Comparison Methods

| Method | Description |
|--------|-------------|
| **CKME** | Two-stage adaptive design + split conformal prediction |
| **DCP-DR** | Distributional CP with density ratio (R, `dcp` package) |
| **hetGP** | Heteroscedastic GP + normal predictive interval (R, `hetGP` package) |

### Evaluation Metrics

- **Coverage**: P(L ≤ Y ≤ U), target = 1 − α = 0.90
- **Width**: E[U − L]
- **Interval Score (Winkler)**:
  ```
  IS(L, U, Y) = (U − L) + (2/α)[L − Y]₊ + (2/α)[Y − U]₊
  ```
  Lower is better.

### Previous Results (Old Simulators, n_macroreps = 1)

Old simulators had different parameterizations. Shown for reference only.

| Simulator | Method | Coverage | Width | IS |
|-----------|--------|----------|-------|----|
| A1 (Student-t, ν=3) | CKME | 0.910 | 8.371 | 11.878 |
| A1 | DCP-DR | 0.892 | 7.232 | 12.406 |
| A1 | hetGP | 0.922 | 8.694 | 13.166 |
| B2 (Gamma, k=2) | CKME | 0.884 | 7.211 | 9.603 |
| B2 | DCP-DR | 0.902 | 6.392 | 9.034 |
| B2 | hetGP | 0.940 | 6.572 | 8.850 |
| C1 (Gauss mix) | CKME | 0.890 | 0.361 | 1.144 |
| C1 | DCP-DR | 0.892 | 2.369 | 2.973 |
| C1 | hetGP | 0.960 | 0.747 | 1.298 |

### How to Run

```bash
# CV hyperparameter tuning (run once)
python exp_nongauss/pretrain_params.py

# Main experiment (adjust --n_macro as needed)
python exp_nongauss/run_nongauss_compare.py --n_macro 5 --method lhs

# Plots
python exp_nongauss/plot_nongauss.py --save exp_nongauss/output/cov_width.png
python exp_nongauss/plot_nongauss_noise.py --mode hist --save exp_nongauss/output/noise_hist.png
python exp_nongauss/plot_nongauss_noise.py --mode var  --save exp_nongauss/output/noise_var.png
```

### Output Structure

```
exp_nongauss/output/
  macrorep_{k}/
    case_{sim}_{method}/
      per_point.csv       # per test point: x, L, U, covered, width, DCP-DR/hetGP cols
      benchmarks.csv      # R benchmark results
      macrorep_0/         # raw X0/Y0/X1/Y1/X_test/Y_test CSVs
  nongauss_compare_summary.csv   # aggregated coverage/width/IS per sim × method
```

---

## Experiment 2 — One-Sided Quantile Bounds (`exp_onesided`)

### Research Question

Does CKME's CDF-first approach yield more accurate plug-in one-sided quantile bounds than
linear quantile regression (QR), especially under non-Gaussian conditional distributions?

### Method Comparison (No Calibration — Estimation Level Only)

| Method | Primary Object | How q̂_τ(x) is obtained |
|--------|---------------|--------------------------|
| **QR** | q̂_τ(x) directly | Linear quantile regression at level τ |
| **CKME** | F̂(y\|x) first | CDF inversion: q̂_τ(x) = inf{y : F̂(y\|x) ≥ τ} |

CKME is CDF-first (like DCP-DR); QR is quantile-first (like CQR). No conformal calibration
is applied — this is a pure plug-in quantile accuracy comparison.

### Simulators

| Name | True Function | Noise | Purpose |
|------|--------------|-------|---------|
| `exp1` | MG1 queue ζ(x) = 1.5x²/(1−x) | Heteroscedastic Gaussian | Sanity check |
| `exp2` | ζ(x) = x + sin(πx) | Heteroscedastic Gaussian | Smooth 1D |
| `nongauss_B2L` | f(x) = exp(x/10)·sin(x) | Centered Gamma k=2 (strong skew) | Hardest case for QR |

### Target Quantile Levels

- τ = 0.05 → **lower bound** L(x) = q̂_{0.05}(x), target P(Y ≥ L(X)) ≈ 0.95
- τ = 0.95 → **upper bound** U(x) = q̂_{0.95}(x), target P(Y ≤ U(X)) ≈ 0.95

### Evaluation Metrics

Reported separately for lower (τ = 0.05) and upper (τ = 0.95) bounds:

- **Empirical coverage**: P(Y ≥ L(X)) or P(Y ≤ U(X)) vs nominal 0.95
- **Pinball loss** at level τ:
  ```
  ρ_τ(y, q̂) = (τ−1)(y − q̂)·1{y < q̂} + τ(y − q̂)·1{y ≥ q̂}
  ```
- **Sup quantile error**: sup_x |q̂(x) − q_true(x)|
- **Mean quantile error**: mean_x |q̂(x) − q_true(x)|
- **Conditional coverage by x-bin**: reveals local adaptation failure
- **Tail probability diagnostics**: P(Y > t | X=x) curves (CKME vs true)

### Configuration

| Parameter | Value |
|-----------|-------|
| n_train | ~250 (from config) |
| r_train | ~10 |
| n_test | 1,000 |
| tau levels | 0.05, 0.95 |
| alpha | 0.1 |
| t_grid_size | 500 |

### How to Run

```bash
# CV hyperparameter tuning (run once)
python exp_onesided/pretrain_params.py

# Main experiment
python exp_onesided/run_onesided_compare.py --n_macro 5

# Plots
python exp_onesided/plot_onesided.py --mode bounds   --save exp_onesided/output/bounds.png
python exp_onesided/plot_onesided.py --mode coverage  --save exp_onesided/output/coverage.png
python exp_onesided/plot_onesided.py --mode sup_error --save exp_onesided/output/sup_error.png
python exp_onesided/plot_onesided.py --mode tailprob  --save exp_onesided/output/tailprob.png
```

### Output Structure

```
exp_onesided/output/
  summary.csv            # coverage + pinball loss per (sim, tau, method, macrorep)
  sup_error.csv          # sup_x |q̂ − q_true| per macrorep
  tailprob_summary.csv   # P(Y > q_true | X) per simulator
  tailprob_curve.csv     # P(Y > t | X=x) for fixed t values
```

---

## Paper Positioning

### exp_nongauss (Two-Sided Intervals)

> CKME uses a two-stage adaptive design to collect Stage 2 calibration data at sites where
> uncertainty is highest, then constructs prediction intervals via split conformal prediction.
> Under non-Gaussian noise (heavy tails, skew, contamination), CKME maintains valid coverage
> while remaining competitive in interval width compared to DCP-DR and hetGP.

### exp_onesided (One-Sided Quantile Bounds)

> We compare CKME and linear QR at the estimation level — no calibration applied. CKME
> estimates the full conditional CDF F̂(y|x) and inverts it for quantiles; QR directly fits
> conditional quantiles. The key question is whether CKME's nonparametric CDF estimate yields
> more accurate plug-in quantile bounds under skewed or non-Gaussian conditional laws where
> linear QR is misspecified.

---

## Key Differences Between the Two Experiments

| Aspect | exp_nongauss | exp_onesided |
|--------|-------------|--------------|
| Goal | Two-sided prediction intervals | One-sided quantile bounds |
| Calibration | Split conformal (Stage 2 data) | None (plug-in only) |
| Competitors | DCP-DR, hetGP (R) | Linear QR (Python, statsmodels) |
| Adaptive design | Yes (two-stage) | No (single-stage) |
| Non-Gaussianity | All 6 simulators | nongauss_B2L (hardest case) |
| Primary metric | Coverage + Width + IS | Coverage + Pinball loss |
| Output | Per-point CSVs + R benchmarks | Per-macrorep CSVs |
