# Non-Gaussian Experiment Summary

> **Note (2026-04-08)**: The simulator parameterizations below are **outdated**.
> The current 6-simulator design (A1S/A1L/B2S/B2L/C1S/C1L with shared σ_tar) and
> up-to-date configuration are documented in **`exp_cmapss/summary.md`** (Experiment 1
> section). Results from new runs should be recorded there.
> This file is kept for historical reference only.

---

## Experiment Settings (Previous Run — Archived)

### Data Sizes
| Parameter | Value |
|-----------|-------|
| Stage 1 sites (n_0) | 200 |
| Stage 1 reps per site (r_0) | 5 |
| Stage 1 total samples | 1,000 |
| Stage 2 sites (n_1) | 100 |
| Stage 2 reps per site (r_1) | 5 |
| Stage 2 total samples | 500 |
| Test points (n_test) | 500 |
| Test reps (r_test) | 1 |
| Candidate pool (n_cand) | 1,000 |
| t_grid_size | 500 |

### Hyperparameters (fixed, not CV-tuned)
| Parameter | Value |
|-----------|-------|
| ell_x | 1.0 |
| lam | 0.01 |
| h | 0.3 |
| alpha (CP level) | 0.1 (target 90% coverage) |

### Site Selection Method
- method = mixed  (gamma = 0.7 LHS + 0.3 adaptive sampling via S^0 score)

---

## Data Generating Processes

All three simulators share the same **true function** (exp2):

```
f(x) = exp(x/10) * sin(x),   x in [0, 2*pi]
```

The noise distributions differ:

### A1 — Heteroscedastic Student-t (heavy tails)
```
Y = f(x) + sigma(x) * T_nu
T_nu ~ t-distribution with nu = 3 degrees of freedom
sigma(x) = 0.05 + 0.5 * x       (increasing std)
Var(Y|x) = sigma^2(x) * nu / (nu - 2) = sigma^2(x) * 3
```

### B2 — Centered Gamma (skewed)
```
Y = f(x) + G - k * theta(x)
G ~ Gamma(k=2, theta(x)),   theta(x) = 0.1 + 0.4 * x
E[Y|x] = f(x)  (centered)
Var(Y|x) = k * theta^2(x) = 2 * theta^2(x)
Skewness = 2 / sqrt(k) = sqrt(2)  (right-skewed, constant)
```

### C1 — Gaussian Mixture (contamination / outliers)
```
Y = f(x) + epsilon
epsilon ~ (1 - pi) * N(0, s1^2) + pi * N(0, s2^2)
pi = 0.05,  s1 = 0.05 (inlier std),  s2 = 1.0 (outlier std)
Var(Y|x) = (1-pi)*s1^2 + pi*s2^2 = 0.05225  (homoscedastic)
```

---

## Previous Run Results (n_macroreps = 1)

Target coverage = 90% (alpha = 0.1)

| Simulator | Method | Coverage | Width | Interval Score |
|-----------|--------|----------|-------|----------------|
| A1 (Student-t) | CKME   | 0.910 | 8.371 | 11.878 |
| A1 (Student-t) | DCP-DR | 0.892 | 7.232 | 12.406 |
| A1 (Student-t) | hetGP  | 0.922 | 8.694 | 13.166 |
| B2 (Gamma)     | CKME   | 0.884 | 7.211 |  9.603 |
| B2 (Gamma)     | DCP-DR | 0.902 | 6.392 |  9.034 |
| B2 (Gamma)     | hetGP  | 0.940 | 6.572 |  8.850 |
| C1 (Gauss mix) | CKME   | 0.890 | 0.361 |  1.144 |
| C1 (Gauss mix) | DCP-DR | 0.892 | 2.369 |  2.973 |
| C1 (Gauss mix) | hetGP  | 0.960 | 0.747 |  1.298 |

**Notes:**
- n_macroreps = 1; standard deviations are not available.
- Interval Score = Width + (2/alpha) * undercoverage penalty; lower is better.
- C1 shows dramatically narrower widths because the dominant noise component
  has std = 0.05 (inliers, 95%), so well-calibrated intervals are very tight.

---

## Updated Settings (Current)

| Parameter | Old | New |
|-----------|-----|-----|
| n_0 | 200 | 250 |
| r_0 | 5 | 20 |
| Stage 1 total | 1,000 | 5,000 |
| n_1 | 100 | 500 |
| r_1 | 5 | 10 |
| Stage 2 total | 500 | 5,000 |
| n_test | 500 | 1,000 |
| Hyperparams | Fixed (config) | CV-tuned per simulator (pretrained_params.json) |

### CV-Tuned Hyperparameters (from pretrain_params.py)
| Simulator | ell_x | lam | h |
|-----------|-------|-----|---|
| A1 | 2.0 | 0.001 | 0.1 |
| B2 | 1.0 | 0.001 | 0.1 |
| C1 | 0.5 | 0.001 | 0.1 |
