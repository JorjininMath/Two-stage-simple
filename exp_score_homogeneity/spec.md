# Experiment Specification: Score Homogeneity as the Mechanism for Conditional Coverage

## 1. Experimental Objective

This experiment tests a specific mechanistic hypothesis about CKME-CP:

> **Hypothesis**: In CKME-CP, adaptive bandwidth improves conditional coverage
> primarily by homogenizing the conformal score distribution across x, not by
> improving CDF/quantile estimation accuracy. These two effects can decouple:
> adaptive h may worsen pointwise CDF error while improving score homogeneity,
> and it is homogeneity — not accuracy — that drives conditional coverage.

The experiment is structured in two phases:

| Phase | DGP | Purpose |
|-------|-----|---------|
| Phase 1 | Location-scale Gaussian (exp2) | Exact case: adaptive h(x) = c·σ(x) makes the estimation problem structurally identical across x → score homogeneity should be near-perfect |
| Phase 2 | Non-location-scale (nongauss_B2L, centered Gamma) | Degradation: shape parameter varies effectively with x → score homogeneity degrades, but should still improve over fixed h |

### Theoretical grounding

The hypothesis rests on a three-level chain:

```
Level 1 (known, RCP-style):
  τ(x) = Q_{1-α}(s(x,Y)|X=x) ≈ const  →  conditional coverage

Level 2 (standard):
  ‖F̂ - F‖∞ small  →  τ(x) ≈ const

Level 3 (this experiment):
  τ(x) ≈ const  ←  does NOT require ‖F̂ - F‖∞ small!
  Sufficient condition: CDF bias δ(x,t) uniform across x.
  For location-scale DGP + h(x) = c·σ(x), this holds exactly.
```

Levels 1→2 are a monotone chain (better estimation → better coverage). This
experiment tests Level 3: score homogeneity and estimation accuracy can
**decouple**, and it is homogeneity that determines conditional coverage.

### Why location-scale is the natural starting point

For Y = f(x) + σ(x)·ε with ε independent of x, adaptive h(x) = c·σ(x) makes
the kernel smoothing operate at the same effective scale relative to the local
noise at every x. The standardized estimation problem (t − f(x))/σ(x) is
structurally identical across x, so the CDF bias δ(x,t) — expressed in
standardized units — does not depend on x. This is not an artificial corner
case; it is the design point of adaptive bandwidth.

---

## 2. Methods Compared

This experiment compares **bandwidth settings**, not methods (no DCP-DR / hetGP).

### Bandwidth configurations

| Label | h(x) | Description |
|-------|-------|-------------|
| `fixed_small` | h = 0.05 | Under-smoothed (high variance, low bias) |
| `fixed_cv` | h = h_CV | CV-optimal scalar h (from pretrained_params.json) |
| `fixed_large` | h = 0.5 | Over-smoothed (low variance, high bias) |
| `adaptive_c{c}` | h(x) = c · σ(x) | Adaptive, oracle σ(x); c scanned |

### c-scan values

c ∈ {0.5, 1.0, 1.5, 2.0, 3.0, 5.0}

This gives 3 fixed + 6 adaptive = 9 bandwidth configurations total.

### Score function

s(x, y) = |F̂(y|x) − 0.5|  (abs_median, the only score type in the CP module)

### CP calibration

Standard split-CP with a single global calibration quantile q̂. This is
critical: the experiment tests whether a **global** q̂ works uniformly across x,
which is exactly the conditional coverage question.

---

## 3. Simulators

### Phase 1: exp2 (Gaussian location-scale)

- Input: x ∈ [0, 2π], d = 1
- True mean: f(x) = exp(x/10) · sin(x)
- Noise std: σ(x) = 0.01 + 0.2·(x − π)²
- Noise: Y = f(x) + σ(x)·ε, ε ~ N(0,1) i.i.d.
- Oracle CDF: F(t|x) = Φ((t − f(x)) / σ(x))
- Oracle quantiles: q_τ(x) = f(x) + σ(x)·Φ⁻¹(τ)

σ(x) ranges from 0.01 (at x=π) to ~2.0 (at boundaries), providing strong
heteroscedasticity — the ideal setting to see bandwidth effects.

### Phase 2: nongauss_B2L (non-location-scale)

- Input: x ∈ [0, 2π], d = 1
- True mean: f(x) = exp(x/10) · sin(x) (same as exp2)
- Noise: centered Gamma with shape k=2 (strong right skew)
  - θ(x) = σ_tar(x) / √k,  σ_tar(x) = 0.1 + 0.1·(x − π)²
  - ε = Gamma(k, θ(x)) − k·θ(x), so E[ε]=0, Var(ε)=σ_tar²(x)
- Skewness = 2/√k = √2 ≈ 1.41 (constant across x)
- BUT: the Gamma shape is the same k everywhere, so skewness is constant.
  The non-location-scale property comes from the fact that higher moments
  (kurtosis) interact differently with the kernel smoothing at different
  noise levels σ_tar(x).

Oracle quantiles for nongauss_B2L are available via Gamma inverse CDF:
  q_τ(x) = f(x) + GammaInvCDF(τ; k, θ(x)) − k·θ(x)

---

## 4. Experimental Setup

### Budget (fixed, not scanning n)

This experiment uses a single budget to isolate the bandwidth effect:

- n_0 = 512 sites × r_0 = 1 rep = 512 Stage 1 observations
- n_1 = 512 sites × r_1 = 1 rep = 512 Stage 2 observations (calibration)
- Site selection: LHS (space-filling, not adaptive — isolate bandwidth effect)
- t_grid_size = 500

r_0 = r_1 = 1 because this experiment uses oracle σ(x) for adaptive h (no need
for local variance estimation from replicates). More sites with r=1 gives
better spatial coverage for kernel regression than fewer sites with r>1.

### Hyperparameters

- ell_x, lam: loaded from pretrained_params.json (exp_nongauss or dedicated
  CV tuning). Shared across all bandwidth configurations.
- h: varies by configuration (the independent variable of this experiment)

### Evaluation grid

- M_eval = 50 equally-spaced x-points in [0, 2π]
- B_test = 2000 fresh draws per eval point per macrorep
  (used for both score ECDF estimation and MC coverage estimation)
- Representative x-points for ECDF plots: 5 points at
  x ∈ {0.3, π/2, π, 3π/2, 5.7} (spanning low/medium/high σ regions)

### Macroreplications

- n_macro = 20 (more than exp_conditional_coverage's 10, because we need
  stable homogeneity metrics and Pareto plots)
- base_seed = 42

---

## 5. Metrics

### A. Score Homogeneity (core new metric)

For each bandwidth configuration and macrorep:

1. At each eval point x_m, generate B fresh draws Y_b ~ P(Y|x_m)
2. Compute scores s_b = |F̂(Y_b|x_m) − 0.5| for b = 1,...,B
3. This gives M empirical score CDFs: Ĝ_m(t) for m = 1,...,M

**Pooled score ECDF**: G̅(t) = (1/M) Σ_m Ĝ_m(t)

**Homogeneity metrics**:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| `ks_max` | max_m sup_t \|Ĝ_m(t) − G̅(t)\| | Worst-case deviation from pooled (primary) |
| `ks_mean` | mean_m sup_t \|Ĝ_m(t) − G̅(t)\| | Average deviation from pooled |
| `ks_pairwise` | mean_{m<m'} KS(Ĝ_m, Ĝ_{m'}) | Average pairwise KS distance |

Lower = more homogeneous. Perfect homogeneity = 0.

### B. Conditional Coverage Gap

Standard split-CP with global q̂, then MC evaluation:

| Metric | Definition |
|--------|-----------|
| `cov_gap_sup` | max_m \|Cov(x_m) − (1−α)\| |
| `cov_gap_mae` | mean_m \|Cov(x_m) − (1−α)\| |
| `width_mean` | mean_m (U(x_m) − L(x_m)) |
| `width_std_x` | std_m (U(x_m) − L(x_m)) (width variation across x) |

### C. Estimation Accuracy

| Metric | Definition |
|--------|-----------|
| `q_err_sup_lo` | max_m \|q̂_{α/2}(x_m) − q_{α/2}(x_m)\| |
| `q_err_sup_hi` | max_m \|q̂_{1−α/2}(x_m) − q_{1−α/2}(x_m)\| |
| `q_err_mae_lo` | mean_m \|q̂_{α/2}(x_m) − q_{α/2}(x_m)\| |
| `q_err_mae_hi` | mean_m \|q̂_{1−α/2}(x_m) − q_{1−α/2}(x_m)\| |
| `cdf_err_mean` | mean_m mean_t \|F̂(t\|x_m) − F(t\|x_m)\| (requires oracle CDF) |

---

## 6. Output Structure

```
exp_score_homogeneity/
├── spec.md                          ← this file
├── config.txt                       ← experimental parameters
├── run_homogeneity.py               ← main script
├── plot_homogeneity.py              ← all figures
├── output/
│   ├── results_exp2.csv             ← per (macrorep, bandwidth, x_eval) rows
│   ├── results_nongauss_B2L.csv
│   ├── score_ecdfs_exp2.csv         ← per (macrorep, bandwidth, x_eval, score_bin) rows
│   ├── score_ecdfs_nongauss_B2L.csv
│   ├── summary_exp2.csv             ← per bandwidth: mean ± SD of all metrics
│   ├── summary_nongauss_B2L.csv
│   ├── fig1_score_ecdf_overlay.png
│   ├── fig2_coverage_curve.png
│   ├── fig3_cscan_pareto.png
│   └── fig4_decoupling.png
```

### results CSV columns

| Column | Description |
|--------|-------------|
| macrorep | Macrorep index |
| bandwidth | Configuration label (e.g., "fixed_cv", "adaptive_c2.0") |
| x_eval | Evaluation point x_m |
| h_at_x | Actual bandwidth used at x_m |
| cov_mc | MC conditional coverage at x_m |
| L | CP lower bound |
| U | CP upper bound |
| q_hat | Global CP calibration quantile |
| q_lo_hat | Pre-CP quantile estimate q̂_{α/2}(x_m) |
| q_hi_hat | Pre-CP quantile estimate q̂_{1−α/2}(x_m) |
| q_lo_oracle | Oracle lower quantile |
| q_hi_oracle | Oracle upper quantile |
| ks_from_pooled | sup_t \|Ĝ_m(t) − G̅(t)\| at this x_m |

### summary CSV columns

| Column | Description |
|--------|-------------|
| bandwidth | Configuration label |
| ks_max_mean / _sd | Score homogeneity (worst-case KS) |
| ks_mean_mean / _sd | Score homogeneity (average KS) |
| cov_gap_sup_mean / _sd | Conditional coverage gap (sup) |
| cov_gap_mae_mean / _sd | Conditional coverage gap (MAE) |
| width_mean_mean / _sd | Mean interval width |
| q_err_sup_lo_mean / _sd | Quantile error (sup, lower) |
| q_err_sup_hi_mean / _sd | Quantile error (sup, upper) |
| cdf_err_mean_mean / _sd | CDF estimation error (mean) |

---

## 7. Figures

### Fig 1: Score ECDF Overlay

- Layout: 2 panels (left = fixed_cv, right = adaptive_c2.0)
- Each panel: 5 curves, one per representative x-point, color-coded by σ(x) level
- x-axis: score value t, y-axis: Ĝ_m(t)
- Expected: curves much more overlapping in the right panel (adaptive)
- Averaged over macroreps (mean ECDF ± shaded SD band)

### Fig 2: Coverage Curve vs x

- Layout: 1 panel, 3–4 curves
- Lines: fixed_small, fixed_cv, fixed_large, adaptive_c2.0
- x-axis: x, y-axis: local empirical coverage Cov(x)
- Horizontal dashed line at 1−α = 0.9
- Expected: adaptive curve closest to 0.9 line, fixed curves deviate especially
  at high/low σ(x) regions
- Mean ± shaded SD across macroreps

### Fig 3: c-scan Pareto Plot

- Layout: 1 panel
- x-axis: ks_max (score homogeneity, lower = better)
- y-axis: cov_gap_sup (conditional coverage gap, lower = better)
- Points: one per bandwidth configuration (3 fixed + 6 adaptive)
- Labels: bandwidth name next to each point
- Expected: monotone relationship — lower ks_max ↔ lower cov_gap_sup
- Error bars: ± 1 SD across macroreps

### Fig 4: Decoupling Plot

- Layout: 1 panel
- x-axis: q_err_sup (quantile estimation error, using max of lo/hi)
- y-axis: cov_gap_sup (conditional coverage gap)
- Points: one per bandwidth configuration
- Expected: NOT monotone — some adaptive configs have higher q_err_sup but
  lower cov_gap_sup than fixed_cv. This is the decoupling phenomenon.
- If points fall on a monotone curve, the hypothesis is not supported.

### Phase 2 extension

Repeat Fig 1–4 for nongauss_B2L. Expected differences:
- Fig 1: score ECDFs less overlapping even with adaptive h (shape heterogeneity)
- Fig 3: Pareto curve shifts up-right (worse homogeneity and coverage)
- Fig 4: decoupling may still appear but less pronounced

---

## 8. Scripts

| File | Role | CLI |
|------|------|-----|
| `spec.md` | This file | — |
| `config.txt` | Parameters | — |
| `run_homogeneity.py` | Main experiment | `--simulator exp2`, `--n_macro 20`, `--quick` |
| `plot_homogeneity.py` | All 4 figures | `--input_dir output/`, `--simulator exp2` |

### run_homogeneity.py workflow

For each simulator × macrorep × bandwidth configuration:
1. Train CKME (Stage 1) with shared (ell_x, lam), configuration-specific h
2. Collect Stage 2 calibration data, calibrate CP (using the configuration's h)
3. At each eval x_m:
   a. Compute score ECDF from B fresh MC draws
   b. Compute MC conditional coverage
   c. Compute pre-CP quantile estimates
4. Compute homogeneity metrics from score ECDFs
5. Save per-point results and aggregated summary

---

## 9. Connections to Paper

This experiment provides empirical evidence for the paper's key mechanistic claim:

> The primary role of adaptive bandwidth in CKME-CP is not to improve
> conditional CDF estimation accuracy, but to homogenize the conformal score
> distribution across the input space, enabling a single global calibration
> quantile to achieve approximately uniform conditional coverage.

This is distinct from:
- **exp_conditional_coverage**: tests asymptotic consistency (does coverage
  converge as n → ∞?). This experiment tests the **mechanism** at fixed n.
- **exp_onesided**: tests estimation-level quantile accuracy (no CP). This
  experiment explicitly decouples estimation from calibration.
- **RCP (Ben Taieb 2025)**: establishes that rectified scores → conditional
  coverage. This experiment shows that CKME's adaptive h achieves a similar
  effect from the estimation layer, without explicit score rectification.

### Theoretical implication

If the decoupling (Fig 4) is observed, it suggests that the appropriate quality
criterion for CDF-based conformal methods is not ‖F̂ − F‖∞ (pointwise accuracy)
but rather the **uniformity** of estimation error across x — a weaker and more
achievable requirement.

---

## 10. Success Criteria

The experiment **succeeds** if:

1. **Phase 1 (location-scale)**:
   - Fig 1 shows near-perfect score ECDF overlap for adaptive h
   - Fig 3 shows monotone relationship: ks_max ↔ cov_gap_sup
   - Fig 4 shows decoupling: at least one adaptive config with higher q_err
     but lower cov_gap than fixed_cv

2. **Phase 2 (non-location-scale)**:
   - Score homogeneity degrades relative to Phase 1, but adaptive h still
     improves over fixed h
   - Decoupling (Fig 4) is less pronounced but still visible

The experiment **does not succeed** if:
- ks_max and q_err_sup are perfectly correlated (no decoupling)
- adaptive h improves both homogeneity AND accuracy simultaneously
  (then we cannot distinguish the mechanisms)

---

## 11. Known Limitations

- Only d=1 simulators. Extension to d=2 (Branin) would test whether the
  mechanism holds in higher dimensions.
- Phase 1 uses oracle σ(x) for adaptive h. In practice, σ̂(x) must be
  estimated, introducing additional error. Testing with estimated σ̂ is a
  natural extension.
- Score type is limited to abs_median. CQR-style scores (max(q̂_lo−y, y−q̂_hi))
  would test whether the mechanism is score-type dependent.
- The c-scan uses 6 values. Finer resolution (e.g., 20 values) would give a
  smoother Pareto frontier but at higher computational cost.
- n_macro = 20 may not give very tight error bars for KS-based metrics. If
  variance is too high, increase to 50.
