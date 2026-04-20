# Experiment Specification: Asymptotic Consistency of CKME-CP

## 1. Experimental Objective

This experiment evaluates whether CKME-CP achieves asymptotic consistency at three
hierarchical levels as the total design budget n grows:

| Level | Property | Formal statement |
|-------|----------|-----------------|
| L1 (strongest) | Quantile consistency | q̂_τ(x) → q_τ(x) uniformly in x, for fixed τ |
| L2 | Interval endpoint consistency | L_n(x) → q_{α/2}(x) and U_n(x) → q_{1−α/2}(x) pointwise |
| L3 (weakest) | Conditional coverage consistency | Cov_n(x) → 1−α pointwise |

These three levels form an implication chain: L1 → L2 → L3. The experiment is designed
to empirically verify all three, and to compare convergence rates between a fixed-bandwidth
(CV-tuned) CKME and an adaptive-bandwidth CKME where h(x) = c · σ̂(x).

The secondary question is whether the adaptive-h strategy accelerates convergence relative
to the fixed-h baseline, and at which level (L1, L2, or L3) the gap is most visible.

---

## 2. Methods Compared

### Method A: CKME-CP with fixed h (CV-tuned)
- Stage 1: train CKMEModel with CV-optimal (ell_x, lam, h); h is a scalar constant
- Stage 2: collect calibration data via LHS or mixed site selection; calibrate CP with
  abs-median score: s(x, y) = |F̂(y|x) − 0.5|
- Quantile inversion: q̂_τ(x) = inf{y : F̂(y|x) ≥ τ}

### Method B: CKME-CP with adaptive h(x) = c · σ̂(x)
- Same as Method A, except h varies per evaluation point
- σ̂(x) = kernel-weighted local standard deviation of training Y values at x
- c = c_scale (default 2.0, tunable)
- All other hyperparameters (ell_x, lam) are shared with Method A

### No baselines (DCP-DR / hetGP) in this experiment
This experiment isolates the CKME estimator's asymptotic behavior. Baseline comparison
(against DCP-DR, hetGP) is done in separate experiments (exp_nongauss, exp_branin).
Adding R-based baselines here would require matching oracle quantiles for every baseline,
which is not standardized.

---

## 3. Simulator

### Primary: exp1 (MG1 queue)
- Input: x ∈ [0.1, 0.9], dimension d = 1
- True mean: f(x) = 1.5 x² / (1 − x)
- Noise: ε ~ N(0, σ²(x)), heteroscedastic variance σ²(x) known analytically
- Oracle quantiles: q_τ(x) = f(x) + σ(x) · Φ⁻¹(τ), available in closed form

exp1 is chosen as the primary simulator because:
1. Oracle quantiles are available in closed form → L1/L2 consistency can be measured exactly
2. Gaussian heteroscedastic noise is the "easiest" setting → establishes a consistency
   baseline before testing on non-Gaussian simulators

### Secondary: exp2 (Gaussian heteroscedastic, damped sine)
- Input: x ∈ [0, 2π], dimension d = 1
- True mean: f(x) = exp(x/10) · sin(x)
- Noise std: σ(x) = 0.01 + 0.2 (x − π)²  (smallest at x = π, growing toward both boundaries)
- Noise: ε ~ N(0, σ²(x)), heteroscedastic variance σ²(x) known analytically
- Oracle quantiles: q_τ(x) = f(x) + σ(x) · Φ⁻¹(τ), available in closed form
- Rationale: tests consistency on a different noise-variance profile (distinct from exp1)
  while keeping oracle quantiles tractable

---

## 4. Experimental Setup

### Budget and design
- Stage 1: n_0 sites × r_0 = 1 rep each (no reps needed; consistency experiment, not
  estimation-variance experiment)
- Stage 2: n_1 = n_0 sites × r_1 = 1 rep each (matched budget: Stage 1 = Stage 2)
- Site selection method: lhs (space-filling) — the goal here is consistency, not
  demonstrating adaptive-design benefit; mixed or sampling would conflate the two effects
- n_vals: [64, 128, 512, 2048, 8192]  (log-spaced, ~4× steps, 5 points for rate estimation)
- Timing test: run n=8192 with n_macro=1 first to measure wall time before formal run

### Hyperparameter tuning
- Tune once via two-stage CV (pretrain_params.py) at each n in n_vals
- CV is performed on a dedicated pilot dataset (seed = base_seed + 999999) separate from
  macrorep data to avoid information leakage
- For n > cv_max_n (default 1024): reuse the params tuned at cv_max_n to avoid O(n³) CV
- Pretrained params are saved to pretrained_params.json and loaded at runtime

### Evaluation grid
- M_eval = 100 equally-spaced x-points in [0.1, 0.9]
- B_test = 2000 fresh draws per eval point per macrorep (Monte Carlo coverage estimate)
- Evaluation grid is fixed across all macroreps (same x-points, different fresh Y draws)

### Macroreplications
- n_macro = 10 (balanced: enough for SD estimation, light enough to run in <1 hour)
- base_seed = 42

---

## 5. Metrics

### L3: Conditional coverage consistency
For each eval point x_m and macrorep k:
  Cov_{n,k}(x_m) = (1/B) Σ_b 1{ Y_b(x_m) ∈ [L_{n,k}(x_m), U_{n,k}(x_m)] }

Aggregate across macroreps:
  - MAE-Cov(n) = mean_k { (1/M) Σ_m |Cov_{n,k}(x_m) − (1−α)| }
  - SupErr(n)  = mean_k { max_m |Cov_{n,k}(x_m) − (1−α)| }
  (SD across macroreps also reported)

Target: both metrics → 0 as n → ∞.

### L2: Interval endpoint consistency
For each eval point x_m and macrorep k, after computing L_{n,k}(x_m) and U_{n,k}(x_m)
from the CP interval:

  EndpointErr_L(n) = mean_k { (1/M) Σ_m |L_{n,k}(x_m) − q_{α/2}(x_m)| }
  EndpointErr_U(n) = mean_k { (1/M) Σ_m |U_{n,k}(x_m) − q_{1−α/2}(x_m)| }
  SupEndpointErr(n) = mean_k { max_m max(|L_{n,k}−q_lo|, |U_{n,k}−q_hi|) }

Requires: oracle quantiles q_{α/2}(x), q_{1−α/2}(x) available (exp1: yes).
Target: all endpoint errors → 0 as n → ∞.

### L1: Quantile consistency (estimation level, pre-CP)
For τ ∈ {α/2, 1−α/2}, evaluate the CKME quantile estimate before CP calibration:
  q̂_τ(x) = inf{ y : F̂(y|x) ≥ τ }

  QuantileErr_τ(n) = mean_k { (1/M) Σ_m |q̂_{τ,n,k}(x_m) − q_τ(x_m)| }
  SupQuantileErr_τ(n) = mean_k { max_m |q̂_{τ,n,k}(x_m) − q_τ(x_m)| }

Requires: oracle quantiles (exp1: yes).
Note: L1 is the estimator-level consistency before CP; it does not involve q_hat or Stage 2.
Target: QuantileErr_τ → 0 as n → ∞.

### Convergence rate (derived)
For each metric M(n), fit log M(n) = a + b · log(n) via OLS over the n_vals points.
Report estimated slope b̂ and overlay the fitted line in log-log plots.
Reference slope: nonparametric kernel regression theory predicts b ≈ −2/5 for d=1
under standard smoothness assumptions (Nadaraya-Watson rate). The CKME setting may
differ; b̂ serves as an empirical characterization, not a formal test.

---

## 6. Output Structure

Following the standard layout from experiment_code_skills:

```
exp_conditional_coverage/output_{h_mode}/
  pretrained_params.json        ← per-n CV-tuned params (from pretrain_params.py)
  summary_exp1.csv              ← all metrics aggregated: per n, mean ± SD across macroreps
  results_exp1.csv              ← per (macrorep, n, x_eval) rows (L, U, cov_mc, q̂_lo, q̂_hi, ...)
  fig_mae_cov_vs_n.png          ← MAE-Cov and SupErr vs n (log-log)
  fig_endpoint_err_vs_n.png     ← EndpointErr_L/U vs n (log-log)
  fig_quantile_err_vs_n.png     ← QuantileErr_τ vs n (log-log)
  fig_coverage_curves.png       ← Cov_n(x) vs x for each n (largest macrorep mean ± SD)
  fig_compare_fixed_vs_adaptive.png  ← side-by-side comparison (generated by plot_consistency.py)
```

h_mode is either `fixed` or `adaptive_c{c_scale}` (e.g. `adaptive_c2.00`).

### results_exp1.csv columns

| Column | Description |
|--------|-------------|
| macrorep | Macrorep index k |
| n_0 | Design budget label |
| x_eval | Evaluation point x_m |
| L | Lower CP bound L_n(x_m) |
| U | Upper CP bound U_n(x_m) |
| q_hat | CP calibration quantile (scalar, same for all x in same macrorep×n) |
| cov_mc | Monte Carlo conditional coverage estimate |
| q_lo_hat | CKME quantile estimate q̂_{α/2}(x_m) before CP |
| q_hi_hat | CKME quantile estimate q̂_{1−α/2}(x_m) before CP |
| q_lo_oracle | Oracle lower quantile q_{α/2}(x_m) |
| q_hi_oracle | Oracle upper quantile q_{1−α/2}(x_m) |
| h_at_x | Bandwidth used at x_m (scalar for fixed-h; per-point for adaptive-h) |

---

## 7. Scripts

| File | Role |
|------|------|
| `spec.md` | This file. Written before running. |
| `config.txt` | Standard config (parsed by `Two_stage.config_utils`) |
| `pretrain_params.py` | One-time CV tuning per n; writes pretrained_params.json |
| `run_consistency.py` | Main script: `--h_mode {fixed,adaptive}`, `--quick`, standard skeleton |
| `plot_consistency.py` | All figures: coverage curves, endpoint errors, quantile errors, log-log rates |

Legacy scripts (`run_cond_cov.py`, `run_cond_cov_adaptive_h.py`, `plot_cond_cov.py`,
`plot_adaptive_vs_fixed.py`) are retained for reference but superseded by the above.

---

## 8. Connections to Paper

This experiment provides empirical evidence for the paper's core theoretical claims:

- **Theorem (asymptotic conditional coverage)**: Under regularity conditions, the CKME-CP
  interval achieves Cov_n(x) → 1−α as n → ∞. This experiment tests L3.
- **Corollary (interval shrinkage)**: As n → ∞, the interval converges to the oracle
  interval. This experiment tests L2.
- **Underlying estimator**: L1 verifies that the CKME quantile estimator is consistent
  before CP calibration, which is the estimator-level premise for both the corollary and
  the theorem.

Per the report writing skill, all coverage consistency claims in the paper should be
phrased as "consistent with asymptotic theory" or "empirically suggestive", not as
formal proofs. The experiment cannot establish conditional coverage guarantees; it
provides finite-sample evidence at n ∈ {128, 512, 2048}.

---

## 9. Known Limitations

- Only d=1 simulators (exp1, exp2). Extension to d=2 (Branin) is a separate experiment.
- Both simulators use Gaussian noise; non-Gaussian consistency is left to a future extension.
- n_vals = {64, 128, 512, 2048, 8192} gives 5 points for rate estimation, but the n=8192
  point requires ~4× more compute than n=2048; wall time is verified with a 1-macrorep
  timing run before the formal experiment.
- r_0 = r_1 = 1: local sigma estimation (knn mode) requires r_0 ≥ 2; adaptive-h must use
  sigma_mode=oracle (true σ(x) from simulator). This is a limitation for real applications
  where σ(x) is unknown.
- The CP interval uses abs-median score, which is two-sided. One-sided consistency is not
  evaluated here; see exp_onesided.
- L1 quantile errors are computed at τ = α/2 and 1−α/2 only. Performance at other τ
  values (especially tail quantiles) may differ; see exp_onesided for the full τ-scan.
