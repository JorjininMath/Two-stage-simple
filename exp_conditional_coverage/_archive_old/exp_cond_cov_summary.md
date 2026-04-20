# Conditional Coverage Experiment Summary

## Experiment Goal

Verify the theoretical proposition that CKME-CP achieves **asymptotic conditional coverage**:

```
Cov_n(x) = P(Y_new ∈ C_{n,1-α}(x) | X_new = x)  →  1 − α   for a.e. x,  as n → ∞
```

This is strictly stronger than the marginal coverage guarantee of standard split-CP
(which only ensures the unconditional P(Y ∈ C(X)) ≥ 1−α).

---

## Experiment Settings

| Parameter | Value |
|-----------|-------|
| Simulators | `exp2`, `nongauss_B2L` |
| n_vals (Stage 1 = Stage 2 sites) | 50, 100, 200, 400, 800 |
| r_0 (Stage 1 reps per site) | 10 |
| r_1 (Stage 2 reps per site) | 10 |
| alpha | 0.1 (target coverage 90%) |
| t_grid_size | 300 |
| Site selection method | mixed (γ=0.7 LHS + 0.3 adaptive) |
| M_eval (eval grid points) | 100 |
| B_test (fresh draws per eval point) | 2000 |
| n_macro | 20 |
| Hyperparameters | CV-tuned per (simulator, n) via two-stage grid search |

### Hyperparameter Search Grid (Two-Stage CV)

**Stage 1** — fix `h = h_default`, search over `(ell_x, lam)`:

| Parameter | Grid |
|-----------|------|
| ell_x | 0.1, 0.5, 1.0, 2.0, 3.0 |
| lam   | 1e-4, 1e-3, 1e-2, 1e-1 |
| h (fixed) | 0.2 |

**Stage 2** — fix best `(ell_x, lam)`, search over `h`:

| Parameter | Grid |
|-----------|------|
| h | 0.05, 0.1, 0.2, 0.3, 0.5 |

### Conditional Coverage Estimation (Score-Based)

For each eval point x^(m) and each macrorep:
1. Compute q_hat from Stage 2 CP calibration (score = |F̂(Y|x) − 0.5|)
2. Generate B=2000 fresh Y_b ~ P(·|X=x^(m)) from the true simulator
3. For each Y_b, compute F̂(Y_b|x^(m)) directly using training Y values as indicator knots
   (mirrors `calibration.py` exactly — no t_grid inversion, no clipping)
4. Estimate: Cov_n(x^(m)) = (1/B) Σ_b 1{|F̂(Y_b|x^(m)) − 0.5| ≤ q_hat}

---

## Simulators

### exp2 — Gaussian Heteroscedastic

```
Y = f(x) + ε,     ε ~ N(0, σ²(x))
f(x) = exp(x/10) * sin(x),   x ∈ [0, 2π]
σ(x) = 0.01 + 0.2*(x − π)²
```

Key property: σ(x) is **extremely small near x=π** (σ(π) = 0.01) and grows to ~2.0
near x=0 and x=2π. This 200× variation across the domain creates a severe
challenge for any method with a **globally fixed bandwidth h**.

### nongauss_B2L — Centered Gamma (Strong Skew)

```
Y = f(x) + G − k*θ(x),   G ~ Gamma(k=2, θ(x))
f(x) = exp(x/10) * sin(x),   x ∈ [0, 2π]
σ_tar(x) = 0.1 + 0.1*(x − π)²,   θ(x) = σ_tar(x) / √2
```

Skewness = 2/√k = √2 ≈ 1.41 (constant right skew). Noise spread variation is
much more moderate (σ ranges ~0.1 to ~0.4) compared to exp2.

---

## Results

### Summary Table — MAE-Cov (mean ± SD over 20 macroreps)

**exp2 (Gaussian heteroscedastic)**

| n   | MAE-Cov mean | MAE-Cov SD | SupErr mean | SupErr SD |
|-----|-------------|------------|-------------|-----------|
| 50  | 0.0613      | 0.0078     | 0.165       | 0.048     |
| 100 | 0.0550      | 0.0052     | 0.118       | 0.020     |
| 200 | 0.0498      | 0.0051     | 0.107       | 0.015     |
| 400 | 0.0475      | 0.0027     | 0.100       | ≈0        |
| 800 | 0.0467      | 0.0025     | 0.100       | ≈0        |

**nongauss_B2L (Gamma strong skew)**

| n   | MAE-Cov mean | MAE-Cov SD | SupErr mean | SupErr SD |
|-----|-------------|------------|-------------|-----------|
| 50  | 0.0569      | 0.0077     | 0.135       | 0.044     |
| 100 | 0.0498      | 0.0053     | 0.111       | 0.023     |
| 200 | 0.0471      | 0.0034     | 0.103       | 0.010     |
| 400 | 0.0470      | 0.0020     | 0.105       | 0.019     |
| 800 | 0.0480      | 0.0017     | 0.097       | 0.003     |

---

## Key Observation: Coverage Plateau and the h-vs-σ Problem

### What we observe

**Figure 1: Conditional coverage curves x → Cov_n(x)**

![Figure 1: Conditional coverage curves](output/fig1_coverage_curves.png)

1. **n=50 → 200**: MAE-Cov decreases for both simulators — consistent with the
   asymptotic conditional coverage proposition.

2. **n=400 → 800**: MAE-Cov stagnates around 0.047–0.048. The error **does not
   continue decreasing** with more data. This is a **floor caused by systematic
   bias, not estimation variance** (SD also stabilizes near zero).

3. **exp2 Figure 1**: The conditional coverage curve x → Cov_n(x) shows a
   prominent **spike to ≈1.0 in the middle region (x ≈ 2–4, i.e., near π)**.
   This spike does not shrink as n grows — it is present and equally pronounced
   for n=400 and n=800.

4. **SupErr for exp2 at n=400/800 = exactly 0.100 with SD≈0**: Every single
   macrorep achieves the same maximum error, localized at the same x region.
   This is a deterministic structural failure, not random fluctuation.

### Root cause: globally fixed h vs. locally tiny σ(x)

The nonconformity score is based on F̂(Y|x), estimated with a smooth indicator
of bandwidth h. Near x=π, the true noise is σ(π) = 0.01, but the CV-tuned
bandwidth h ≈ 0.08–0.2 (tuned to be appropriate over the full domain).

The ratio h/σ(π) ≈ 8–20×: the indicator bandwidth is **far larger than the
actual spread of Y|x=π**.

**Consequence:**
- For test draws Y_b ~ N(0, 0.01²), all values lie in a tiny range ≈ ±0.03
- The smooth indicator g(Y_b − Y_train) varies negligibly over this range
  (its scale is h >> σ(π))
- Therefore F̂(Y_b | x=π) ≈ constant ≈ 0.5 for all Y_b
- Score_b = |F̂ − 0.5| ≈ 0 << q_hat for all Y_b
- **Cov_n(π) → 1.0 regardless of n**

This is confirmed by the quantile estimation diagnostic below:
at x=π, the CKME estimated quantile function spans ≈ [−0.25, 0.25] while the
true distribution is concentrated in ≈ [−0.03, 0.03]. The CDF estimate is
entirely dominated by h, not by the data.

**Figure 2: CKME quantile estimation at three representative x values (exp2)**

![Figure 2: CKME quantile estimation](../exp_onesided/output/exp2_quantile.png)

### Connection to asymptotic theory

The asymptotic conditional coverage proposition requires h → 0 as n → ∞
(or adaptive local bandwidth). With **fixed h**, the CDF estimate at
low-noise regions converges to a biased limit, not to the true F(y|x).
The systematic over-smoothing at x=π is not reducible by increasing n
— it is an O(h) approximation bias that persists in the limit.

This means the plateau in MAE-Cov is **not a failure of the theoretical
proposition**, but a consequence of violating its bandwidth assumption.

---

## exp2 vs nongauss_B2L: Why B2L behaves better

| Property | exp2 | nongauss_B2L |
|----------|------|-------------|
| σ(x) range | 0.01 – 2.0 (200× variation) | 0.10 – 0.40 (4× variation) |
| h/σ worst case | ~8–20× | ~0.5–2× |
| Over-smoothing | Severe at x=π | Mild, no single collapsed region |
| Coverage plateau | Yes, SupErr pinned at 0.10 | No pinning, mild plateau |

nongauss_B2L has a much more moderate σ variation, so the CV-tuned h is
roughly appropriate everywhere. The remaining plateau in B2L is likely
due to residual bias from the logistic indicator's O(h) approximation error
at the distribution tails, which also does not vanish with fixed h.

---

## Implications

1. **For the proposition**: The convergence Cov_n(x) → 1−α requires h→0
   (or local bandwidth selection). The current implementation uses fixed h,
   which is sufficient for mild heteroscedasticity (B2L) but fails for
   extreme heteroscedasticity (exp2 near x=π).

2. **For practice**: When σ(x) varies over orders of magnitude, a globally
   fixed h inevitably over-smooths the low-noise region. Adaptive bandwidth
   (e.g., h(x) ∝ local IQR of training Y near x) or variable kernel width
   would be needed for uniform conditional coverage convergence.

3. **For the paper**: exp2 illustrates a useful limitation to discuss explicitly.
   nongauss_B2L is the cleaner demonstration of the proposition because the
   bandwidth mismatch is mild. The exp2 result motivates future work on
   adaptive h selection.

---

## Next Steps

- [ ] Re-run with larger n_macro (50) on nongauss_B2L to confirm convergence trend
- [x] Try adaptive h(x) = local σ̂(x) * constant for exp2 to see if plateau disappears → **done, see below**
- [ ] Consider replacing exp2 with a simulator that has moderate heteroscedasticity
      (e.g., nongauss_A1L or a custom Gaussian with σ variation ≤ 5×)
- [ ] Add CQR-style score variant to `run_cond_cov.py` for comparison

---

## Adaptive-h Experiment

**Date**: 2026-04-08  
**Script**: `exp_conditional_coverage/run_cond_cov_adaptive_h.py`  
**Plots**: `exp_conditional_coverage/output_adaptive_h_c2.00/fig_compare_*.png`

### Design

Instead of a globally CV-tuned h, use a **point-wise adaptive bandwidth**:

```
h(x) = c_scale × σ̂(x)
```

where σ̂(x) is estimated via **k-NN** (k=5) from Stage 1 training replicates:

1. Compute per-site empirical std from model.Y (shape n_sites × r_0)
2. For query x, find k nearest training sites by Euclidean distance
3. Average their per-site stds → σ̂(x)
4. h(x) = max(c_scale × σ̂(x), h_min)

Applied **consistently** to both calibration (q_hat computation) and coverage check (test scores).

| Parameter | Value |
|-----------|-------|
| c_scale   | 2.0   |
| h_min     | 1e-3  |
| n_neighbors | 5   |
| n_macro   | 5     |
| Simulator | exp2  |

### h(x) Profile

The adaptive h(x) correctly reflects exp2's noise structure (Figure C):
- h(π) ≈ 0.02 (two-fold c_scale × σ(π)=0.01) — much smaller than CV-tuned h ≈ 0.05–0.2
- h(0) ≈ h(2π) ≈ 4.0 — large, appropriate for the high-noise boundary region

### Results — MAE-Cov (n_macro = 5)

**exp2 — Adaptive h (c=2.0) vs Fixed h (CV-tuned, 20 macroreps)**

| n   | Adaptive MAE-Cov | Adaptive SD | Fixed MAE-Cov | Fixed SD | Adaptive SupErr | Fixed SupErr |
|-----|-----------------|-------------|---------------|----------|-----------------|--------------|
| 50  | 0.0483          | 0.0096      | 0.0613        | 0.0078   | 0.137           | 0.165        |
| 100 | 0.0479          | 0.0065      | 0.0550        | 0.0052   | 0.128           | 0.118        |
| 200 | 0.0410          | 0.0047      | 0.0498        | 0.0051   | 0.113           | 0.107        |
| 400 | 0.0377          | 0.0014      | 0.0475        | 0.0027   | 0.109           | **0.100**    |
| 800 | 0.0394          | 0.0030      | 0.0467        | 0.0025   | 0.119           | **0.100**    |

### Key Findings

1. **x=π spike eliminated** (Figure B): Fixed-h coverage curve has a prominent spike to
   Cov_n(π)=1.0 that does not shrink with n. With adaptive h, the spike disappears — the
   curve is much flatter and no longer pinned at 1.0 near π.

2. **SupErr no longer deterministically pinned**: Fixed-h SupErr at n=400/800 is exactly
   0.100 with SD≈0 (every macrorep hits the same ceiling). Adaptive-h SupErr fluctuates
   (0.109–0.119) with SD > 0, indicating the structural lock is broken.

3. **MAE-Cov marginally better at large n**: Adaptive h achieves 0.037–0.039 at n≥400,
   vs 0.047–0.048 for fixed h. Improvement is ~0.009, not a dramatic reduction.

4. **Plateau persists**: MAE-Cov does not clearly decrease from n=400 to n=800 under
   adaptive h either. The plateau is only partially resolved.

### Root Cause Analysis: Why Adaptive h Is Not Sufficient

Even with h(π) ≈ 0.02, the CKME coefficient vector c(x=π) mixes contributions from
**all n_sites training sites** via the kernel matrix inversion. Distant sites with large
σ contribute ~Bernoulli(0.5) to the indicator regardless of the test draw Y_b:

```
F̂(Y_b | x=π) = Σⱼ c_j(π) · g_{Y_b}(Y_train,j)
```

Sites far from π have |Y_train| >> h(π), so g_{Y_b}(Y_train,j) is already near 0 or 1;
their contribution is a constant offset independent of Y_b. The effective local support
(training sites within ~h(π) of x=π in the output space) is small relative to the total
coefficient mass. Therefore |F̂(Y_b|π) − 0.5| remains small even with adaptive h.

**Conclusion**: Fixing h(x) alone is insufficient. The kernel length scale ell_x also
needs to be locally adapted (smaller near π) to restrict coefficient mass to truly
local training sites. Alternatively, replace exp2 with a simulator of moderate
heteroscedasticity where this mismatch does not arise.

### Output Files

```
exp_conditional_coverage/output_adaptive_h_c2.00/
  results_exp2.csv               # per-(macrorep, n, x_eval) rows with cov_mc, h_at_x
  summary_exp2.csv               # MAE-Cov mean/SD per n
  fig_compare_mae_cov.png        # Figure A: MAE-Cov vs n, fixed vs adaptive
  fig_compare_coverage_curves.png # Figure B: Cov_n(x) curves, 2-row comparison
  fig_h_profile.png              # Figure C: h(x) profile across domain
```
