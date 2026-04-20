---
editor_options: 
  markdown: 
    wrap: 72
---

# exp2 Quantile Error Analysis

## Objective

This document investigates why CKME's **sup absolute quantile error at
τ=0.95** does not decrease as n_train grows from 500 → 1000 → 2000 →
5000 on the `exp2` simulator. The sup error remains roughly flat across
all training sizes, which is unexpected if the estimator is consistent.
We systematically examine four candidate error sources, assess the
evidence for each, and identify the root cause.

------------------------------------------------------------------------

## Error 1: t_grid Right-Boundary Clipping

### Description

When the estimated CDF F̂(t\|x) never reaches τ=0.95 within the t_grid,
the quantile inversion falls back to `t_grid[-1]`:

``` python
# exp2_quantile_error.py:170-171
has_valid = mask.any(axis=1)
q_ckme    = np.where(has_valid, t_grid[mask.argmax(axis=1)], t_grid[-1])
```

This happens because: - t_grid is built from a **single global
percentile** of all training Y (0.5th–99.5th percentile + 10% margin),
shared across all x - At x ≈ 0 and x ≈ 2π, noise variance σ²(x) =
(0.01 + 0.2(x−π)²)² is large, so the true q₀.₉₅(x) can exceed
t_grid[-1] - When q_true \> t_grid[-1], the returned q̂ = t_grid[-1] is a
fixed constant regardless of n_train, producing large, persistent error

### Evidence (from perpoint data, 20 macroreps × 500 test points each)

| n_train | Fraction clipped (at_right_bnd=1) |
|---------|-----------------------------------|
| 500     | 2.61%                             |
| 1000    | 2.60%                             |
| 2000    | 2.67%                             |
| 5000    | 2.64%                             |

-   Clipping fraction is **essentially constant** across n_train
    (\~2.6%)
-   Mean abs_err when clipped: **0.87**
-   Mean abs_err when not clipped: **0.18** (5× lower)
-   Clipping concentrates at **x ≈ 0 and x ≈ 2π** (high-σ regions)

### Why sup does not decrease

The sup error takes the max over all test points. Even though 97%+ of
test points improve with more data, the \~2.6% clipped points always
produce error \~0.87, so the sup is always dominated by them — pinned
regardless of n_train.

------------------------------------------------------------------------

## Error 2: Logistic Indicator Smoothing Bias (Structural)

### Description

The logistic smooth indicator approximates 1{y ≤ t}:

```         
g_t(y) = 1 / (1 + exp(-(t - y) / h))
```

With h=0.079, `g_t(y) = 0.95` occurs at `t - y = h · ln(19) ≈ 0.23`.
This means the estimated CDF F̂(t\|x) is evaluated at a **shifted**
threshold — introducing a systematic bias of \~0.23 units in the
quantile estimate.

This bias is **deterministic** (does not shrink with more data).

### Concrete example

Consider a single training observation Y = y near the true 0.95-quantile
q\*:

| Function | Value at t = y | Value at t = y + 0.23 | Reaches 1 at |
|----|----|----|----|
| True indicator 1{y ≤ t} | 1 (jumps immediately) | 1 | t = y |
| Logistic g_t(y), h=0.079 | 0.5 | 0.95 | t → ∞ only |

The logistic function needs t to be **0.23 units above y** before it
"votes" 0.95. With the true indicator, t = y is already enough to vote
1.

Because of this, F̂(q\* \| x) \< 0.95 — the estimated CDF is depressed
below the true CDF at the true quantile. To find where F̂ ≥ 0.95, CKME
must search further right, returning q̂₀.₉₅ \> q\* (systematic
overestimate by roughly h · ln(19) ≈ 0.23).

### Does smaller h help?

At first glance, yes: smaller h → h · ln(19) shrinks → less bias at
τ=0.95. But this is a **bias–variance tradeoff**:

-   **h too small**: logistic approaches the true step function → bias
    ↓, but the G_bar matrix becomes nearly discontinuous → CDF estimate
    is noisy and non-smooth, harder for CKME to generalize across sites.
-   **h too large**: smoother CDF estimate → variance ↓, but systematic
    bias at tails ↑.

The pretrained h = 0.079 was tuned to minimize average CRPS (a
full-distribution loss). A smaller h might reduce tail bias but would
require re-tuning lam and ell_x jointly. Using pinball loss at τ=0.95 as
the tuning objective would directly target this tradeoff.

### Evidence

-   Pretrained params: h = 0.079 (from `pretrained_params.json`)
-   Bias magnitude: h · ln(19) ≈ 0.079 × 2.944 ≈ 0.23
-   Params were tuned to minimize average CRPS (a full-distribution
    loss), not specifically for τ=0.95 tail accuracy

### Experiment: replacing logistic with step indicator

To test whether this bias is actually driving the non-decreasing sup
error at τ=0.95, the logistic indicator was replaced with the exact step
function `1{y ≤ t}` (no smoothing bias at all). Results (20 macroreps,
`exp2_raw_step.csv`):

| metric    | τ    | logistic                 | step                   | Improved? |
|-----------|------|--------------------------|------------------------|-----------|
| Sup error | 0.05 | flat \~0.6→0.2 partially | clear decrease 0.6→0.2 | Yes       |
| Sup error | 0.95 | flat \~0.87              | flat \~0.87            | **No**    |

**Conclusion**: switching to step indicator fixes sup error at τ=0.05
but has no effect on sup error at τ=0.95. Therefore smoothing bias is
**not** the cause of the τ=0.95 non-decreasing behavior. Error 1 (t_grid
clipping) is the dominant cause.

------------------------------------------------------------------------

## Error 3: Hyperparameters Not Optimized for Tail Quantiles

### Description

The fixed params (ell_x=0.239, lam=2.78e-4, h=0.079) were selected by
minimizing average CRPS across the full conditional distribution. CRPS
is a proper scoring rule that rewards accuracy across all quantile
levels equally — it does not prioritize the tails.

As a result: - **h** (indicator bandwidth) controls the smoothing bias;
CRPS-optimal h may be too small for accurate tail CDF estimation -
**lam** (regularization) controls coefficient smoothness; with
lam=2.78e-4 the coefficients are weakly regularized and may produce
non-monotone F̂ in sparse regions - Adding more training data improves
center-of-distribution accuracy but does not correct for the wrong h or
ell_x choice for τ=0.95

### Evidence

-   Params source: `pretrained_params.json`, key "exp2", tuned by CRPS
    in `pretrain_params.py`
-   No tail-specific tuning was performed

------------------------------------------------------------------------

## Error 4: Systematic CDF Under-estimation in High-Noise Regions (Root Cause)

### Description

Even after removing the t_grid right-boundary clip (by widening t_grid
from 0.5/99.5 to 0.1/99.9 percentile), the sup error at τ=0.95 does
**not** decrease with n_train and is actually **larger** (\~2.0) than
under the narrow grid (\~0.87).

This reveals that the clip was a symptom, not the root cause. The
fundamental problem is:

-   X is drawn uniformly from [0, 2π], so x≈0 and x≈2π receive the same
    density of training sites as x≈π
-   But σ(x) = 0.01 + 0.2(x−π)² at x≈0 is \~197× larger than at x≈π
-   The CDF at the far tail (τ=0.95) in high-σ regions requires far more
    data to estimate accurately — uniform sampling does not provide this
-   With wider t_grid, the clipping is removed but the CDF estimate in
    the extended tail region (few training Y values there) is highly
    noisy → quantile inversion returns an inaccurate, variable result
    that can be worse than the old clip

### Evidence (wide t_grid experiment, 20 macroreps, `exp2_raw_logistic_wide.csv`)

| n_train | mean sup (τ=0.95) | std    |
|---------|-------------------|--------|
| 500     | 1.9961            | 0.2395 |
| 1000    | 1.9108            | 0.2136 |
| 2000    | 1.9178            | 0.1526 |
| 5000    | 1.9921            | 0.0890 |

-   Mean sup stays flat (\~2.0) across all n_train — not decreasing
-   Std **does** decrease (0.24 → 0.09), showing the distribution is
    concentrating but the mean is pinned by the hardest test points
-   Mean abs error **does** decrease: 0.158 → 0.138 → 0.129 → 0.123
    (bulk of test points improve)

### Connection to the Two-Stage Framework

This is precisely the motivation for the adaptive two-stage design:

-   **Stage 1** (LHS uniform): provides a coarse CDF estimate everywhere
-   **S⁰ score**: measures local uncertainty, proportional to
    conditional variance / IQR → assigns high score to x≈0 and x≈2π
-   **Stage 2** (adaptive): concentrates the additional sample budget on
    high-S⁰ regions, improving CDF estimation where it matters most

In `exp_onesided`, there is no Stage 2 — it is a one-shot uniform
design. The sup error at τ=0.95 stagnating is thus a direct consequence
of uniform design failing heteroscedastic problems, and is the exact
scenario the two-stage framework is designed to address.

------------------------------------------------------------------------

## Summary Table

| \# | Error | Status for τ=0.95 sup | Key evidence |
|----|----|----|----|
| 1 | t_grid right-boundary clipping | **Symptom, not root cause** | Widening t_grid removes clip but sup stays \~2.0 (larger than clipped \~0.87); clipping was masking a deeper problem |
| 2 | Logistic indicator smoothing bias | **Ruled out for τ=0.95** (affects τ=0.05) | Step indicator experiment: sup error at τ=0.95 unchanged (\~0.87); sup error at τ=0.05 improved |
| 3 | Params optimized for CRPS, not τ=0.95 | Plausible but secondary | No direct experiment yet; fixed params from pretrained_params.json |
| 4 | CDF under-estimation in high-σ regions (uniform design) | **Confirmed root cause** | Wide t_grid experiment: sup flat at \~2.0, std shrinking; mean error does decrease; adaptive Stage 2 sampling is the intended fix |

------------------------------------------------------------------------

## Conclusion

The non-decreasing sup error at τ=0.95 is not caused by grid truncation
or indicator smoothing bias. The root cause is that uniform Stage 1
design fails to allocate sufficient sample density to high-variance
boundary regions (x ≈ 0 and x ≈ 2π), where the true upper-tail quantile
lies well beyond what the global t_grid covers accurately. Widening the
grid only exposes this underlying estimation difficulty more clearly.
The mean absolute error does decrease with n_train, confirming that CKME
improves for the bulk of the input space; the sup error is dominated by
the single hardest test point and cannot be reduced without either
adaptive Stage 2 sampling or a larger, targeted local budget. This is
precisely the scenario that the two-stage adaptive design framework is
intended to address.

------------------------------------------------------------------------

## Supplementary: Sup Error vs τ (exp2_sup_vs_tau.py)

To understand at which τ the non-convergence begins, we swept τ ∈ {0.5,
0.6, 0.7, 0.8, 0.9, 0.95} with n_train ∈ {500, 1000, 2000, 5000} using
CKME (step indicator, oracle t_grid).

### Findings

| τ    | Sup error trend with n_train      |
|------|-----------------------------------|
| 0.5  | Decreasing — convergence observed |
| 0.6  | Decreasing — convergence observed |
| 0.7  | Decreasing — convergence observed |
| 0.8  | Flat / no clear convergence       |
| 0.9  | Flat / no clear convergence       |
| 0.95 | Flat / no clear convergence       |

**Threshold**: convergence breaks down around τ ≈ 0.8. For τ ≤ 0.7, the
sup error decreases as n_train grows; for τ ≥ 0.8, it stagnates.

### Interpretation

This is consistent with the root cause identified in Error 4: high-σ
boundary regions (x ≈ 0, x ≈ 2π) dominate the sup error. As τ increases
toward 1, the true quantile moves further into the tail where density is
sparse and uniform design provides insufficient local coverage. The
transition at τ ≈ 0.8 reflects where the tail becomes "hard" under
uniform sampling for exp2's heteroscedastic noise profile.

Output CSV:
`/Users/zhaojin/Dropbox/Jin/CKME/Two-stage-simple/exp_onesided/output_sup_tau/sup_vs_tau_raw_step.csv`

![Sup error vs
tau](/Users/zhaojin/Dropbox/Jin/CKME/Two-stage-simple/exp_onesided/output_sup_tau/sup_vs_tau.png)
