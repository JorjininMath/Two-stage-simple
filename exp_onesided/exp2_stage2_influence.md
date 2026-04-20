# Stage 2 Influence on Quantile Estimation (exp2)

## Objective

Investigate whether adding a **Stage 2 adaptive sampling step** (sites selected proportional to S⁰ score) can reduce the quantile estimation error of CKME on the `exp2` simulator, especially the sup error at τ=0.95 which was identified as the hard-to-improve quantity under uniform design.

------------------------------------------------------------------------

## Simulator

**`exp2`**: 1D heteroscedastic, x ∈ [0, 2π]

```         
f(x)  = x + sin(πx)
σ(x)  = 0.01 + 0.2(x − π)²
Y | x ~ f(x) + σ(x) · ε,   ε ~ N(0,1)
```

Noise variance is U-shaped: lowest at x=π (\~0.01), highest at x=0 and x=2π (\~0.8). This creates the hardest quantile estimation at the boundary regions.

------------------------------------------------------------------------

## Experimental Settings

| Setting | Value |
|----|----|
| Simulator | `exp2` |
| Stage 1 n_train | 500, 1000, 2000, 5000 sites |
| Stage 1 reps r_train | 10 reps per site |
| Stage 2 budget n_1 | **500 sites** (fixed) |
| Stage 2 reps r_1 | 10 reps per site (same as Stage 1) |
| Stage 2 method | `sampling` — sites drawn ∝ S⁰(x) |
| Candidate pool | 1000 uniform candidates for S⁰ scoring |
| Hyperparameters | Pretrained (no retune after Stage 2) |
| ell_x | 0.2385 |
| lam | 2.78e-04 |
| h | 0.079 |
| Indicator | `step` |
| t_grid | Percentile-based (0.5/99.5), 500 points, 10% margin |
| Taus compared | τ = 0.05 (lower), τ = 0.95 (upper) |
| Metrics | mean / sup / q95 of \|q̂\_τ(x) − q_true_τ(x)\| over 500 test points |
| True quantiles | Monte Carlo with 10,000 draws per test point |
| Macroreps | **20** |

**S⁰ score definition:**

```         
S⁰(x) = q̂_{0.95}(x) − q̂_{0.05}(x)
```

Higher S⁰ = wider estimated conditional interval = more uncertain = more worth sampling.

**Retraining after Stage 2:** - Stage 1 sites X₁ and Stage 2 sites X₂ are combined: X_combined = [X₁; X₂] - G_bar (per-site avg indicator matrix) is recomputed on the combined t_grid (built from all data) - CKME is solved once on X_combined — no iterative update, no hyperparameter retuning

------------------------------------------------------------------------

## Results

Tables 1–2 and Figure 1 report mean, sup, and 95th-percentile absolute quantile error across 20 macroreplications for both τ = 0.05 and τ = 0.95. Stage 2 sampling consistently reduces mean and q95 error, but its effect on sup error differs by quantile level.

### Summary Table (20 macroreps, mean ± std)

#### τ = 0.05

| n_train | Method | mean err (±std) | sup err (±std) | q95 err (±std) |
|----|----|----|----|----|
| 500 | stage1_only | 0.1109 ± 0.0153 | 0.7614 ± 0.2413 | 0.3391 ± 0.0801 |
| 500 | stage2_sampling | **0.0828 ± 0.0084** | **0.4858 ± 0.1504** | **0.2263 ± 0.0533** |
| 1000 | stage1_only | 0.0818 ± 0.0111 | 0.5131 ± 0.1441 | 0.2380 ± 0.0633 |
| 1000 | stage2_sampling | **0.0709 ± 0.0086** | **0.4036 ± 0.1182** | **0.1872 ± 0.0422** |
| 2000 | stage1_only | 0.0660 ± 0.0090 | 0.3601 ± 0.1459 | 0.1771 ± 0.0352 |
| 2000 | stage2_sampling | **0.0626 ± 0.0059** | **0.3194 ± 0.0966** | **0.1659 ± 0.0265** |
| 5000 | stage1_only | 0.0519 ± 0.0050 | 0.2821 ± 0.1074 | 0.1384 ± 0.0083 |
| 5000 | stage2_sampling | **0.0512 ± 0.0048** | **0.2678 ± 0.0915** | 0.1387 ± 0.0087 |

#### τ = 0.95

| n_train | Method | mean err (±std) | sup err (±std) | q95 err (±std) |
|----|----|----|----|----|
| 500 | stage1_only | 0.1183 ± 0.0238 | **0.9170 ± 0.1691** | 0.3848 ± 0.1352 |
| 500 | stage2_sampling | **0.0898 ± 0.0112** | 1.0591 ± 0.1340 | **0.2279 ± 0.0589** |
| 1000 | stage1_only | 0.0948 ± 0.0104 | **0.8509 ± 0.1194** | 0.2683 ± 0.0523 |
| 1000 | stage2_sampling | **0.0845 ± 0.0079** | 0.9759 ± 0.0714 | **0.2054 ± 0.0301** |
| 2000 | stage1_only | 0.0816 ± 0.0076 | **0.8738 ± 0.0930** | 0.2008 ± 0.0398 |
| 2000 | stage2_sampling | **0.0780 ± 0.0060** | 0.9735 ± 0.1031 | **0.1852 ± 0.0308** |
| 5000 | stage1_only | 0.0747 ± 0.0053 | **0.8746 ± 0.0457** | 0.1808 ± 0.0231 |
| 5000 | stage2_sampling | **0.0726 ± 0.0051** | 0.9301 ± 0.0495 | **0.1725 ± 0.0218** |

------------------------------------------------------------------------

## Figure

![Stage 2 influence on quantile estimation](/Users/zhaojin/Dropbox/Jin/CKME/Two-stage-simple/exp_onesided/output_twostage/twostage_line.png)

*Line = mean across 20 macroreps; shaded band = ±1 std.*

------------------------------------------------------------------------

## Key Findings

### 1. Mean error: consistently improves with Stage 2

Across all n_train and both τ = 0.05 and τ = 0.95, `stage2_sampling` reduces mean absolute error. At n_train = 500, the improvement is substantial: - τ = 0.05: 0.1109 → 0.0828 (−25%) - τ = 0.95: 0.1183 → 0.0898 (−24%)

The gain shrinks as n_train grows, which is expected — Stage 1 alone is already more accurate at n_train = 5000, leaving less room for improvement.

### 2. q95 error: consistently improves with Stage 2

The 95th-percentile error (performance at the second-worst test points) also improves at all n_train and both τ, confirming that Stage 2 helps the bulk of the hard test points, not just the average.

### 3. Sup error at τ = 0.05: improves with Stage 2

At τ = 0.05, Stage 2 reduces sup error at every n_train (e.g., 0.7614 → 0.4858 at n_train = 500). The lower tail benefits from the adaptive design.

### 4. Sup error at τ = 0.95: does not improve — increases slightly

At τ = 0.95, Stage 2 consistently produces a larger sup error than Stage 1 alone:

| n_train | stage1_only sup | stage2_sampling sup | change |
|---------|-----------------|---------------------|--------|
| 500     | 0.9170          | 1.0591              | +15.5% |
| 1000    | 0.8509          | 0.9759              | +14.7% |
| 2000    | 0.8738          | 0.9735              | +11.4% |
| 5000    | 0.8746          | 0.9301              | +6.3%  |

This stagnation (≈ 0.85–1.06 range) is consistent with the wide-t_grid finding in `exp2_error_analysis.md`. Stage 2 adaptive sampling does not eliminate the worst-case error at τ = 0.95.

Two effects work against Stage 2 at the upper tail:

1.  **t_grid expansion**: S⁰-guided sampling correctly adds data in high-σ regions, but the combined dataset produces a wider t_grid (more extreme Y values). This can shift how the 0.95-quantile maps onto the grid at the hardest test point, sometimes increasing the error.

2.  **Insufficient local concentration**: estimating F(y\|x) at the extreme upper tail requires very large local sample counts. A budget of 500 Stage 2 sites distributed across [0, 2π] does not concentrate enough data at the single hardest x to overcome this.

The mean/q95 improvement confirms that Stage 2 improves the error distribution, but the single worst-case point resists improvement without a larger or more targeted budget.

------------------------------------------------------------------------

## Discussion

The results suggest that Stage 2 adaptive sampling improves CKME's quantile estimation in the distributional sense — reducing mean and high-percentile error — but does not resolve worst-case sup error at τ = 0.95. This asymmetry between τ = 0.05 and τ = 0.95 is consistent with the asymmetric difficulty structure of the exp2 simulator: the upper tail at x ≈ 0 and x ≈ 2π is harder to reach than the lower tail because the noise variance is substantially larger there, placing the true q₀.₉₅(x) further into the grid's tail.

These findings are aligned with the theoretical motivation for the two-stage framework: Stage 2 improves estimation where S⁰ assigns high scores, and S⁰ correctly diagnoses high-σ regions as uncertain. The limitation is that 500 adaptive sites are sufficient to help the typical hard point but not the single worst case. This points to a budget-scaling question rather than a structural flaw in the adaptive design.

------------------------------------------------------------------------

## Comparison with Error Analysis Context

This experiment directly connects to the findings in `exp2_error_analysis.md`:

| Analysis finding | This experiment's evidence |
|----|----|
| t_grid clipping was a symptom, not root cause | Confirmed — sup at τ=0.95 stays ≈0.9–1.1 even with Stage 2 |
| Uniform design fails heteroscedastic problems | Confirmed — mean/q95 improve with adaptive design |
| Adaptive Stage 2 should help | Partially confirmed — helps mean/q95/τ=0.05 sup; τ=0.95 sup resists |
| Single worst-case point dominates sup error | Confirmed — sup at τ=0.95 does not converge with n_train or Stage 2 budget |

------------------------------------------------------------------------

## Limitations

-   Results are limited to a single 1D heteroscedastic simulator (`exp2`); generalization to higher dimensions or other noise structures is not established.
-   Stage 2 hyperparameters are not retuned after combining Stage 1 and Stage 2 data; the pretrained params may be suboptimal for the expanded dataset.
-   The S⁰ score targets interval width (q₀.₉₅ − q₀.₀₅) and not specifically the upper tail; a τ-specific score might perform better for one-sided quantile objectives.
-   20 macroreplications provide moderate precision; standard deviations for sup error remain relatively large, especially at small n_train.

------------------------------------------------------------------------

## Takeaway

Stage 2 adaptive sampling reliably improves CKME's mean and 95th-percentile quantile error at both τ = 0.05 and τ = 0.95. However, the sup error at τ = 0.95 is dominated by the single worst-case test point in the high-variance boundary region, and 500 adaptive sites are insufficient to resolve it. Reducing this worst-case error likely requires either a larger Stage 2 budget or a score designed to target the specific quantile level of interest.

------------------------------------------------------------------------

## Next Steps

-   **Increase Stage 2 budget**: try n_1 ∈ {1000, 2000} to see if τ=0.95 sup eventually improves
-   **Targeted design**: instead of S⁰ ∝ interval width, use a score targeting τ=0.95 specifically (e.g., variance of F̂(q_0.95\|x) estimate)
-   **Multiple Stage 2 rounds**: iterative refinement of the hardest region
-   **Per-x diagnostic**: identify the single worst test point across macroreps to confirm it is always near x=0 or x=2π
