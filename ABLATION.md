# Ablation Study Plan

This document records ablation ideas for the two-stage CKME framework.
Status: **PLANNING** — experiments not yet run. Update each entry when results are available.

---

## Framework

All ablations use:
- Same macro-rep seeds and total sample budget (n_0 × r_0 + n_1 × r_1 = fixed)
- Metrics: coverage, width, interval score (IS), **max bin deviation** (group coverage)
- Simulators: 6 nongauss sims (A1S/A1L/B2S/B2L/C1S/C1L)
- Theory reference: Note 2 (score homogeneity), Note 3 (group coverage, T-G1 through T-G4)

Full model label: **CKME-full** = Stage 2 + S0-guided site selection + smooth indicator (CV-tuned h)

---

## Ablation 1: S^0-Guided Site Selection vs Uniform LHS

**Question**: Does the S^0 uncertainty score actually help, or is space-filling sampling good enough?

| Variant | Description |
|---------|-------------|
| `CKME-full` | Stage 2 sites selected by S^0 score (method=mixed) |
| `CKME-no-S0` | Stage 2 sites selected by pure LHS (method=lhs) |

**Config change**: `method=lhs` instead of `method=mixed` in `run_nongauss_compare.py`

**Expected outcome**: CKME-full should give narrower intervals with similar coverage, especially in heteroscedastic regions where S^0 concentrates samples.

**Theory link**: S^0 sampling concentrates Stage 2 data in high-σ bins, increasing n_min in T-G2's binomial floor. Pure LHS may leave high-variance bins under-sampled, increasing max bin deviation.

**Status**: DONE (see `exp_design/report.md` and `project_exp_design.md`)

**Notation**:
- `B_total` — total simulator-call budget (main x-axis, log-spaced)
- `B_1=n_0·r_0`, `B_2=n_1·r_1`; balanced split `B_1=B_2=B_total/2`, `r_0=r_1=10`
- Δ IS = IS(LHS) − IS(adaptive variant); positive ⇒ adaptive wins
- `ρ = n_0 / n_0^*(d)` (fill-factor) — `n_0^*(d)` = Stage-1 site count at 90% asymptotic
  CRPS. Predefined regime classifier: `ρ<0.3` starved / `0.3≤ρ<1` intermediate /
  `ρ≥1` saturated. (`n_0^*(d)` calibration is a side experiment, pending.)

**Central claim — mechanism-dependent gain curve**:
The shape of Δ IS vs `B_total` depends on *what* starves Stage-1:
- **Heavy-tail mechanism** → ∩-shape (negative left, positive middle, 0 right)
- **High-d mechanism** → monotone decay (positive left, 0 right)
- **Saturated regime always → Δ IS ≈ 0** regardless of mechanism

Mechanism: adaptive compensates Stage-1 CDF error where largest *and* reliably
detectable; heavy-tail breaks detectability at low n₀, high-d amplifies the
underlying error.

**S⁰ variant strategy**: main sweep uses `tail` only (cheapest). `epistemic` is
appendix-level diagnostic on A1L_raw + gibbs_s1_d5 (existing data). Reason: d=5
result shows `epistemic` loses to `tail` in high-d starved — not a universal
better default; bootstrap overhead (30×) is impractical.

**Claim → Evidence → Figure**:

| # | Claim | Evidence | Figure |
|---|-------|----------|--------|
| C1 | Adaptive gain curve exists; shape mechanism-dependent | control: exp2_gauss; heavy-tail: A1L_raw + B2L_raw; high-d: gibbs_s1_d5 + gibbs_s2_d5 | Δ IS ± 1 SE vs `B_total` (6 curves, grouped by mechanism) |
| C2 | `ρ` predicts regime within each mechanism | all DGPs + `n_0^*(d)` calibration | Δ IS vs `ρ` (2-panel: heavy-tail, high-d) |
| C3 | Practical decision rule | derived from C1–C2 | (mechanism × regime) → action table |
| A1 | (Appendix) S⁰ variant sanity check | A1L_raw, gibbs_s1_d5 existing data | `tail` vs `epistemic` Δ IS on 2 DGPs |

**Empirical findings (2026-04-14)**:
- 1D Gauss: saturated throughout `B_total∈[500,10000]` → flat Δ IS. Control.
- 1D Student-t ν=3 (`nongauss_A1L_raw`): tail Δ=−0.48 @ B=500, rising to 0 @
  B=2000. ∩ shape confirmed.
- d=5 (`gibbs_s1_d5`): tail Δ=+0.29 @ B=1000, decaying to 0 @ B=10000. Monotone
  decay confirmed.

**Runner**: `exp_design/run_saturation_sweep.py` — balanced B₁:B₂=1:1 sweep,
outputs `delta_is_curve.csv` with `delta_<variant>_se` for ±1 SE bands.
Main sweep: `--methods lhs:tail sampling:tail` only. Planned: `B_total ∈
{500,1000,2000,5000,10000}` for 1D; `{1000,2000,5000,10000,20000}` for d=5.

---

## Ablation 2: Two-Stage vs One-Stage (Is Stage 2 Necessary?)

**Question**: Does adaptive Stage 2 data collection improve CP calibration over using Stage 1 data alone?

| Variant | Description |
|---------|-------------|
| `CKME-full` | Stage 1 (n_0, r_0) + Stage 2 (n_1, r_1); CP calibrated on Stage 2 |
| `CKME-1stage` | Stage 1 only with matched total budget; CP calibrated on Stage 1 data |

**Budget matching**: n_0_large × r_0_large ≈ (n_0 × r_0) + (n_1 × r_1)

**Expected outcome**: Two-stage should improve coverage in tail/heteroscedastic regions because Stage 2 concentrates calibration samples where coverage is hardest to achieve.

**Theory link**: In 1-stage, CP calibration uses training data → potential overfitting bias in scores. Two-stage uses independent Stage 2 data → clean exchangeability for CP. This is the p_{n+1} term in T-G1.

**Status**: PLANNED

**Results**: *(fill in after experiment)*

---

## Ablation 3: Smooth Indicator vs Hard Step Function

**Question**: Does the smooth logistic indicator (bandwidth h) matter for CDF estimation quality?

| Variant | Description |
|---------|-------------|
| `CKME-smooth` | Indicator = logistic function, h tuned by CV |
| `CKME-hard` | Indicator = Heaviside step function (h → 0) |

**Config change**: Pass `indicator="step"` to CKMEModel (or set h to very small value)

**Note**: `exp_conditional_coverage` already explored this partially — can reuse results for 1D cases.

**Expected outcome**: Smooth indicator reduces gradient noise in the CRPS loss, enabling better CV tuning. Hard step may perform comparably for large n but worse for small n.

**Theory link**: Note 2 Prop 1 requires smooth g(·) for the score homogeneity bound. Step function has g''=0 a.e. (zero bias in δ(c,n)), but CDF estimate has higher variance. The trade-off is bias (smooth) vs variance (step) in the structural term.

**Status**: PLANNED (partial data from exp_conditional_coverage)

**Results**: *(fill in after experiment)*

---

## Ablation 4: CV Hyperparameter Tuning vs Fixed Params

**Question**: How much does cross-validated (h, ell_x, lam) selection contribute vs using fixed defaults?

| Variant | Description |
|---------|-------------|
| `CKME-CV` | h, ell_x, lam selected by k-fold CRPS CV (pretrain_params.py) |
| `CKME-fixed` | Fixed default params: ell_x=1.0, lam=0.01, h=0.1 |

**Expected outcome**: CV tuning matters more for non-Gaussian simulators (B2L, C1L) where default h may over-smooth or under-smooth the CDF.

**Status**: PLANNED — lower priority, mostly sanity check

**Results**: *(fill in after experiment)*

---

## Ablation 5: Sample Budget Allocation Between Stages

**Question**: Given fixed total budget B = n_0 r_0 + n_1 r_1, what is the optimal split?

| Variant | n_0 | r_0 | n_1 | r_1 | Total |
|---------|-----|-----|-----|-----|-------|
| More Stage 1 | 400 | 20 | 250 | 10 | 10500 |
| Balanced (current) | 250 | 20 | 500 | 10 | 10000 |
| More Stage 2 | 150 | 20 | 700 | 10 | 10000 |

**Expected outcome**: Some optimal balance exists — too little Stage 1 → poor S^0 score → bad site selection; too little Stage 2 → insufficient CP calibration data.

**Status**: PLANNED

**Results**: *(fill in after experiment)*

---

## Ablation 6: Bin Count K (Partition Granularity)

**Question**: Does CKME's group coverage advantage hold across different partition granularities?

| Variant | K | Approx n_min per bin |
|---------|---|---------------------|
| Coarse | 5 | ~200 |
| Default | 10 | ~100 |
| Fine | 20 | ~50 |
| Very fine | 40 | ~25 |

**Config change**: Post-hoc re-binning of existing `per_point.csv` — **no new experiment needed**.

**Expected outcome**: T-G2 predicts binomial floor = sqrt(log(2K/η)/(2 n_min)). As K grows, floor rises (0.12 → 0.17 → 0.24 → 0.34). CKME should stay below the floor at all K. DCP-DR's structural gap (T-G3) is population-level and should persist regardless of K.

**Theory link**: Directly tests T-G1 (partition-free) and T-G2 (worst-bin decomposition). A plot of max_bin_dev vs K with the theoretical floor overlaid would be a strong visual for the paper.

**Status**: PLANNED — zero cost, pure post-hoc analysis

**Results**: *(fill in after experiment)*

---

## Ablation 7: Adaptive h(x) = cσ(x) vs Fixed h

**Question**: Does the adaptive bandwidth assumed by T-G1 actually improve group coverage over fixed CV-tuned h?

| Variant | Description |
|---------|-------------|
| `CKME-fixed-h` | Fixed h from CV tuning (current default) |
| `CKME-adaptive-h` | h(x) = c · σ_tar(x), oracle σ, c = 2.0 |

**Config**: Use `exp_conditional_coverage/run_consistency.py`'s `_adaptive_h` machinery with `oracle_var_fn`. Simulators: B2L, C1L (strongest non-Gaussianity). n_macro = 10.

**Expected outcome**: Adaptive h should reduce max bin deviation (structural term δ(c,n) → 0 by Note 2 Prop 1). Coverage at low-σ(x) bins (near x=π) should improve most, eliminating the U-shape seen with fixed h. The improvement may be modest since fixed-h CKME already sits below the binomial floor.

**Theory link**: This is the core assumption of T-G1/T-G2. Note 3 §5.1 flags this as a caveat ("Corollary 1 requires h(x)=cσ(x), while the main experiment uses fixed h"). This ablation directly resolves that caveat.

**Status**: PLANNED

**Results**: *(fill in after experiment)*

---

## Ablation 8: σ Dynamic Range (ρ_max Sensitivity)

**Question**: Does DCP-DR's group coverage gap scale with ρ_max = σ_max/σ_min as T-G3 predicts?

| Variant | σ_tar(x) | σ_min | σ_max | ρ_max |
|---------|----------|-------|-------|-------|
| Current | 0.1 + 0.1(x−π)² | 0.1 | 1.09 | 10.9 |
| Flat | 0.5 + 0.01(x−π)² | 0.5 | 0.60 | 1.2 |
| Medium | 0.3 + 0.05(x−π)² | 0.3 | 0.79 | 2.6 |

**Config change**: Requires new simulator variants with modified `_sigma_tar`. The true function f(x) stays the same — only noise scale changes.

**Expected outcome**: T-G3 predicts d_TV(S, ρS) → 0 as ρ → 1. So DCP-DR's max bin deviation should shrink dramatically for the flat variant (ρ=1.2), closing the gap with CKME. This would confirm that DCP-DR's poor group coverage is not a tuning failure but a structural consequence of heteroscedasticity + Y-space scores.

**Theory link**: Directly tests Proposition 5 (T-G3) and Remark 6. A scatter plot of empirical max_bin_dev vs predicted d_TV(S, ρS) would be a compelling validation.

**Status**: PLANNED — requires new simulator variants, higher effort

**Results**: *(fill in after experiment)*

---

## Summary Table (fill in after experiments)

| Ablation | Variant | Coverage (mean) | Width (mean) | IS (mean) | Max Bin Dev | vs CKME-full |
|----------|---------|----------------|--------------|-----------|-------------|--------------|
| 1 | CKME-full (mixed) | — | — | — | — | baseline |
| 1 | CKME-no-S0 (lhs) | — | — | — | — | — |
| 2 | CKME-full (2-stage) | — | — | — | — | baseline |
| 2 | CKME-1stage | — | — | — | — | — |
| 3 | CKME-smooth | — | — | — | — | baseline |
| 3 | CKME-hard | — | — | — | — | — |
| 4 | CKME-CV | — | — | — | — | baseline |
| 4 | CKME-fixed | — | — | — | — | — |
| 6 | K=5 | — | — | — | — | — |
| 6 | K=10 (default) | — | — | — | — | baseline |
| 6 | K=20 | — | — | — | — | — |
| 6 | K=40 | — | — | — | — | — |
| 7 | CKME-fixed-h | — | — | — | — | baseline |
| 7 | CKME-adaptive-h | — | — | — | — | — |
| 8 | ρ_max=10.9 (current) | — | — | — | — | baseline |
| 8 | ρ_max=2.6 | — | — | — | — | — |
| 8 | ρ_max=1.2 | — | — | — | — | — |

---

## Priority Order

1. **Ablation 6** (K variation) — zero cost post-hoc, directly validates T-G1/T-G2 partition-free claim
2. **Ablation 1** (S^0 score) — core two-stage claim; one config change
3. **Ablation 7** (adaptive h) — resolves note3 §5.1 caveat; validates T-G1 assumption
4. **Ablation 3** (smooth vs step indicator) — partial data exists from exp_conditional_coverage
5. **Ablation 2** (two-stage necessity) — reviewer will ask; needs new 1-stage runner
6. **Ablation 5** (budget allocation) — exploratory; run if time permits
7. **Ablation 8** (ρ_max sensitivity) — strongest theory validation (T-G3); needs new simulators
8. **Ablation 4** (CV tuning) — sanity check; lowest priority
