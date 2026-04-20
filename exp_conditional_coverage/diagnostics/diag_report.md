# Diagnostic Report: Why does L1 fail to converge under adaptive `h(x) = 2σ(x)`?

**Parent experiment:** `exp_conditional_coverage` (see `../report.md`).
**Diagnostic spec:** `diag_spec.md`.
**Date:** 2026-04-09.

---

## A. Motivation

The main consistency experiment found a striking asymmetry in how the two
bandwidth modes converge:

| Metric (exp1, `n = 8192`)         | Fixed `h` (CV-tuned) | Adaptive `h(x) = 2σ(x)` |
|------------------------------------|----------------------|--------------------------|
| L1 — `MAE` of `q̂_{0.05}` (pre-CP) | converging           | **flat at ≈ 1.47**       |
| L1 — `MAE` of `q̂_{0.95}` (pre-CP) | converging           | **flat at ≈ 1.96**       |
| L3 — MAE conditional coverage      | slope ≈ −0.15        | slope ≈ −0.39            |

The adaptive setup achieves *better coverage convergence* with a *non-converging
estimator*. Section E of the main report attributed this to "CP absorbing the
over-smoothing bias", but the over-smoothing claim itself was not verified —
we only knew that L1 was flat, not *why*.

This diagnostic asks the narrower question:

> **Is the L1 failure under adaptive `h` caused by `h(x) = c·σ(x)` being too
> large, in the sense that the kernel covers a non-trivial fraction of the
> conditional support and biases `F̂(y|x)` toward an over-smoothed shape?**

If the answer is yes, the bias should be **structural** — i.e. flat in `n`,
with the variance shrinking but the estimator converging to the wrong limit.

---

## B. Diagnostics performed

Diagnostics 1, 2, 4 are **pure post-processing** of existing
`results_{exp1,exp2}.csv` from `output_consistency_fixed/` and
`output_consistency_adaptive_c2.00/`. Diagnostic 3 is a small additional
training experiment at `n = 512` over five `c` values.

| # | Question | What we measure | What "yes" looks like |
|---|----------|-----------------|-----------------------|
| 1 | Is `h(x)` large compared to the spread of `Y|x`? | `h_at_x` and `q_hi_oracle − q_lo_oracle` along `x` | Adaptive ratio not « 1 |
| 2 | Does the bias shrink with `n`? | `q̂_τ(x)` curves at `n=64` and `n=8192` overlaid on oracle | Adaptive curves overlap (no shrinkage) and lie outside oracle |
| 3 | Is the L1/L3 conflict monotone in `c`? | L1, L3 at `n=512` for `c ∈ {0.25, 0.5, 1, 2, 4}` | L1 grows in `c`, L3 falls in `c` |
| 4 | Is the error bias-dominated and structural? | bias² and variance of `q̂_τ` across 10 macroreps, vs `n` | Adaptive bias² flat / increasing; variance shrinks |

Diagnostic 5 (CDF overlay) is still deferred — we have not needed it.

Scripts: `run_diag_postprocess.py` (D1/D2/D4), `run_consistency.py …
--c_scale c` and `plot_cscan.py` (D3). Outputs in `output/diag{1,2,3,4}_*`
and `output_cscan/c_<c>/`.

---

## C. Results

### C.1 Diagnostic 1 — `h(x)` vs the spread of `Y|x`

![](output/diag1_h_vs_spread.png)

For adaptive `h`, the ratio

$$
\frac{h(x)}{q_{0.95}(x) - q_{0.05}(x)}
\;=\;
\frac{2\,\sigma(x)}{2\,\Phi^{-1}(0.95)\,\sigma(x)}
\;=\;
\frac{1}{\Phi^{-1}(0.95)}
\;\approx\; 0.608
$$

is **constant in `x` and equal to ≈ 0.608 in both simulators** — directly
verified in `diag1_h_vs_spread.csv`. Concretely, the adaptive smoothing kernel
covers a window roughly equal to **60 % of the central 90 % interval of
`Y|x`**. In absolute units the bandwidth ranges over

- **exp1**: `h_at_x ∈ [0.043, 14.2]`, median `0.42`
- **exp2**: `h_at_x ∈ [0.020, 3.97]`, median `1.03`

By contrast, the fixed-`h` mode uses `h = 0.05` everywhere, giving a ratio
that is mostly small (median 0.07 on exp1, 0.03 on exp2) except in
low-noise regions where the conditional spread is itself tiny — in particular
on exp2 around `x = π` the fixed ratio crosses 1, mirroring the localised dip
discussed in the main report.

**Reading.** Adaptive `h` is, by construction, aggressive: a kernel covering
60 % of the central 90 % of the conditional density cannot resolve tail
quantiles without a substantial smoothing bias.

### C.2 Diagnostic 2 — Quantile curves at `n = 64` vs `n = 8192`

![](output/diag2_quantile_curves.png)

For each simulator, the two panels show the macrorep-mean estimate
`q̂_{0.05}(x)` and `q̂_{0.95}(x)` against the oracle (black). Fixed-`h` curves
are blue (dashed `n=64`, solid `n=8192`); adaptive-`h` curves are orange.

The visual pattern is unambiguous:

- **Fixed `h`**: dashed and solid blue curves are clearly *different*; the
  solid `n=8192` curve sits much closer to the oracle than the dashed `n=64`
  curve, especially in the high-noise tails.
- **Adaptive `h`**: dashed and solid orange curves are **nearly
  indistinguishable**, and both lie **outside** the oracle (lower curve below,
  upper curve above) — a systematic *outward* bias of the same magnitude at
  `n=64` and at `n=8192`.

The geometric form of the bias is consistent with over-smoothing of `F̂`:
when the kernel is wide, `F̂(y|x)` becomes a more diffuse function of `y`,
its inversion at `τ = 0.05` is pushed further to the left, and at `τ = 0.95`
further to the right.

### C.3 Diagnostic 4 — Bias / variance decomposition

![](output/diag4_bias_var.png)

For each `(sim, mode, n, x_eval)` we compute, across 10 macroreps,

$$
\mathrm{bias}^2(x) \;=\; \big(\overline{\hat q_\tau}(x) - q_\tau(x)\big)^{2},
\qquad
\mathrm{var}(x) \;=\; \mathrm{Var}_k\!\big(\hat q_\tau(x)\big),
$$

then average each over the 100-point evaluation grid. Full table in
`diag4_bias_var.csv`; the headline numbers are:

| sim  | mode      | `n`  | `bias²(q̂_lo)` | `var(q̂_lo)` | `bias²(q̂_hi)` | `var(q̂_hi)` |
|------|-----------|------|----------------|--------------|----------------|--------------|
| exp1 | fixed     | 64   | 0.548          | 0.122        | 6.772          | 1.758        |
| exp1 | fixed     | 8192 | **0.006**      | **0.019**    | 0.559          | 0.012        |
| exp1 | adaptive  | 64   | 3.614          | 0.027        | 10.982         | 1.302        |
| exp1 | adaptive  | 8192 | **3.888**      | **0.0005**   | **8.743**      | 0.013        |
| exp2 | fixed     | 64   | 0.214          | 0.521        | 0.153          | 0.491        |
| exp2 | fixed     | 8192 | **0.013**      | **0.009**    | 0.021          | 0.010        |
| exp2 | adaptive  | 64   | 2.165          | 0.287        | 2.418          | 0.234        |
| exp2 | adaptive  | 8192 | **2.899**      | **0.008**    | **3.103**      | 0.007        |

Three observations:

1. **Adaptive bias² is flat or slightly *increasing* in `n`.** On exp1 q_lo
   it goes 3.61 → 3.89, on exp2 q_lo 2.16 → 2.90 — i.e., the more data we
   collect, the more confidently we land on the wrong answer. This is the
   textbook signature of an estimator that converges to a **biased limit**:
   variance shrinks, bias does not.
2. **Adaptive variance shrinks correctly.** On exp1 q_lo it falls
   `0.027 → 0.0005`, a ~55× reduction over a 128× increase in `n` — close
   to the `1/n` rate. The estimation machinery is working; it is just
   pointed at the wrong target distribution.
3. **Fixed `h` is bias-and-variance shrinking, except for one tail.** The
   one residual is exp1 `q̂_hi`, where bias² plateaus at ≈ 0.56 while
   variance crashes to 0.012 — a milder version of the same pathology
   driven by the modestly oversized scalar `h = 0.05` in the high-noise
   region near `x = 0.9`. (Worth noting because it contextualises the
   slope-`−0.15` rate seen in Section D.1 of the main report: even fixed
   `h` is not bias-free at high `x` for the upper tail.)

### C.4 Diagnostic 3 — `c`-scan at fixed `n = 512`

![](output/diag3_cscan.png)

To turn the over-smoothing story into a quantitative trade-off, we re-run the
adaptive consistency experiment at a single training size `n = 512` and
five bandwidth multipliers `c ∈ {0.25, 0.5, 1.0, 2.0, 4.0}`, with 5 macroreps
each. Script: `run_consistency.py --h_mode adaptive --c_scale c …`. Outputs
in `output_cscan/c_<c>/`, aggregated by `plot_cscan.py` into
`output/diag3_cscan.{png,csv}`.

| sim  | `c`  | L1 `MAE q̂_lo` | L1 `MAE q̂_hi` | L3 `MAE cov` |
|------|------|----------------|----------------|--------------|
| exp1 | 0.25 | **0.143**      | **0.382**      | 0.0299       |
| exp1 | 0.5  | 0.322          | 0.512          | 0.0220       |
| exp1 | 1.0  | 0.887          | 0.945          | 0.0155       |
| exp1 | 2.0  | 1.457          | 1.905          | 0.0126       |
| exp1 | 4.0  | 1.870          | 3.410          | **0.0120**   |
| exp2 | 0.25 | **0.199**      | **0.240**      | 0.0517       |
| exp2 | 0.5  | 0.338          | 0.406          | 0.0453       |
| exp2 | 1.0  | 0.739          | 0.866          | 0.0372       |
| exp2 | 2.0  | 1.211          | 1.387          | 0.0323       |
| exp2 | 4.0  | 1.630          | 1.845          | **0.0276**   |

The trade-off is monotone and clean on **both** simulators:

- **L1 quantile error grows roughly linearly in `c`.** From `c = 0.25` to
  `c = 4` it increases ≈ 13× on exp1 (`q̂_lo`: 0.14 → 1.87) and ≈ 8× on exp2
  (`q̂_lo`: 0.20 → 1.63). This is fully consistent with the over-smoothing
  picture: bigger kernel → bigger structural bias on tail quantiles.
- **L3 conditional coverage error *decreases* in `c`.** On exp1 it falls
  0.030 → 0.012 (≈ 2.5×); on exp2 0.052 → 0.028 (≈ 1.9×). Larger `c` gives CP
  more bias to absorb but also a smoother, more *learnable* score distribution.
- **The two effects are decoupled.** There is no value of `c` that wins on
  both axes — the L1-best (`c = 0.25`) and L3-best (`c = 4`) lie at opposite
  ends of the scan. Picking `c` is therefore a deliberate choice between
  estimator fidelity (L1) and calibrated coverage (L3).

The default `c = 2` used in Section D of the main report sits firmly on the
"L3-favouring" side of the curve. If we cared about L1 we would shrink to
`c ≈ 0.25–0.5`, where `h(x)` would be ≈ 7–15 % of the central 90 % interval —
back in the regime where the fixed-`h` mode already converges.

---

## D. Conclusion

All four diagnostics confirm the over-smoothing hypothesis cleanly:

- **D1.** Adaptive `h(x)` covers ≈ 60 % of the central 90 % conditional
  interval — by construction, not by accident, since the ratio is exactly
  `1/Φ⁻¹(0.95)` whenever `c = 2` and the noise is Gaussian.
- **D2.** The macrorep-mean quantile curves at `n = 64` and `n = 8192` are
  visually identical under adaptive `h`, and both sit systematically outside
  the oracle — exactly what an asymptotically biased estimator looks like.
- **D3.** Sweeping `c ∈ {0.25, …, 4}` at fixed `n = 512` traces a clean,
  monotone L1/L3 trade-off on both simulators: shrinking `c` recovers L1
  (≈ 13× drop in `MAE q̂_lo` on exp1) at the price of L3 (≈ 2.5× rise in
  `MAE cov`). There is no `c` that wins on both axes.
- **D4.** Bias² is flat or increasing with `n` while variance shrinks at the
  expected rate. The L1 failure is **structural**: the adaptive estimator
  converges, but to a wrong limit.

Combined with Section E of the main report, the picture is now complete:

1. Adaptive `h(x) = 2σ(x)` over-smooths `F̂(y|x)`, producing a systematic
   *outward* bias on tail quantiles whose magnitude is set by `h`, not by `n`.
2. This bias is symmetric around the conditional median (Gaussian noise) and
   smooth in `x` — exactly the shape that a *scalar* CP correction `q̂` can
   absorb by translating the inversion levels from `0.5 ± (1−α)/2` to
   `0.5 ± q̂`.
3. So the post-CP interval converges to the oracle (L2 in the report's
   current definition is post-CP, hence converges; L3 likewise), even though
   the underlying CKME quantile estimator does not. The improved L3 slope
   under adaptive `h` is **not** an estimator improvement — it is CP
   compensating for a tractable structural bias.

The L1 → L2 → L3 implication chain promised in `spec.md` therefore breaks
under adaptive `h`. Under fixed `h` it holds, because all three quantities
share the same bias/variance source.

---

## E. What this means for "fixing" L1

The c-scan (C.4) confirms that the L1-vs-L3 conflict is not an artefact of
`c = 2` — it is a genuine, monotone trade-off across the entire bandwidth
range we tested. Any "fix" therefore has to either (i) move along the
L1/L3 curve to a different operating point, or (ii) escape the curve by
changing how the bandwidth interacts with the CP correction. Directions in
order of intrusiveness:

1. **Smaller `c` (move along the curve).** From C.4, `c ∈ {0.25, 0.5}` gives
   L1 errors comparable to fixed `h` at the same `n`, at the cost of L3
   roughly doubling. Cheap and immediate; useful if a paper figure wants to
   show "adaptive can also recover L1 convergence".
2. **`n`-shrinking `c`.** Use `c(n) = c_0 (n/n_{\rm ref})^{-1/5}` so the
   bandwidth follows the classical nonparametric rate. This should give
   L1 convergence at the predicted `n^{−2/5}` *and* keep CP's heteroscedasticity
   awareness at small `n`. Implementation cost: trivial — one line in
   `run_consistency.py`. This is the natural follow-up.
3. **Decouple CP from estimation.** Use adaptive `h` only inside the CP
   score, while keeping the CDF estimate at fixed `h`. This makes CP "see"
   the heteroscedasticity directly rather than via a biased CDF. Implementation
   cost: moderate (need a second `F̂` instance with the smaller bandwidth).
4. **Bias-corrected quantile inversion** (jackknife / undersmoothing). High
   complexity, low expected gain in 1-D — not worth pursuing first.

---

## F. Limitations

- Both simulators use **Gaussian** noise, which is exactly the case where the
  smoothing bias of `F̂` is symmetric around the median and CP absorption
  works best. Under skewed or heavy-tailed noise the same "L1 broken, L3 fine"
  phenomenon may not hold — the CP scalar `q̂` would have to absorb both bias
  and asymmetry, which a single number cannot do well.
- The diagnostic only looks at `τ ∈ {0.05, 0.95}`; other quantiles may show a
  different bias profile. `exp_onesided` has scanned `τ` but for fixed `h`.
- Variance estimates are based on 10 macroreps, which is enough to see the
  order-of-magnitude difference in D4 but not enough for tight error bars on
  small `var` values.

---

## G. Update needed in the parent report

- Section C of `report.md` should explicitly state that the current "L2
  endpoint error" uses the **post-CP** interval endpoints
  `L_n(x) = F̂⁻¹(0.5 − q̂ | x)` and
  `U_n(x) = F̂⁻¹(0.5 + q̂ | x)`,
  not `F̂⁻¹(α/2 | x)` and `F̂⁻¹(1−α/2 | x)`. Hence L2 inherits CP's bias
  correction and is **not** a direct consequence of L1.
- Section E should reference this diagnostic as the empirical verification of
  the over-smoothing claim.
- Section F should note that the L1 → L2 → L3 chain holds for fixed `h` but
  is broken by adaptive `h`, and that the adaptive L3 improvement comes from
  CP absorbing structural bias rather than from better estimation.
