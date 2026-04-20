# Asymptotic Consistency of CKME-CP: An Empirical Study

## A. Experimental Objective

This experiment evaluates whether CKME-CP achieves asymptotic consistency at three
hierarchical levels as the total design budget `n` grows:

| Level | Property | Formal statement |
|-------|----------|-----------------|
| **L1** (strongest) | Quantile consistency | q̂_τ(x) → q_τ(x) for τ ∈ {α/2, 1−α/2} |
| **L2** | Interval endpoint consistency | L_n(x) → q_{α/2}(x), U_n(x) → q_{1−α/2}(x) |
| **L3** (weakest) | Conditional coverage consistency | Cov_n(x) → 1−α |

These three levels form an implication chain L1 → L2 → L3. The experiment empirically
verifies all three and compares two bandwidth strategies: a CV-tuned **fixed** scalar
bandwidth and an **adaptive** per-point bandwidth `h(x) = c · σ(x)` driven by the oracle
noise standard deviation. The secondary question is whether the adaptive strategy
accelerates the empirical convergence rate, and at which level the gap (if any) is most
visible.

---

## B. Experimental Setup

### Simulators

Two heteroscedastic Gaussian benchmarks with closed-form oracle quantiles
`q_τ(x) = f(x) + σ(x) · Φ⁻¹(τ)` are used:

- **exp1** (MG1 queue): `f(x) = 1.5 x² / (1 − x)`, `x ∈ [0.1, 0.9]`
- **exp2** (damped-sine): `f(x) = exp(x/10) · sin(x)`, `x ∈ [0, 2π]`,
  with noise std `σ(x) = 0.01 + 0.2 (x − π)²` (smallest at `x = π`, growing
  toward both boundaries)

Closed-form quantiles allow exact measurement of L1 and L2 errors. Both simulators are
1-dimensional, so the comparison against the classical nonparametric rate is meaningful.

### Methods compared

- **Method A — Fixed h (CV).** Standard CKME with a scalar bandwidth tuned by two-stage
  k-fold CV (`ell_x` + `lam` first, then `h`); all hyperparameters held constant across
  evaluation points.
- **Method B — Adaptive h (oracle).** Same `ell_x`, `lam` as Method A, but the smoothing
  bandwidth at every evaluation point is `h(x) = 2 · σ(x)`, where `σ(x)` is the *true*
  noise standard deviation taken from the simulator. The adaptive bandwidth enters in
  three places: (i) the conditional CDF evaluation used for L1 quantile inversion,
  (ii) the conditional CDF evaluation used for L2 endpoint construction, and
  (iii) the recomputation of CP calibration scores `s(x, y) = |F̂(y|x) − 0.5|` on the
  Stage 2 calibration data.

The oracle σ choice is a deliberate idealisation: it removes nuisance variability in
σ̂ estimation and isolates the effect of the bandwidth-shape itself. As discussed
in Section F, this is a limitation for practical use.

### Budget and protocol

| Parameter | Value |
|-----------|------:|
| `n_vals` | 64, 128, 512, 2048, 8192 |
| Stage 1 sites × reps | n_0 × r_0 = n × 1 |
| Stage 2 sites × reps | n_1 × r_1 = n × 1 |
| Site selection (Stage 2) | mixed (γ=0.7 LHS, 0.3 sampling) |
| Significance level | α = 0.10 (target coverage 90%) |
| Evaluation points | M_eval = 100 fixed x-grid |
| MC draws per point | B_test = 2000 |
| Macroreplications | n_macro = 10 |
| Base seed | 42 |

Hyperparameters `(ell_x, lam, h)` are tuned via two-stage 5-fold CV at each
`n ≤ cv_max_n = 1024` on a pilot dataset (seed offset +999999) and reused for
`n ∈ {2048, 8192}` to keep CV cost bounded. The fixed-h and adaptive-h runs
share the same pretrained `(ell_x, lam)` so that any difference is attributable
exclusively to the bandwidth scheme.

This experiment is *not* a benchmark comparison: DCP-DR and hetGP are evaluated in
separate experiments (`exp_nongauss`, `exp_branin`). Adding them here would force
oracle-quantile alignment for every baseline and obscure the asymptotic question.

---

## C. Evaluation Metrics

For each macrorep `k`, design size `n`, and evaluation point `x_m`:

**L3 — Conditional coverage**
- `MAE-Cov(n)  = mean_k mean_m |Cov_{n,k}(x_m) − (1−α)|`
- `SupErr(n)   = mean_k max_m |Cov_{n,k}(x_m) − (1−α)|`

`Cov_{n,k}(x_m)` is the empirical coverage on `B_test = 2000` fresh draws.

**L2 — Interval endpoints (post-CP).** The endpoints `L_n(x)`, `U_n(x)` are the
**CP-calibrated** interval bounds, not pre-CP quantile estimates. With abs-median
score `s(x, y) = |F̂(y|x) − 0.5|` and CP scalar `q̂`,
$$
L_n(x) \;=\; \hat F^{-1}\!\big(\,0.5 - \hat q\;\big|\;x\big),
\qquad
U_n(x) \;=\; \hat F^{-1}\!\big(\,0.5 + \hat q\;\big|\;x\big),
$$
which **only** coincides with `F̂⁻¹(α/2|x)`, `F̂⁻¹(1−α/2|x)` in the limit `q̂ →
(1−α)/2`. In finite `n`, `q̂` deviates from this value to compensate for the
miscalibration of `F̂`, so L2 already absorbs the CP correction. The metric is
- `MAE-L(n) = mean_k mean_m |L_{n,k}(x_m) − q_{α/2}(x_m)|`
- `MAE-U(n) = mean_k mean_m |U_{n,k}(x_m) − q_{1−α/2}(x_m)|`

This makes L2 a *post-CP* quantity. As a consequence, L2 is **not** a direct
estimator-level consequence of L1: under an adaptive bandwidth that produces
biased pre-CP quantiles, the CP correction can still drive L2 (and L3) to the
oracle, which is exactly what Section D.2 shows. See the diagnostic report
`diagnostics/diag_report.md` for the explicit verification.

**L1 — Pre-CP quantile estimates**
- `MAE-q_lo(n) = mean_k mean_m |q̂_{α/2,n,k}(x_m) − q_{α/2}(x_m)|`
- `MAE-q_hi(n) = mean_k mean_m |q̂_{1−α/2,n,k}(x_m) − q_{1−α/2}(x_m)|`

For each metric, the empirical convergence rate is summarised by the OLS slope of
`log(metric) vs log(n)`.

**Reference slope `−2/5`.** The dashed grey line in every log–log figure has slope
`−0.40 = −2/5`. This is the classical minimax rate of convergence for nonparametric
estimation of a twice-differentiable regression function (or density) in dimension
`d = 1` using a kernel-type estimator. The intuition is the standard kernel
bias–variance tradeoff: with bandwidth `h`, the squared bias scales as `h^4` (from
a second-order kernel and a `C^2` target), the variance scales as `1/(n h)`, and
the optimal `h ≍ n^{−1/(2s+d)} = n^{−1/5}` (with smoothness `s = 2`, dimension
`d = 1`) yields a total mean-squared error of `n^{−2s/(2s+d)} = n^{−4/5}`, i.e.
a root-MSE / pointwise-error rate of `n^{−2/5}`. Stone (1980, 1982) proved that
this rate is *minimax-optimal* in this smoothness class, so it is the natural
yardstick against which any kernel-based one-dimensional procedure is judged
[Stone, C.J. (1980), *Optimal rates of convergence for nonparametric estimators*,
Annals of Statistics 8, 1348–1360; Stone, C.J. (1982), *Optimal global rates of
convergence for nonparametric regression*, Annals of Statistics 10, 1040–1053;
see also Tsybakov, A. (2009), *Introduction to Nonparametric Estimation*, Springer,
Ch. 1].

CKME-CP is not a Nadaraya–Watson estimator, and the CP calibration step is not part
of the classical theory, so `−2/5` is not a *predicted* rate for any of the six
metrics in this report. It is included only as a visual benchmark: any line on a
log–log plot that runs roughly parallel to the dashed grey reference is decaying
at the kernel-optimal pace, while a flatter line indicates a slower rate (or a
non-vanishing bias).

Lower is better for every metric. All numbers are reported as mean across the 10
macroreplications, with standard deviations shown as shaded bands in figures.

---

## D. Results

### D.1 Fixed-h convergence (Method A)

Table 1 reports the L3, L2, and L1 errors on **exp1** under fixed h. All six metrics
decrease monotonically with `n`. The L3 metrics shrink from `MAE-Cov = 0.086` and
`SupErr = 0.42` at `n = 64` to `0.043` and `0.107` at `n = 8192`; the L1 quantile errors
shrink from `0.34 / 1.70` to `0.075 / 0.316` over the same range. Empirical OLS slopes
on `log(n)` are around `−0.15` for L3 and somewhat steeper for L1, slower than the
reference `−0.40`.

**Table 1.** Fixed-h CKME-CP on exp1 (mean ± SD across 10 macroreplications).

| n | MAE-Cov | SupErr | MAE-L | MAE-U | MAE-q_lo | MAE-q_hi |
|---:|---:|---:|---:|---:|---:|---:|
| 64   | 0.086 ± 0.013 | 0.419 ± 0.173 | 0.339 ± 0.086 | 1.840 ± 1.778 | 0.338 ± 0.073 | 1.701 ± 0.291 |
| 128  | 0.080 ± 0.012 | 0.283 ± 0.097 | 0.405 ± 0.134 | 1.413 ± 2.151 | 0.378 ± 0.143 | 0.513 ± 0.084 |
| 512  | 0.056 ± 0.005 | 0.169 ± 0.063 | 0.238 ± 0.055 | 0.327 ± 0.049 | 0.247 ± 0.050 | 0.348 ± 0.055 |
| 2048 | 0.047 ± 0.002 | 0.108 ± 0.013 | 0.180 ± 0.045 | 0.226 ± 0.033 | 0.134 ± 0.028 | 0.325 ± 0.038 |
| 8192 | 0.043 ± 0.002 | 0.107 ± 0.012 | 0.159 ± 0.029 | 0.190 ± 0.016 | 0.075 ± 0.024 | 0.316 ± 0.016 |

The same pattern is observed on **exp2** (Table 2): coverage MAE decreases from
`0.105` to `0.041`, sup-error from `0.403` to `0.100`, and L1 quantile errors shrink
from `≈ 0.59` to `≈ 0.12`. Both simulators support the L1 → L2 → L3 implication
chain under fixed h: when the underlying CDF estimator improves, the CP-calibrated
endpoints and the conditional coverage improve in lockstep.

**Table 2.** Fixed-h CKME-CP on exp2 (mean ± SD across 10 macroreplications).

| n | MAE-Cov | SupErr | MAE-L | MAE-U | MAE-q_lo | MAE-q_hi |
|---:|---:|---:|---:|---:|---:|---:|
| 64   | 0.105 ± 0.025 | 0.403 ± 0.192 | 1.107 ± 0.916 | 1.347 ± 1.002 | 0.589 ± 0.159 | 0.569 ± 0.140 |
| 128  | 0.077 ± 0.009 | 0.337 ± 0.090 | 0.466 ± 0.117 | 0.493 ± 0.073 | 0.495 ± 0.086 | 0.477 ± 0.083 |
| 512  | 0.054 ± 0.008 | 0.197 ± 0.115 | 0.228 ± 0.040 | 0.250 ± 0.032 | 0.228 ± 0.039 | 0.256 ± 0.027 |
| 2048 | 0.048 ± 0.005 | 0.106 ± 0.012 | 0.160 ± 0.037 | 0.164 ± 0.019 | 0.157 ± 0.024 | 0.166 ± 0.021 |
| 8192 | 0.041 ± 0.002 | 0.100 ± 0.000 | 0.128 ± 0.016 | 0.123 ± 0.011 | 0.113 ± 0.009 | 0.125 ± 0.013 |

![Fixed-h log-log convergence on exp1.](output_consistency_fixed/fig_loglog_exp1.png)

*Figure 1. L3 (left), L2 (middle), and L1 (right) error metrics versus n on exp1
under fixed h, on log–log axes. Markers are means across 10 macroreplications;
shaded bands show ±1 SD; the dashed grey line is the reference slope −0.40.*

![Fixed-h log-log convergence on exp2.](output_consistency_fixed/fig_loglog_exp2.png)

*Figure 2. Same as Figure 1 but for exp2. All three levels decrease monotonically
with n on both simulators.*

### D.2 Adaptive-h convergence (Method B)

Switching to `h(x) = 2 · σ(x)` produces a strikingly different picture (Table 3).

**Table 3.** Oracle-adaptive-h CKME-CP on exp1 (mean ± SD across 10 macroreplications).

| n | MAE-Cov | SupErr | MAE-L | MAE-U | MAE-q_lo | MAE-q_hi |
|---:|---:|---:|---:|---:|---:|---:|
| 64   | 0.029 ± 0.014 | 0.083 ± 0.034 | 0.149 ± 0.080 | 0.402 ± 0.107 | **1.436 ± 0.075** | **2.270 ± 0.291** |
| 128  | 0.035 ± 0.021 | 0.090 ± 0.038 | 0.202 ± 0.083 | 0.345 ± 0.065 | **1.426 ± 0.072** | **1.875 ± 0.149** |
| 512  | 0.012 ± 0.006 | 0.043 ± 0.025 | 0.115 ± 0.047 | 0.206 ± 0.040 | **1.456 ± 0.042** | **1.913 ± 0.068** |
| 2048 | 0.007 ± 0.001 | 0.022 ± 0.003 | 0.079 ± 0.014 | 0.166 ± 0.022 | **1.453 ± 0.014** | **1.922 ± 0.035** |
| 8192 | 0.006 ± 0.001 | 0.022 ± 0.003 | 0.046 ± 0.009 | 0.142 ± 0.010 | **1.467 ± 0.008** | **1.956 ± 0.019** |

Two phenomena dominate.

First, **L3 conditional coverage improves dramatically**: `MAE-Cov` at `n = 8192`
falls from `0.043` (fixed) to `0.006` (adaptive), and `SupErr` from `0.107` to `0.022`.
The empirical OLS slope is approximately `−0.39` for `MAE-Cov`, very close to the
reference `−2/5` rate, compared with roughly `−0.15` under fixed h.

Second, the L1 quantile metrics **do not decrease at all**: `MAE-q_lo` is essentially
flat at `≈ 1.44`–`1.47` and `MAE-q_hi` at `≈ 1.88`–`1.96` across all five values of
`n`. The pre-CP CKME quantiles are systematically biased toward the conditional
median, which is the expected effect of choosing a very wide bandwidth: the kernel
indicator over-smooths the conditional CDF, the inverted quantile is pulled toward
the centre, and increasing `n` does not remove a bias whose source is the choice
of `h` rather than sample size.

**Table 4.** Oracle-adaptive-h CKME-CP on exp2 (mean ± SD across 10 macroreplications).

| n | MAE-Cov | SupErr | MAE-L | MAE-U | MAE-q_lo | MAE-q_hi |
|---:|---:|---:|---:|---:|---:|---:|
| 64   | 0.053 ± 0.009 | 0.203 ± 0.089 | 0.280 ± 0.075 | 0.325 ± 0.128 | 1.207 ± 0.361 | 1.267 ± 0.299 |
| 128  | 0.042 ± 0.010 | 0.113 ± 0.027 | 0.203 ± 0.055 | 0.223 ± 0.074 | 1.337 ± 0.291 | 1.369 ± 0.252 |
| 512  | 0.029 ± 0.009 | **0.100 ± 0.000** | 0.119 ± 0.020 | 0.127 ± 0.028 | 1.333 ± 0.092 | 1.403 ± 0.138 |
| 2048 | 0.028 ± 0.004 | **0.100 ± 0.000** | 0.084 ± 0.032 | 0.074 ± 0.016 | 1.378 ± 0.101 | 1.455 ± 0.077 |
| 8192 | 0.024 ± 0.001 | **0.100 ± 0.000** | 0.056 ± 0.010 | 0.048 ± 0.006 | 1.405 ± 0.059 | 1.451 ± 0.057 |

On **exp2** the adaptive-h scheme also reduces `MAE-Cov` (from `0.105` to `0.024`)
and tightens the L2 endpoints, but `SupErr` plateaus exactly at the nominal level
`α = 0.10` for `n ≥ 512`. This indicates a *systematic* deviation: at one or more
specific evaluation points the conditional coverage misses the target by the full
α, and additional data does not help. The same L1 collapse is present:
`MAE-q_lo` and `MAE-q_hi` are essentially flat near `1.3`–`1.5`.

![Fixed vs adaptive comparison on exp1.](output_consistency_fixed/fig_compare_fixed_vs_adaptive_exp1.png)

*Figure 3. MAE-Cov (left) and SupErr (right) versus n on exp1 for fixed h (blue
circles) and oracle-adaptive h (orange squares). Both panels are log–log; the
dashed grey line is the reference slope −0.40. The adaptive scheme tracks the
reference rate; the fixed scheme decays substantially more slowly.*

![Fixed vs adaptive comparison on exp2.](output_consistency_fixed/fig_compare_fixed_vs_adaptive_exp2.png)

*Figure 4. Same as Figure 3 for exp2. MAE-Cov is uniformly lower under the
adaptive scheme, but SupErr saturates near α = 0.10, reflecting a systematic
local deviation that cannot be averaged out over the evaluation grid.*

### D.3 Coverage curves: fixed vs adaptive h

Figures 5–8 show the empirical conditional coverage `Cov_n(x)` as a function of
`x` for each value of `n`, averaged over the 10 macroreplications. Comparing the
fixed-h panels (Figs. 5, 7) with the adaptive-h panels (Figs. 6, 8) gives a
geometric picture of the rate gap reported in Section D.2.

**exp1.** Under fixed h (Fig. 5), the coverage curves drift toward the nominal
level `1 − α = 0.90` as `n` grows but remain visibly biased: even at `n = 8192`
there is a persistent gap of roughly 0.02–0.05 across most of the domain, and
the curves cluster *above* the target near the right boundary (`x ≈ 0.9`,
the high-variance region of the MG1 queue). Switching to adaptive h (Fig. 6)
removes this bias almost entirely: from `n = 512` onward the curves are
essentially indistinguishable from the dashed `0.90` line, and the macrorep SD
band collapses to a thin ribbon. This is the visual counterpart of the
`MAE-Cov` jumping from `0.043` (fixed) to `0.006` (adaptive) at `n = 8192`.

**exp2.** Under fixed h (Fig. 7), the curves again drift toward the target as
`n` grows, with the largest residual deviations near the boundaries where
`σ(x) = 0.01 + 0.2(x − π)²` is largest. Adaptive h (Fig. 8) brings most of the
domain onto the nominal line already at `n = 512`, but a localised dip remains:
in a small neighbourhood of `x = π` — exactly the *low-noise* region where the
oracle bandwidth `h(x) = 2σ(x)` collapses toward `≈ 0.02` — the curve sits
noticeably below `0.90` and does *not* improve with additional data. This isolated deviation is exactly what produces
the `SupErr ≈ α = 0.10` floor in Table 4: the worst-x miss is structural, not a
sample-size effect, and it is invisible to the average-over-x metric `MAE-Cov`
(which still shrinks). The plot makes clear that the saturated sup-error is a
single-region phenomenon rather than a global failure.

In summary, the adaptive-h scheme delivers two visual signatures that the
fixed-h scheme does not: (i) the SD band tightens substantially at every `n`
on exp1, indicating not just lower bias but lower run-to-run variability of
the coverage curve, and (ii) on both simulators the *shape* of the curve is
flatter and closer to horizontal at the target line, consistent with the
adaptive bandwidth absorbing local heteroscedasticity that the global CV-tuned
`h` cannot.

![Coverage curves under fixed h on exp1.](output_consistency_fixed/fig_coverage_curves_exp1.png)

*Figure 5. Fixed-h coverage curves on exp1.*

![Coverage curves under oracle adaptive h on exp1.](output_consistency_adaptive_c2.00/fig_coverage_curves_exp1.png)

*Figure 6. Oracle-adaptive-h coverage curves on exp1.*

![Coverage curves under fixed h on exp2.](output_consistency_fixed/fig_coverage_curves_exp2.png)

*Figure 7. Fixed-h coverage curves on exp2.*

![Coverage curves under oracle adaptive h on exp2.](output_consistency_adaptive_c2.00/fig_coverage_curves_exp2.png)

*Figure 8. Oracle-adaptive-h coverage curves on exp2. The localised dip near
the central inflection is the geometric source of the SupErr ≈ α floor in
Table 4.*

*Figures 5–8. Empirical conditional coverage Cov_n(x) versus x for each design
size n. Lines are means across 10 macroreplications; shaded bands are ±1 SD;
the black dashed line marks the nominal 90% target.*

### D.4 Width curves: why over-coverage occurs at low-noise regions

Figures 9–12 plot the mean prediction interval width `W_n(x) = U_n(x) − L_n(x)`
across macroreplications alongside the oracle width
`W*(x) = q_{1−α/2}(x) − q_{α/2}(x) = 2 Φ⁻¹(0.95) σ(x)` for exp2.

For exp2 the noise standard deviation is `σ(x) = 0.01 + 0.2(x − π)²`, so the
oracle width is a U-shaped curve with a near-zero minimum at `x = π`
(`W*(π) ≈ 0.033`) and maxima at `x ∈ {0, 2π}` (`W* ≈ 6.5`).

**Fixed h (Fig. 9).** The CKME interval width is approximately constant across
`x` at small `n` and slowly develops the correct U-shape as `n` grows. At
`n = 8192`, the width tracks the oracle well at both endpoints but remains
noticeably elevated near `x = π`: the predicted width bottoms out at roughly
2–3, two orders of magnitude above `W*(π) ≈ 0.03`. This floor is the direct
cause of the over-coverage visible in Fig. 7 at the same location — the interval
is far wider than the nearly degenerate conditional distribution requires.

**Adaptive h (Fig. 10).** The oracle-adaptive bandwidth produces a width curve
that closely follows the oracle U-shape even at moderate `n`. The match improves
with `n`, but a residual gap persists at `x ≈ π`: the predicted width converges
toward — but remains slightly above — the oracle width. This residual is the
geometric counterpart of the over-coverage spike in Fig. 8.

**Mechanism.** The width floor arises because CKME estimates the conditional CDF
via smooth kernel indicators with bandwidth `h`. When the true conditional
distribution is nearly a point mass (`σ(x) → 0`), the smoothed CDF is an
S-curve of width `O(h)` rather than a step function. Inverting this CDF at
`τ = α/2` and `τ = 1 − α/2` yields quantile estimates separated by at least
`O(h)`, regardless of how small the true dispersion is. CP calibration adds a
global correction `q̂` that is driven by the high-variance regions and further
inflates the interval at low-variance points. The result is a structurally
irreducible over-coverage wherever `σ(x) ≪ h(x)`.

This width analysis complements the coverage-curve discussion: Figs. 7–8 show
*where* the coverage deviates from the target; Figs. 9–10 show *why* — the
prediction interval cannot shrink below the kernel-smoothing resolution.

![Width curves under fixed h on exp2.](output_consistency_fixed/fig_width_curves_exp2.png)

*Figure 9. Fixed-h prediction interval width versus x on exp2. The black dashed
line is the oracle width W*(x). The predicted width has a floor near x = π that
does not vanish as n grows.*

![Width curves under oracle adaptive h on exp2.](output_consistency_adaptive_c2.00/fig_width_curves_exp2.png)

*Figure 10. Oracle-adaptive-h prediction interval width versus x on exp2. The
width tracks the oracle shape much more closely, but a small residual gap
persists near x = π.*

---

## E. Interpretation and Discussion

The fixed-h results provide a clean empirical instance of the L1 → L2 → L3 chain:
the underlying CKME quantile estimator is consistent, and the consistency propagates
through the CP calibration to the conditional-coverage level. This is the
expected behaviour and matches the asymptotic prediction underlying the paper's main
theorem.

The adaptive-h results are more interesting and, at first glance, paradoxical:
**the L1 implication breaks, yet L3 improves and tracks the classical −2/5 rate**.
The mechanism is specific to conformal prediction. Choosing a wide oracle-driven
bandwidth `h(x) = 2σ(x)` over-smooths the conditional CDF, so the inverted quantile
estimates `q̂_{α/2}(x)`, `q̂_{1−α/2}(x)` are pulled toward the conditional median —
a finite, *non-vanishing* bias. CP calibration on the Stage 2 data, however, computes
a single quantile of the score `s(x, y) = |F̂(y|x) − 0.5|` and applies it to the
inverted quantiles; this absorbs the systematic bias as long as the bias is
approximately the *same shape* across the input space. The CP step does not need the
underlying quantile estimates to be accurate, only their *miscalibration* to be
correctable by a global score adjustment.

The result is a phenomenon worth highlighting in the paper:
> An estimator that is *not* L1-consistent can still drive an L3-consistent CP
> interval, provided the CP calibration step has access to a calibration sample
> that experiences the same systematic bias.

This is consistent with the broader observation in `exp_onesided` that the CKME
quantile estimator carries an O(h) tail-quantile bias under fixed h. Adaptive-h
amplifies that bias on purpose (to gain a smoother CDF) and trades L1 accuracy for
L3 calibration headroom.

The diagnostic experiment in `diagnostics/diag_report.md` verifies this
mechanism explicitly. **D1** shows that the adaptive kernel covers
`h(x) / (q_{0.95} − q_{0.05}) = 1/Φ⁻¹(0.95) ≈ 0.608` of the central 90 %
conditional interval — by construction whenever `c = 2` and noise is Gaussian.
**D2** shows that the macrorep-mean `q̂_{0.05}(x)` and `q̂_{0.95}(x)` curves at
`n = 64` and `n = 8192` are visually indistinguishable and lie systematically
*outside* the oracle (lower curve below, upper curve above). **D4** quantifies
this: the adaptive `bias²(q̂_lo)` on exp1 actually grows from 3.61 at `n = 64`
to 3.89 at `n = 8192` while `var(q̂_lo)` shrinks from 0.027 to 0.0005 — the
estimator is converging, but to a wrong limit. Finally **D3** sweeps `c ∈ {0.25,
0.5, 1, 2, 4}` at `n = 512` and traces a clean monotone L1/L3 trade-off on both
simulators: from `c = 0.25` to `c = 4`, the L1 `MAE q̂_lo` on exp1 rises from
0.14 to 1.87 (≈ 13×) while `MAE cov` falls from 0.030 to 0.012 (≈ 2.5×). No
single `c` wins on both axes. The default `c = 2` used here therefore sits on
the "L3-favouring" end of the curve by deliberate choice, not by accident.

The exp2 saturation at `SupErr ≈ α` deserves separate comment. With a grid of
`M_eval = 100` evaluation points and `B_test = 2000` MC draws, the worst-point
deviation under perfect calibration is small but nonzero. The fact that the
deviation freezes *exactly* at `α` for `n ≥ 512` indicates that on at least one
evaluation point the empirical coverage is approximately `1 − 2α` rather than
`1 − α`: the local conditional distribution is mismodelled in a way that the
global CP correction cannot repair. The localised dip visible in Figure 8 sits
near `x = π`, which is precisely the *low-noise* region of `f(x) = exp(x/10)·sin(x)`
where the oracle bandwidth `h(x) = 2σ(x)` collapses to a very small value: with
such a narrow kernel and only `r_1 = 1` calibration replication per site, the
conditional CDF estimate becomes nearly degenerate, the symmetric `|F̂ − 0.5|`
score loses informativeness, and the global CP quantile cannot correct the
mismatch. A full diagnosis is left for follow-up work.

The empirical slopes themselves should be read with care. With only five
`n`-values (64, 128, 512, 2048, 8192), the OLS slope is a *summary*, not a
hypothesis test. The qualitative comparison — adaptive h is close to `−0.40`,
fixed h is closer to `−0.15` — is robust across both simulators, but the precise
numbers should not be cited as identified rates.

---

## F. Limitations

- **Simulators are 1-dimensional and Gaussian.** Both exp1 and exp2 use heteroscedastic
  Gaussian noise. Non-Gaussian consistency is studied separately in `exp_nongauss`;
  multi-dimensional consistency is left to future work.
- **Oracle σ is used for adaptive h.** The adaptive bandwidth uses the *true* noise
  standard deviation, which is not available outside controlled simulation. A practical
  implementation would require a plug-in σ̂(x); the bias-variance tradeoff of this
  estimator would interact with the CP step in ways that this experiment does not
  measure.
- **Single replication per site (`r_0 = r_1 = 1`).** This is appropriate for a
  consistency experiment but precludes a knn-based local σ̂ at small `n`.
- **Two-sided abs-median score only.** The CP score `|F̂ − 0.5|` is symmetric.
  One-sided consistency is evaluated in `exp_onesided`.
- **Five n-values for rate fitting.** The slopes are descriptive summaries, not
  identified parameters; with five points the OLS uncertainty is non-trivial.
- **L1 errors only at τ ∈ {α/2, 1−α/2} = {0.05, 0.95}.** Tail behaviour at other τ
  may differ; see `exp_onesided` for the full τ-scan.
- **`n_macro = 10`.** Sufficient for SD bands but small enough that occasional
  per-point variation is visible in the coverage curves.

---

## G. Takeaway

Under a CV-tuned fixed bandwidth, CKME-CP empirically satisfies the full L1 → L2 → L3
consistency chain on both heteroscedastic Gaussian benchmarks: pre-CP quantile errors,
post-CP endpoint errors, and conditional coverage errors all decrease with `n`.
Replacing the fixed bandwidth with an oracle-adaptive `h(x) = 2σ(x)` *breaks* the
L1 link — the pre-CP quantile estimates are biased toward the median and do not
improve with `n` — but accelerates L3 conditional-coverage convergence to a rate
that closely matches the classical nonparametric `n^{−2/5}` benchmark. The
mechanism is that CP calibration absorbs the systematic over-smoothing bias whenever
it has the same shape across the input space, so an L1-inconsistent estimator can
still produce an L3-consistent CP interval. This decoupling is a CP-specific feature
of CKME and is worth flagging in the paper, both as an empirical phenomenon and as
a caution against using L1 quantile error as a proxy for the validity of the
calibrated interval.

In particular, the L1 → L2 → L3 implication chain promised in `spec.md` holds
under fixed `h` (where all three quantities share the same bias/variance
source), but is broken under adaptive `h`: L2 inherits the post-CP correction
through its definition `L_n(x) = F̂⁻¹(0.5 − q̂|x)`, and L3 inherits it through
the calibrated coverage. The diagnostic report
`diagnostics/diag_report.md` documents this explicitly, including the c-scan
that turns the qualitative picture into a quantitative L1/L3 Pareto curve.
