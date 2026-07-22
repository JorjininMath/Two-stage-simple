# experiments/coverage_mechanism — Specification

Status: spec for implementation, 2026-07-14. Implements the falsifiable claims
from `notes/theory/aleatoric-epistemic-score-decomposition.md` (local note)
and feeds the theory
and ablation sections of the journal draft.

Three experiments:

- **E1 — Distortion law**: quantitative test of the second-order local
  coverage distortion formula (and its breakdown regime), plus the
  tail-quantile bias law (former E4, folded in as E1b).
- **E2 — Shape ablation**: adaptive-h CKME-DCP vs normalized-residual CP;
  honest tie on location-scale Gaussian, separation on shape-varying DGPs.
- **E3 — Validity ablation**: design–calibration separation (T2); proper
  two-stage calibration vs two leakage violations.

General conventions (match
`experiments/adaptive_h/planned_final_benchmark_spec.md` where applicable):

- alpha = 0.1 (90% intervals), DCP score `|Fhat(y|x) - 1/2|`.
- BASE_SEED = 20260714; macrorep k uses seed BASE_SEED + k.
- CLI: `--n_macro`, `--n_workers`, `--analyze_only`, `--output_dir`.
  Smoke run must work with `--n_macro 2`.
- Outputs (CSV + PNG) to `output_e1/`, `output_e2/`, `output_e3/`
  (gitignored). Every figure regenerable from CSVs via `--analyze_only`.
- Plug-in scale estimator = site sample SD + Gaussian Nadaraya-Watson
  smoothing (`s_hat_NW`), exactly as in the final `experiments/adaptive_h` spec.
  No IQR.
- Do not modify `src/CKME/`, `src/CP/`, `src/Two_stage/` core APIs. New
  simulators are registered in `src/Two_stage/sim_functions/` + `__init__.py`.
  English code and
  comments.

Implementation-alignment requirements (checked 2026-07-15, must hold in all
new runners):

- **Query-specific bandwidth (confirmed in `experiments/adaptive_h/adaptive_bandwidth.py`):**
  the adaptive arms evaluate Fhat(.|x_query) with ONE bandwidth h(x_query)
  applied to ALL training responses. This matches the analyzed estimator
  (note §0.2). Keep this convention; do NOT assign per-training-site
  bandwidths h(x_i).
- **Monotone projection before quantile extraction:** the existing utils clip
  Fhat to [0,1] but then apply `np.searchsorted`, which silently assumes
  monotonicity in t. All new runners must enforce monotonicity explicitly
  (running max `np.maximum.accumulate` over t_grid, then clip) before any
  quantile/level-set extraction, in every arm. This matches the projected
  estimator Ftilde the theory statements refer to.
- **Two-layer evaluation protocol
  (`notes/decisions/2026-07-15-cdf-legality-and-interval-reporting-policy.md` §6):**
  (i) GUARANTEE layer: calibrate q_hat on raw point-evaluated scores
  S = |Fhat_raw(y_cal|x) - 1/2| (grid-free), and report score-set coverage
  1{S(y_test) <= q_hat} as the PRIMARY coverage metric in every experiment —
  this is the exact object of the split-CP theorem. (ii) REPORTING layer:
  intervals [L, U] from the monotone-projected CDF via the shared helper,
  used for width, interval score, bin profiles, and baseline comparability;
  interval coverage is reported as a secondary descriptive metric next to
  the score-interval wedge. Policy B (projected scores end-to-end) remains
  available via t_grid parameters as a consistency variant only.
- **t_grid coverage rule:** the t-grid must cover the pooled Stage-1 +
  Stage-2 (and test) response range with margin (interval reporting is
  truncated at the grid ends, and the optional Policy B variant saturates
  scores under a narrow grid — see the policy note §5). Runners must assert
  `t_grid[0] <= min(Y) and t_grid[-1] >= max(Y)` on calibration data.

---

## E1 — Quantitative distortion law (mostly analytic; no CKME in E1a)

**Goal.** E1's target is the note's §3.1 (population-calibrated first-order
expansion) and §3.2 (smoothing-only distortion law), in their stated validity
regime — NOT a generic "fixed h is bad" demonstration. Three claims:

1. (small-rho quadratic law) For fixed h, population-calibrated conditional
   coverage satisfies

       c(x) - (1-alpha) ≈ kappa_2 * |f_eps'(z_a)| * [rho(x)^2 - E rho(X)^2],
       rho(x) = h / s(x),  z_a = z_{1-alpha/2},

   with the predicted slope, when sup_x rho(x) is small.
2. (exact benchmark at any rho) The EXACT smoothing-only coverage is
   C(rho; r) = F_eps(H_rho^{-1}(1/2 + r)) - F_eps(H_rho^{-1}(1/2 - r)) with
   H_rho(z) = E[Psi((z - eps)/rho)]; the Gaussian closed form below is its
   special case. The quadratic law must be shown to depart from this exact
   curve for rho >~ 1 — this documents the attribution rule (evaluate the
   exact benchmark before blaming input smoothing for large-rho residuals).
3. (cure) Adaptive h(x) = c*s(x) gives a flat profile (exact homogeneity),
   and finite calibration adds only a common shift across x.

**Analytic device.** Use the *Gaussian-CDF indicator* (`gaussian_cdf` in
`src/CKME/indicators.py`, g0 = standard normal, kappa_2 = 1) and Gaussian noise.
Then output smoothing is exact variance inflation:

    F_h°(t|x) = Phi( (t - m(x)) / sqrt(s(x)^2 + h^2) ).

With score threshold r, the oracle-smoothed prediction set endpoints are
`m(x) ± sqrt(s^2+h^2) * Phi^{-1}(1/2 + r)` and TRUE conditional coverage is
closed-form:

    c(x; r) = 2 * Phi( sqrt(1 + rho(x)^2) * Phi^{-1}(1/2 + r) ) - 1.

(Sanity check: depends on x only through rho(x).)

**DGP-E1.** x in [0,1], q_X = Unif[0,1], m(x) = 0,
`s(x) = 0.20 + 0.15 * sin(2*pi*x)` (s in [0.05, 0.35]).

**E1a — population-calibration analytic arm (no fitting).**

1. For h in {0.015, 0.03, 0.12}: solve for the population threshold r° by
   1-D root finding on `mean_x[c(x; r)] = 0.9` (x on a 2001-point grid).
2. Output the exact deviation curve `c(x; r°) - 0.9` vs x and vs
   `rho(x)^2 - mean(rho^2)`.
3. Overlay the theory line with slope
   `kappa_2 * z_a * phi(z_a) = 1.645 * phi(1.645) ≈ 0.1699` (alpha = 0.1).
4. Breakdown panel: exact deviation vs rho over rho in [0, 2.5]
   (drive with h = 0.12), showing where the second-order line departs.
5. Finite-calibration arm: n_cal in {100, 1000}, 500 Monte Carlo calibration
   draws; verify the empirical threshold produces a common vertical shift
   O(n_cal^{-1/2}) without changing the profile shape (report shift SD).

**E1b — tail-quantile bias law (analytic + overlay).** Exact smoothed
quantile: `q_{tau,h}(x) = m + sqrt(s^2+h^2) * z_tau`, so

    q_{tau,h} - q_tau = (sqrt(s^2+h^2) - s) * z_tau ≈ (s * rho^2 / 2) * z_tau.

Plot bias vs tau in {0.5, 0.75, 0.9, 0.95, 0.99} at s in {0.05, 0.2, 0.35}
for h in {0.03, 0.12}; verify the h^2 scaling and the f_eps'/f_eps = z_tau
amplification. No estimation needed.

**E1c — does the law survive the full estimator?** Fit real CKME on DGP-E1
with a dense design so that input-direction error is small:

- Stage 1: n_0 = 200 LHS sites, r_0 = 20; `gaussian_cdf` indicator;
  (ell_x, lam) by CV once (pretrain step), then frozen.
- Arms: fixed h = 0.03; adaptive h(x) = 0.3 * s_hat_NW(x).
- Calibration: 1000 iid Unif[0,1] points, 1 rep. Test: x-grid of 101 points
  x 500 fresh reps for per-x coverage.
- n_macro = 20.
- Compare mean per-x coverage profile to the E1a analytic curve (fixed arm)
  and to a flat line (adaptive arm).

**Acceptance.**

- E1a: regression of exact deviation on `[rho^2 - mean(rho^2)]` over the
  sub-range sup rho <= 0.6 gives slope within 15% of 0.1699 and R^2 > 0.95.
- Breakdown panel clearly shows departure for rho >~ 1 (documents the regime
  condition for the paper).
- E1c fixed-arm empirical profile within MC error bands of the analytic
  curve on the interior of the domain; adaptive-arm profile flat within
  bands. If E1c deviates systematically, report — it measures how much
  input-direction error contaminates the smoothing-only law.

**Outputs.** `e1a_curves.csv` (x, s, rho, h, c_exact, c_theory),
`e1a_slope.csv`, `e1b_quantile_bias.csv`, `e1c_profiles.csv`,
figures `e1a_deviation_vs_rho2.png`, `e1a_breakdown.png`,
`e1b_tail_bias.png`, `e1c_profiles.png`.

---

## E2 — Score-layer collapse and shape ablation (the key new comparison)

**Goal.** E2's target is the note's §4.2–4.3 (standardized-score reduction
and SYMMETRIC rank collapse) and its positioning consequence: CKME's
defensible empirical territory is x-VARYING conditional shape. The goal is
NOT "CKME beats normalized residuals" — it is "each method succeeds/fails
exactly where the reduction theory says". Four graded predictions:

- P-EXACT (theorem-level sanity check): on the symmetric location-scale DGP,
  `ckme_oracle` and `normres_oracle` produce IDENTICAL prediction sets up to
  grid tolerance (symmetric rank collapse), and `ckme_oracle` sets are
  c-independent (run c in {0.5, 2.0}, compare sets). Failure here is an
  implementation bug, not an interesting result.
- P-LS (honest tie): on the symmetric location-scale DGP, plug-in arms
  `ckme_plugin`, `normres_plugin`, `normres_cdf` are statistically
  indistinguishable.
- P-FS (fixed asymmetric shape): on `fixed_gamma_ls` (shape asymmetric but
  x-invariant), `ckme_plugin` ≈ `normres_cdf` (both capture the fixed shape;
  balanced one-sided misses) while `normres_plugin` pays for shape-wrong
  symmetric intervals (unbalanced one-sided misses, worse IS). CKME earns no
  advantage over the pooled-CDF baseline here — report this honestly.
- P-XS (x-varying shape, CKME's territory): on `shape_t_dfx` and
  `shape_gamma_kx`, `ckme_plugin` beats BOTH `normres_plugin` and
  `normres_cdf` on bin coverage deviation and one-sided balance, because a
  pooled residual CDF cannot adapt to x-varying shape.

**DGPs** (all x in [0, 2*pi], m(x) = exp(x/10)*sin(x), q_X = Unif):

| Name | eps(x) | Scale | Role |
|---|---|---|---|
| `raised_floor_gauss` (existing) | N(0,1) | s(x) = 0.10 + 0.20 (x-pi)^2 | P-EXACT + P-LS |
| `fixed_gamma_ls` (NEW) | (Gamma(2,1) - 2)/sqrt(2), fixed for all x (skew ≈ 1.41) | s(x) = 0.10 + 0.20 (x-pi)^2 | P-FS: asymmetric, x-invariant shape |
| `shape_t_dfx` (NEW) | T_{nu(x)} / sqrt(nu(x)/(nu(x)-2)) | s = 0.3 const | P-XS: tail weight varies |
| `shape_gamma_kx` (NEW) | (Gamma(k(x),1) - k(x)) / sqrt(k(x)) | s = 0.3 const | P-XS: skewness varies |

with `nu(x) = exp( (1-u)*ln(30) + u*ln(3) )`, `k(x) = 16^(1-u)`,
`u = x/(2*pi)`. Both eps(x) have mean 0, variance 1 for all x (shape-only
variation; unit-variance standardization is essential — document it in the
simulator docstrings). Register both in `src/Two_stage/sim_functions/`.

**Arms** (identical data per macrorep; same (ell_x, lam) for all CKME arms):

| Arm | Score | Notes |
|---|---|---|
| `ckme_fixed` | DCP score, scalar h from Stage-1 CV | internal baseline |
| `ckme_plugin` | DCP score, h(x) = c * s_hat_NW(x) | our method (c from experiments/adaptive_h default) |
| `ckme_oracle` | DCP score, h(x) = c * s(x) | mechanism reference |
| `normres_plugin` | \|y - m_hat_NW(x)\| / s_hat_NW(x) | key comparator |
| `normres_oracle` | \|y - m(x)\| / s(x) | oracle reference (P-EXACT partner) |
| `normres_cdf` | \|F_hat_eps((y - m_hat_NW(x)) / s_hat_NW(x)) - 1/2\| with F_hat_eps = pooled ECDF of standardized Stage-1 residuals (training fold only) | shape-aware standardized baseline (required by note §4.4.3) |
| `ckme_hard` | DCP score from hard-indicator KRR (same weights, h = 0, i.e. site ECDFs), followed by monotone projection onto the CDF cone | why-smooth ablation |

`ckme_hard` note: identical input-side KRR weights `(K + n*lam*I)^{-1} k_x`
applied to `1{y_ij <= t}` instead of the smooth indicator; clip + isotonic
projection in t before quantile extraction. Purpose: test the practical value
of smooth output evaluation and the adaptive-h dial against the no-surrogate
alternative (which is consistent but has no scale-adaptive dial). Do not
present it as an invalid method — only as a differently-smoothed estimator.

Fairness requirement: `m_hat_NW` and `s_hat_NW` use the same Stage-1
replications and the same NW bandwidth selection rule as the plug-in scale in
`experiments/adaptive_h` — both methods see identical information.

**Protocol.** Stage 1: n_0 = 100 LHS sites, r_0 = 10. Calibration: n_1 = 500
iid from q_X, 1 rep (matched-target; no adaptive site selection — keep design
non-adaptive to avoid confounds). Test: 5000 iid points, 1 rep, K = 10
equal-count x-bins; plus x-grid 101 x 200 reps for profile figures.
n_macro = 50.

**Metrics** (per macrorep, per DGP, per arm): marginal coverage (PRIMARY =
score-set coverage `1{S(y_test) <= q_hat}`; interval coverage secondary),
mean width, interval score; per bin: coverage, one-sided miss rates
`P(Y < L)`, `P(Y > U)`, mean width; aggregates: max and RMS bin coverage
deviation from 0.9; paired per-macrorep deltas `ckme_plugin - normres_plugin`
for IS and max-bin-deviation, with SE over macroreps.
Width cross-check: also compute the score-native set size
`sum_k dt * 1{score(t_k) <= q_hat}` (Lebesgue measure of the score set);
report mean |measure - (U - L)| — values beyond ~one grid step flag
dip-straddling points and must be reported per DGP/arm.
Metric-object rule: coverage (primary) refers to the conformal SCORE SET;
width and Winkler interval score refer to (and are only defined for) the
PROJECTED INTERVAL. The two coincide iff the score set is connected; report
the disconnection rate per DGP/arm, and never present Winkler as an
evaluation of the guaranteed set.

**Acceptance** (mapped to the graded predictions above).

- A0 (P-EXACT): on `raised_floor_gauss`, `ckme_oracle` vs `normres_oracle`
  interval endpoints agree within t-grid resolution for >= 99% of test
  points, and `ckme_oracle` with c = 0.5 vs c = 2.0 likewise. Hard gate:
  treat failure as an implementation bug (monotone projection, query-specific
  h, tie handling) and fix before interpreting anything else.
- A1 (P-LS, report prominently — the tie is evidence FOR the reduction
  theory, not failure): |paired IS delta| <= 1 SE among the three plug-in
  arms; bin profiles overlap.
- A2 (P-FS): on `fixed_gamma_ls`, `normres_plugin` shows unbalanced one-sided
  misses (miss_left/miss_right ratio bounded away from 1) and worse IS than
  both `ckme_plugin` and `normres_cdf` (paired Wilcoxon p < 0.05);
  `ckme_plugin` vs `normres_cdf` is expected to tie — report the tie.
- A3 (P-XS): on both shape DGPs, `ckme_plugin` beats `normres_plugin` AND
  `normres_cdf` on max-bin coverage deviation (paired Wilcoxon p < 0.05),
  with balanced one-sided misses; `normres_cdf` beats `normres_plugin`
  (it fixes shape on average but not its x-variation).
- A4: every arm's marginal coverage within the Beta(k, n+1-k) band for
  n_cal = 500.
- Falsification note: the sharp test is A3 against `normres_cdf`. If
  `normres_cdf` matches `ckme_plugin` on the shape DGPs, the "x-varying
  shape" defense fails empirically — flag immediately rather than tuning
  DGPs until it passes; shape variation may be strengthened (nu down to 2.5,
  k down to 0.5) only as a reported revision of this spec.

**Outputs.** `e2_per_point.csv` (macrorep, dgp, arm, x, y, L, U, covered),
`e2_marginal.csv`, `e2_bins.csv`, `e2_paired.csv`; figures: bin-coverage
profiles per DGP (5 arms), one-sided miss profiles, paired-delta forest plot.

---

## E3 — Validity ablation: design–calibration separation (T2)

**Claim tested.** Two-stage adaptivity is safe iff all data-dependent
decisions freeze before calibration; violating the separation breaks marginal
coverage.

**DGP.** `raised_floor_gauss`. Stage 1: n_0 = 50 x r_0 = 10.

**Arms.**

| Arm | Construction | Expected marginal coverage |
|---|---|---|
| `proper` | S0-guided Stage-2 site selection (`method="sampling"`) from Stage-1 model; FRESH responses at selected sites; calibrate only on Stage-2 | ~0.90 (in band) |
| `leak_reuse` | calibration set = Stage-2 + Stage-1 training responses | < 0.90 (overfit scores too small => q_hat too small) |
| `leak_select` | draw candidate responses first, then keep the n_1 points with SMALLEST \|y - m_hat(x)\| as the calibration set | << 0.90 (selection on calibration responses) |

n_1 = 200 in all arms (equalize calibration size; for `leak_reuse` subsample
the union to 200 so size is not a confound). Test: 2000 iid points.
n_macro = 100 (cheap).

**Acceptance.** `proper` mean coverage within the exact Beta band for
n_cal = 200; both leak arms outside the band in the predicted direction with
p < 0.01 (one-sample test across macroreps). Report full coverage
distributions (violin/strip plot), not only means.

**Outputs.** `e3_coverage.csv` (macrorep, arm, coverage, width, q_hat),
figure `e3_validity_ablation.png`.

---

## E4 — Score-choice ablation (piloted 2026-07-16; see `pilot_score_design/`)

**Goal.** Test the layer-separation thesis at the score layer: efficiency
promises made by smarter conformal scores are bounded by estimation-layer
quality, and the paper's aleatoric-normalization principle ports to other
score families. Two sub-studies, both piloted with runnable results archived
in `experiments/coverage_mechanism/pilot_score_design/`.

**E4a — PIT-band cut-point family** (arms: `ET` theta = alpha/2; `opt`
per-x width-minimizing theta; `bopt` opt + density-floor constraint
gamma = 0.15). DGP: fixed skewed Gamma (k=2) location-scale; budgets
B in {600, 2000}; 50 macroreps each. Pilot findings (record, then reproduce
at final scale): all arms valid at 0.90; opt's oracle prize (-12.3% width vs
ET) inverts in the MEAN at both budgets (+14.0% / +7.6%) due to catastrophic
tail-chasing failures (rate 20% / 10%, extremes +172% / +224%) while the
MEDIAN stays mildly negative (~ -3%); bopt halves the failure rate but never
beats ET. Acceptance: reproduce validity + the heavy-tailed risk profile;
report means AND medians AND failure rates (> +20% vs ET).

**E4b — PCP arms (Wang et al. 2023) + scale normalization** (arms: `ET`
reference; `pcp` score E = min_k |y - yhat_k|, K = 40 inverse-transform
samples from the projected CDF; `pcp_scaled` score E / shat_NW(x), the
paper's replication-based scale normalization ported to PCP). DGPs: skewed
Gamma (homoscedastic) + raised-floor heteroscedastic Gaussian; 50 macroreps.
Set size = Lebesgue measure of the union (two-layer protocol); report mean
component count. Pilot findings: all arms valid; on the heteroscedastic DGP
raw PCP's bin coverage tilts hard with s(x) (Spearman -0.98, bin range
0.725-0.999) exactly as the score-homogeneity analysis predicts
(E scales with s(x)); `pcp_scaled` compresses the tilt 4.4x (range 0.062),
flatter than fixed-h ET (0.109) — direct evidence that the
estimation-layer normalization principle transfers to sample-based scores.
Efficiency cost: on unimodal DGPs PCP pays a 16-20% measure premium over the
ET band with 3-4.5 disconnected components on average (PCP's advantage is
multimodality, absent here). Acceptance: validity; tilt-range ratio
pcp / pcp_scaled >= 3; honest reporting of the measure premium.

Falsification notes: if `pcp_scaled` fails to flatten the tilt on the final
DGPs, the portability claim is dropped (not re-tuned); if opt's failure mode
vanishes at final scale, revisit the T5-based explanation.

## Runtime and execution notes

- E1a/E1b are analytic (seconds). E1c ~ 20 macroreps x 2 arms, n = 4000
  training rows — hours on a laptop, trivially parallel by macrorep.
- E2 is the heavy one: 50 macroreps x 3 DGPs x shared fit + 5 scoring arms.
  CKME fit is shared within (macrorep, DGP): fit once per h-rule (fixed /
  plugin / oracle differ only in indicator bandwidth at score time — reuse
  weights where the implementation allows, as in experiments/adaptive_h). Parallel
  by macrorep (`--n_workers`); SLURM array optional, follow
  `experiments/gibbs_compare/run_all_gibbs_arc.sh` pattern if needed.
- Every table in the eventual paper must be generated by `summarize.py` from
  the CSVs — no hand-typed numbers.

## Deliverables checklist for the implementer

1. `src/Two_stage/sim_functions/sim_shape_t_dfx.py`,
   `sim_shape_gamma_kx.py`,
   `sim_fixed_gamma_ls.py` + registry entries (unit-variance standardization
   tested by a quick moment check in a `__main__` block or test).
2. `experiments/coverage_mechanism/run_e1_distortion.py` (E1a/E1b/E1c via
   `--part {a,b,c,all}`), `run_e2_shape.py`, `run_e3_validity.py`,
   `summarize.py`, `plot_e1.py`, `plot_e2.py`, `plot_e3.py`.
3. `experiments/coverage_mechanism/pretrain_params.py` producing
   `pretrained_params.json` for DGP-E1 (E1c) and the three E2 DGPs.
4. Smoke test: `--n_macro 2` end-to-end for E1c/E2/E3 finishing in minutes.
5. README.md in the folder summarizing commands (follow
   `experiments/framing_validation/README.md` style).
