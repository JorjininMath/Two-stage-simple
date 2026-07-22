# Gate 4 Specification: Fixing Epistemic Error via Targeted Stage-2 Sampling

Status: SPEC ONLY (not yet implemented). Companion to the three framing-validation
gates in this folder (see `README.md`). Gate 3 established that the bootstrap
tail-quantile variance `u_tail(x)` localizes an epistemic hotspot far better than
CP interval width (separation index 112 vs 4.9). Gate 4 closes the loop:

> diagnose (u_tail) -> act (allocate Stage-2 budget) -> verify (error drops,
> CP width tax is refunded) -> and, when sampling does NOT fix it, detect that
> the model itself is the binding constraint (lack-of-fit signal) and fix the
> model instead.

This spec is self-contained so it can be handed to another implementer. The
implementation should reuse the helper functions of
`gate3_epistemic_diagnostics.py` (`simulate`, `lhs_1d`, `fit_ckme`,
`raw_scores`, `monotone_cdf`, `invert_cdf`, `separation_index`) either by
import or by copy — do NOT re-derive them.

---

## 1. Claims under test

- **C1 (G4a, fixable hotspot)**: when the epistemic hotspot is caused by local
  data scarcity, allocating Stage-2 sites proportionally to `u_tail(x)` reduces
  the hotspot CDF error substantially more than uniform LHS at the same budget,
  and deflates the global split-CP threshold `q_hat` (the "width tax" of Gate 3
  is refunded), while marginal coverage stays at 1 - alpha.
- **C2 (G4b, capacity-limited hotspot)**: when the hotspot is caused by a
  feature the kernel length scale cannot resolve, targeted sampling reduces
  `u_tail` (variance goes down) but NOT the true CDF error (bias floor). A
  replication-based lack-of-fit statistic detects this residual misfit from
  data alone; re-fitting with a smaller / CV-tuned `ell_x` then fixes it.
- **C3 (decision rule)**: the two-step rule
  1. `u_tail` hotspot -> add data there;
  2. after adding data, lack-of-fit still high there -> change the model
     (here: shrink `ell_x`),
  correctly routes both scenarios without using any oracle quantity.

C1 is the headline result. C2/C3 make the story honest: they show the
diagnostic does not just say "sample more" blindly.

---

## 2. Common setup (both parts)

All settings identical to Gate 3 unless stated otherwise.

```text
x in [0, 1],  sigma = 0.10 constant (flat aleatoric noise, Gaussian)
alpha = 0.1
CKME params (default arm): ell_x = 0.1, lam = 1e-3, h = 0.05, logistic indicator
r = 10 replications per site EVERYWHERE (stage 1, stage 2, calibration),
    so the pooled data always satisfies the constant-r requirement of
    CKMEModel.fit(X, Y, params=..., r=10) in distinct-sites mode
N_TGRID = 500, t_grid = linspace(Y_pool.min() - 0.5, Y_pool.max() + 0.5, 500)
    (recompute the t_grid AFTER pooling stage-2 data, per arm)
X_GRID  = linspace(0.02, 0.98, 200)   # diagnostic grid, same as Gate 3
N_BOOT  = 30 site-bootstrap refits for u_tail (pre and post)
N_MACRO = 20 (run a 3-macrorep pilot first; see Section 8)
BASE_SEED = 20260708
```

### Seed layout (per macrorep k)

```text
seed = BASE_SEED + k * 1000
rng_train  = default_rng(seed)          # stage-1 site noise
rng_cal    = default_rng(seed + 1)      # calibration sites + noise (shared by all arms)
rng_test   = default_rng(seed + 2)      # test set (shared by all arms)
rng_boot0  = default_rng(seed + 3)      # PRE-stage-2 bootstrap (u_tail flag)
rng_alloc  = default_rng(seed + 4)      # stage-2 site allocation draws
rng_lof    = default_rng(seed + 5)      # fresh lack-of-fit validation set (shared by all arms)
rng_sim2_a = default_rng(seed + 40 + arm_index)   # stage-2 simulator noise, per arm
rng_boot1_a = default_rng(seed + 60 + arm_index)  # POST-stage-2 bootstrap, per arm
```

Everything computed BEFORE stage-2 allocation (stage-1 model, pre u_tail,
calibration set, test set, LOF set) is computed ONCE per macrorep and shared
across arms, so arm comparisons are paired.

### Stage-2 allocation policies (the "arms")

Given a weight curve `w(x) >= 0` on `X_GRID`, select `n_1` distinct sites from
a candidate pool:

```python
cand = lhs_1d(2000, rng_alloc)                      # candidate pool, once per macrorep
w_cand = np.interp(cand, X_GRID, w)                 # interpolate weight curve
p = GAMMA / len(cand) + (1 - GAMMA) * w_cand / w_cand.sum()
p = p / p.sum()
sites = rng_alloc.choice(cand, size=N_1, replace=False, p=p)
```

with `GAMMA = 0.2` (uniform exploration floor, mirroring the `mixed` method of
`src/Two_stage/site_selection.py`). The arms differ only in `w`:

| Arm      | Weight curve `w(x)`                              | Role                     |
|----------|--------------------------------------------------|--------------------------|
| `lhs`    | constant (equivalently: plain `lhs_1d(N_1)`)     | baseline                 |
| `utail`  | pre-stage-2 `u_tail(x)`                          | practical, data-only     |
| `oracle` | pre-stage-2 `cdf_l2(x)` (true CDF L2 gap)        | upper bound (G4a only)   |

For `lhs`, implement literally as `lhs_1d(N_1, rng_alloc)` (not the weighted
sampler with constant w) so the baseline matches the rest of the repo.

### Refit and post-diagnostics (per arm)

1. Simulate stage-2 data at the selected sites: `r = 10` reps, noise from
   `rng_sim2_a`.
2. Pool: `X_pool = concat(stage1 sites, stage2 sites)` (each repeated 10x,
   consecutively), `Y_pool` likewise. Refit `CKMEModel` with the arm's params
   (Section 3/4) and `r = 10`.
3. Recompute the split-CP threshold `q_hat_post` on the SAME calibration set
   as pre-stage-2 (scores from the refit model). Also record `q_hat_pre`
   (stage-1 model) once per macrorep.
4. Marginal coverage of the refit model + `q_hat_post` on the shared test set
   (1000 uniform x, fresh y).
5. Post diagnostics on `X_GRID`: CP width, oracle `cdf_l2`, `u_tail` via
   `N_BOOT` site bootstraps of the POOLED site set (resample all
   `n_0 + n_1` sites with replacement, keep their rep blocks).
6. Lack-of-fit statistic (data-only, fresh validation set, shared across arms):

   ```text
   LOF set: n_lof = 40 LHS sites x r_lof = 20 reps  (rng_lof)
   For each LOF site x_i:
       ybar_i  = mean of the 20 reps
       s2_i    = within-site sample variance (ddof=1)
       m_hat_i = refit-model median at x_i  (invert monotone CDF at tau=0.5)
       LOF_i   = r_lof * (ybar_i - m_hat_i)^2 / s2_i
   ```

   Under a well-specified fit, `LOF_i` is approximately F(1, 19)-distributed
   (order 1, 95% quantile approx 4.4); systematic local bias inflates it by
   `r_lof * bias^2 / sigma^2` — a bias of one sigma at `r_lof = 20` gives
   LOF around 20. Report the max of `LOF_i` over hotspot-window sites and the
   median over background sites.

### Metrics recorded per (macrorep, arm)

| Metric              | Definition                                                        |
|---------------------|-------------------------------------------------------------------|
| `hot_cdf_l2`        | mean of post `cdf_l2(x)` over the hotspot window                  |
| `bg_cdf_l2`         | median of post `cdf_l2(x)` outside the window                     |
| `hot_utail_post`    | mean of post `u_tail(x)` over the window                          |
| `si_utail_post`     | separation index of post `u_tail` (Gate-3 definition)             |
| `q_hat_post`        | split-CP threshold after refit                                    |
| `coverage`          | marginal coverage on test set                                     |
| `mean_width`        | mean CP interval width over `X_GRID`                              |
| `hot_lof_max`       | max LOF over window LOF-sites                                     |
| `bg_lof_med`        | median LOF over background LOF-sites                              |
| `n1_in_window`      | number of stage-2 sites that landed in the hotspot window         |

Also record once per macrorep (pre-stage-2): `q_hat_pre`, pre `u_tail` curve,
pre `cdf_l2` curve, pre SI values.

---

## 3. Part G4a — data-scarcity hotspot (fixable by sampling)

### DGP

```text
f(x) = sin(2*pi*x) + 1.5 * exp(-(x - 0.5)^2 / (2 * 0.10^2))   # WIDE bump, sd = 0.10
sigma = 0.10 constant
```

The bump sd (0.10) equals `ell_x`, so the feature IS resolvable by the kernel
— given enough local data.

### Stage-1 design: thinned, not empty

```text
Outside [0.35, 0.65]: 50 equally spaced sites
    (25 on linspace(0.0, 0.34, 25), 25 on linspace(0.66, 1.0, 25))
Inside  [0.35, 0.65]: 6 equally spaced sites, linspace(0.37, 0.63, 6)
n_0 = 56 sites x r_0 = 10 reps
```

IMPORTANT — why thinned and not a hole: `u_tail` is a bootstrap variance. In a
region with NO sites at all, every bootstrap refit extrapolates identically, so
the variance is LOW and the diagnostic goes blind. With ~20% relative density,
the few interior sites are resampled in/out across bootstraps and `u_tail`
spikes. This caveat must be stated in the report; it is a real limitation of
bootstrap epistemic diagnostics (deserts need a density-based guard, out of
scope here).

### Stage-2 budget and arms

```text
N_1 = 60 new sites x R_1 = 10 reps  (budget comparable to stage 1)
Arms: lhs, utail, oracle   (all three; same PARAMS as Gate 3 for fit and refit)
Hotspot window: [0.35, 0.65]
```

### Expected qualitative outcome

- Pre: `u_tail` and `cdf_l2` both spike on [0.35, 0.65]; `q_hat_pre` inflated
  relative to a well-fit model (Gate-3 mechanism).
- Post `utail` arm: most of the 60 sites land in the window
  (`n1_in_window` >> 60 * 0.3); `hot_cdf_l2` drops toward `bg_cdf_l2`;
  `q_hat_post` drops toward the well-specified value (approx 0.30 from Gate 2
  oracle arms); mean width shrinks toward `2 * 1.645 * sigma approx 0.33`.
- Post `lhs` arm: only ~30% of budget lands in the window; partial improvement.
- `oracle` arm: at least as good as `utail`; the utail-vs-oracle gap measures
  how much is lost by using the data-only diagnostic.
- LOF: low everywhere post-fix (the model can represent f) — this is the
  contrast with G4b.

### Success criteria (C1)

Paired over macroreps (utail vs lhs), Wilcoxon signed-rank:

1. `hot_cdf_l2(utail) < hot_cdf_l2(lhs)` with p < 0.01 and median ratio <= 0.6.
2. `q_hat_post(utail) < q_hat_post(lhs)` with p < 0.05.
3. Marginal coverage in [0.87, 0.93] for every arm (guaranteed by split-CP;
   this is a sanity check, not a comparison).
4. `si_utail_post(utail)` <= 3 (hotspot flattened), vs pre-SI >> 10.

If (1) holds but (2) does not, the width-tax-refund claim must be dropped from
the paper text (report it as observed); (1), (3) are the hard gates.

---

## 4. Part G4b — capacity-limited hotspot (sampling is not enough)

### DGP and stage-1 design

Exactly the Gate-3 bump arm:

```text
f(x) = sin(2*pi*x) + 1.5 * exp(-(x - 0.5)^2 / (2 * 0.03^2))   # NARROW bump, sd = 0.03
Stage 1: n_0 = 100 grid sites (linspace(0, 1, 100)) x r_0 = 10
Hotspot window: [0.41, 0.59]   (0.5 +/- 3 * 0.03, as in Gate 3)
```

### Arms

```text
N_1 = 60 x R_1 = 10, allocation always utail-weighted (GAMMA = 0.2); arms
differ in the REFIT model:

| Arm          | Allocation | Refit params                                       |
|--------------|-----------|-----------------------------------------------------|
| lhs_fixed    | lhs       | ell_x = 0.1 (unchanged)          — do-nothing baseline |
| utail_fixed  | utail     | ell_x = 0.1 (unchanged)          — "just add data"     |
| utail_retune | utail     | ell_x CV-selected from {0.02, 0.05, 0.1}, lam and h fixed |
```

For `utail_retune`, select `ell_x` by 5-fold CV with CRPS on the pooled data
(reuse `src/CKME/tuning.py` with `ParamGrid(ell_x_list=[0.02, 0.05, 0.1],
lam_list=[1e-3], h_list=[0.05])`), folds split BY SITE (keep rep blocks
together). If wiring `tuning.py` into the standalone harness is awkward,
a hand-rolled site-level 5-fold CRPS loop over the three candidates is
acceptable — record which `ell_x` won per macrorep.

### Expected qualitative outcome

- `utail_fixed`: post `u_tail` drops in the window (more data -> less
  variance), but post `cdf_l2` stays high (smoothing-bias floor at
  `ell_x = 0.1 >> 0.03`), and LOF stays high in the window
  (`hot_lof_max` >> 4.4 while `bg_lof_med` approx 1). This is the
  "variance fixed, bias remains" signature.
- `utail_retune`: CV picks `ell_x = 0.02` (or 0.05) in most macroreps; post
  `cdf_l2` AND LOF both drop in the window.
- `q_hat_post`: stays inflated for `utail_fixed`, deflates for `utail_retune`.

### Success criteria (C2, C3)

1. `hot_cdf_l2(utail_fixed) / hot_cdf_l2(lhs_fixed)` median in [0.6, 1.5]
   — i.e., targeted sampling alone does NOT substantially fix the hotspot
   (allow mild improvement; the point is it does not close the gap).
2. `hot_cdf_l2(utail_retune) < hot_cdf_l2(utail_fixed)`, Wilcoxon p < 0.01,
   median ratio <= 0.3.
3. LOF separates the two: `hot_lof_max(utail_fixed) > 10` in >= 80% of
   macroreps AND `hot_lof_max(utail_retune) < 10` in >= 80% of macroreps.
   (10 is a placeholder threshold approx 2x the F(1,19) 95% quantile; if the
   pilot shows it mis-calibrated, set the threshold to the 99% quantile of
   background LOF values pooled across macroreps and document the change.)
4. Coverage in [0.87, 0.93] for every arm.

Criterion 3 is what licenses the decision rule (C3): a practitioner who can
compute only `u_tail` and LOF — no oracle — reaches the right action in both
parts.

---

## 5. Outputs

```text
experiments/framing_validation/output_gate4a/
    gate4a_metrics.csv        one row per (macrorep, arm), all Section-2 metrics
    gate4a_curves.csv         per-x mean/SE of pre+post u_tail, cdf_l2, width, per arm
    gate4a_fix_epistemic.png
experiments/framing_validation/output_gate4b/
    gate4b_metrics.csv, gate4b_curves.csv, gate4b_capacity_limit.png
    gate4b_ellx_selected.csv  per-macrorep CV-selected ell_x (retune arm)
```

All outputs are gitignored (`**/output_*/` already covers them).

### Figure G4a (2 x 3 panels)

1. DGP `f(x)` + rug of stage-1 sites (thinned region visible) + hotspot window.
2. PRE diagnostics: `u_tail(x)` and `cdf_l2(x)` (log-y, twin axes or normalized)
   — "the flag".
3. Stage-2 site histograms per arm (lhs / utail / oracle), 20 bins — "the action".
4. POST `cdf_l2(x)` per arm (log-y) + pre curve in gray — "the verification".
5. Paired scatter: `hot_cdf_l2` lhs (x-axis) vs utail (y-axis), one point per
   macrorep, y = x reference line.
6. `q_hat`: box/strip plot of pre, lhs, utail, oracle; horizontal line at the
   Gate-2 oracle-arm level (approx 0.30) for reference.

### Figure G4b (1 x 4 panels)

1. POST `cdf_l2(x)` for the three arms (log-y) + pre curve.
2. POST `u_tail(x)` for the three arms (log-y) — shows utail_fixed DOES drop.
3. LOF per LOF-site (x-axis = site location), three arms, threshold line —
   "the second flag".
4. `q_hat` box/strip plot per arm + pre.

---

## 6. Implementation notes and pitfalls

- **Rep-block bootstrap**: post-stage-2 `u_tail` resamples SITES of the pooled
  design; a resampled site keeps its whole 10-rep block. Same as Gate 3
  (`Y_sites[idx].ravel()`, `np.repeat(x_sites[idx], 10)`), just with the
  pooled site list.
- **Do not let arms share simulator noise streams**: stage-2 sites differ
  across arms, so common random numbers are impossible there; use the per-arm
  substreams of Section 2 and rely on pairing at the macrorep level.
- **`t_grid` must be recomputed per arm after pooling** (stage-2 y-values can
  extend the range, especially in the window where f has the bump).
- **Candidate pool is shared across arms within a macrorep** (drawn once with
  `rng_alloc`); only the sampling probabilities differ. This reduces
  between-arm Monte Carlo noise.
- **`choice(..., replace=False, p=...)` requires p to sum to 1** and can be
  slow for large pools — 2000 candidates is fine.
- **Runtime**: each CKME fit here is a Cholesky on <= 160 sites — milliseconds.
  The bootstrap dominates: per macrorep, G4a costs 1 pre-boot (30 fits) +
  3 arms x (1 refit + 30 post-boot fits) approx 125 fits; x 20 macroreps
  x 2 parts approx 5k fits plus CDF evaluations. Expect < 1 hour single-core;
  add `--n_workers` process parallelism over macroreps only if needed.
- **Numerical guard**: if `s2_i` in the LOF statistic is tiny (unlucky reps),
  floor it at `1e-8`. With sigma = 0.1 and 20 reps this should never bind.
- **Language/units in figures**: English labels, consistent with Gates 1-3.

## 7. File layout

```text
experiments/framing_validation/gate4_fix_epistemic.py
    --part {a,b,all}     default all
    --n_macro INT        default 20
    --analyze_only       re-read metrics/curves CSVs and re-plot only
```

Reuse Gate-3 helpers via `from gate3_epistemic_diagnostics import ...` (they
have no side effects at import; config constants are module-level so import is
safe) or copy them into the new file with a comment noting the origin. Copying
is acceptable if it keeps the two gates independently runnable; do not modify
`gate3_epistemic_diagnostics.py`.

## 8. Pilot before full run

Run `--n_macro 3` first and check:

1. G4a pre `u_tail` SI over [0.35, 0.65] is >> 10 (the flag works on the
   thinned design). If not, thin the interior further (6 -> 4 sites) before
   scaling up.
2. G4a `utail` arm puts > 50% of stage-2 sites in the window.
3. G4b `utail_retune` CV actually selects `ell_x < 0.1` (if it keeps choosing
   0.1, check the site-level fold split and the CRPS grid range).
4. LOF background median is O(1). If it is systematically >> 1, estimation
   error of `m_hat` is contaminating the statistic — increase stage-2 budget
   or report LOF with this caveat.

Only after the pilot passes, run the full `--n_macro 20` and fill the results
into `README.md` (new "Gate 4" rows) and
`notes/planning/current/paper/paper-storyline-and-merge-decision.md` Section 6.
