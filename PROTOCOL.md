# Locked Estimator & Evaluation Protocol

This document freezes the estimator and the evaluation protocol used by all
paper-bound experiments. Any experiment that deviates from this protocol must
say so explicitly in its `spec.md`. Changes to this file are logged in
`CHANGELOG.md`.

## 1. Estimator

**CKME conditional CDF.**
`F_hat(t|x) = sum_i w_i(x) * (1/r_i) * sum_j Psi((t - y_ij)/h)`, with
`w(x) = (K + n*lam*I)^{-1} k_x`. Replicated designs are fitted in
distinct-site mode (`CKMEModel.fit(..., r=r_0)`): the n_0 x n_0 system with
per-site averaged indicators is mathematically identical to the flattened
n_0*r_0 system (block structure of K_X), at O(r_0^2) memory savings.

**Scale-adaptive bandwidth.** The adaptive variant uses a query-specific
bandwidth `h(x) = c * s_hat(x)` applied to ALL training responses at
evaluation time (NOT training-site-specific h_i — that is a different
estimator). `s_hat(x)` is the plug-in scale: per-site sample SD smoothed by
Nadaraya-Watson regression. `h(x)` is floored (see
`exp_adaptive_h/adaptive_h_utils.py`) to avoid degenerate indicators.

## 2. Hyperparameter tuning

k-fold CV with CRPS loss. With replicated training data, folds are
**site-grouped** (`GroupKFold`; all replications of a site stay in one fold).
Plain shuffled KFold on flattened replicated rows leaks replications across
train/validation and biases CV toward under-smoothing. The main pipeline does
this automatically (`fit(..., r=r_0)` passes site groups); direct callers of
`tune_ckme_params` / `cross_validate_ckme` on replicated data must pass
`groups=np.repeat(np.arange(n_sites), r)`.

All tuning decisions are frozen BEFORE calibration data is drawn.

## 3. Calibration (split conformal)

**iid q_X calibration** (`run_stage2(method="iid")`, the default):
calibration inputs are drawn iid from the target law q_X (uniform over the
design box unless a custom `qx_sampler` is given), each with exactly ONE
fresh simulator output (`r_1 = 1`). With the model frozen, calibration pairs
and a fresh test pair are iid from q_X x F(.|x); the split-CP finite-sample
guarantee then holds exactly.

Replications belong in Stage-1 training (scale learning, per-site ECDFs,
distinct-site compression), not in calibration: the conformal quantile is a
marginal object, and for a fixed budget iid spreading dominates replicating
(effective calibration size under clustering is the number of sites, not
sites x reps). Legacy design-selected / replicated calibration modes
("lhs", "sampling", "mixed") are retained only to reproduce older
experiments and emit a `UserWarning`.

Design freedom: the TRAINING design is unconstrained (space-filling,
replicated, adaptive, multi-round) — validity only requires that adaptivity
stops before the calibration draw.

**Conformal quantile.** `k = ceil((1-alpha)*(n_cal+1))`; if `k > n_cal`,
`q_hat = +inf` (predict the whole space). Truncating to the max score would
break the guarantee.

## 4. Two-layer evaluation

- **Guarantee layer (primary coverage metric):** raw point-evaluated DCP
  scores `|F_hat(y|x) - 1/2|`, calibrated and tested on the SAME raw-score
  object. This is the exact split-CP quantity; coverage is reported from
  score membership `score(y) <= q_hat`.
- **Reporting layer (width / interval score):** the monotone-projected
  interval — running max over the t-grid, clip to [0,1], generalized inverse
  (`CP/interval.py::projected_quantile_interval`, the single shared
  implementation). Width and Winkler interval score are ONLY defined on this
  interval. The raw score set's Lebesgue measure coincides with the interval
  width iff the set is connected; discrepancies are reported, not hidden.

Rationale and edge cases (saturation, non-monotone raw CDFs, searchsorted on
non-monotone arrays): `notes/planning/cdf_legality_policy.md` (local),
summarized in the relevant experiment specs.

## 5. Experiment conventions

- **Macroreps:** >= 50 for any number that appears in a table or figure
  destined for the paper. Pilots may use 10 but must be labeled as pilots
  (8-macrorep artifacts have produced sign-flipping conclusions).
- **Seeds:** one base seed per experiment, macrorep k uses `base + k`;
  design and simulator noise seeded independently.
- **Metrics:** coverage (guarantee layer), width and Winkler interval score
  (reporting layer), plus CRPS / CDF error where the spec calls for it.
- **Configs:** every experiment directory carries a `config.txt` or
  `spec.md` stating n_0, r_0, n_cal, alpha, grids, and arm definitions.
