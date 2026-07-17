# Changelog

Correctness fixes and protocol changes. Newest first. Each entry states what
was wrong, why it mattered, and what changed.

## 2026-07-17 — Task 1: correctness & estimator lock

### iid q_X calibration mode (exchangeability fix)
- **Wrong:** `run_stage2` built the calibration set from design-selected
  sites (LHS / S^0-driven) with `r_1` replications each, flattened. Clustered
  scores are not exchangeable with a single-draw test point, and the
  calibration X-law was not q_X — the split-CP theorem's assumptions did not
  match the code.
- **Fix:** new default `method="iid"`: calibration inputs iid ~ q_X
  (`Two_stage/design.py::sample_iid_qx`, custom `qx_sampler` supported),
  exactly one fresh output each (`r_1=1` enforced). Legacy modes retained
  with a `UserWarning`. Files: `Two_stage/stage2.py`, `Two_stage/design.py`,
  `Two_stage/__init__.py`.

### Site-grouped CV folds (replicate-leakage fix)
- **Wrong:** `CKME/tuning.py` used shuffled `KFold` on flattened replicated
  rows; replications of one site landed in both train and validation folds,
  rewarding interpolation of site noise and biasing CV toward
  under-smoothing.
- **Fix:** `groups` parameter threaded through `tune_ckme_params` /
  `cross_validate_ckme` / `_evaluate_params_cv` (GroupKFold when given);
  `CKMEModel.fit` passes `np.repeat(np.arange(n_sites), r)` automatically
  when `r > 1`. Direct tuning callers on replicated data
  (`exp_design/pretrain_params.py`, `exp_onesided/pretrain_params.py`)
  still need `groups` — flagged, not yet migrated.

### Conformal quantile edge case (k > n_cal)
- **Wrong:** `k = min(ceil((1-alpha)(n_cal+1)), n_cal)` silently truncated to
  the max calibration score when n_cal was too small for the level alpha;
  the split-CP proof requires predicting the whole space in that case.
- **Fix:** `q_hat = +inf` when `k > n_cal`. Files: `CP/calibration.py`,
  `exp_adaptive_h/adaptive_h_utils.py::adaptive_recalibrate_q`.

### PROTOCOL.md added
Locked estimator + evaluation protocol (query-specific h(x), site-grouped
CV, iid calibration, raw-score guarantee layer vs projected reporting layer,
+inf edge case, >=50 macroreps for paper numbers).

## 2026-07-16 — CDF legality / interval extraction

### Shared monotone-projection interval helper
- **Wrong:** two inconsistent interval-extraction conventions coexisted
  (`CP/interval.py` rightmost-True search vs generalized inverse in
  experiment code), and `np.searchsorted` was applied to non-monotone raw
  CDF rows — a silent error (binary search assumes sorted input), moving up
  to ~60% of interval endpoints for small fixed h.
- **Fix:** single canonical helper
  `CP/interval.py::projected_quantile_interval` — running-max projection,
  clip to [0,1], generalized inverse — used by `CP.predict_interval` and
  `exp_adaptive_h/adaptive_h_utils.py::adaptive_predict_interval`.

### Raw-score guarantee layer confirmed as default
- Projected-score calibration (Policy B) is available behind an optional
  `t_grid` argument (`CP/calibration.py`, `CP/cp.py`) but NOT the default:
  with t-grids narrower than the response range, projected scores saturate
  at 0.5 and the conformal quantile degenerates. Default calibration uses
  raw point-evaluated scores; intervals for reporting go through the
  projection (two-layer protocol, see PROTOCOL.md §4).
