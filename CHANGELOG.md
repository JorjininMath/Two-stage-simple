# Changelog

Correctness fixes and protocol changes. Newest first. Each entry states what
was wrong, why it mattered, and what changed.

## 2026-07-22 — Research workspace and provenance reorganization

- Moved the public Python packages to `src/{CKME,CP,Two_stage}` while
  preserving the `CKME`, `CP`, and `Two_stage` import APIs.
- Added editable-install packaging, stable project-path helpers, and core
  import/path/two-stage smoke tests.
- Moved experiment implementations under `experiments/` and retained
  lightweight wrappers for selected old `exp_*` commands.
- Renamed the implementable adaptive-scale code and Exp4 entrypoints to state
  the sample-SD plus Nadaraya-Watson estimator explicitly.
- Retired IQR response-scale outputs to
  `_archive/01_experiments/diagnostics/adaptive_h_iqr_response_scale/`; they are
  excluded from the result registry and manuscript exporter.
- Added project-status, result/claim registry, advisor-update,
  advisor-feedback, manuscript-asset provenance, and indexed-archive layers.
- Moved R benchmarks to `benchmarks/dcp/`, the SLURM dispatcher to `hpc/`, and
  paper snapshots into `paper/current/` versus `paper/archive/`.
- No research payload was permanently deleted.

## 2026-07-17 — Task 1: correctness & estimator lock

### iid q_X calibration mode (exchangeability fix)
- **Wrong:** `run_stage2` built the calibration set from design-selected
  sites (LHS / S^0-driven) with `r_1` replications each, flattened. Clustered
  scores are not exchangeable with a single-draw test point, and the
  calibration X-law was not q_X — the split-CP theorem's assumptions did not
  match the code.
- **Fix:** new default `method="iid"`: calibration inputs iid ~ q_X
  (`src/Two_stage/design.py::sample_iid_qx`, custom `qx_sampler` supported),
  exactly one fresh output each (`r_1=1` enforced). Legacy modes retained
  with a `UserWarning`. Files: `src/Two_stage/stage2.py`,
  `src/Two_stage/design.py`, `src/Two_stage/__init__.py`.

### Site-grouped CV folds (replicate-leakage fix)
- **Wrong:** `src/CKME/tuning.py` used shuffled `KFold` on flattened replicated
  rows; replications of one site landed in both train and validation folds,
  rewarding interpolation of site noise and biasing CV toward
  under-smoothing.
- **Fix:** `groups` parameter threaded through `tune_ckme_params` /
  `cross_validate_ckme` / `_evaluate_params_cv` (GroupKFold when given);
  `CKMEModel.fit` passes `np.repeat(np.arange(n_sites), r)` automatically
  when `r > 1`. Direct tuning callers on replicated data
  (`experiments/design/pretrain_params.py`,
  `experiments/onesided/pretrain_params.py`)
  still need `groups` — flagged, not yet migrated.

### Conformal quantile edge case (k > n_cal)
- **Wrong:** `k = min(ceil((1-alpha)(n_cal+1)), n_cal)` silently truncated to
  the max calibration score when n_cal was too small for the level alpha;
  the split-CP proof requires predicting the whole space in that case.
- **Fix:** `q_hat = +inf` when `k > n_cal`. Files:
  `src/CP/calibration.py`,
  `experiments/adaptive_h/adaptive_bandwidth.py::adaptive_recalibrate_q`.

### PROTOCOL.md added
Locked estimator + evaluation protocol (query-specific h(x), site-grouped
CV, iid calibration, raw-score guarantee layer vs projected reporting layer,
+inf edge case, >=50 macroreps for paper numbers).

## 2026-07-16 — CDF legality / interval extraction

### Shared monotone-projection interval helper
- **Wrong:** two inconsistent interval-extraction conventions coexisted
  (`src/CP/interval.py` rightmost-True search vs generalized inverse in
  experiment code), and `np.searchsorted` was applied to non-monotone raw
  CDF rows — a silent error (binary search assumes sorted input), moving up
  to ~60% of interval endpoints for small fixed h.
- **Fix:** single canonical helper
  `src/CP/interval.py::projected_quantile_interval` — running-max projection,
  clip to [0,1], generalized inverse — used by `CP.predict_interval` and
  `experiments/adaptive_h/adaptive_bandwidth.py::adaptive_predict_interval`.

### Raw-score guarantee layer confirmed as default
- Projected-score calibration (Policy B) is available behind an optional
  `t_grid` argument (`src/CP/calibration.py`, `src/CP/cp.py`) but NOT the default:
  with t-grids narrower than the response range, projected scores saturate
  at 0.5 and the conformal quantile degenerates. Default calibration uses
  raw point-evaluated scores; intervals for reporting go through the
  projection (two-layer protocol, see PROTOCOL.md §4).
