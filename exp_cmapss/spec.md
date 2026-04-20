# C-MAPSS Experiment Spec (exp_cmapss)

## Goal

Apply CKME+CP to C-MAPSS FD001 for RUL uncertainty prediction.
CKME estimates the conditional distribution P(RUL | X); split conformal prediction
constructs calibrated prediction intervals.

**Y = RUL** (remaining useful life = max_cycle − current_cycle, capped at 125).
**X** = window-based features computed from the last 20 cycles (mean, std, min, max, slope,
last-minus-first per retained sensor) + life_progress.
No replications are needed. Each snapshot is one independent (X, Y) pair;
CKME handles varying X via kernel smoothing.

---

## Pipeline overview

```
FD001 data
  → engine-wise split (train / cal / test engines)
  → sparse snapshot selection (life-progress-based, per engine)
  → window features
  → standardize (train statistics only)
  → CKME fit on X_train, Y_train
  → CP calibrate on X_cal, Y_cal
  → evaluate on X_test, Y_test
```

---

## Preprocessing design

### 1. Engine-wise splitting ✅ implemented
Split training engines into train / calibration by engine ID (not by row or cycle).
FD001 test engines are kept as the held-out test set.
Default: 80% of training engines → train, 20% → calibration.

### 2. Window-based features ✅ implemented
For each selected snapshot cycle, compute features from the last `window` cycles:
mean, std, min, max, slope (linear trend), last-minus-first per retained sensor column.
Also append `life_progress = cycle / max_cycle`.
Near-constant sensors (training std < 1e-4) are dropped automatically.

### 3. Sparse snapshot selection ❌ not yet implemented
**Current behavior**: `_window_features` generates one row per cycle per engine
(dense, highly autocorrelated).

**Target behavior**: select a small number of snapshots per engine at target
life-progress levels.

- Training engines: 6 snapshots at progress ≈ [0.20, 0.35, 0.50, 0.70, 0.85, 0.95]
- Calibration engines: 3 snapshots at progress ≈ [0.30, 0.60, 0.90]
- Test engines: all cycles (for full evaluation curve), or sparse at same 6 levels

For each target level, find the cycle in that engine with the closest
`cycle / max_cycle` and extract window features there.

This ensures: (a) each engine contributes a fixed number of samples (balanced),
(b) samples are well-separated within each engine (reduced autocorrelation),
(c) samples are comparable across engines of different total lifetimes.

### 4. Return metadata ❌ not yet implemented
Preprocessing should return per-sample metadata alongside X and Y:
- `unit_id`   — engine index
- `cycle`     — cycle number of the snapshot
- `life_progress` — cycle / max_cycle at the snapshot

Needed for: per-stage coverage analysis, engine-balanced diagnostics,
and group-aware CV tuning.

### 5. Standardize ✅ implemented
Fit mean/std on X_train only; apply the same transform to X_cal and X_test.

---

## Run pipeline (current)

```bash
# CV-tune CKME hyperparameters once
python exp_cmapss/pretrain_params.py

# Run full experiment
python exp_cmapss/run_cmapss.py
```

Methods compared:
- **Ridge + split CP** — linear baseline with residual conformal calibration
- **CKME Stage 1 CP** — CKME calibrated on all calibration samples
- **CKME Stage 2 CP** — CKME calibrated on high-risk subset (RUL ≤ late_stage_rul)

---

## Config parameters (exp_cmapss/config.txt)

| Parameter | Default | Description |
|---|---|---|
| `rul_cap` | 125 | Cap RUL at this value (standard for FD001) |
| `window_size` | 20 | Lookback window in cycles |
| `train_frac` | 0.8 | Fraction of training engines for CKME fit |
| `t_grid_size` | 300 | Number of CDF threshold points |
| `alpha` | 0.1 | CP significance level (target 90% coverage) |
| `late_stage_rul` | 30 | High-risk threshold for Stage 2 CP |
| `ell_x` | 1.0 | RBF length scale (overridden by pretrained_params.json) |
| `lam` | 0.01 | Tikhonov regularization |
| `h` | 0.3 | Indicator bandwidth |

---

## Output

```
exp_cmapss/output/
  tables/
    cmapss_results.csv         — coverage / width / IS / RMSE per method × subset
    cmapss_results_all.csv     — all-test subset
    cmapss_results_RULle30.csv — late-stage (RUL ≤ 30) subset
    cmapss_predictions.csv     — per-sample L, U, Y_pred for all methods
  figures/
    (plots go here)
```

---

## Known issues / TODO

- [ ] Implement sparse snapshot selection in `preprocess.py` (items 3, 4 above)
- [ ] Return `unit_id`, `cycle`, `life_progress` metadata from `load_and_preprocess`
- [ ] Add per-RUL-bin coverage plot in `plot_cmapss.py`
- [ ] Evaluate whether sparser calibration set (3 snapshots/engine) affects CP validity
