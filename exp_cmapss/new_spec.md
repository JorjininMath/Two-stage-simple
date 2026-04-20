# C-MAPSS Preprocess Design for CKME-CP

## Goal
Design a C-MAPSS preprocessing pipeline that is more compatible with CKME-CP by preserving short-term degradation history while reducing within-engine dependence and sample redundancy.

## 1. Treat engines as the primary sampling units
Use **engine-wise splitting** for proper train / calibration / test.  
Do not split by row or by cycle.  
This avoids leakage from the same engine appearing in multiple sets.

## 2. Keep window-based state representation
Do not use raw single-cycle sensor values directly as CKME inputs.  
For each selected cycle, construct features from a recent window (for example, the last 20 cycles), such as mean, standard deviation, min, max, slope, and last-minus-first for each retained variable.  
This lets \(X\) represent local degradation history instead of a noisy snapshot.

## 3. Replace dense per-cycle samples with sparse snapshots
Do **not** create one training sample for every cycle.  
Instead, sample a small number of representative cycles from each engine.  
This reduces strong dependence caused by highly overlapping adjacent windows and prevents one engine from contributing many near-duplicate samples.

## 4. Use progress-based snapshot selection
Prefer selecting snapshots by **life progress** rather than fixed cycle spacing.  
For example, choose cycles closest to target progress levels such as 0.20, 0.35, 0.50, 0.70, 0.85, and 0.95.  
This makes sample locations more comparable across engines with different total lifetimes.

## 5. Use fewer and more separated snapshots in calibration
Calibration samples should be more sparse than training samples.  
For example, training engines may contribute 5--6 snapshots, while calibration engines contribute only 2--3 well-separated snapshots.  
This makes the calibration set closer to the engine-level independence structure desired by conformal prediction.

## 6. Return engine metadata together with samples
The preprocessing output should include not only `X` and `Y`, but also sample-level metadata such as:
- `unit_id`
- `cycle`
- `life_progress`

This is necessary for later group-aware tuning, engine-balanced diagnostics, and coverage analysis by degradation stage.

## 7. Keep engine contributions balanced
The design should avoid giving long trajectories disproportionately large influence.  
This can be done either by forcing each engine to contribute the same number of snapshots, or later by using equal-engine weighting in model fitting or validation.

## 8. Recommended final principle
The new preprocess should not aim to make C-MAPSS strictly i.i.d.  
Instead, it should:
- preserve local degradation information,
- reduce within-engine redundancy,
- and align the sample structure more closely with the engine-level independence assumption behind CKME-CP.