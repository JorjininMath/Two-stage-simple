"""
preprocess.py

Load and preprocess C-MAPSS FD001 for the CKME+CP pipeline.

Steps
-----
1. Load train_FD001.txt and test_FD001.txt (+ RUL_FD001.txt)
2. Compute RUL labels; optionally cap at rul_cap
3. Drop near-constant sensors (std < variance_threshold)
4. Build window-based tabular features (lookback L cycles)
5. Standardize features using training statistics only
6. Split training engines into train / calibration by engine id

Output
------
A dict with keys:
    X_train, Y_train          : CKME training data
    X_cal, Y_cal, rul_cal     : calibration data with RUL labels
    X_test, Y_test, rul_test  : test data with RUL labels
    feature_names             : list of feature name strings
    scaler_mean, scaler_std   : for reference
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict


# FD001 column names (26 total)
_RAW_COLS = (
    ["unit", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors known to be near-constant in FD001 (kept here for reference;
# auto-detection is still applied on top)
_KNOWN_CONSTANT_FD001 = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}


def _load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=_RAW_COLS)
    return df


def _add_rul(df: pd.DataFrame, rul_cap: Optional[int] = 125) -> pd.DataFrame:
    """Add RUL column. RUL = max_cycle_in_engine - current_cycle."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = max_cycle - df["cycle"]
    if rul_cap is not None:
        df["rul"] = df["rul"].clip(upper=rul_cap)
    return df


def _select_sensors(
    df_train: pd.DataFrame,
    variance_threshold: float = 1e-4,
) -> list[str]:
    """
    Return list of sensor + op columns to keep.
    Drop near-constant sensors based on training data std.
    """
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    op_cols = ["op1", "op2", "op3"]

    keep = []
    for col in op_cols + sensor_cols:
        if df_train[col].std() > variance_threshold:
            keep.append(col)
    return keep


def _window_features(
    df: pd.DataFrame,
    selected_cols: list[str],
    window: int = 20,
    snapshot_levels: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build window-based feature matrix.

    For each selected cycle t, use the window [max(1, t-window+1), ..., t].
    Features per selected column:
        mean, std, min, max, slope (linear trend), last-minus-first
    Plus: normalized life progress (cycle / max_cycle_in_engine)

    Parameters
    ----------
    snapshot_levels : list of float or None
        If provided, each engine contributes exactly len(snapshot_levels) samples,
        one per target life-progress level. For each level, the cycle with the
        closest life_progress = cycle/max_cycle is selected.
        If None, every cycle is used (dense, for test sets or backward compat).

    Returns
    -------
    X    : ndarray, shape (N, n_features)
    Y    : ndarray, shape (N,)   -- RUL
    meta : ndarray, shape (N, 3) -- columns: unit_id, cycle, life_progress
    """
    rows_X, rows_Y, rows_meta = [], [], []

    for unit_id, grp in df.groupby("unit"):
        grp = grp.sort_values("cycle").reset_index(drop=True)
        max_cycle = grp["cycle"].max()
        progress = grp["cycle"].values / max_cycle  # life progress for every row

        if snapshot_levels is not None:
            # For each target level, find the row index with closest life progress
            selected_indices = []
            for level in snapshot_levels:
                idx = int(np.argmin(np.abs(progress - level)))
                selected_indices.append(idx)
            # Deduplicate while preserving order (short engines may map to same cycle)
            seen = set()
            selected_indices = [i for i in selected_indices if not (i in seen or seen.add(i))]
        else:
            selected_indices = list(range(len(grp)))

        for i in selected_indices:
            start = max(0, i - window + 1)
            window_data = grp.iloc[start : i + 1]

            feat = []
            for col in selected_cols:
                vals = window_data[col].values.astype(float)
                feat.append(vals.mean())
                feat.append(vals.std() if len(vals) > 1 else 0.0)
                feat.append(vals.min())
                feat.append(vals.max())
                if len(vals) > 1:
                    t_idx = np.arange(len(vals), dtype=float)
                    slope = np.polyfit(t_idx, vals, 1)[0]
                else:
                    slope = 0.0
                feat.append(slope)
                feat.append(float(vals[-1] - vals[0]))  # last - first

            lp = float(grp.iloc[i]["cycle"]) / max_cycle
            feat.append(lp)

            rows_X.append(feat)
            rows_Y.append(float(grp.iloc[i]["rul"]))
            rows_meta.append([float(unit_id), float(grp.iloc[i]["cycle"]), lp])

    X    = np.array(rows_X,   dtype=float)
    Y    = np.array(rows_Y,   dtype=float)
    meta = np.array(rows_meta, dtype=float)  # columns: unit_id, cycle, life_progress
    return X, Y, meta


def _build_feature_names(selected_cols: list[str]) -> list[str]:
    names = []
    suffixes = ["mean", "std", "min", "max", "slope", "last_minus_first"]
    for col in selected_cols:
        for suf in suffixes:
            names.append(f"{col}_{suf}")
    names.append("life_progress")
    return names


def _standardize(
    X_train: np.ndarray,
    *others: np.ndarray,
) -> Tuple:
    """Standardize using training mean/std. Returns (X_train_scaled, *others_scaled, mean, std)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    scaled = [(X - mean) / std for X in [X_train, *others]]
    return (*scaled, mean, std)


_DEFAULT_TRAIN_LEVELS = [0.10, 0.22, 0.34, 0.47, 0.59, 0.71, 0.83, 0.95]
_DEFAULT_CAL_LEVELS   = [0.80, 0.90, 1.0]   # late-stage only, to align with test (last cycle)
_DEFAULT_TEST_LEVELS  = [1.0]                # last observed cycle per engine (benchmark task)


def load_and_preprocess(
    data_dir: str = "exp_cmapss/data",
    rul_cap: int = 125,
    window: int = 20,
    train_frac: float = 0.8,
    late_stage_rul: int = 30,
    variance_threshold: float = 1e-4,
    random_state: int = 42,
    train_snapshot_levels: Optional[list] = None,
    cal_snapshot_levels: Optional[list] = None,
    test_snapshot_levels: Optional[list] = None,
    verbose: bool = True,
) -> Dict:
    """
    Full preprocessing pipeline for FD001.

    Parameters
    ----------
    data_dir : str
        Directory containing train_FD001.txt, test_FD001.txt, RUL_FD001.txt.
    rul_cap : int
        Cap RUL at this value (standard: 125).
    window : int
        Lookback window size.
    train_frac : float
        Fraction of training engines used for CKME training (rest = calibration).
    late_stage_rul : int
        RUL threshold for high-risk / late-stage subset.
    variance_threshold : float
        Drop sensors with training std below this.
    random_state : int
        Seed for engine-level train/cal split.
    train_snapshot_levels : list of float or None
        Life-progress levels at which to extract snapshots from training engines.
        Defaults to [0.20, 0.35, 0.50, 0.70, 0.85, 0.95] (6 per engine).
        Pass None to use all cycles (dense, not recommended for CKME).
    cal_snapshot_levels : list of float or None
        Life-progress levels for calibration engines.
        Defaults to [0.30, 0.60, 0.90] (3 per engine, sparser for CP validity).
        Pass None to use all cycles.
    verbose : bool

    Returns
    -------
    dict with keys:
        X_train, Y_train, meta_train
        X_cal, Y_cal, rul_cal, meta_cal
        X_test, Y_test, rul_test, meta_test
        X_cal_highrisk, Y_cal_highrisk, rul_cal_highrisk
        feature_names, scaler_mean, scaler_std
        n_train_engines, n_cal_engines

    meta arrays have columns: [unit_id, cycle, life_progress]
    """
    if train_snapshot_levels is None:
        train_snapshot_levels = _DEFAULT_TRAIN_LEVELS
    if cal_snapshot_levels is None:
        cal_snapshot_levels = _DEFAULT_CAL_LEVELS
    if test_snapshot_levels is None:
        test_snapshot_levels = _DEFAULT_TEST_LEVELS
    data_dir = Path(data_dir)
    train_path = data_dir / "train_FD001.txt"
    test_path  = data_dir / "test_FD001.txt"
    rul_path   = data_dir / "RUL_FD001.txt"

    for p in [train_path, test_path, rul_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Missing: {p}\n"
                f"Download C-MAPSS FD001 from NASA and place files in {data_dir}/"
            )

    # --- Load raw data ---
    df_train_raw = _load_raw(train_path)
    df_test_raw  = _load_raw(test_path)

    # --- Add RUL to training data ---
    df_train_raw = _add_rul(df_train_raw, rul_cap=rul_cap)

    # --- Add RUL to test data ---
    # RUL_FD001.txt: one value per engine = RUL at the LAST observed cycle
    rul_test_last = pd.read_csv(rul_path, header=None, names=["rul_last"])["rul_last"].values
    df_test_raw = df_test_raw.copy()
    # compute RUL for all cycles: RUL_at_cycle_t = rul_at_last + (max_cycle - t)
    def _add_test_rul(grp, rul_last_val, rul_cap):
        grp = grp.sort_values("cycle").reset_index(drop=True)
        max_cycle = grp["cycle"].max()
        rul = rul_last_val + (max_cycle - grp["cycle"])
        if rul_cap is not None:
            rul = rul.clip(upper=rul_cap)
        grp["rul"] = rul.values
        return grp

    test_units = df_test_raw["unit"].unique()
    test_parts = []
    for uid, rul_val in zip(sorted(test_units), rul_test_last):
        grp = df_test_raw[df_test_raw["unit"] == uid]
        test_parts.append(_add_test_rul(grp, rul_val, rul_cap))
    df_test_raw = pd.concat(test_parts, ignore_index=True)

    # --- Select informative sensors ---
    selected_cols = _select_sensors(df_train_raw, variance_threshold=variance_threshold)
    if verbose:
        print(f"Selected {len(selected_cols)} columns: {selected_cols}")

    # --- Split train engines into train / calibration ---
    rng = np.random.default_rng(random_state)
    all_train_units = df_train_raw["unit"].unique()
    rng.shuffle(all_train_units)
    n_train_eng = int(len(all_train_units) * train_frac)
    train_units = set(all_train_units[:n_train_eng])
    cal_units   = set(all_train_units[n_train_eng:])

    df_train = df_train_raw[df_train_raw["unit"].isin(train_units)]
    df_cal   = df_train_raw[df_train_raw["unit"].isin(cal_units)]

    if verbose:
        print(f"Train engines: {len(train_units)}, Cal engines: {len(cal_units)}, Test engines: {len(test_units)}")

    # --- Build window features ---
    if verbose:
        print("Building window features...")
        print(f"  Train snapshot levels: {train_snapshot_levels} ({len(train_snapshot_levels)} per engine)")
        print(f"  Cal snapshot levels:   {cal_snapshot_levels} ({len(cal_snapshot_levels)} per engine, late-stage)")
        print(f"  Test snapshot levels:  {test_snapshot_levels} (last cycle per engine)")

    X_train_raw, Y_train, meta_train = _window_features(
        df_train, selected_cols, window, snapshot_levels=train_snapshot_levels
    )
    X_cal_raw, Y_cal, meta_cal = _window_features(
        df_cal, selected_cols, window, snapshot_levels=cal_snapshot_levels
    )
    X_test_raw, Y_test, meta_test = _window_features(
        df_test_raw, selected_cols, window, snapshot_levels=test_snapshot_levels
    )

    # RUL labels are the same as Y for filtering purposes
    rul_cal  = Y_cal.copy()
    rul_test = Y_test.copy()

    # --- Standardize ---
    X_train, X_cal, X_test, scaler_mean, scaler_std = _standardize(
        X_train_raw, X_cal_raw, X_test_raw
    )

    # --- High-risk calibration subset ---
    mask_highrisk = rul_cal <= late_stage_rul
    X_cal_highrisk   = X_cal[mask_highrisk]
    Y_cal_highrisk   = Y_cal[mask_highrisk]
    rul_cal_highrisk = rul_cal[mask_highrisk]

    feature_names = _build_feature_names(selected_cols)

    if verbose:
        print(f"X_train: {X_train.shape}  ({len(train_units)} engines × {len(train_snapshot_levels)} levels)")
        print(f"X_cal:   {X_cal.shape}  ({len(cal_units)} engines × {len(cal_snapshot_levels)} late-stage levels, "
              f"high-risk: {X_cal_highrisk.shape[0]})")
        print(f"X_test:  {X_test.shape}  ({len(test_units)} engines × last cycle)")
        print(f"RUL range train: [{Y_train.min():.0f}, {Y_train.max():.0f}]")

    return dict(
        X_train=X_train,
        Y_train=Y_train,
        meta_train=meta_train,
        X_cal=X_cal,
        Y_cal=Y_cal,
        rul_cal=rul_cal,
        meta_cal=meta_cal,
        X_cal_highrisk=X_cal_highrisk,
        Y_cal_highrisk=Y_cal_highrisk,
        rul_cal_highrisk=rul_cal_highrisk,
        X_test=X_test,
        Y_test=Y_test,
        rul_test=rul_test,
        meta_test=meta_test,
        feature_names=feature_names,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        n_train_engines=len(train_units),
        n_cal_engines=len(cal_units),
        selected_cols=selected_cols,
    )
