"""
preprocess.py

Load and preprocess Kenneth French daily market returns.

Response Y  : daily market return  Mkt = MktRF + RF  (units: %)
Covariate X : 22-day lagged realized volatility        (units: %)
              lrealvol[t] = sqrt(sum_{s=t-22}^{t-1} Mkt[s]^2)

The first 22 rows are dropped (insufficient lag history).

Usage:
    from exp_returns.preprocess import load_returns_data, get_round_splits
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data_us_08172021.CSV"
VOL_LAG   = 22          # number of past days used to compute realized vol


def load_returns_data(data_path: Path | str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load daily returns and compute lagged realized volatility.

    Returns
    -------
    Y : ndarray, shape (T,)
        Daily market returns (%).
    X : ndarray, shape (T, 1)
        22-day lagged realized volatility (%).
    """
    if data_path is None:
        data_path = DATA_PATH
    df = pd.read_csv(data_path)
    df["Mkt"] = df["MktRF"] + df["RF"]

    mkt = df["Mkt"].values.astype(float)
    T   = len(mkt)

    lrealvol = np.empty(T)
    lrealvol[:] = np.nan
    for t in range(VOL_LAG, T):
        lrealvol[t] = math.sqrt(float(np.sum(mkt[t - VOL_LAG : t] ** 2)))

    # Drop first VOL_LAG rows (no lag history)
    Y = mkt[VOL_LAG:]
    X = lrealvol[VOL_LAG:].reshape(-1, 1)
    return Y, X


def get_round_splits(
    T: int,
    n_rounds: int = 5,
    holdout_frac: float = 0.10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate (ind_cp, ind_test) index pairs for n_rounds rolling windows.

    Mirrors the R code structure:
        ind_cp   = [(r-1)*T_ho .. floor(T*0.5) + (r-1)*T_ho]
        ind_test = [max(ind_cp)+1 .. max(ind_cp)+T_ho]

    Parameters
    ----------
    T : int
        Total number of observations.
    n_rounds : int
        Number of rolling rounds.
    holdout_frac : float
        Fraction of T used as the holdout (test) window per round.

    Returns
    -------
    List of (ind_cp, ind_test) where each is a numpy integer array.
    """
    T_ho       = int(math.floor(T * holdout_frac))
    base_train = int(math.floor(T * 0.50))
    splits     = []
    for r in range(n_rounds):
        start_cp  = r * T_ho
        end_cp    = base_train + r * T_ho + 1   # exclusive upper bound
        start_tst = end_cp
        end_tst   = end_cp + T_ho
        if end_tst > T:
            break
        splits.append((np.arange(start_cp, end_cp), np.arange(start_tst, end_tst)))
    return splits


def split_d0_d1(
    ind_cp: np.ndarray,
    split_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a training-window index array into D0 (fit) and D1 (calibrate).

    The first split_ratio fraction goes to D0, the rest to D1.
    Temporal order is preserved.
    """
    mid   = int(math.floor(len(ind_cp) * split_ratio))
    ind_d0 = ind_cp[:mid]
    ind_d1 = ind_cp[mid:]
    return ind_d0, ind_d1
