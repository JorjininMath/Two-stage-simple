"""
scores.py

Nonconformity score computation for Conformal Prediction.

This module provides pure functions for computing nonconformity scores from CDF values.
These functions are used internally by calibration and interval construction modules.

Key idea
--------
A nonconformity score A(x, y) measures how "unlikely" or "nonconforming" a
pair (x, y) is under the model. Higher scores indicate higher nonconformity.

For CKME models, we use the CDF-based score:
- A(x, y) = |F(y | x) - 0.5|: Distance from CDF value to 0.5
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray

# Valid score types (centralized definition)
VALID_SCORE_TYPES = ["abs_median"]


def score_from_cdf(F: ArrayLike, score_type: str = "abs_median") -> ArrayLike:
    """
    Compute nonconformity scores from CDF values.

    This is a pure function that takes CDF values and returns scores.
    It is used internally by calibration and interval construction modules.

    Parameters
    ----------
    F : array-like, shape (n,) or (n, m)
        CDF values F(y | x). Can be a 1D array for single point or 2D array
        for multiple points/grid.

    score_type : str, default="abs_median"
        Type of nonconformity score: "abs_median" = |F(y | x) - 0.5|

    Returns
    -------
    scores : ndarray, same shape as F
        Nonconformity scores. Higher scores indicate higher nonconformity.

    Raises
    ------
    ValueError
        If score_type is not supported.
    """
    F = np.asarray(F, dtype=float)

    if score_type == "abs_median":
        return np.abs(F - 0.5)
    else:
        raise ValueError(
            f"score_type must be one of {VALID_SCORE_TYPES}, "
            f"got {score_type}"
        )
