"""
interval.py

Interval construction layer for Conformal Prediction.

This module provides functions for constructing prediction intervals using
level set search on a grid of Y values.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CKME.ckme import CKMEModel

from .scores import score_from_cdf

ArrayLike = np.ndarray


def predict_interval(
    model: "CKMEModel",
    X_query: ArrayLike,
    t_grid: ArrayLike,
    q_hat: float,
    score_type: str = "abs_median",
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Construct conformal intervals [L(x_j), U(x_j)] for batch of query points.

    For each query point x, this function:
    1. Computes F(t | x) for all t in t_grid
    2. Computes scores A(x, t) from CDF values
    3. Finds the leftmost and rightmost t where A(x, t) ≤ q̂

    Parameters
    ----------
    model : CKMEModel
        Trained CKME model. Must be fitted before use.

    X_query : array-like, shape (q, d) or (d,)
        Query input points. Can be a single point (1D) or multiple points (2D).

    t_grid : array-like, shape (M,)
        Dense grid of Y values for level set search. Should cover the
        range of possible Y values.

    q_hat : float
        Conformal quantile threshold (from calibration).

    score_type : str, default="abs_median"
        Type of nonconformity score to use. Must match the score type used
        in calibration.

    Returns
    -------
    L : ndarray, shape (q,)
        Left bounds of prediction intervals.

    U : ndarray, shape (q,)
        Right bounds of prediction intervals.

    Notes
    -----
    If no valid interval is found for a point (no t satisfies A(x, t) ≤ q̂),
    returns the full grid range [t_grid[0], t_grid[-1]] as a fallback.
    """
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    t_grid = np.asarray(t_grid, dtype=float).ravel()

    # Batch compute CDF for all query points at once
    # F_all shape: (q, M) where F_all[j, m] = F(t_m | x_j)
    F_all = model.predict_cdf(X_query, t_grid)  # shape (q, M)

    # Batch compute scores for all query points
    # scores_all shape: (q, M) where scores_all[j, m] = A(x_j, t_m)
    scores_all = score_from_cdf(F_all, score_type=score_type)  # shape (q, M)

    # Vectorized: first and last indices where A <= q_hat
    mask = scores_all <= q_hat
    idx_L = mask.argmax(axis=1)
    idx_U = mask.shape[1] - 1 - mask[:, ::-1].argmax(axis=1)
    # Degenerate: no True in row -> argmax returns 0, idx_U = M-1; L=t[0], U=t[-1] is correct
    L = t_grid[idx_L]
    U = t_grid[idx_U]

    return L, U

