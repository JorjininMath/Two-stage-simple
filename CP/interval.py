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

ArrayLike = np.ndarray


def projected_quantile_interval(
    F_all: ArrayLike,
    t_grid: ArrayLike,
    q_hat: float,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Shared interval-extraction helper (canonical convention, 2026-07-15).

    Monotone-projects each CDF row (running max + clip to [0, 1]) and returns
    generalized-inverse endpoints at levels 1/2 - q_hat and 1/2 + q_hat:

        L = inf{t : Ftilde(t) >= 1/2 - q_hat},
        U = inf{t : Ftilde(t) >= 1/2 + q_hat}.

    Rationale: raw KRR-weighted CDFs can be non-monotone (negative weights),
    which makes level-set hulls include spurious dip regions and makes
    binary-search inversion silently incorrect. Projection matches the
    estimator Ftilde used in the theory. See
    notes/planning/cdf_legality_policy.md.

    Parameters
    ----------
    F_all : array-like, shape (q, M) or (M,)
        Raw CDF values per query row over t_grid.
    t_grid : array-like, shape (M,)
    q_hat : float

    Returns
    -------
    L, U : ndarray, shape (q,)
        Fallback t_grid[-1] when a level is never reached.
    """
    F_all = np.atleast_2d(np.asarray(F_all, dtype=float))
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    T = t_grid.shape[0]

    F_proj = np.clip(np.maximum.accumulate(F_all, axis=1), 0.0, 1.0)
    tau_lo = float(np.clip(0.5 - q_hat, 0.0, 1.0))
    tau_hi = float(np.clip(0.5 + q_hat, 0.0, 1.0))

    mask_lo = F_proj >= tau_lo
    idx_L = np.where(mask_lo.any(axis=1), mask_lo.argmax(axis=1), T - 1)
    mask_hi = F_proj >= tau_hi
    idx_U = np.where(mask_hi.any(axis=1), mask_hi.argmax(axis=1), T - 1)

    return t_grid[idx_L], t_grid[idx_U]


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
    if score_type != "abs_median":
        raise ValueError(
            "predict_interval currently supports score_type='abs_median' only; "
            "the projected generalized-inverse extraction is defined for the "
            "DCP score |F - 1/2|."
        )

    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    t_grid = np.asarray(t_grid, dtype=float).ravel()

    # Batch compute CDF for all query points at once
    # F_all shape: (q, M) where F_all[j, m] = F(t_m | x_j)
    F_all = model.predict_cdf(X_query, t_grid)  # shape (q, M)

    # Canonical extraction (2026-07-15): monotone projection + generalized
    # inverse, replacing the raw level-set hull. On monotone rows the change
    # is at most one grid cell; on non-monotone rows it removes spurious dip
    # regions. See notes/planning/cdf_legality_policy.md.
    return projected_quantile_interval(F_all, t_grid, q_hat)

