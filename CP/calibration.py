"""
calibration.py

Calibration layer for Conformal Prediction.

This module provides functions for calibrating CP models using a calibration set.
Given calibration data (X_cal, Y_cal), it computes nonconformity scores and
determines the conformal quantile q̂.
"""

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CKME.ckme import CKMEModel

from .scores import score_from_cdf

ArrayLike = np.ndarray


def calibrate(
    model: "CKMEModel",
    X_cal: ArrayLike,
    Y_cal: ArrayLike,
    alpha: float,
    score_type: str = "abs_median",
    verbose: bool = False,
) -> tuple[float, ArrayLike]:
    """
    Calibrate the CP model using a calibration set.

    This function:
    1. Computes nonconformity scores for all (x, y) pairs in the calibration set
    2. Computes the conformal quantile q̂ = (1-α) quantile of calibration scores
    3. Returns q̂ and calibration scores

    Parameters
    ----------
    model : CKMEModel
        Trained CKME model. Must be fitted before use.

    X_cal : array-like, shape (n_cal, d)
        Calibration input points.

    Y_cal : array-like, shape (n_cal,)
        Calibration output values.

    alpha : float
        Significance level. The prediction intervals will have coverage
        approximately (1 - alpha).

    score_type : str, default="abs_median"
        Type of nonconformity score: "abs_median" = |F(y | x) - 0.5|

    verbose : bool, default=False
        If True, print calibration information.

    Returns
    -------
    q_hat : float
        Conformal quantile threshold.

    calibration_scores : ndarray, shape (n_cal,)
        Nonconformity scores for calibration data.

    Notes
    -----
    The conformal quantile is computed as:
        q̂ = quantile(scores, (1-α) * (n_cal + 1) / n_cal)

    This ensures finite-sample coverage guarantees.
    """
    X_cal = np.atleast_2d(np.asarray(X_cal, dtype=float))
    Y_cal = np.asarray(Y_cal, dtype=float).ravel()

    if X_cal.shape[0] != Y_cal.shape[0]:
        raise ValueError(
            f"X_cal and Y_cal must have the same number of samples, "
            f"got {X_cal.shape[0]} and {Y_cal.shape[0]}"
        )

    # Compute CDF values F(y_i | x_i) for calibration set (CKME model only)
    n_cal = len(X_cal)
    if not (hasattr(model, 'L') and hasattr(model, 'kx') and hasattr(model, 'X') and hasattr(model, 'Y') and hasattr(model, 'indicator')):
        raise ValueError("calibrate only supports CKMEModel")

    from CKME.coefficients import compute_ckme_coeffs
    C_cal = compute_ckme_coeffs(model.L, model.kx, model.X, X_cal)  # (n_sites, n_cal)
    if getattr(model, 'r', 1) > 1:
        # Distinct-sites mode: average g_t(Y) over replicates per site
        # OLD: G_cal = model.indicator.g_matrix(model.Y, Y_cal)  # (n_0*r, n_cal)
        Y_flat = model.Y.ravel()                                  # (n_sites * r,)
        G_all  = model.indicator.g_matrix(Y_flat, Y_cal)          # (n_sites*r, n_cal)
        G_cal  = G_all.reshape(model.n, model.r, -1).mean(axis=1) # (n_sites, n_cal)
    else:
        G_cal = model.indicator.g_matrix(model.Y, Y_cal)          # (n, n_cal)
    F_cal = np.sum(C_cal * G_cal, axis=0)  # shape (n_cal,)
    np.clip(F_cal, 0.0, 1.0, out=F_cal)

    # Compute nonconformity scores from CDF values
    calibration_scores = score_from_cdf(F_cal, score_type=score_type)

    # Compute conformal quantile: k = ceil((1-alpha)*(1+n)), q̂ = sort(scores)[k-1]
    k = int(np.ceil((1 - alpha) * (1 + n_cal)))
    k = min(k, n_cal)
    sorted_scores = np.sort(calibration_scores)
    q_hat = float(sorted_scores[k - 1])

    if verbose:
        print(f"Calibration completed:")
        print(f"  Calibration set size: {n_cal}")
        print(f"  Alpha: {alpha}")
        print(f"  Conformal quantile q̂: {q_hat:.6f}")
        print(f"  Score range: [{calibration_scores.min():.6f}, "
              f"{calibration_scores.max():.6f}]")

    return q_hat, calibration_scores

