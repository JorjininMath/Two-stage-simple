"""
cdf.py

CDF computation utilities for CKME.

This module provides functions for computing conditional CDFs F(t | x) from
CKME coefficients and smooth indicator functions.

The computation involves:
1. Computing coefficient vectors c(x) (using coefficients.py)
2. Evaluating smooth indicators g_t(Y) on training outputs
3. Computing CDF: F(t | x) = E[g_t(Y) | X = x] = g_t(Y)^T c(x)

These functions are stateless and can be used independently by:
- ckme.py (for the model class)
- tuning.py (for parameter tuning without creating model instances)
- Other wrappers that need CDF computation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from .coefficients import build_cholesky_factor, compute_ckme_coeffs
from .indicators import BaseIndicator

if TYPE_CHECKING:
    from .parameters import Params

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# CDF computation from coefficients
# ---------------------------------------------------------------------------

def compute_cdf_from_coeffs(
    C: ArrayLike,
    Y: ArrayLike,
    indicator: BaseIndicator,
    t_grid: ArrayLike,
    clip: bool = True,
) -> ArrayLike:
    """
    Compute conditional CDF F(t | x) from coefficient matrix and indicator.

    This function computes:
        F(t_m | x_j) = E[g_{t_m}(Y) | X = x_j] = g_{t_m}(Y)^T c(x_j)

    Parameters
    ----------
    C : ndarray, shape (n, q)
        CKME coefficient matrix for q query points.

    Y : ndarray, shape (n,)
        Training output values.

    indicator : BaseIndicator
        Smooth indicator object for computing g_t(Y).

    t_grid : ndarray, shape (M,)
        Threshold grid at which F(t_m | x) is evaluated.

    clip : bool, default=True
        If True, clip the resulting CDF values to [0, 1]. This is sometimes
        useful numerically because the RKHS representation can produce
        values slightly outside [0, 1].

    Returns
    -------
    F : ndarray, shape (q, M)
        Matrix of CDF values with entries F[j, m] = F(t_m | x_j).
    """
    # Build G(Y, t_m) matrix: shape (n, M), columns = g_{t_m}(Y)
    G = indicator.g_matrix(Y, t_grid)

    # Compute CDF: F(t_m | x_j) = g_{t_m}(Y)^T c(x_j) = (C^T @ G)[j, m]
    # Shape: (q, M) = (n, q)^T @ (n, M)
    F = C.T @ G  # shape (q, M)

    if clip:
        np.clip(F, 0.0, 1.0, out=F)
    return F


# ---------------------------------------------------------------------------
# Complete CDF computation pipeline (for parameter tuning)
# ---------------------------------------------------------------------------

def compute_ckme_cdf(
    X_train: ArrayLike,
    Y_train: ArrayLike,
    params: "Params",  # Forward reference
    X_query: ArrayLike,
    t_grid: ArrayLike,
    indicator: Optional[BaseIndicator] = None,
    indicator_type: str = "logistic",
    clip: bool = True,
) -> ArrayLike:
    """
    Complete CDF computation pipeline.

    This function performs the full computation:
    1. Build kernel matrix and Cholesky factor
    2. Compute coefficients for query points
    3. Compute CDF using smooth indicators

    This is designed for parameter tuning where you want to evaluate
    models without creating CKMEModel instances.

    Parameters
    ----------
    X_train : ndarray, shape (n, d)
        Training input points.

    Y_train : ndarray, shape (n,)
        Training output values.

    params : Params
        CKME hyperparameters (ell_x, lam, h).

    X_query : ndarray, shape (q, d) or (d,)
        Query input points.

    t_grid : ndarray, shape (M,)
        Threshold grid at which F(t_m | x) is evaluated.

    indicator : BaseIndicator, optional
        Smooth indicator object. If None, creates one using params and indicator_type.

    indicator_type : str, default="logistic"
        Type of indicator to use if indicator is None.
        Options: "logistic", "gaussian_cdf", "softplus", "step".

    clip : bool, default=True
        If True, clip the resulting CDF values to [0, 1].

    Returns
    -------
    F : ndarray, shape (q, M)
        Matrix of CDF values with entries F[j, m] = F(t_m | x_j).

    Notes
    -----
    This function is stateless and can be called multiple times with
    different parameters for parameter tuning.
    """
    from .parameters import Params
    from .kernels import make_x_rbf_kernel
    from .indicators import make_indicator

    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    Y_train = np.asarray(Y_train, dtype=float).ravel()
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    t_grid = np.asarray(t_grid, dtype=float).ravel()

    n = X_train.shape[0]

    # Build X-kernel
    kx = make_x_rbf_kernel(params.ell_x)

    # Precompute training Gram matrix K_X
    K_X = kx(X_train, X_train)  # shape (n, n)

    # Build Cholesky factor
    L = build_cholesky_factor(K_X, n, params.lam)

    # Compute coefficients for query points
    C = compute_ckme_coeffs(L, kx, X_train, X_query)  # shape (n, q)

    # Create indicator if not provided
    if indicator is None:
        indicator = make_indicator(indicator_type, params.h)

    # Compute CDF using indicator method
    F = compute_cdf_from_coeffs(C, Y_train, indicator, t_grid, clip=clip)

    return F
