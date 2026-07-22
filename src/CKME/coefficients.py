"""
coefficients.py

Coefficient computation utilities for CKME.

This module provides functions for computing CKME coefficient vectors c(x),
which are the core of the conditional kernel mean embedding representation.

The computation involves:
1. Building the Cholesky factor of the regularized kernel matrix
2. Solving the linear system to obtain coefficients for query points

These functions are stateless and can be used independently by:
- ckme.py (for the model class)
- cdf.py (for CDF computation)
- Other wrappers that need coefficient computation
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.linalg import cholesky as scipy_cholesky, solve_triangular

from .kernels import KernelFunc

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Linear algebra utilities
# ---------------------------------------------------------------------------

def build_cholesky_factor(K_X: ArrayLike, n: int, lam: float) -> ArrayLike:
    """
    Build Cholesky factor L of (K_X + n * lam * I).

    This is used to solve the regularized linear system:
        (K_X + n * lam * I) c(x) = k_X(X, x)

    Parameters
    ----------
    K_X : ndarray, shape (n, n)
        Training Gram matrix K_X[i, j] = k_X(X[i], X[j]).

    n : int
        Number of training samples.

    lam : float
        Ridge regularization parameter. Must be positive.

    Returns
    -------
    L : ndarray, shape (n, n)
        Lower triangular Cholesky factor such that L @ L.T = K_X + n * lam * I.

    Raises
    ------
    ValueError
        If lam is not positive.
    """
    lam = float(lam)
    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}")

    # Memory-efficient: avoid creating full identity matrix
    # Instead, copy K_X and add regularization only to diagonal
    A = K_X.copy()
    A.flat[::n+1] += n * lam  # Add n*lam to diagonal elements only
    # In-place Cholesky via LAPACK; reuses A's buffer for L.
    L = scipy_cholesky(A, lower=True, overwrite_a=True, check_finite=False)
    return L


def build_cholesky_from_X(
    X: ArrayLike,
    ell_x: float,
    n_lam: float,
    dtype: Optional[type] = None,
) -> ArrayLike:
    """
    Build the lower Cholesky factor of (K_X + n*lam*I) directly from X.

    Equivalent to ``build_cholesky_factor(rbf_kernel_x(X, X, ell_x), n, lam)``
    but allocates only ONE (n, n) buffer instead of three (D2, K_X, copy),
    and runs Cholesky in place. For n=2^14 this drops peak memory from
    ~6-8x to ~1x of n^2.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Training inputs.
    ell_x : float
        RBF lengthscale.
    n_lam : float
        Pre-multiplied ridge n*lam to add to the diagonal.
    dtype : numpy dtype, optional
        Working precision. Defaults to X's dtype. Pass np.float32 to halve
        peak memory for large n at the cost of mild numerical precision.

    Returns
    -------
    L : ndarray, shape (n, n), lower triangular
        Cholesky factor with L @ L.T = K_X + n_lam * I.
    """
    if ell_x <= 0:
        raise ValueError(f"ell_x must be positive, got {ell_x}")
    n_lam = float(n_lam)
    if n_lam <= 0:
        raise ValueError(f"n_lam must be positive, got {n_lam}")

    X = np.ascontiguousarray(X, dtype=dtype) if dtype is not None else np.ascontiguousarray(X)
    n = X.shape[0]

    # Single (n, n) allocation: start from -2 X X^T, then add row/col norms.
    A = X @ X.T                                  # (n, n)
    sq = np.einsum("ij,ij->i", X, X)             # (n,)
    A *= -2.0
    A += sq[:, None]
    A += sq[None, :]
    np.maximum(A, 0.0, out=A)
    A *= -0.5 / (ell_x * ell_x)
    np.exp(A, out=A)
    A.flat[::n + 1] += n_lam

    # In-place Cholesky: reuses A's buffer, no extra (n, n) allocation.
    L = scipy_cholesky(A, lower=True, overwrite_a=True, check_finite=False)
    return L


def solve_ckme_system(L: ArrayLike, K_Xq: ArrayLike) -> ArrayLike:
    """
    Solve the CKME linear system (L L^T) C = K_Xq for C.

    Given the Cholesky factor L where L @ L.T = K_X + n * lam * I,
    this solves for the coefficient matrix C.

    Parameters
    ----------
    L : ndarray, shape (n, n)
        Lower triangular Cholesky factor.

    K_Xq : ndarray, shape (n, q)
        Cross-kernel matrix between training points and query points.
        K_Xq[i, j] = k_X(X[i], X_query[j]).

    Returns
    -------
    C : ndarray, shape (n, q)
        Coefficient matrix whose j-th column is c(x_j) for the j-th query.
        These coefficients are NOT normalized and can have negative values.
    """
    # Solve (L L^T) C = K_Xq via two triangular solves.
    # solve_triangular avoids the LU re-factorization that np.linalg.solve does.
    U = solve_triangular(L, K_Xq, lower=True, check_finite=False)
    C = solve_triangular(L.T, U, lower=False, check_finite=False)
    return C


# ---------------------------------------------------------------------------
# Coefficient computation
# ---------------------------------------------------------------------------

def compute_ckme_coeffs(
    L: ArrayLike,
    kx: KernelFunc,
    X_train: ArrayLike,
    X_query: ArrayLike,
) -> ArrayLike:
    """
    Compute CKME coefficient vectors for one or more query inputs.

    This is a convenience function that combines kernel evaluation and
    linear system solving.

    Parameters
    ----------
    L : ndarray, shape (n, n)
        Lower triangular Cholesky factor of (K_X + n * lam * I).

    kx : callable
        X-kernel function kx(X1, X2) that computes kernel matrices.

    X_train : ndarray, shape (n, d)
        Training input points.

    X_query : ndarray, shape (q, d) or (d,)
        Query input points. A single point can be passed as a 1D array.

    Returns
    -------
    C : ndarray, shape (n, q)
        Coefficient matrix whose j-th column is c(x_j) for the j-th query.
        These coefficients are NOT normalized and can have negative values.
    """
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    # Cross-kernel between training and query points: shape (n, q)
    K_Xq = kx(X_train, X_query)
    # Solve the linear system
    C = solve_ckme_system(L, K_Xq)
    return C

