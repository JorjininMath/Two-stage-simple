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

from typing import Callable

import numpy as np

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
    # A small jitter could be added here if needed for numerical stability.
    L = np.linalg.cholesky(A)
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
    # Solve (L L^T) C = K_Xq for C
    # First solve L U = K_Xq
    U = np.linalg.solve(L, K_Xq)
    # Then solve L^T C = U
    C = np.linalg.solve(L.T, U)
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

