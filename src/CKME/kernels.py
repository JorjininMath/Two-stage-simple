"""
kernels.py

Kernel functions used in the CKME core.

In the current CKME-CDF setup, we only need an X-kernel k_X(x, x'),
which is typically chosen as a Gaussian / RBF kernel. This module
implements a simple RBF kernel and a small factory to create
callable kernel functions with fixed bandwidth.

If, in the future, you decide to re-introduce Y-side kernels
(e.g., for MMD-based losses), they can also be added here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

ArrayLike = np.ndarray
KernelFunc = Callable[[ArrayLike, ArrayLike], ArrayLike]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _squared_euclidean_distances(X1: ArrayLike, X2: ArrayLike) -> ArrayLike:
    """
    Compute the matrix of squared Euclidean distances between rows of X1 and X2.

    Parameters
    ----------
    X1 : array-like, shape (n1, d)
        First set of points.

    X2 : array-like, shape (n2, d)
        Second set of points.

    Returns
    -------
    D2 : ndarray, shape (n1, n2)
        Matrix of squared distances, where
            D2[i, j] = || X1[i, :] - X2[j, :] ||^2.
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)

    # Compute squared norms
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    X1_sq = np.sum(X1 ** 2, axis=1).reshape(n1, 1)  # (n1, 1)
    X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, n2)  # (1, n2)

    # Use (x - y)^2 = ||x||^2 + ||y||^2 - 2 x^T y
    D2 = X1_sq + X2_sq - 2.0 * X1 @ X2.T
    # Numerical safety: eliminate tiny negative values due to floating-point
    np.maximum(D2, 0.0, out=D2)
    return D2


# ---------------------------------------------------------------------------
# RBF (Gaussian) kernel for X
# ---------------------------------------------------------------------------

def rbf_kernel_x(X1: ArrayLike, X2: ArrayLike, ell_x: float) -> ArrayLike:
    """
    Gaussian / RBF kernel on the input space X.

    This kernel is defined as
        k_X(x, x') = exp( - ||x - x'||^2 / (2 * ell_x^2) ).

    Parameters
    ----------
    X1 : array-like, shape (n1, d)
        First set of input points.

    X2 : array-like, shape (n2, d)
        Second set of input points.

    ell_x : float
        Bandwidth (lengthscale) parameter for the RBF kernel. Larger values
        correspond to smoother functions in X, smaller values to more local
        behavior.

    Returns
    -------
    K : ndarray, shape (n1, n2)
        Kernel matrix with entries k_X(X1[i], X2[j]).
    """
    if ell_x <= 0:
        raise ValueError(f"ell_x must be positive, got {ell_x}")

    D2 = _squared_euclidean_distances(X1, X2)
    K = np.exp(-D2 / (2.0 * ell_x ** 2))
    return K


# ---------------------------------------------------------------------------
# RBF (Gaussian) kernel for Y
# ---------------------------------------------------------------------------

def rbf_kernel_y(Y1: ArrayLike, Y2: ArrayLike, ell_y: float) -> ArrayLike:
    """
    Gaussian / RBF kernel on the output space Y.

    This kernel is defined as
        k_Y(y, y') = exp( - (y - y')^2 / (2 * ell_y^2) ).

    Parameters
    ----------
    Y1 : array-like, shape (n1,)
        First set of output values.

    Y2 : array-like, shape (n2,)
        Second set of output values.

    ell_y : float
        Bandwidth (lengthscale) parameter for the RBF kernel. Larger values
        correspond to smoother functions in Y, smaller values to more local
        behavior.

    Returns
    -------
    K : ndarray, shape (n1, n2)
        Kernel matrix with entries k_Y(Y1[i], Y2[j]).
    """
    if ell_y <= 0:
        raise ValueError(f"ell_y must be positive, got {ell_y}")

    Y1 = np.asarray(Y1, dtype=float).ravel()
    Y2 = np.asarray(Y2, dtype=float).ravel()

    # Compute squared distances: (y1 - y2)^2
    Y1_2d = Y1.reshape(-1, 1)  # (n1, 1)
    Y2_2d = Y2.reshape(1, -1)  # (1, n2)
    D2 = (Y1_2d - Y2_2d) ** 2  # (n1, n2)

    K = np.exp(-D2 / (2.0 * ell_y ** 2))
    return K


# ---------------------------------------------------------------------------
# Small factory to create a fixed-bandwidth X-kernel
# ---------------------------------------------------------------------------

def make_x_rbf_kernel(ell_x: float) -> KernelFunc:
    """
    Create an RBF X-kernel function with fixed bandwidth ell_x.

    This is a small convenience factory. Instead of passing ell_x around
    everywhere, you can construct a callable kernel and store it in the
    CKME model:

        kx = make_x_rbf_kernel(ell_x)
        K_X = kx(X, X)  # training Gram matrix
        K_Xq = kx(X, X_query)  # cross-kernel with test points

    Parameters
    ----------
    ell_x : float
        Bandwidth for the RBF X-kernel.

    Returns
    -------
    kx : callable
        A function kx(X1, X2) that computes the RBF kernel matrix with
        the given ell_x.
    """
    def kx(X1: ArrayLike, X2: ArrayLike) -> ArrayLike:
        return rbf_kernel_x(X1, X2, ell_x=ell_x)

    return kx


# ---------------------------------------------------------------------------
# Small factory to create a fixed-bandwidth Y-kernel
# ---------------------------------------------------------------------------

def make_y_rbf_kernel(ell_y: float) -> Callable[[ArrayLike, ArrayLike], ArrayLike]:
    """
    Create an RBF Y-kernel function with fixed bandwidth ell_y.

    This is a convenience factory for the Y-kernel used in the alternative
    CDF estimation method (Y-kernel + logistic link).

    Parameters
    ----------
    ell_y : float
        Bandwidth for the RBF Y-kernel.

    Returns
    -------
    ky : callable
        A function ky(Y1, Y2) that computes the RBF kernel matrix with
        the given ell_y.
    """
    def ky(Y1: ArrayLike, Y2: ArrayLike) -> ArrayLike:
        return rbf_kernel_y(Y1, Y2, ell_y=ell_y)

    return ky
