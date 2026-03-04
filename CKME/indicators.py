"""
indicators.py

Smooth indicator functions g_t(y) used to approximate the step function
1{y <= t} when constructing conditional CDFs from the CKME embedding.

In the CKME-CDF setting, the conditional CDF is estimated via
    F(t | x) = < g_t , mu_{Y|X=x} >,
where g_t is a smooth surrogate for the discontinuous indicator 1{y <= t}.

This module implements several choices of g_t(y) and a small factory
to construct indicator objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


IndicatorType = Literal["logistic", "gaussian_cdf", "softplus", "step"]


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseIndicator(ABC):
    """
    Abstract base class for smooth indicator families g_t(y).

    The main role of an indicator object is to provide vectorized evaluations
    of g_t(y) over:
    - a vector of Y values for a single threshold t, and
    - a vector of thresholds t_grid for a fixed Y sample.
    """

    def __init__(self, h: float):
        if h <= 0:
            raise ValueError(f"Indicator bandwidth h must be positive, got {h}")
        self.h = float(h)

    @abstractmethod
    def g_vector(self, Y: np.ndarray, t: float) -> np.ndarray:
        """
        Evaluate g_t(y) on a vector of Y values for a single threshold t.

        Parameters
        ----------
        Y : ndarray, shape (n,)
            Observed Y-values from the training sample.

        t : float
            Threshold at which the smooth indicator is evaluated.

        Returns
        -------
        g : ndarray, shape (n,)
            Vector with entries g_t(Y[i]).
        """
        raise NotImplementedError

    def g_matrix(self, Y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """
        Evaluate g_t(y) on a vector of Y values and a grid of thresholds.

        This is a convenience method that stacks calls to g_vector for
        each t in t_grid.

        Parameters
        ----------
        Y : ndarray, shape (n,)
            Observed Y-values from the training sample.

        t_grid : ndarray, shape (M,)
            Grid of thresholds t_m.

        Returns
        -------
        G : ndarray, shape (n, M)
            Matrix with entries G[i, m] = g_{t_m}(Y[i]).
        """
        Y = np.asarray(Y).ravel()
        t_grid = np.asarray(t_grid).ravel()
        n = Y.shape[0]
        M = t_grid.shape[0]
        G = np.empty((n, M), dtype=float)
        for m, t in enumerate(t_grid):
            G[:, m] = self.g_vector(Y, t)
        return G


# ---------------------------------------------------------------------------
# Logistic smooth indicator
# ---------------------------------------------------------------------------

class LogisticIndicator(BaseIndicator):
    r"""
    Logistic smooth indicator:

        g_t(y) = 1 / (1 + exp( -(t - y) / h )).

    This function approximates 1{y <= t} as h -> 0, and is smooth for any
    fixed h > 0. It is numerically stable and often used as a default choice.
    """

    def g_vector(self, Y: np.ndarray, t: float) -> np.ndarray:
        Y = np.asarray(Y).ravel()
        z = (t - Y) / self.h
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    def g_matrix(self, Y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Vectorized: G[i,m] = g_{t_m}(Y[i])."""
        Y = np.asarray(Y).ravel()[:, None]
        t_grid = np.asarray(t_grid).ravel()[None, :]
        z = np.clip((t_grid - Y) / self.h, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Gaussian CDF smooth indicator
# ---------------------------------------------------------------------------

class GaussianCDFIndicator(BaseIndicator):
    r"""
    Gaussian CDF-based smooth indicator:

        g_t(y) = Phi( (t - y) / h ),

    where Phi is the standard normal CDF. This choice is infinitely smooth
    and pairs naturally with Gaussian kernels, though it requires evaluating
    the normal CDF.
    """

    def __init__(self, h: float):
        super().__init__(h)
        # Import here to avoid making scipy a hard dependency if not used.
        try:
            from scipy.stats import norm  # type: ignore
        except ImportError as e:
            raise ImportError(
                "GaussianCDFIndicator requires scipy. "
                "Please install scipy or use a different indicator type."
            ) from e
        self._norm = norm

    def g_vector(self, Y: np.ndarray, t: float) -> np.ndarray:
        Y = np.asarray(Y).ravel()
        z = (t - Y) / self.h
        return self._norm.cdf(z)

    def g_matrix(self, Y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Vectorized: G[i,m] = g_{t_m}(Y[i])."""
        Y = np.asarray(Y).ravel()[:, None]
        t_grid = np.asarray(t_grid).ravel()[None, :]
        z = (t_grid - Y) / self.h
        return self._norm.cdf(z)


# ---------------------------------------------------------------------------
# Softplus-based smooth indicator
# ---------------------------------------------------------------------------

class SoftplusIndicator(BaseIndicator):
    r"""
    Softplus-based smooth indicator:

        g_t(y) = log(1 + exp((t - y) / h)) / log(1 + exp(1 / h)).

    This is a smooth, monotone function that transitions from 0 to 1
    and can be seen as a smoothed "ramp" approximation. It is less
    directly interpretable as a CDF but can have useful numerical
    properties in some settings.
    """

    def g_vector(self, Y: np.ndarray, t: float) -> np.ndarray:
        Y = np.asarray(Y).ravel()
        z = (t - Y) / self.h
        z = np.clip(z, -50.0, 50.0)
        num = np.log1p(np.exp(z))
        den = np.log1p(np.exp(1.0 / self.h))
        return num / den

    def g_matrix(self, Y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Vectorized: G[i,m] = g_{t_m}(Y[i])."""
        Y = np.asarray(Y).ravel()[:, None]
        t_grid = np.asarray(t_grid).ravel()[None, :]
        z = np.clip((t_grid - Y) / self.h, -50.0, 50.0)
        num = np.log1p(np.exp(z))
        den = np.log1p(np.exp(1.0 / self.h))
        return num / den


# ---------------------------------------------------------------------------
# Standard step function (Heaviside) indicator
# ---------------------------------------------------------------------------

class StepIndicator(BaseIndicator):
    r"""
    Standard step function (Heaviside function):

        g_t(y) = 1{y <= t}

    This is the true indicator function, not a smooth approximation.
    The h parameter is not used for this indicator type (kept for interface consistency).
    """

    def __init__(self, h: float):
        # Override parent validation to allow any h value (not used for step function)
        # Store h but don't validate it
        self.h = float(h)

    def g_vector(self, Y: np.ndarray, t: float) -> np.ndarray:
        Y = np.asarray(Y).ravel()
        return (Y <= t).astype(float)

    def g_matrix(self, Y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
        """Vectorized: G[i,m] = 1{Y[i] <= t_m}."""
        Y = np.asarray(Y).ravel()[:, None]
        t_grid = np.asarray(t_grid).ravel()[None, :]
        return (Y <= t_grid).astype(float)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_indicator(indicator_type: IndicatorType, h: float) -> BaseIndicator:
    """
    Factory function to create a smooth indicator object.

    Parameters
    ----------
    indicator_type : {"logistic", "gaussian_cdf", "softplus", "step"}
        Name of the indicator family.
        - "logistic", "gaussian_cdf", "softplus": Smooth approximations
        - "step": Standard step function 1{y <= t} (h parameter is ignored)

    h : float
        Smoothing bandwidth h > 0 (for smooth indicators).
        For "step" indicator, h is accepted but not used.

    Returns
    -------
    indicator : BaseIndicator
        An instance of the requested indicator class.
    """
    if indicator_type == "logistic":
        return LogisticIndicator(h)
    elif indicator_type == "gaussian_cdf":
        return GaussianCDFIndicator(h)
    elif indicator_type == "softplus":
        return SoftplusIndicator(h)
    elif indicator_type == "step":
        return StepIndicator(h)
    else:
        raise ValueError(f"Unknown indicator_type: {indicator_type}")
