"""
sim_nongauss_A1.py

Non-Gaussian DGP A1: exp2 true function + heteroscedastic Student-t noise
(un-normalized: scale = sigma_tar(x) directly, variance grows with heavy tails).

  Y = f(x) + sigma_tar(x) * T_nu,   T_nu ~ t_nu
  sigma_tar(x) = 0.01 + 0.2*(x - pi)^2

  Var(Y|x) = sigma_tar(x)^2 * nu/(nu-2)     (finite only for nu > 2)
    nu=3  -> Var = 3  * sigma_tar^2
    nu=10 -> Var = 1.25 * sigma_tar^2

  Lower nu -> heavier tails. nu=10 light, nu=3 heavy.

Domain: x in [0, 2*pi]

Note: the variance-normalized version (Var(Y|x) = sigma_tar(x)^2 exactly) is
archived at _archive/sim_nongauss_A1_normalized.py.
"""
from __future__ import annotations
import numpy as np
from math import pi as _PI
from .exp2 import exp2_true_function, EXP2_X_BOUNDS


def _sigma_tar(x: np.ndarray) -> np.ndarray:
    return 0.01 + 0.2 * (x - _PI) ** 2


def make_nongauss_A1_simulator(nu: float = 3.0):
    """Return a simulator with Student-t noise of nu degrees of freedom.

    Scale = sigma_tar(x) directly (no variance normalization).
    """
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x, dtype=float)
        y_true = exp2_true_function(x)
        noise = rng.standard_t(df=nu, size=x.shape) * _sigma_tar(x)
        return y_true + noise
    return simulator


def nongauss_A1_noise_variance(x: np.ndarray, nu: float = 3.0) -> np.ndarray:
    """Var(Y|x) = sigma_tar(x)^2 * nu/(nu-2)  (finite only for nu > 2)."""
    x = np.asarray(x, dtype=float)
    if nu <= 2:
        return np.full_like(x, np.inf)
    return _sigma_tar(x) ** 2 * (nu / (nu - 2))
