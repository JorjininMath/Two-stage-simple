"""
sim_nongauss_A1.py

Non-Gaussian DGP A1: exp2 true function + heteroscedastic Student-t noise.
  Y = f(x) + s_nu(x) * T_nu,   T_nu ~ t_nu
  s_nu(x) = sigma_tar(x) * sqrt((nu-2)/nu)   =>  Var(Y|x) = sigma_tar(x)^2
  sigma_tar(x) = 0.1 + 0.1*(x - pi)^2

  nu: degrees of freedom (lower -> heavier tails; nu=10 light, nu=3 heavy)

Domain: x in [0, 2*pi]
"""
from __future__ import annotations
import numpy as np
from math import pi as _PI, sqrt
from .exp2 import exp2_true_function, EXP2_X_BOUNDS


def _sigma_tar(x: np.ndarray) -> np.ndarray:
    return 0.1 + 0.1 * (x - _PI) ** 2


def make_nongauss_A1_simulator(nu: float = 3.0):
    """Return a simulator with Student-t noise of nu degrees of freedom.

    Scale s_nu(x) = sigma_tar(x) * sqrt((nu-2)/nu) ensures Var(Y|x) = sigma_tar(x)^2.
    """
    scale_factor = sqrt((nu - 2) / nu)

    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x, dtype=float)
        y_true = exp2_true_function(x)
        scale = _sigma_tar(x) * scale_factor
        noise = rng.standard_t(df=nu, size=x.shape) * scale
        return y_true + noise
    return simulator


def nongauss_A1_noise_variance(x: np.ndarray, nu: float = 3.0) -> np.ndarray:
    """Var(Y|x) = sigma_tar(x)^2  (by construction)."""
    return _sigma_tar(np.asarray(x, dtype=float)) ** 2
