"""
sim_nongauss_A1.py

Non-Gaussian DGP A1: exp2 true function + heteroscedastic Student-t noise.
  Y = f(x) + sigma(x) * T_nu,   T_nu ~ t_nu
  sigma(x) = 0.05 + 0.5 * x     (increasing std)
  nu: degrees of freedom (tunable; lower -> heavier tails)

Domain: x in [0, 2*pi]
"""
from __future__ import annotations
import numpy as np
from scipy.stats import t as student_t
from .exp2 import exp2_true_function, EXP2_X_BOUNDS


def _sigma(x: np.ndarray) -> np.ndarray:
    # σ(x) = 0.05 + 0.5x  (increasing, always positive)
    return 0.05 + 0.5 * x


def make_nongauss_A1_simulator(nu: float = 3.0):
    """Return a simulator with Student-t noise of nu degrees of freedom."""
    def simulator(x, random_state=None):
        x = np.asarray(x, dtype=float)
        y_true = exp2_true_function(x)
        sigma = _sigma(x)
        noise = student_t.rvs(df=nu, loc=0.0, scale=sigma,
                              size=y_true.shape, random_state=random_state)
        return y_true + noise
    return simulator


def nongauss_A1_noise_variance(x: np.ndarray, nu: float = 3.0) -> np.ndarray:
    """v(x) = sigma^2(x) * nu / (nu - 2),  valid for nu > 2."""
    return _sigma(np.asarray(x, dtype=float)) ** 2 * nu / (nu - 2)
