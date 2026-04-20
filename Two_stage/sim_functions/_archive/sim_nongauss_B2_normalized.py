"""
sim_nongauss_B2.py

Non-Gaussian DGP B2: exp2 true function + centered Gamma noise.
  G ~ Gamma(k, theta(x)),  theta(x) = sigma_tar(x) / sqrt(k)
  epsilon = G - k * theta(x)   (centered so E[epsilon|x] = 0)
  Y = f(x) + epsilon
  Var(epsilon|x) = k * theta^2(x) = sigma_tar(x)^2  (by construction)
  sigma_tar(x) = 0.1 + 0.1*(x - pi)^2

  k: shape parameter (lower k -> more skewed; skewness = 2/sqrt(k))
  k=9: mild skew, k=2: strong skew

Domain: x in [0, 2*pi]
"""
from __future__ import annotations
import numpy as np
from math import pi as _PI, sqrt
from .exp2 import exp2_true_function, EXP2_X_BOUNDS


def _sigma_tar(x: np.ndarray) -> np.ndarray:
    return 0.1 + 0.1 * (x - _PI) ** 2


def make_nongauss_B2_simulator(k: float = 2.0):
    """Return a simulator with centered Gamma noise of shape k.

    theta(x) = sigma_tar(x) / sqrt(k)  =>  Var(Y|x) = sigma_tar(x)^2.
    """
    sqrt_k = sqrt(k)

    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x, dtype=float)
        y_true = exp2_true_function(x)
        theta = _sigma_tar(x) / sqrt_k
        eps = rng.gamma(shape=k, scale=theta, size=x.shape) - k * theta
        return y_true + eps
    return simulator


def nongauss_B2_noise_variance(x: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Var(Y|x) = sigma_tar(x)^2  (by construction)."""
    return _sigma_tar(np.asarray(x, dtype=float)) ** 2
