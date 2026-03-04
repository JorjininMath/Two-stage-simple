"""
sim_nongauss_B2.py

Non-Gaussian DGP B2: exp2 true function + centered Gamma noise.
  G ~ Gamma(k, theta(x)),  theta(x) = 0.1 + 0.4 * x
  epsilon = G - k * theta(x)   (centered so E[epsilon|x] = 0)
  Y = f(x) + epsilon
  Var(epsilon|x) = k * theta^2(x)

  k: shape parameter (tunable; lower k -> more skewed, skewness = 2/sqrt(k))

Domain: x in [0, 2*pi]
"""
from __future__ import annotations
import numpy as np
from .exp2 import exp2_true_function, EXP2_X_BOUNDS


def _theta(x: np.ndarray) -> np.ndarray:
    # theta(x) = 0.1 + 0.4x  (scale, always positive)
    return 0.1 + 0.4 * x


def make_nongauss_B2_simulator(k: float = 2.0):
    """Return a simulator with centered Gamma noise of shape k."""
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x, dtype=float)
        y_true = exp2_true_function(x)
        theta = _theta(x)
        g = rng.gamma(shape=k, scale=theta, size=x.shape)
        eps = g - k * theta  # center: E[G] = k*theta
        return y_true + eps
    return simulator


def nongauss_B2_noise_variance(x: np.ndarray, k: float = 2.0) -> np.ndarray:
    """v(x) = k * theta^2(x)."""
    return k * _theta(np.asarray(x, dtype=float)) ** 2
