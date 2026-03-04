"""
sim_nongauss_C1.py

Non-Gaussian DGP C1: exp2 true function + Gaussian mixture noise (constant outlier rate).
  epsilon ~ (1-pi)*N(0, s1^2) + pi*N(0, s2^2)
  Y = f(x) + epsilon
  Var(epsilon|x) = (1-pi)*s1^2 + pi*s2^2   (constant in x)

  s1 = 0.05  (inlier std, tight component)
  s2 = 1.0   (outlier std, heavy component)
  pi: outlier probability (tunable; controls contamination rate)

Domain: x in [0, 2*pi]
"""
from __future__ import annotations
import numpy as np
from .exp2 import exp2_true_function, EXP2_X_BOUNDS


def make_nongauss_C1_simulator(pi: float = 0.05, s1: float = 0.05, s2: float = 1.0):
    """Return a simulator with Gaussian mixture noise.

    Parameters
    ----------
    pi : outlier probability, in (0, 1)
    s1 : std of the inlier (tight) component
    s2 : std of the outlier component; s2 >> s1 for visible contamination
    """
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x, dtype=float)
        y_true = exp2_true_function(x)
        n = x.shape[0]
        is_outlier = rng.uniform(size=n) < pi          # True with prob pi
        std = np.where(is_outlier, s2, s1)
        eps = rng.normal(loc=0.0, scale=std, size=n)
        return y_true + eps
    return simulator


def nongauss_C1_noise_variance(pi: float = 0.05, s1: float = 0.05, s2: float = 1.0) -> float:
    """v = (1-pi)*s1^2 + pi*s2^2  (constant, independent of x)."""
    return (1 - pi) * s1 ** 2 + pi * s2 ** 2
