"""
sim_nongauss_C1.py

Non-Gaussian DGP C1: exp2 true function + shifted Gaussian mixture noise
(un-normalized: component stds and shifts proportional to sigma_tar(x) without
a variance-correction factor).

  epsilon | x ~ (1-pi) * N(mu1(x), (s1 * sigma_tar(x))^2)
               +   pi  * N(mu2(x), (s2 * sigma_tar(x))^2)

  mu1(x) = -pi   * delta * sigma_tar(x)     (inlier, left-shifted)
  mu2(x) =  (1-pi) * delta * sigma_tar(x)   (outlier, right-shifted)
  => E[epsilon|x] = 0  exactly.

  sigma_tar(x) = 0.1 + 0.1*(x - pi)^2
  Fixed shape: s1=0.35, s2=1.0, delta=4.0

  Var[epsilon|x] = [(1-pi)*s1^2 + pi*s2^2 + pi*(1-pi)*delta^2] * sigma_tar(x)^2

  pi=0.02 -> light contamination, pi=0.10 -> heavy contamination.

Domain: x in [0, 2*pi]

Note: the variance-normalized version (Var(Y|x) = sigma_tar(x)^2 exactly) is
archived at _archive/sim_nongauss_C1_normalized.py.
"""
from __future__ import annotations
import numpy as np
from math import pi as _PI
from .exp2 import exp2_true_function, EXP2_X_BOUNDS

_S1_DEFAULT    = 0.35
_S2_DEFAULT    = 1.0
_DELTA_DEFAULT = 4.0


def _sigma_tar(x: np.ndarray) -> np.ndarray:
    return 0.1 + 0.1 * (x - _PI) ** 2


def make_nongauss_C1_simulator(
    pi: float = 0.10,
    s1: float = _S1_DEFAULT,
    s2: float = _S2_DEFAULT,
    delta: float = _DELTA_DEFAULT,
):
    """Return a simulator with shifted Gaussian mixture noise (un-normalized)."""
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x   = np.asarray(x, dtype=float)
        sig = _sigma_tar(x)
        y_true = exp2_true_function(x)
        is_outlier = rng.uniform(size=x.shape) < pi
        mu  = np.where(is_outlier,  (1 - pi) * delta * sig,
                                   -pi       * delta * sig)
        std = np.where(is_outlier, s2 * sig, s1 * sig)
        eps = rng.normal(loc=mu, scale=std, size=x.shape)
        return y_true + eps

    return simulator


def nongauss_C1_noise_variance(
    x: np.ndarray,
    pi: float = 0.10,
    s1: float = _S1_DEFAULT,
    s2: float = _S2_DEFAULT,
    delta: float = _DELTA_DEFAULT,
) -> np.ndarray:
    """Var(Y|x) = [(1-pi)*s1^2 + pi*s2^2 + pi*(1-pi)*delta^2] * sigma_tar(x)^2."""
    var_factor = (1 - pi) * s1 ** 2 + pi * s2 ** 2 + pi * (1 - pi) * delta ** 2
    return var_factor * _sigma_tar(np.asarray(x, dtype=float)) ** 2
