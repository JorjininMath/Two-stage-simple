"""
sim_nongauss_C1.py

Non-Gaussian DGP C1: exp2 true function + shifted Gaussian mixture noise.

  epsilon | x ~ (1-pi)*N(mu1(x), (a*s1*sigma_tar(x))^2)
               +    pi *N(mu2(x), (a*s2*sigma_tar(x))^2)

  sigma_tar(x) = 0.1 + 0.1*(x - pi)^2   (shared heteroscedastic target)
  a(pi) = 1 / sqrt((1-pi)*s1^2 + pi*s2^2 + pi*(1-pi)*delta^2)
  mu1(x) = -a(pi)*pi*delta*sigma_tar(x)        (inlier, left-shifted)
  mu2(x) =  a(pi)*(1-pi)*delta*sigma_tar(x)    (outlier, right-shifted)

  => E[epsilon|x] = 0,  Var[epsilon|x] = sigma_tar(x)^2  (exact, all x)

  Fixed shape params: s1=0.35, s2=2.5, delta=1.8
  pi: outlier probability (0.02 = light contamination, 0.10 = heavy)

Domain: x in [0, 2*pi]
"""
from __future__ import annotations
import numpy as np
from math import pi as _PI, sqrt
from .exp2 import exp2_true_function, EXP2_X_BOUNDS

_S1_DEFAULT    = 0.35
_S2_DEFAULT    = 1.0
_DELTA_DEFAULT = 4.0


def _sigma_tar(x: np.ndarray) -> np.ndarray:
    return 0.1 + 0.1 * (x - _PI) ** 2


def _mixture_a(pi: float, s1: float, s2: float, delta: float) -> float:
    """Scale so Var(epsilon|x) = sigma_tar(x)^2 for all x."""
    return 1.0 / sqrt((1 - pi) * s1 ** 2 + pi * s2 ** 2 + pi * (1 - pi) * delta ** 2)


def make_nongauss_C1_simulator(
    pi: float = 0.10,
    s1: float = _S1_DEFAULT,
    s2: float = _S2_DEFAULT,
    delta: float = _DELTA_DEFAULT,
):
    """Return a simulator with shifted Gaussian mixture noise.

    Parameters
    ----------
    pi    : outlier probability (0.02 = light, 0.10 = heavy contamination)
    s1    : relative inlier std
    s2    : relative outlier std
    delta : shift strength in units of sigma_tar(x)
    """
    a = _mixture_a(pi, s1, s2, delta)

    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x   = np.asarray(x, dtype=float)
        sig = _sigma_tar(x)
        y_true = exp2_true_function(x)
        is_outlier = rng.uniform(size=x.shape) < pi
        mu  = np.where(is_outlier,  a * (1 - pi) * delta * sig,
                                   -a * pi        * delta * sig)
        std = np.where(is_outlier, a * s2 * sig, a * s1 * sig)
        eps = rng.normal(loc=mu, scale=std, size=x.shape)
        return y_true + eps

    return simulator
