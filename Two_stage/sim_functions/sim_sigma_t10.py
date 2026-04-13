"""
sim_sigma_t10.py

Simulator: exp3_alloc
  True function : f(x) = exp(x/10) * sin(x)   (exp2 true function)
  Noise scale   : sigma(x) = 0.01 + 0.2*(x - pi)^2  (exp2 noise std)
  Noise         : Y = f(x) + sigma(x) * T_3,   T_3 ~ t(nu=3), zero-mean
  Domain        : x in [0, 2*pi]
"""
from __future__ import annotations
import numpy as np
from math import pi as _PI
from .exp2 import exp2_true_function, EXP2_X_BOUNDS

EXP3_ALLOC_X_BOUNDS = EXP2_X_BOUNDS

_NU = 3.0


def _sigma(x: np.ndarray) -> np.ndarray:
    """exp2 noise std: sigma(x) = 0.01 + 0.2*(x - pi)^2."""
    return 0.01 + 0.2 * (x - _PI) ** 2


def exp3_alloc_simulator(x: np.ndarray, random_state=None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    noise = rng.standard_t(df=_NU, size=x.shape) * _sigma(x)
    return exp2_true_function(x) + noise


def exp3_alloc_true_function(x: np.ndarray) -> np.ndarray:
    return exp2_true_function(np.asarray(x, dtype=float))
