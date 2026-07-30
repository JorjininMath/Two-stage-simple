"""
sim_branin_gauss.py

DGP branin_gauss: Branin-Hoo true function + heteroscedastic Gaussian noise.
  Y = f(x) + epsilon,
  epsilon ~ N(0, sigma(x)^2)
  sigma(x) = 0.4 * (4 * x1_scaled + 1)     [linear in x1; range 0.4 to 2.0]
  x1_scaled = (x1 - (-5)) / (10 - (-5)),   x1 = x[:, 0]

Domain: x1 in [-5, 10],  x2 in [0, 15]  (EXP3_X_BOUNDS)
"""
from __future__ import annotations

import numpy as np

from .exp3 import exp3_true_function, EXP3_X_BOUNDS

BRANIN_GAUSS_X_BOUNDS = EXP3_X_BOUNDS


def _x1_scaled(x: np.ndarray) -> np.ndarray:
    x_2d = np.atleast_2d(np.asarray(x, dtype=float))
    lo, hi = -5.0, 10.0
    return (x_2d[:, 0] - lo) / (hi - lo)


def _sigma(x: np.ndarray) -> np.ndarray:
    return 0.4 * (4.0 * _x1_scaled(x) + 1.0)


def branin_gauss_simulator(x, random_state=None):
    """Branin-Hoo + heteroscedastic Gaussian noise."""
    rng = np.random.default_rng(random_state)
    x_2d = np.atleast_2d(np.asarray(x, dtype=float))
    sigma = _sigma(x_2d)
    return exp3_true_function(x_2d) + rng.normal(0.0, sigma)
