"""
sim_branin_student.py

DGP branin_student: Branin-Hoo true function + heteroscedastic Student-t noise.
  Y = f(x) + epsilon,
  epsilon ~ t_{nu(x)}(0, sigma(x))
  nu(x)    = max(2.0, 6 - 4 * x1_scaled)    [ranges 6 at x1=-5 down to 2 at x1=10]
  sigma(x) = 0.4 * (4 * x1_scaled + 1)      [ranges 0.4 at x1=-5 up to 2.0 at x1=10]
  x1_scaled = (x1 - (-5)) / (10 - (-5)),    x1 = x[:, 0]

Note: nu(x) reaches 2.0 at x1=10, where Student-t variance is infinite.
Sampling uses the z / sqrt(chi2(nu)/nu) * sigma decomposition, which is
vectorized via numpy's array-valued chisquare (numpy >= 1.17).
Use percentile-based t_grid bounds (not min/max) to handle extreme outliers.

Domain: x1 in [-5, 10],  x2 in [0, 15]  (EXP3_X_BOUNDS)
"""
from __future__ import annotations

import numpy as np

from .exp3 import exp3_true_function, EXP3_X_BOUNDS

BRANIN_STUDENT_X_BOUNDS = EXP3_X_BOUNDS


def _x1_scaled(x: np.ndarray) -> np.ndarray:
    x_2d = np.atleast_2d(np.asarray(x, dtype=float))
    lo, hi = -5.0, 10.0
    return (x_2d[:, 0] - lo) / (hi - lo)


def _sigma(x: np.ndarray) -> np.ndarray:
    return 0.4 * (4.0 * _x1_scaled(x) + 1.0)


def _nu(x: np.ndarray) -> np.ndarray:
    return np.maximum(2.0, 6.0 - 4.0 * _x1_scaled(x))


def branin_student_simulator(x, random_state=None):
    """Branin-Hoo + heteroscedastic Student-t noise.

    Sampling: eps = z / sqrt(v) * sigma,  v = chi2(nu)/nu,  z ~ N(0,1).
    This is equivalent to t_{nu}(0, sigma) and is vectorized over nu.
    """
    rng = np.random.default_rng(random_state)
    x_2d = np.atleast_2d(np.asarray(x, dtype=float))
    n = x_2d.shape[0]
    nu_arr    = _nu(x_2d)      # shape (n,)
    sigma_arr = _sigma(x_2d)   # shape (n,)
    z = rng.standard_normal(n)
    v = rng.chisquare(nu_arr) / nu_arr   # numpy supports array-valued df
    eps = z / np.sqrt(v) * sigma_arr
    return exp3_true_function(x_2d) + eps
