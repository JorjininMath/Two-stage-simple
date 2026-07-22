"""
sim_gibbs_s1.py

Gibbs et al. (RLCP paper) Setting 1 — location-scale DGP (1D or d-dim).

1D (d=1):
  f(x) = 0.5 * x
  sigma(x) = |sin(x)|
  Y = f(x) + sigma(x) * eps,  eps ~ N(0, 1)

d-dim extension (d >= 1):
  f(x) = 0.5 * mean(x)
  sigma(x) = |sin(x_1)|         (depends only on first coordinate)
  Y = f(x) + sigma(x) * eps,  eps ~ N(0, 1)

The d>1 variant creates irrelevant directions x_2,...,x_d that let us test
whether S^0-guided sampling concentrates budget in the informative direction.

Reference: Gibbs et al. (2023) RLCP, simu_sett.R setting=1.

Domain: x in [-3, 3]^d  (X ~ N(0,1), truncated at 3 sigma).
"""
from __future__ import annotations
import numpy as np

GIBBS_S1_X_BOUNDS = (np.array([-3.0]), np.array([3.0]))


def _gibbs_s1_sigma(x1, sigma_scale: float = 1.0):
    return sigma_scale * np.abs(np.sin(x1))


def make_gibbs_s1_simulator(sigma_scale: float = 1.0):
    """1D: Y = 0.5*x + sigma_scale*|sin(x)| * N(0,1)."""
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x, dtype=float)
        sigma = _gibbs_s1_sigma(x, sigma_scale)
        return 0.5 * x + sigma * rng.standard_normal(x.shape)
    return simulator


def gibbs_s1_simulator(x, random_state=None):
    """RLCP Setting 1 (original scale): Y = 0.5*x + |sin(x)| * N(0,1)."""
    return make_gibbs_s1_simulator(sigma_scale=1.0)(x, random_state=random_state)


def make_gibbs_s1_d_simulator(d: int, sigma_scale: float = 1.0):
    """d-dim: Y = 0.5 * mean(x) + sigma_scale*|sin(x_1)| * N(0,1)."""
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != d:
            x = x.reshape(-1, d)
        sigma = _gibbs_s1_sigma(x[:, 0], sigma_scale)
        return 0.5 * x.mean(axis=1) + sigma * rng.standard_normal(x.shape[0])

    return simulator


def gibbs_s1_d_bounds(d: int):
    return (np.full(d, -3.0), np.full(d, 3.0))
