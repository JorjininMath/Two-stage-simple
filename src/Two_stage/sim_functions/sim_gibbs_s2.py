"""
sim_gibbs_s2.py

Gibbs et al. (RLCP paper) Setting 2 — location-scale DGP (1D or d-dim).

1D (d=1):
  f(x) = 0.5 * x
  sigma(x) = sigma_scale * 2 * phi(x / 1.5)   (2 * N(0, 1.5) pdf)
  Y = f(x) + sigma(x) * eps,  eps ~ N(0, 1)

d-dim extension (d >= 1):
  f(x) = 0.5 * mean(x)
  sigma(x) = sigma_scale * 2 * phi(x_1 / 1.5)   (depends only on first coordinate)
  Y = f(x) + sigma(x) * eps,  eps ~ N(0, 1)

Reference: Gibbs et al. (2023) RLCP, simu_sett.R setting=2.

sigma(x_1) peaks at x_1=0: sigma(0) ~ 0.532, decays to near 0 for |x_1| > 3.

Domain: x in [-3, 3]^d.
"""
from __future__ import annotations
import numpy as np

GIBBS_S2_X_BOUNDS = (np.array([-3.0]), np.array([3.0]))

_SCALE = 1.5
_COEFF = 2.0 / (np.sqrt(2.0 * np.pi) * _SCALE)


def _sigma_1d(x, sigma_scale: float = 1.0):
    return sigma_scale * _COEFF * np.exp(-0.5 * (x / _SCALE) ** 2)


def make_gibbs_s2_simulator(sigma_scale: float = 1.0):
    """1D: Y = 0.5*x + sigma_scale * 2*dnorm(x, 0, 1.5) * N(0,1)."""
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x, dtype=float)
        return 0.5 * x + _sigma_1d(x, sigma_scale) * rng.standard_normal(x.shape)
    return simulator


def gibbs_s2_simulator(x, random_state=None):
    """RLCP Setting 2 (original scale)."""
    return make_gibbs_s2_simulator(sigma_scale=1.0)(x, random_state=random_state)


def make_gibbs_s2_d_simulator(d: int, sigma_scale: float = 1.0):
    """d-dim: Y = 0.5 * mean(x) + sigma_scale * 2*dnorm(x_1, 0, 1.5) * N(0,1)."""
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != d:
            x = x.reshape(-1, d)
        sigma = _sigma_1d(x[:, 0], sigma_scale)
        return 0.5 * x.mean(axis=1) + sigma * rng.standard_normal(x.shape[0])

    return simulator


def gibbs_s2_d_bounds(d: int):
    return (np.full(d, -3.0), np.full(d, 3.0))
