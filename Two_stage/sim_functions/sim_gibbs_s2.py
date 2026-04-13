"""
sim_gibbs_s2.py

Gibbs et al. (RLCP paper) Setting 2 — univariate location-scale DGP:
  f(x) = 0.5 * x
  sigma(x) = 2 * phi(x / 1.5)  where phi = standard normal pdf
           = 2 * dnorm(x, 0, 1.5)  in R notation
  Y = f(x) + sigma(x) * eps,  eps ~ N(0, 1)

Reference: Gibbs et al. (2023) RLCP, simu_sett.R setting=2.

Domain: x in [-3, 3]  (X ~ N(0,1), truncated at 3 sigma)

sigma(x) peaks at x=0: sigma(0) = 2/sqrt(2*pi*1.5^2) ~ 0.532,
and decays to near 0 for |x| > 3.
"""
from __future__ import annotations
import numpy as np

import numpy as _np
GIBBS_S2_X_BOUNDS = (_np.array([-3.0]), _np.array([3.0]))

_SCALE = 1.5
_COEFF = 2.0 / (np.sqrt(2.0 * np.pi) * _SCALE)  # 2 * N(0, 1.5) pdf prefactor


def gibbs_s2_simulator(x, random_state=None):
    """Y = 0.5*x + 2*dnorm(x, 0, 1.5) * N(0,1)."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    sigma = _COEFF * np.exp(-0.5 * (x / _SCALE) ** 2)
    return 0.5 * x + sigma * rng.standard_normal(x.shape)
