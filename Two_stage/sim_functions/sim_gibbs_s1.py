"""
sim_gibbs_s1.py

Gibbs et al. (RLCP paper) Setting 1 — univariate location-scale DGP:
  f(x) = 0.5 * x
  sigma(x) = |sin(x)|
  Y = f(x) + sigma(x) * eps,  eps ~ N(0, 1)

Reference: Gibbs et al. (2023) RLCP, simu_sett.R setting=1.

Domain: x in [-3, 3]  (X ~ N(0,1), truncated at 3 sigma)

Note: sigma(x) = 0 at x = k*pi, so Y|X=k*pi is degenerate (variance 0).
      This is a stress test for heteroscedastic methods.
"""
from __future__ import annotations
import numpy as np

import numpy as _np
GIBBS_S1_X_BOUNDS = (_np.array([-3.0]), _np.array([3.0]))


def gibbs_s1_simulator(x, random_state=None):
    """Y = 0.5*x + |sin(x)| * N(0,1)."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    sigma = np.abs(np.sin(x))
    return 0.5 * x + sigma * rng.standard_normal(x.shape)
