"""Stationary M/M/1 FCFS system-time simulator.

The input is the traffic intensity rho in (0, 1). With service rate mu = 1
and arrival rate lambda = rho, the stationary customer system time satisfies

    Y | rho ~ Exponential(rate=1-rho).

Consequently, both the conditional mean and conditional standard deviation are
1 / (1-rho). The registered benchmark domain stays away from instability:
rho in [0.1, 0.9].
"""
from __future__ import annotations

import numpy as np

MM1_SERVICE_RATE = 1.0
MM1_SOJOURN_X_BOUNDS = (np.array([0.1]), np.array([0.9]))


def _validate_rho(rho: np.ndarray) -> np.ndarray:
    """Return rho as a float array after checking the stable-queue domain."""
    rho_arr = np.asarray(rho, dtype=float)
    if not np.all(np.isfinite(rho_arr)):
        raise ValueError("rho must contain only finite values")
    if np.any((rho_arr <= 0.0) | (rho_arr >= 1.0)):
        raise ValueError("rho must lie strictly between 0 and 1")
    return rho_arr


def mm1_sojourn_mean(rho: np.ndarray) -> np.ndarray:
    """Conditional mean E[Y | rho] for the stationary M/M/1 system time."""
    rho_arr = _validate_rho(rho)
    return 1.0 / (MM1_SERVICE_RATE * (1.0 - rho_arr))


def mm1_sojourn_scale(rho: np.ndarray) -> np.ndarray:
    """Conditional standard deviation SD[Y | rho]."""
    return mm1_sojourn_mean(rho)


def mm1_sojourn_simulator(
    rho: np.ndarray,
    random_state: int | None = None,
) -> np.ndarray:
    """Draw stationary M/M/1 system times at the supplied traffic intensities."""
    scale = mm1_sojourn_scale(rho)
    rng = np.random.default_rng(random_state)
    return rng.exponential(scale=scale, size=scale.shape)
