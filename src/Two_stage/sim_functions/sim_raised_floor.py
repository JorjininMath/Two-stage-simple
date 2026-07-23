"""Raised-floor location-scale simulators for the adaptive-h benchmark.

Both DGPs use

    m(x) = exp(x/10) sin(x),
    s(x) = 0.10 + 0.20 (x-pi)^2,

on x in [0, 2*pi]. The Gaussian arm has standard-normal noise. The
non-Gaussian arm uses T_3 / sqrt(3), so both DGPs have conditional standard
deviation s(x) while differing in tail shape.
"""
from __future__ import annotations

import numpy as np

RAISED_FLOOR_X_BOUNDS = (np.array([0.0]), np.array([2.0 * np.pi]))
RAISED_FLOOR_T_DOF = 3.0


def _validate_x(x: np.ndarray) -> np.ndarray:
    """Return x as a float array after checking the benchmark domain."""
    x_arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must contain only finite values")
    lower = float(RAISED_FLOOR_X_BOUNDS[0][0])
    upper = float(RAISED_FLOOR_X_BOUNDS[1][0])
    if np.any((x_arr < lower) | (x_arr > upper)):
        raise ValueError(f"x must lie in [{lower}, {upper}]")
    return x_arr


def raised_floor_mean(x: np.ndarray) -> np.ndarray:
    """Conditional mean m(x) = exp(x/10) sin(x)."""
    x_arr = _validate_x(x)
    return np.exp(x_arr / 10.0) * np.sin(x_arr)


def raised_floor_scale(x: np.ndarray) -> np.ndarray:
    """Conditional standard deviation s(x) with a positive floor of 0.10."""
    x_arr = _validate_x(x)
    return 0.10 + 0.20 * (x_arr - np.pi) ** 2


def raised_floor_gauss_simulator(
    x: np.ndarray,
    random_state: int | None = None,
) -> np.ndarray:
    """Draw from m(x) + s(x) Z with Z standard normal."""
    mean = raised_floor_mean(x)
    scale = raised_floor_scale(x)
    rng = np.random.default_rng(random_state)
    return mean + scale * rng.standard_normal(size=mean.shape)


def raised_floor_t3_simulator(
    x: np.ndarray,
    random_state: int | None = None,
) -> np.ndarray:
    """Draw from m(x) + s(x) T_3/sqrt(3), which has conditional SD s(x)."""
    mean = raised_floor_mean(x)
    scale = raised_floor_scale(x)
    rng = np.random.default_rng(random_state)
    standardized_noise = (
        rng.standard_t(df=RAISED_FLOOR_T_DOF, size=mean.shape)
        / np.sqrt(RAISED_FLOOR_T_DOF)
    )
    return mean + scale * standardized_noise
