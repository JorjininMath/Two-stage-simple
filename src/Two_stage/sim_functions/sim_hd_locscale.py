"""High-dimensional location-scale DGP with irrelevant covariates.

For d >= 1:
    X_j ~ Uniform(-1.5, 1.5)
    Y = 0.3 * X_1 + s(X_1) * eps
    s(X_1) = sqrt(1 + 0.3 * |X_1|)
    eps ~ N(0, 1)

Only the first coordinate controls the conditional distribution. The remaining
coordinates stress CKME smoothing in irrelevant dimensions.
"""
from __future__ import annotations

import numpy as np


def hd_locscale_bounds(d: int):
    return (np.full(d, -1.5), np.full(d, 1.5))


def hd_locscale_scale(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x1 = x_arr
    else:
        x1 = x_arr[:, 0]
    return np.sqrt(1.0 + 0.3 * np.abs(x1))


def make_hd_locscale_simulator(d: int):
    """d-dimensional location-scale simulator with scale depending on x_1."""
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        if x_arr.shape[1] != d:
            x_arr = x_arr.reshape(-1, d)
        x1 = x_arr[:, 0]
        scale = hd_locscale_scale(x_arr)
        return 0.3 * x1 + scale * rng.standard_normal(x_arr.shape[0])

    return simulator
