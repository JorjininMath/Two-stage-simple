"""
exp3.py

Exp3: Branin–Hoo (2D), Student-t noise.
Input: x ∈ [-5, 10] × [0, 15]
"""
import numpy as np

EXP3_X_BOUNDS = (np.array([-5.0, 0.0]), np.array([10.0, 15.0]))


def _to_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        if len(x) % 2 != 0:
            raise ValueError("1D input must have even length for 2D coordinates")
        return x.reshape(-1, 2)
    return x


def exp3_true_function(x):
    """Branin–Hoo function."""
    x_2d = _to_2d(x)
    x1, x2 = x_2d[:, 0], x_2d[:, 1]
    # Standard Branin-Hoo benchmark parameters from the global optimization literature.
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def _exp3_x1_scaled(x):
    x_2d = _to_2d(x)
    x1 = x_2d[:, 0]
    lo, hi = EXP3_X_BOUNDS[0][0], EXP3_X_BOUNDS[1][0]
    return (x1 - lo) / (hi - lo)


def exp3_noise_std_function(x):
    """σ(x1) = 0.4 * (4 x1_scaled^2 + 1)"""
    x1_scaled = _exp3_x1_scaled(x)
    return 0.4 * (4.0 * x1_scaled**2 + 1.0)


def exp3_noise_variance_function(x):
    return exp3_noise_std_function(x) ** 2


def exp3_noise_dof_function(x):
    """ν(x1) = max(3, 6 - 4 x1_scaled)"""
    x1_scaled = _exp3_x1_scaled(x)
    return np.maximum(3.0, 6.0 - 4.0 * x1_scaled)
