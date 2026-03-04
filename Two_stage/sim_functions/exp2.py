"""
exp2.py

Exp2: ζ(x) = exp(x/10) * sin(x), heteroscedastic noise.
Input: x ∈ [0, 2π]
"""
import numpy as np

EXP2_X_BOUNDS = (np.array([0.0]), np.array([2 * np.pi]))


def exp2_true_function(x):
    return np.exp(x / 10) * np.sin(x)


def exp2_noise_variance_function(x):
    return (0.01 + 0.2 * (x - np.pi) ** 2) ** 2
