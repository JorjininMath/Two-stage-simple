"""
exp_test.py

Exp_test: Branin–Hoo (2D), fixed Gaussian noise N(0, 3).
Input: x ∈ [-5, 10] × [0, 15]
"""
from .exp3 import EXP3_X_BOUNDS

EXP_TEST_X_BOUNDS = EXP3_X_BOUNDS  # Same as exp3


def exp_test_noise_variance_function(x):
    """Fixed variance σ² = 9, i.e. ε ~ N(0, 3)."""
    return 9.0  # scalar broadcasts for any x
