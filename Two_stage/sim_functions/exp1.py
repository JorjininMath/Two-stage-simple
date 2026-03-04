"""Exp1: MG1 queue. ζ(x)=1.5x²/(1-x), heteroscedastic noise."""
import numpy as np
EXP1_X_BOUNDS = (np.array([0.1]), np.array([0.9]))


def exp1_true_function(x):
    return 1.5 * x**2 / (1 - x)


def exp1_noise_variance_function(x):
    # Analytical noise variance of the M/G/1 queue model.
    # Polynomial coefficients (20, 121, 116, 29) and scale (2500) are
    # derived from the Pollaczek-Khinchine formula for this queue setting.
    num = x * (20 + 121*x - 116*x**2 + 29*x**3)
    den = 4 * (1 - x)**4 * 2500
    return num / den
