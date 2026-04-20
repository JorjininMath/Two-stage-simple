"""
sim_exp2_gauss.py

exp2 true function + heteroscedastic Gaussian noise with tunable sigma slope.

  f(x) = exp(x/10) * sin(x),   x in [0, 2*pi]
  sigma(x) = sigma_base + sigma_slope * (x - pi)^2
  Y = f(x) + sigma(x) * eps,  eps ~ N(0, 1)

Two preset levels:
  low:  sigma_base=0.1, sigma_slope=0.05  -> sigma in [0.1, 0.59], rho ~ 5.9
  high: sigma_base=0.1, sigma_slope=0.20  -> sigma in [0.1, 2.07], rho ~ 20.7

Used in exp_design as Gaussian controls (smooth f, no heavy tail, no high-d).
"""
from __future__ import annotations
import numpy as np
from .exp2 import exp2_true_function, EXP2_X_BOUNDS

EXP2_GAUSS_X_BOUNDS = EXP2_X_BOUNDS
_PI = np.pi

EXP2_GAUSS_LOW_PARAMS = {"sigma_base": 0.1, "sigma_slope": 0.05}
# HIGH = 5 * LOW so SNR(low)/SNR(high) ~ 5 (empirically 5.0x on U[0,2pi]).
EXP2_GAUSS_HIGH_PARAMS = {"sigma_base": 0.5, "sigma_slope": 0.25}


def exp2_gauss_noise_std(x, sigma_base=0.1, sigma_slope=0.1):
    x = np.asarray(x, dtype=float)
    return sigma_base + sigma_slope * (x - _PI) ** 2


def make_exp2_gauss_simulator(sigma_base: float = 0.1, sigma_slope: float = 0.1):
    """Gaussian simulator with sigma(x) = sigma_base + sigma_slope*(x-pi)^2."""
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x, dtype=float)
        sigma = exp2_gauss_noise_std(x, sigma_base, sigma_slope)
        return exp2_true_function(x) + rng.normal(0.0, sigma, size=x.shape)
    return simulator
