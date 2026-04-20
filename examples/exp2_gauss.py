"""
exp2_gauss: Heteroscedastic Gaussian noise on exp2 true function.

True function: f(x) = exp(x/10) * sin(x), x in [0, 2*pi]
Noise: Y = f(x) + sigma(x) * eps, eps ~ N(0, 1)
Noise std: sigma(x) = sigma_base + sigma_slope * (x - pi)^2

Two noise levels:
  low:  sigma_base=0.1, sigma_slope=0.05  =>  sigma in [0.1, 0.59]
  high: sigma_base=0.1, sigma_slope=0.20  =>  sigma in [0.1, 2.07]

Heteroscedasticity ratio rho = sigma_max / sigma_min:
  low:  rho ~  5.9
  high: rho ~ 20.7
"""
import numpy as np

X_BOUNDS = (np.array([0.0]), np.array([2 * np.pi]))
D = 1

_PI = np.pi


def true_function(x):
    """f(x) = exp(x/10) * sin(x)."""
    return np.exp(x / 10) * np.sin(x)


def noise_std(x, sigma_base=0.1, sigma_slope=0.1):
    """sigma(x) = sigma_base + sigma_slope * (x - pi)^2."""
    return sigma_base + sigma_slope * (x - _PI) ** 2


def noise_variance(x, sigma_base=0.1, sigma_slope=0.1):
    """sigma(x)^2."""
    return noise_std(x, sigma_base, sigma_slope) ** 2


def make_simulator(sigma_base=0.1, sigma_slope=0.1):
    """Create a Gaussian simulator with given noise parameters."""
    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x, dtype=float)
        y_true = true_function(x)
        sigma = noise_std(x, sigma_base, sigma_slope)
        return y_true + rng.normal(0, sigma, size=y_true.shape)
    return simulator


# Pre-built variants
VARIANTS = {
    "exp2_gauss_low": {
        "sigma_base": 0.1,
        "sigma_slope": 0.05,
        "description": "Low noise: sigma in [0.1, 0.59], rho ~ 5.9",
    },
    "exp2_gauss_high": {
        "sigma_base": 0.1,
        "sigma_slope": 0.20,
        "description": "High noise: sigma in [0.1, 2.07], rho ~ 20.7",
    },
}

# Registry entries (compatible with Two_stage/sim_functions interface)
REGISTRY = {}
for name, cfg in VARIANTS.items():
    REGISTRY[name] = {
        "simulator": make_simulator(cfg["sigma_base"], cfg["sigma_slope"]),
        "bounds": X_BOUNDS,
        "d": D,
        "sigma_base": cfg["sigma_base"],
        "sigma_slope": cfg["sigma_slope"],
    }
