"""Simulators for experiments: exp1, exp2, exp3, exp_test, and non-Gaussian variants."""
from __future__ import annotations
import numpy as np
from .simulator import make_simulator, make_student_t_simulator
from .exp1 import exp1_true_function, exp1_noise_variance_function, EXP1_X_BOUNDS
from .exp2 import exp2_true_function, exp2_noise_variance_function, EXP2_X_BOUNDS
from .exp3 import (
    exp3_true_function,
    exp3_noise_variance_function,
    exp3_noise_std_function,
    exp3_noise_dof_function,
    EXP3_X_BOUNDS,
)
from .exp_test import exp_test_noise_variance_function, EXP_TEST_X_BOUNDS
from .sim_nongauss_A1 import make_nongauss_A1_simulator
from .sim_nongauss_B2 import make_nongauss_B2_simulator
from .sim_nongauss_C1 import make_nongauss_C1_simulator

exp1_simulator = make_simulator(exp1_true_function, exp1_noise_variance_function)
exp2_simulator = make_simulator(exp2_true_function, exp2_noise_variance_function)
exp3_simulator = make_student_t_simulator(
    exp3_true_function, exp3_noise_std_function, exp3_noise_dof_function
)
exp_test_simulator = make_simulator(exp3_true_function, exp_test_noise_variance_function)


def exp2_test_simulator(x: np.ndarray, random_state: int | None = None) -> np.ndarray:
    """exp2_test: exp2 true function + skew / heavy-tail mixture Gaussian noise.

    New noise (from your slide):
        Y = ζ(x) + ε(x),
        ε(x) ~ (1 - π(x)) N(-δ(x), σ²(x)) + π(x) N(δ(x), σ²(x)).

    We implement:
      σ(x) = 0.3 * (1 + |x - π|)          # local scale
      δ(x) = c * σ(x), c = 3.0           # distance between the two modes
      π(x) = logit^{-1}(a sin x), a = 2  # mixing weight varying with x

    Old noise (kept here for reference, now disabled):
        # σ(x) = sqrt(exp2_noise_variance_function(x))
        # ε(x) = σ(x) * (Z1, U < p; c * Z2, U ≥ p),  p = 0.5, c = 4.0
    """
    if random_state is not None:
        np.random.seed(random_state)
    x_arr = np.asarray(x, dtype=float)
    y_true = exp2_true_function(x_arr)
    # New mixture noise:
    sigma = 0.3 * (1.0 + np.abs(x_arr - np.pi))  # σ(x)
    c = 3.0
    delta = c * sigma                            # δ(x)
    a = 2.0
    pi_x = 1.0 / (1.0 + np.exp(-a * np.sin(x_arr)))  # π(x)
    U = np.random.uniform(0.0, 1.0, size=x_arr.shape)
    Z = np.random.normal(0.0, 1.0, size=x_arr.shape)
    mean = np.where(U < pi_x, delta, -delta)     # choose +δ or -δ
    eps = mean + sigma * Z
    return y_true + eps

_EXPERIMENT_REGISTRY = {
    "exp1": {"simulator": exp1_simulator, "bounds": EXP1_X_BOUNDS, "d": 1},
    "exp2": {"simulator": exp2_simulator, "bounds": EXP2_X_BOUNDS, "d": 1},
    "exp2_test": {"simulator": exp2_test_simulator, "bounds": EXP2_X_BOUNDS, "d": 1},
    "exp3": {"simulator": exp3_simulator, "bounds": EXP3_X_BOUNDS, "d": 2},
    "exp_test": {"simulator": exp_test_simulator, "bounds": EXP_TEST_X_BOUNDS, "d": 2},
    # Non-Gaussian variants (exp2 true function, default noise params)
    "nongauss_A1": {"simulator": make_nongauss_A1_simulator(nu=3.0), "bounds": EXP2_X_BOUNDS, "d": 1},
    "nongauss_B2": {"simulator": make_nongauss_B2_simulator(k=2.0),  "bounds": EXP2_X_BOUNDS, "d": 1},
    "nongauss_C1": {"simulator": make_nongauss_C1_simulator(pi=0.05), "bounds": EXP2_X_BOUNDS, "d": 1},
}


def get_experiment_config(name: str):
    if name not in _EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown experiment: {name}. Valid: {list(_EXPERIMENT_REGISTRY)}")
    return _EXPERIMENT_REGISTRY[name]
