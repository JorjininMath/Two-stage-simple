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
from .sim_branin_gauss import branin_gauss_simulator, BRANIN_GAUSS_X_BOUNDS
from .sim_branin_student import branin_student_simulator, BRANIN_STUDENT_X_BOUNDS
from .sim_sigma_t10 import exp3_alloc_simulator, EXP3_ALLOC_X_BOUNDS
from .sim_gibbs_s1 import (
    gibbs_s1_simulator, GIBBS_S1_X_BOUNDS,
    make_gibbs_s1_simulator, make_gibbs_s1_d_simulator, gibbs_s1_d_bounds,
)
from .sim_gibbs_s2 import (
    gibbs_s2_simulator, GIBBS_S2_X_BOUNDS,
    make_gibbs_s2_simulator, make_gibbs_s2_d_simulator, gibbs_s2_d_bounds,
)
from .sim_exp2_gauss import (
    make_exp2_gauss_simulator,
    EXP2_GAUSS_LOW_PARAMS, EXP2_GAUSS_HIGH_PARAMS,
    EXP2_GAUSS_X_BOUNDS,
)

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
    # Non-Gaussian variants: Student-t only (exp2 true function, scale = sigma_tar(x))
    # Gamma (B2) and Gaussian-mixture (C1) variants archived to _archive/.
    # A1S / A1L play the "low / high" roles for this family (shape axis = heavy-tail
    # severity, not amplitude): both share sigma_tar(x); only nu differs
    # (nu=10 weak tail -> low, nu=3 strong tail -> high). SNR gap is ~1.5x (vs 5x
    # on the amplitude-axis pairs), because the contrast is tail shape, not scale.
    "nongauss_A1S": {"simulator": make_nongauss_A1_simulator(nu=10.0), "bounds": EXP2_X_BOUNDS, "d": 1},
    "nongauss_A1L": {"simulator": make_nongauss_A1_simulator(nu=3.0),  "bounds": EXP2_X_BOUNDS, "d": 1},
    # Branin-Hoo (2D) with two noise settings
    "branin_gauss":   {"simulator": branin_gauss_simulator,   "bounds": BRANIN_GAUSS_X_BOUNDS,   "d": 2},
    "branin_student": {"simulator": branin_student_simulator, "bounds": BRANIN_STUDENT_X_BOUNDS,  "d": 2},
    # exp3_alloc: exp2 true function + Student-t nu=3 noise (sigma(x) scale)
    "exp3_alloc":     {"simulator": exp3_alloc_simulator,     "bounds": EXP3_ALLOC_X_BOUNDS,      "d": 1},
    # Gibbs et al. (RLCP) DGPs — location-scale, X in [-3,3]
    # Low = original RLCP setting (sigma_scale=1.0); High scales sigma by 2.0.
    "gibbs_s1":       {"simulator": gibbs_s1_simulator,       "bounds": GIBBS_S1_X_BOUNDS,         "d": 1},
    "gibbs_s2":       {"simulator": gibbs_s2_simulator,       "bounds": GIBBS_S2_X_BOUNDS,         "d": 1},
    "gibbs_s1_high":  {"simulator": make_gibbs_s1_simulator(sigma_scale=5.0), "bounds": GIBBS_S1_X_BOUNDS, "d": 1},
    "gibbs_s2_high":  {"simulator": make_gibbs_s2_simulator(sigma_scale=5.0), "bounds": GIBBS_S2_X_BOUNDS, "d": 1},
    # Gibbs d=5 extensions: sigma depends only on x_1 (high-d mechanism DGPs)
    "gibbs_s1_d5":      {"simulator": make_gibbs_s1_d_simulator(5), "bounds": gibbs_s1_d_bounds(5), "d": 5},
    "gibbs_s2_d5":      {"simulator": make_gibbs_s2_d_simulator(5), "bounds": gibbs_s2_d_bounds(5), "d": 5},
    "gibbs_s1_d5_high": {"simulator": make_gibbs_s1_d_simulator(5, sigma_scale=5.0), "bounds": gibbs_s1_d_bounds(5), "d": 5},
    "gibbs_s2_d5_high": {"simulator": make_gibbs_s2_d_simulator(5, sigma_scale=5.0), "bounds": gibbs_s2_d_bounds(5), "d": 5},
    # exp2 true function + heteroscedastic Gaussian (low / high sigma slope)
    "exp2_gauss_low":  {"simulator": make_exp2_gauss_simulator(**EXP2_GAUSS_LOW_PARAMS),
                        "bounds": EXP2_GAUSS_X_BOUNDS, "d": 1},
    "exp2_gauss_high": {"simulator": make_exp2_gauss_simulator(**EXP2_GAUSS_HIGH_PARAMS),
                        "bounds": EXP2_GAUSS_X_BOUNDS, "d": 1},
    # WSC 2026 paper DGPs: sigma(x) = 0.01 + 0.2*(x-pi)^2
    # wsc_gauss: Exp 1 (Gaussian noise)
    # nongauss_A1L: Exp 2 (Student-t nu=3) — registered above
    "wsc_gauss": {"simulator": make_exp2_gauss_simulator(sigma_base=0.01, sigma_slope=0.20),
                  "bounds": EXP2_GAUSS_X_BOUNDS, "d": 1},
}


def get_experiment_config(name: str):
    if name not in _EXPERIMENT_REGISTRY:
        raise ValueError(f"Unknown experiment: {name}. Valid: {list(_EXPERIMENT_REGISTRY)}")
    return _EXPERIMENT_REGISTRY[name]
