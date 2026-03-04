"""Generic simulator: y = ζ(x) + ε, ε ~ N(0, sqrt(r(x))) or Student-t."""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from scipy.stats import t as student_t


def make_simulator(true_fn, noise_var_fn):
    def simulator(x, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        x = np.asarray(x)
        y_true = true_fn(x)
        noise_var = noise_var_fn(x)
        return y_true + np.random.normal(0, np.sqrt(noise_var), size=y_true.shape)
    return simulator


def make_student_t_simulator(
    true_fn: Callable,
    noise_std_fn: Callable,
    noise_dof_fn: Callable,
):
    """Simulator with Student-t noise: y = ζ(x) + ε, ε ~ t_ν(0, σ(x))."""
    def simulator(x, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        x = np.asarray(x)
        y_true = true_fn(x)
        sigma = noise_std_fn(x)
        nu = noise_dof_fn(x)
        noise = student_t.rvs(df=nu, loc=0.0, scale=sigma, size=y_true.shape, random_state=random_state)
        return y_true + noise
    return simulator
