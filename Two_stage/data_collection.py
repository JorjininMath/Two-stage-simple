"""Collect Stage 1 and Stage 2 data."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from .design import generate_space_filling_design
from .sim_functions import get_experiment_config

ArrayLike = np.ndarray


def collect_stage2_data(
    X_1: ArrayLike,
    r_1: int,
    simulator_func: str,
    random_state: Optional[int] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Collect Stage 2 data D_1 at selected sites X_1.

    Each site in X_1 is evaluated r_1 times.

    Parameters
    ----------
    X_1 : array-like, shape (n_1, d)
        Selected design sites.
    r_1 : int
        Replications per site.
    simulator_func : str
        Experiment name, e.g. "exp1".
    random_state : int, optional
        Random seed.

    Returns
    -------
    X_stage2 : ndarray, shape (n_1 * r_1, d)
    Y_stage2 : ndarray, shape (n_1 * r_1,)
    """
    exp_config = get_experiment_config(simulator_func)
    sim = exp_config["simulator"]
    X_1 = np.atleast_2d(np.asarray(X_1, dtype=float))
    n_1 = X_1.shape[0]
    X_all = np.repeat(X_1, r_1, axis=0)
    Y_all = sim(X_all, random_state=random_state)
    return X_all, np.asarray(Y_all).ravel()


def collect_stage1_data(
    n_0: int,
    d: int,
    r_0: int,
    simulator_func: str,
    X_bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    design_method: str = "lhs",
    random_state: Optional[int] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    exp_config = get_experiment_config(simulator_func)
    sim = exp_config["simulator"]
    if X_bounds is None:
        X_bounds = exp_config["bounds"]
    X_design = generate_space_filling_design(n_0, d, method=design_method, bounds=X_bounds, random_state=random_state)
    X_all = np.repeat(X_design, r_0, axis=0)
    Y_all = sim(X_all, random_state=random_state)
    return X_all, np.asarray(Y_all).ravel()
