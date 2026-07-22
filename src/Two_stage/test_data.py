"""
test_data.py

Generate test data with same distribution as Stage 2, excluding X_1 for exchangeability.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .data_collection import collect_stage2_data
from .design import generate_space_filling_design
from .s0_score import compute_s0_tail_uncertainty
from .sim_functions import get_experiment_config

ArrayLike = np.ndarray


def _exclude_near_x1(
    X_cand: np.ndarray,
    X_1: np.ndarray,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Return indices of X_cand that are not within tolerance of any X_1."""
    if X_1.size == 0:
        return np.arange(len(X_cand))
    dist = cdist(X_cand, X_1)
    min_dist = dist.min(axis=1)
    return np.where(min_dist > tolerance)[0]


def generate_test_data(
    stage2_result: object,
    n_test: int,
    r_test: int,
    X_cand: ArrayLike,
    simulator_func: str = "exp1",
    X_bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    random_state: Optional[int] = None,
    mixed_ratio: float = 0.7,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate test data with same distribution as Stage 2, excluding X_1.

    Test points are drawn using the same method as stage2.selection_method
    (lhs, sampling, or mixed). Points overlapping with X_1 are excluded.

    Parameters
    ----------
    stage2_result : Stage2Result
        From run_stage2 or load_stage2_result.
    n_test : int
        Number of test design sites.
    r_test : int
        Replications per test site.
    X_cand : array-like, shape (n_cand, d)
        Candidate points. Required for sampling/mixed.
    simulator_func : str, default="exp1"
    X_bounds : tuple, optional
        (lower, upper). From experiment config if None.
    random_state : int, optional
    mixed_ratio : float, default=0.7
        For mixed method.
    tolerance : float, default=1e-6
        Min distance to X_1 for exclusion.

    Returns
    -------
    X_test_full : ndarray, shape (n_test * r_test, d)
    Y_test : ndarray, shape (n_test * r_test,)
    """
    method: str = stage2_result.selection_method
    X_1 = stage2_result.X_1
    model = stage2_result.model
    t_grid = stage2_result.t_grid
    alpha = stage2_result.alpha

    exp_config = get_experiment_config(simulator_func)
    if X_bounds is None:
        X_bounds = exp_config["bounds"]
    d = X_1.shape[1]

    if random_state is not None:
        np.random.seed(random_state)

    if method == "lhs":
        # Generate LHS, exclude points near X_1
        buffer = 2 * n_test
        n_gen = 0
        X_test_sites = []
        attempt = 0
        while len(X_test_sites) < n_test and attempt < 10:
            seed = (random_state + attempt * 1000) if random_state is not None else None
            X_lhs = generate_space_filling_design(
                n=n_test + buffer,
                d=d,
                method="lhs",
                bounds=X_bounds,
                random_state=seed,
            )
            keep_idx = _exclude_near_x1(X_lhs, X_1, tolerance)
            for i in keep_idx:
                if len(X_test_sites) >= n_test:
                    break
                X_test_sites.append(X_lhs[i])
            attempt += 1
        if len(X_test_sites) < n_test:
            raise ValueError(
                f"Could not generate {n_test} test points excluding X_1. "
                f"Got {len(X_test_sites)}. Try larger buffer or different seed."
            )
        X_test_sites = np.array(X_test_sites[:n_test])

    else:
        # sampling or mixed: use X_cand, exclude X_1
        X_cand = np.atleast_2d(np.asarray(X_cand, dtype=float))
        remain_idx = _exclude_near_x1(X_cand, X_1, tolerance)
        X_remain = X_cand[remain_idx]
        n_remain = len(X_remain)
        if n_remain < n_test:
            raise ValueError(
                f"Only {n_remain} candidates after excluding X_1. Need n_test={n_test}. "
                "Use larger X_cand or smaller n_test."
            )

        s0 = compute_s0_tail_uncertainty(
            model=model, X_cand=X_remain, t_grid=t_grid, alpha=alpha
        )
        probs = np.maximum(s0, 0.0)
        if np.all(probs <= 0):
            probs = np.ones(n_remain) / n_remain
        else:
            probs = probs / probs.sum()

        if method == "sampling":
            idx = np.random.choice(n_remain, size=n_test, replace=False, p=probs)
            X_test_sites = X_remain[idx]
        else:
            # mixed
            gamma = mixed_ratio
            n_lhs = int(np.round((1.0 - gamma) * n_test))
            n_dens = n_test - n_lhs
            selected = []
            if n_lhs > 0:
                X_lhs = generate_space_filling_design(
                    n=n_lhs, d=d, method="lhs", bounds=X_bounds, random_state=random_state
                )
                dist = cdist(X_lhs, X_remain)
                lhs_idx = np.argmin(dist, axis=1)
                selected = list(np.unique(lhs_idx))
            n_needed = n_test - len(selected)
            if n_needed > 0:
                rem = np.setdiff1d(np.arange(n_remain), selected)
                if len(rem) <= n_needed:
                    selected.extend(rem.tolist())
                else:
                    p_rem = probs[rem] / probs[rem].sum()
                    idx = np.random.choice(len(rem), size=n_needed, replace=False, p=p_rem)
                    selected.extend(rem[idx].tolist())
            selected = np.array(selected[:n_test])
            X_test_sites = X_remain[selected]

    X_test_full, Y_test = collect_stage2_data(
        X_1=X_test_sites,
        r_1=r_test,
        simulator_func=simulator_func,
        random_state=random_state,
    )
    return X_test_full, Y_test
