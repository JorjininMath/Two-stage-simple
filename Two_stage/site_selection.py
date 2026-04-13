"""
site_selection.py

Stage 2 site selection from candidate set X_cand.

Methods: sampling (p ∝ S^0), lhs (ignore S^0), mixed (LHS + S^0 sampling).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cdist

from .design import generate_space_filling_design

ArrayLike = np.ndarray

VALID_SELECTION_METHODS = ["sampling", "lhs", "mixed"]


def select_stage2_sites(
    X_cand: ArrayLike,
    scores: ArrayLike,
    n_1: int,
    method: Literal["sampling", "lhs", "mixed"] = "sampling",
    X_bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    random_state: Optional[int] = None,
    mixed_ratio: float = 0.7,
) -> np.ndarray:
    """
    Select n_1 Stage-2 design sites from X_cand.

    Parameters
    ----------
    X_cand : array-like, shape (n_cand, d)
        Candidate points.
    scores : array-like, shape (n_cand,)
        S^0(x) for each candidate. Higher = more need for data.
    n_1 : int
        Number of sites to select.
    method : {"sampling", "lhs", "mixed"}, default="sampling"
        - "sampling": Sample n_1 sites from p(x) ∝ S^0(x)
        - "lhs": Latin Hypercube Sampling (ignores scores). Requires X_bounds.
        - "mixed": (1-γ)*n_1 from LHS + γ*n_1 from S^0 sampling. Requires X_bounds.
    X_bounds : tuple, optional
        (lower_bounds, upper_bounds), shape (d,). Required for "lhs" and "mixed".
    random_state : int, optional
        Random seed.
    mixed_ratio : float, default=0.7
        γ for "mixed": fraction from S^0 sampling. 1.0=sampling, 0.0=lhs.

    Returns
    -------
    X_selected : ndarray, shape (n_1, d)
    """
    if method not in VALID_SELECTION_METHODS:
        raise ValueError(
            f"method must be one of {VALID_SELECTION_METHODS}, got {method}"
        )

    if method == "lhs":
        if X_bounds is None:
            raise ValueError("X_bounds required for method='lhs'")
        lower = np.asarray(X_bounds[0]).ravel()
        upper = np.asarray(X_bounds[1]).ravel()
        d = len(lower)
        if len(upper) != d:
            raise ValueError("X_bounds lower and upper must have same dimension")
        X_selected = generate_space_filling_design(
            n=n_1, d=d, method="lhs", bounds=X_bounds, random_state=random_state
        )
        return X_selected

    # sampling or mixed: need X_cand and scores
    X_cand = np.atleast_2d(np.asarray(X_cand, dtype=float))
    scores = np.asarray(scores, dtype=float).ravel()
    n_cand, d = X_cand.shape

    if len(scores) != n_cand:
        raise ValueError(
            f"scores length {len(scores)} must match X_cand rows {n_cand}"
        )
    if n_1 > n_cand:
        raise ValueError(f"Cannot select {n_1} sites from {n_cand} candidates")

    weighted_scores = np.maximum(scores, 0.0)
    if np.all(weighted_scores <= 0):
        probs = np.ones(n_cand) / n_cand
    else:
        probs = weighted_scores / np.sum(weighted_scores)

    rng = np.random.default_rng(random_state)

    if method == "sampling":
        idx = rng.choice(n_cand, size=n_1, replace=False, p=probs)
        return X_cand[idx]

    # mixed
    if X_bounds is None:
        raise ValueError("X_bounds required for method='mixed'")
    gamma = mixed_ratio
    n_lhs = int(np.round((1.0 - gamma) * n_1))
    n_density = n_1 - n_lhs

    selected_indices = []

    if n_lhs > 0:
        X_lhs = generate_space_filling_design(
            n=n_lhs,
            d=d,
            method="lhs",
            bounds=X_bounds,
            random_state=random_state,
        )
        dist = cdist(X_lhs, X_cand)
        lhs_idx = np.argmin(dist, axis=1)
        selected_indices = list(np.unique(lhs_idx))

    n_needed = n_1 - len(selected_indices)
    if n_needed > 0:
        remaining = np.setdiff1d(np.arange(n_cand), selected_indices)
        if len(remaining) <= n_needed:
            selected_indices.extend(remaining.tolist())
        else:
            p_rem = probs[remaining]
            p_rem = p_rem / p_rem.sum()
            idx = rng.choice(len(remaining), size=n_needed, replace=False, p=p_rem)
            selected_indices.extend(remaining[idx].tolist())
    if len(selected_indices) > n_1:
        selected_indices = selected_indices[:n_1]
    elif len(selected_indices) < n_1:
        remaining = np.setdiff1d(np.arange(n_cand), selected_indices)
        n_fill = n_1 - len(selected_indices)
        fill_idx = rng.choice(
            len(remaining), size=min(n_fill, len(remaining)), replace=False
        )
        selected_indices.extend(remaining[fill_idx].tolist())

    selected_indices = np.array(selected_indices[:n_1])
    return X_cand[selected_indices]
