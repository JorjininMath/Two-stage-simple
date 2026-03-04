"""Space-filling design."""
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.stats import qmc

ArrayLike = np.ndarray


def generate_space_filling_design(
    n: int,
    d: int,
    method: str = "lhs",
    bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    random_state: Optional[int] = None,
) -> ArrayLike:
    if method == "lhs":
        if bounds is None:
            lower, upper = np.zeros(d), np.ones(d)
        else:
            lower = np.asarray(bounds[0]).ravel()
            upper = np.asarray(bounds[1]).ravel()
        sampler = qmc.LatinHypercube(d=d, seed=random_state)
        return qmc.scale(sampler.random(n=n), lower, upper)
    elif method == "grid":
        if bounds is None:
            lower, upper = np.zeros(d), np.ones(d)
        else:
            lower = np.asarray(bounds[0]).ravel()
            upper = np.asarray(bounds[1]).ravel()
        if d == 1:
            return np.linspace(lower[0], upper[0], n, endpoint=True).reshape(-1, 1)
        n_per_dim = max(2, int(np.round(n ** (1.0 / d))))
        axes = [np.linspace(lower[i], upper[i], n_per_dim, endpoint=True) for i in range(d)]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)
    raise ValueError(f"Unknown method: {method}")
