"""Space-filling design and iid target-law sampling."""
from __future__ import annotations
from typing import Callable, Optional, Tuple
import numpy as np
from scipy.stats import qmc

ArrayLike = np.ndarray


def sample_iid_qx(
    n: int,
    d: int,
    bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    random_state: Optional[int] = None,
    qx_sampler: Optional[Callable[[int, np.random.Generator], ArrayLike]] = None,
) -> ArrayLike:
    """
    Draw n iid inputs from the target law q_X.

    This is the sampler used for the calibration set of split conformal
    prediction: calibration inputs must be an iid sample from the SAME law
    q_X that test points follow. Unlike LHS (stratified) or score-driven
    sampling (data-dependent tilt), an iid sample keeps the calibration
    pairs exchangeable with a fresh test pair, which is what the split-CP
    coverage guarantee requires.

    Parameters
    ----------
    n : int
        Number of iid draws.
    d : int
        Input dimension.
    bounds : tuple (lower, upper), optional
        Box bounds for the default uniform q_X. Required if qx_sampler is
        None. Each of lower/upper is a scalar or length-d array.
    random_state : int, optional
        Random seed.
    qx_sampler : callable, optional
        Custom sampler for non-uniform q_X. Called as
        qx_sampler(n, rng) with rng a numpy Generator; must return an
        array of shape (n, d). Overrides bounds when given.

    Returns
    -------
    X : ndarray, shape (n, d)
        iid sample from q_X.
    """
    rng = np.random.default_rng(random_state)
    if qx_sampler is not None:
        X = np.atleast_2d(np.asarray(qx_sampler(n, rng), dtype=float))
        if X.shape != (n, d):
            raise ValueError(
                f"qx_sampler must return shape ({n}, {d}), got {X.shape}"
            )
        return X
    if bounds is None:
        raise ValueError("bounds is required when qx_sampler is None")
    lower = np.broadcast_to(np.asarray(bounds[0], dtype=float).ravel(), (d,))
    upper = np.broadcast_to(np.asarray(bounds[1], dtype=float).ravel(), (d,))
    return rng.uniform(lower, upper, size=(n, d))


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
