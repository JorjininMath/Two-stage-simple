"""
s0_score.py

Compute need-for-data score S^0 using tail_uncertainty (interval width).

S^0(x) = q_{1-α/2}^{(0)}(x) - q_{α/2}^{(0)}(x)

No CP calibration needed. Uses CKME model only.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from .stage1_train import Stage1TrainResult

ArrayLike = np.ndarray


def compute_s0_tail_uncertainty(
    model,
    X_cand: ArrayLike,
    t_grid: ArrayLike,
    alpha: float = 0.1,
) -> np.ndarray:
    """
    Compute S^0(x) = q_{1-α/2}(x) - q_{α/2}(x) for each candidate point.

    Uses ``model.predict_quantile_solve`` (brentq root-finding for smooth
    indicators; exact weighted-ECDF inversion for step) to avoid the
    discretisation error of grid-based inversion.

    Parameters
    ----------
    model : CKMEModel
        Trained CKME model from Stage 1.
    X_cand : array-like, shape (n_cand, d)
        Candidate points for Stage-2 site selection.
    t_grid : array-like, shape (M,)
        Kept for API compatibility; not used by predict_quantile_solve.
    alpha : float, default=0.1
        Significance level. Uses quantiles at α/2 and 1-α/2.

    Returns
    -------
    scores : ndarray, shape (n_cand,)
        S^0(x) for each candidate. Higher = more uncertainty, more need for data.
    """
    X_cand = np.atleast_2d(np.asarray(X_cand, dtype=float))

    tau_lower = alpha / 2
    tau_upper = 1 - alpha / 2

    q_lower = model.predict_quantile_solve(X_cand, tau_lower)
    q_upper = model.predict_quantile_solve(X_cand, tau_upper)
    return q_upper - q_lower


def compute_s0_predictive_std(model, X_cand: ArrayLike) -> np.ndarray:
    """
    Compute S^0(x) = sqrt(Var[Y|x]) via c(x)-weighted empirical moments.

    Bypasses the smooth indicator g_t entirely — moments are recovered
    directly from the CKME coefficients c(x) = (K_X + λI)^{-1} k_x(X, ·):

        μ̂(x)     = Σ_i c_i(x) · Y_i
        Ê[Y²|x]  = Σ_i c_i(x) · Y_i²
        Var̂(x)   = max(Ê[Y²|x] − μ̂(x)², 0)

    For r > 1 (distinct-sites mode) uses per-site means (Ȳ_j, Ȳ²_j).

    Parameters
    ----------
    model : CKMEModel
        Trained CKME model. Must have attributes L, kx, X, Y, r, params.
    X_cand : array-like, shape (n_cand, d)
        Candidate points.

    Returns
    -------
    scores : ndarray, shape (n_cand,)
        Estimated conditional standard deviation at each candidate point.
    """
    from CKME.coefficients import compute_ckme_coeffs

    X_cand = np.atleast_2d(np.asarray(X_cand, dtype=float))
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_cand)  # (n_sites, q)

    if model.r > 1:
        Ybar = model.Y.mean(axis=1)              # (n_sites,)
        Y2bar = (model.Y ** 2).mean(axis=1)      # (n_sites,)
    else:
        Ybar = np.asarray(model.Y).ravel()
        Y2bar = Ybar ** 2

    mu = C.T @ Ybar                               # (q,)
    mu2 = C.T @ Y2bar                             # (q,)
    var = np.maximum(mu2 - mu ** 2, 0.0)
    return np.sqrt(var)


def compute_s0_bootstrap_epistemic(
    res: Stage1TrainResult,
    X_cand: ArrayLike,
    K: int = 30,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Bootstrap epistemic S^0(x) = sqrt( ∫ Var_k[ F̂_k(t|x) ] dt ).

    Site-level resample: sample n_0 site indices with replacement, each
    keeping all r_0 replications. Refit CKME with the same fixed params
    (no CV) for each of K bootstrap samples. Aggregate variance of the
    estimated CDF over the existing t_grid.

    Parameters
    ----------
    res : Stage1TrainResult
        Stage 1 result. Must have X_all, Y_all, n_0, r_0, t_grid, params.
    X_cand : array-like, shape (n_cand, d)
        Candidate points.
    K : int, default=30
        Number of bootstrap replicates.
    random_state : int, optional
        Seed for site resampling.

    Returns
    -------
    scores : ndarray, shape (n_cand,)
        sqrt of integrated CDF variance across bootstrap replicates.
    """
    from CKME import CKMEModel

    X_cand = np.atleast_2d(np.asarray(X_cand, dtype=float))

    n_0, r_0 = res.n_0, res.r_0
    d = res.d
    indicator_type = getattr(res.model, "indicator_type", "logistic")

    # Reshape flat arrays into per-site blocks
    X_sites = res.X_all.reshape(n_0, r_0, d)
    Y_sites = res.Y_all.reshape(n_0, r_0)

    rng = np.random.default_rng(random_state)
    t_grid = res.t_grid
    M = t_grid.shape[0]
    q = X_cand.shape[0]

    F_stack = np.empty((K, q, M), dtype=float)

    for k in range(K):
        idx = rng.integers(0, n_0, size=n_0)
        Xb = X_sites[idx].reshape(n_0 * r_0, d)
        Yb = Y_sites[idx].reshape(n_0 * r_0)

        model_k = CKMEModel(indicator_type=indicator_type)
        model_k.fit(X=Xb, Y=Yb, params=res.params, r=r_0)
        F_stack[k] = model_k.predict_cdf(X_cand, t_grid)

    var_F = F_stack.var(axis=0, ddof=1)              # (q, M)
    integrated = np.trapz(var_F, t_grid, axis=1)     # (q,)
    return np.sqrt(np.maximum(integrated, 0.0))


def compute_s0(
    res: Stage1TrainResult,
    X_cand: ArrayLike,
    alpha: float = 0.1,
    score_type: str = "tail",
    K: int = 30,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Compute S^0 from loaded Stage 1 result. Convenience wrapper.

    Parameters
    ----------
    res : Stage1TrainResult
        Loaded result from load_stage1_train_result().
    X_cand : array-like, shape (n_cand, d)
        Candidate points.
    alpha : float, default=0.1
        Significance level. Used only for score_type="tail".
    score_type : {"tail", "variance", "epistemic"}, default="tail"
        "tail"      — S^0 = q_{1-α/2} − q_{α/2} (smooth indicator + brentq)
        "variance"  — S^0 = sqrt(Var[Y|x])     (c(x)-weighted moments)
        "epistemic" — bootstrap CDF variance (K refits with fixed params)
    K : int, default=30
        Bootstrap replicates (epistemic only).
    random_state : int, optional
        Seed (epistemic only).

    Returns
    -------
    scores : ndarray, shape (n_cand,)
    """
    if score_type == "tail":
        return compute_s0_tail_uncertainty(
            model=res.model,
            X_cand=X_cand,
            t_grid=res.t_grid,
            alpha=alpha,
        )
    elif score_type == "variance":
        return compute_s0_predictive_std(res.model, X_cand)
    elif score_type == "epistemic":
        return compute_s0_bootstrap_epistemic(
            res, X_cand, K=K, random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown score_type: {score_type!r}")
