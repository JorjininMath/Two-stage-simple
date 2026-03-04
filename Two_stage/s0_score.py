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

    Parameters
    ----------
    model : CKMEModel
        Trained CKME model from Stage 1.
    X_cand : array-like, shape (n_cand, d)
        Candidate points for Stage-2 site selection.
    t_grid : array-like, shape (M,)
        Threshold grid for CDF evaluation.
    alpha : float, default=0.1
        Significance level. Uses quantiles at α/2 and 1-α/2.

    Returns
    -------
    scores : ndarray, shape (n_cand,)
        S^0(x) for each candidate. Higher = more uncertainty, more need for data.
    """
    X_cand = np.atleast_2d(np.asarray(X_cand, dtype=float))
    t_grid = np.asarray(t_grid, dtype=float).ravel()

    tau_lower = alpha / 2
    tau_upper = 1 - alpha / 2

    F_pred = model.predict_cdf(X_cand, t_grid)  # shape (n_cand, M)

    # Vectorized quantile: find segment F[i] <= tau <= F[i+1], then linear interp
    def _quantile_vec(F, tau):
        idx = np.argmax(F >= tau, axis=1) - 1
        idx = np.maximum(0, np.minimum(idx, F.shape[1] - 2))
        f0 = F[np.arange(F.shape[0]), idx]
        f1 = F[np.arange(F.shape[0]), idx + 1]
        t0, t1 = t_grid[idx], t_grid[idx + 1]
        denom = np.where(np.abs(f1 - f0) < 1e-12, 1.0, f1 - f0)
        w = (tau - f0) / denom
        return t0 + w * (t1 - t0)

    q_lower = _quantile_vec(F_pred, tau_lower)
    q_upper = _quantile_vec(F_pred, tau_upper)
    return q_upper - q_lower


def compute_s0(res: Stage1TrainResult, X_cand: ArrayLike, alpha: float = 0.1) -> np.ndarray:
    """
    Compute S^0 from loaded Stage 1 result. Convenience wrapper.

    Parameters
    ----------
    res : Stage1TrainResult
        Loaded result from load_stage1_train_result().
    X_cand : array-like, shape (n_cand, d)
        Candidate points.
    alpha : float, default=0.1
        Significance level.

    Returns
    -------
    scores : ndarray, shape (n_cand,)
    """
    return compute_s0_tail_uncertainty(
        model=res.model,
        X_cand=X_cand,
        t_grid=res.t_grid,
        alpha=alpha,
    )
