"""
adaptive_h_utils.py

Per-DGP oracle scale s(x) dispatch + adaptive-h evaluation utilities.

For each DGP, s(x) is the **scale** function used for h(x) = c * s(x):
  - Gaussian DGPs: s(x) = noise std  (sigma)
  - Student-t DGP: s(x) = scale       (NOT std; std = s * sqrt(nu/(nu-2)))

Adaptive-h evaluation bypasses model.predict_cdf and manually rebuilds the
indicator g_{t,h(x_i)} per query point. This works because the CKME
coefficients C(x) only depend on the kernel k_x (not h); only the indicator
needs to change at eval time.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from CKME.coefficients import compute_ckme_coeffs
from CKME.indicators import make_indicator
from CP.interval import projected_quantile_interval

ArrayLike = np.ndarray
_PI = np.pi
_H_FLOOR = 1e-3  # avoid h = 0 where s(x) = 0 (gibbs_s1)


# ---------------------------------------------------------------------------
# Per-DGP scale functions s(x)
# ---------------------------------------------------------------------------

def _first_col(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        return x_arr
    return x_arr[:, 0]


def _wsc_gauss_scale(x: np.ndarray) -> np.ndarray:
    x = _first_col(x)
    return 0.01 + 0.20 * (x - _PI) ** 2


def _exp2_gauss_low_scale(x: np.ndarray) -> np.ndarray:
    x = _first_col(x)
    return 0.10 + 0.05 * (x - _PI) ** 2


def _exp2_gauss_high_scale(x: np.ndarray) -> np.ndarray:
    x = _first_col(x)
    return 0.50 + 0.25 * (x - _PI) ** 2


def _gibbs_s1_scale(x: np.ndarray) -> np.ndarray:
    x = _first_col(x)
    return np.abs(np.sin(x))


def _exp1_scale(x: np.ndarray) -> np.ndarray:
    x = _first_col(x)
    # M/G/1 queue noise std (Pollaczek-Khinchine).
    num = x * (20 + 121 * x - 116 * x ** 2 + 29 * x ** 3)
    den = 4 * (1 - x) ** 4 * 2500
    return np.sqrt(num / den)


def _nongauss_A1L_scale(x: np.ndarray) -> np.ndarray:
    x = _first_col(x)
    return 0.01 + 0.20 * (x - _PI) ** 2


def _hd_locscale_scale(x: np.ndarray) -> np.ndarray:
    x1 = _first_col(x)
    return np.sqrt(1.0 + 0.3 * np.abs(x1))


ORACLE_SCALE: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "wsc_gauss":    _wsc_gauss_scale,
    "exp2_gauss_low": _exp2_gauss_low_scale,
    "exp2_gauss_high": _exp2_gauss_high_scale,
    "gibbs_s1":     _gibbs_s1_scale,
    "exp1":         _exp1_scale,
    "nongauss_A1L": _nongauss_A1L_scale,
    "hd_locscale_d2": _hd_locscale_scale,
    "hd_locscale_d5": _hd_locscale_scale,
    "hd_locscale_d20": _hd_locscale_scale,
}


def get_oracle_h(simulator: str, x_query: np.ndarray, c_scale: float) -> np.ndarray:
    """h(x) = c_scale * s(x), floored at _H_FLOOR to avoid degenerate indicators."""
    if simulator not in ORACLE_SCALE:
        raise ValueError(f"No oracle scale defined for {simulator}; valid: {list(ORACLE_SCALE)}")
    x_arr = np.asarray(x_query, dtype=float)
    s = np.asarray(ORACLE_SCALE[simulator](x_arr), dtype=float).ravel()
    return np.maximum(c_scale * s, _H_FLOOR)


# ---------------------------------------------------------------------------
# Adaptive-h evaluation primitives
# ---------------------------------------------------------------------------

def _projected_point_scores(
    model,
    X_pts: np.ndarray,
    Y_pts: np.ndarray,
    h_pts: np.ndarray,
    t_grid: np.ndarray | None,
) -> np.ndarray:
    """DCP scores |Ftilde(y|x) - 1/2| with per-point adaptive h.

    Policy B (notes/planning/cdf_legality_policy.md): when t_grid is given,
    the CDF value is read through the monotone projection,
    Ftilde(y) = clip(max(raw F at y, max_{t_k <= y} raw F(t_k)), 0, 1),
    so calibration/evaluation scores use the SAME object as interval
    extraction. t_grid=None falls back to the legacy raw point evaluation.
    """
    X_pts = np.atleast_2d(X_pts)
    Y_pts = np.asarray(Y_pts).ravel()
    h_pts = np.asarray(h_pts).ravel()
    n_pts = len(Y_pts)

    Y_flat = model.Y.ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_pts)
    t_arr = None if t_grid is None else np.asarray(t_grid, dtype=float).ravel()

    scores = np.empty(n_pts)
    for j in range(n_pts):
        ind_j = make_indicator(model.indicator_type, float(h_pts[j]))
        if t_arr is None:
            thresholds = np.array([float(Y_pts[j])])
        else:
            thresholds = np.append(t_arr, float(Y_pts[j]))
        G_j = ind_j.g_matrix(Y_flat, thresholds)
        if getattr(model, "r", 1) > 1:
            G_site = G_j.reshape(model.n, model.r, -1).mean(axis=1)
        else:
            G_site = G_j
        F_vals = C[:, j] @ G_site
        if t_arr is None:
            F_j = float(np.clip(F_vals[0], 0.0, 1.0))
        else:
            raw_at_y = F_vals[-1]
            pos = int(np.searchsorted(t_arr, float(Y_pts[j]), side="right"))
            grid_part = F_vals[:pos].max() if pos > 0 else -np.inf
            F_j = float(np.clip(max(raw_at_y, grid_part), 0.0, 1.0))
        scores[j] = abs(F_j - 0.5)
    return scores


def adaptive_recalibrate_q(
    model,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    h_cal: np.ndarray,
    alpha: float,
    t_grid: np.ndarray | None = None,
) -> float:
    """Recompute split-CP q_hat using per-point adaptive h on calibration data.

    Pass t_grid to score through the monotone-projected CDF (Policy B);
    t_grid=None keeps the legacy raw point evaluation.
    """
    Y_cal = np.asarray(Y_cal).ravel()
    n_cal = len(Y_cal)
    scores = _projected_point_scores(model, X_cal, Y_cal, h_cal, t_grid)
    k = int(np.ceil((1 - alpha) * (1 + n_cal)))
    if k > n_cal:
        return float("inf")  # split-CP edge case: predict the whole space
    return float(np.sort(scores)[k - 1])


def adaptive_predict_interval(
    model,
    X_query: np.ndarray,
    h_query: np.ndarray,
    t_grid: np.ndarray,
    q_hat: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict [L, U] using per-point adaptive h via direct level-set search."""
    X_query = np.atleast_2d(X_query)
    h_query = np.asarray(h_query).ravel()
    M = X_query.shape[0]
    T = len(t_grid)

    Y_flat = model.Y.ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_query)

    # Build raw CDF rows, then extract via the shared canonical helper
    # (monotone projection + generalized inverse); see
    # notes/planning/cdf_legality_policy.md and CP/interval.py.
    F_all = np.empty((M, T))
    for m in range(M):
        ind_m = make_indicator(model.indicator_type, float(h_query[m]))
        G_mat = ind_m.g_matrix(Y_flat, t_grid)
        if getattr(model, "r", 1) > 1:
            G_site = G_mat.reshape(model.n, model.r, T).mean(axis=1)
        else:
            G_site = G_mat
        F_all[m] = C[:, m] @ G_site

    return projected_quantile_interval(F_all, t_grid, q_hat)


def adaptive_score_coverage(
    model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    h_test: np.ndarray,
    q_hat: float,
    t_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Score-based coverage: 1{|F(Y|x) - 0.5| <= q_hat} with adaptive h.

    Pass t_grid to evaluate through the monotone-projected CDF (Policy B),
    consistently with adaptive_recalibrate_q and interval extraction.
    """
    scores = _projected_point_scores(model, X_test, Y_test, h_test, t_grid)
    return (scores <= q_hat).astype(int)
