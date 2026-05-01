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

ArrayLike = np.ndarray
_PI = np.pi
_H_FLOOR = 1e-3  # avoid h = 0 where s(x) = 0 (gibbs_s1)


# ---------------------------------------------------------------------------
# Per-DGP scale functions s(x)
# ---------------------------------------------------------------------------

def _wsc_gauss_scale(x: np.ndarray) -> np.ndarray:
    return 0.01 + 0.20 * (x - _PI) ** 2


def _gibbs_s1_scale(x: np.ndarray) -> np.ndarray:
    return np.abs(np.sin(x))


def _exp1_scale(x: np.ndarray) -> np.ndarray:
    # M/G/1 queue noise std (Pollaczek-Khinchine).
    num = x * (20 + 121 * x - 116 * x ** 2 + 29 * x ** 3)
    den = 4 * (1 - x) ** 4 * 2500
    return np.sqrt(num / den)


def _nongauss_A1L_scale(x: np.ndarray) -> np.ndarray:
    return 0.01 + 0.20 * (x - _PI) ** 2


ORACLE_SCALE: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "wsc_gauss":    _wsc_gauss_scale,
    "gibbs_s1":     _gibbs_s1_scale,
    "exp1":         _exp1_scale,
    "nongauss_A1L": _nongauss_A1L_scale,
}


def get_oracle_h(simulator: str, x_query: np.ndarray, c_scale: float) -> np.ndarray:
    """h(x) = c_scale * s(x), floored at _H_FLOOR to avoid degenerate indicators."""
    if simulator not in ORACLE_SCALE:
        raise ValueError(f"No oracle scale defined for {simulator}; valid: {list(ORACLE_SCALE)}")
    x_1d = np.asarray(x_query).ravel()
    s = ORACLE_SCALE[simulator](x_1d)
    return np.maximum(c_scale * s, _H_FLOOR)


# ---------------------------------------------------------------------------
# Adaptive-h evaluation primitives
# ---------------------------------------------------------------------------

def adaptive_recalibrate_q(
    model,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    h_cal: np.ndarray,
    alpha: float,
) -> float:
    """Recompute split-CP q_hat using per-point adaptive h on calibration data."""
    X_cal = np.atleast_2d(X_cal)
    Y_cal = np.asarray(Y_cal).ravel()
    n_cal = len(Y_cal)
    h_cal = np.asarray(h_cal).ravel()

    Y_flat = model.Y.ravel()
    C_cal = compute_ckme_coeffs(model.L, model.kx, model.X, X_cal)

    scores = np.empty(n_cal)
    for j in range(n_cal):
        ind_j = make_indicator(model.indicator_type, float(h_cal[j]))
        g_j = ind_j.g_matrix(Y_flat, np.array([float(Y_cal[j])]))[:, 0]
        if getattr(model, "r", 1) > 1:
            g_site = g_j.reshape(model.n, model.r).mean(axis=1)
        else:
            g_site = g_j
        F_j = float(np.clip(C_cal[:, j] @ g_site, 0.0, 1.0))
        scores[j] = abs(F_j - 0.5)

    k = int(np.ceil((1 - alpha) * (1 + n_cal)))
    k = min(k, n_cal)
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

    tau_lo = float(np.clip(0.5 - q_hat, 0.0, 1.0))
    tau_hi = float(np.clip(0.5 + q_hat, 0.0, 1.0))

    L_arr = np.empty(M)
    U_arr = np.empty(M)
    for m in range(M):
        ind_m = make_indicator(model.indicator_type, float(h_query[m]))
        G_mat = ind_m.g_matrix(Y_flat, t_grid)
        if getattr(model, "r", 1) > 1:
            G_site = G_mat.reshape(model.n, model.r, T).mean(axis=1)
        else:
            G_site = G_mat
        F_m = np.clip(C[:, m] @ G_site, 0.0, 1.0)
        L_arr[m] = t_grid[min(np.searchsorted(F_m, tau_lo), T - 1)]
        U_arr[m] = t_grid[min(np.searchsorted(F_m, tau_hi), T - 1)]

    return L_arr, U_arr


def adaptive_score_coverage(
    model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    h_test: np.ndarray,
    q_hat: float,
) -> np.ndarray:
    """Score-based coverage: 1{|F_hat(Y|x) - 0.5| <= q_hat} with adaptive h."""
    X_test = np.atleast_2d(X_test)
    Y_test = np.asarray(Y_test).ravel()
    h_test = np.asarray(h_test).ravel()
    n_test = len(Y_test)

    Y_flat = model.Y.ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_test)

    scores = np.empty(n_test)
    for i in range(n_test):
        ind_i = make_indicator(model.indicator_type, float(h_test[i]))
        g_i = ind_i.g_matrix(Y_flat, np.array([float(Y_test[i])]))[:, 0]
        if getattr(model, "r", 1) > 1:
            g_site = g_i.reshape(model.n, model.r).mean(axis=1)
        else:
            g_site = g_i
        F_i = float(np.clip(C[:, i] @ g_site, 0.0, 1.0))
        scores[i] = abs(F_i - 0.5)

    return (scores <= q_hat).astype(int)
