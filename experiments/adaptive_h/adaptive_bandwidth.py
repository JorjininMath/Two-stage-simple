"""Per-DGP oracle scales and adaptive-bandwidth evaluation utilities.

For each DGP, s(x) is the oracle response scale used in h(x) = c * s(x).
It is the conditional standard deviation for the final Gaussian,
variance-normalized Student-t, and M/M/1 DGPs. Historical unnormalized
Student-t simulators retain their original scale parameter for reproducibility.

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
from Two_stage.sim_functions.sim_mm1_sojourn import mm1_sojourn_scale
from Two_stage.sim_functions.sim_raised_floor import raised_floor_scale

ArrayLike = np.ndarray
_PI = np.pi
_H_FLOOR = 1e-3  # avoid h = 0 where s(x) = 0 (gibbs_s1)
_DEFAULT_MAX_BATCH_BYTES = 64 * 1024**2
_FLOAT_BYTES = np.dtype(float).itemsize


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
    "mm1_sojourn": mm1_sojourn_scale,
    "raised_floor_gauss": raised_floor_scale,
    "raised_floor_t3": raised_floor_scale,
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

def _validate_query_bandwidths(
    model,
    X_query: np.ndarray,
    h_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize query arrays and reject inconsistent bandwidth vectors."""
    X_arr = np.atleast_2d(np.asarray(X_query, dtype=float))
    h_arr = np.asarray(h_query, dtype=float).ravel()
    if X_arr.shape[0] != h_arr.size:
        raise ValueError(
            "h_query must contain one bandwidth per query point; "
            f"got {h_arr.size} bandwidths for {X_arr.shape[0]} points"
        )
    if model.indicator_type != "step" and (
        np.any(~np.isfinite(h_arr)) or np.any(h_arr <= 0.0)
    ):
        raise ValueError("Adaptive bandwidths must be finite and positive")
    return X_arr, h_arr


def _temporary_batch_shape(
    n_query: int,
    n_obs: int,
    n_sites: int,
    n_thresholds: int,
    max_batch_bytes: int,
) -> tuple[int, int]:
    """Choose query/threshold batch sizes for temporary indicator arrays.

    The returned shape bounds the main indicator tensor plus the replicated-
    site reduction. The final ``(n_query, n_thresholds)`` CDF output is not
    counted because it is required by the caller regardless of batching.
    """
    if max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be positive")

    # The extra n_obs term is a conservative allowance for ufunc work buffers.
    elements_per_query_threshold = 2 * n_obs + n_sites
    full_query_bytes = (
        elements_per_query_threshold * n_thresholds * _FLOAT_BYTES
    )
    if full_query_bytes <= max_batch_bytes:
        query_batch = max(1, max_batch_bytes // max(1, full_query_bytes))
        return min(n_query, query_batch), n_thresholds

    threshold_batch = max_batch_bytes // (
        max(1, elements_per_query_threshold) * _FLOAT_BYTES
    )
    return 1, max(1, min(n_thresholds, threshold_batch))


def _apply_indicator_transform(
    standardized: np.ndarray,
    indicator_type: str,
    h_batch: np.ndarray,
) -> np.ndarray:
    """Transform standardized ``(t-y)/h`` values in place when possible."""
    if indicator_type == "logistic":
        np.clip(standardized, -50.0, 50.0, out=standardized)
        np.negative(standardized, out=standardized)
        np.exp(standardized, out=standardized)
        standardized += 1.0
        np.reciprocal(standardized, out=standardized)
        return standardized

    if indicator_type == "gaussian_cdf":
        from scipy.special import ndtr

        try:
            ndtr(standardized, out=standardized)
        except TypeError:  # pragma: no cover - compatibility with old SciPy
            standardized[...] = ndtr(standardized)
        return standardized

    if indicator_type == "softplus":
        np.clip(standardized, -50.0, 50.0, out=standardized)
        np.exp(standardized, out=standardized)
        np.log1p(standardized, out=standardized)
        with np.errstate(over="ignore"):
            denominator = np.log1p(np.exp(1.0 / h_batch))
        reshape = (h_batch.size,) + (1,) * (standardized.ndim - 1)
        standardized /= denominator.reshape(reshape)
        return standardized

    raise ValueError(f"Unsupported vectorized indicator type: {indicator_type}")


def _adaptive_grid_indicator(
    indicator_type: str,
    Y_flat: np.ndarray,
    t_grid: np.ndarray,
    h_batch: np.ndarray,
) -> np.ndarray:
    """Return G[b, i, m] for query-specific bandwidths."""
    if indicator_type not in {"logistic", "gaussian_cdf", "softplus"}:
        # CKMEModel currently restricts indicator_type to the known families,
        # but retaining this fallback makes the utility safe for model-like
        # objects used in downstream experiments.
        rows = [
            make_indicator(indicator_type, float(h)).g_matrix(Y_flat, t_grid)
            for h in h_batch
        ]
        return np.stack(rows, axis=0)

    G = np.empty((h_batch.size, Y_flat.size, t_grid.size), dtype=float)
    np.subtract(
        t_grid[None, None, :],
        Y_flat[None, :, None],
        out=G,
    )
    np.divide(G, h_batch[:, None, None], out=G)
    return _apply_indicator_transform(G, indicator_type, h_batch)


def _adaptive_point_indicator(
    indicator_type: str,
    Y_flat: np.ndarray,
    Y_query: np.ndarray,
    h_batch: np.ndarray,
) -> np.ndarray:
    """Return G[b, i] at one query-specific response threshold per point."""
    if indicator_type == "step":
        return (Y_flat[None, :] <= Y_query[:, None]).astype(float)
    if indicator_type not in {"logistic", "gaussian_cdf", "softplus"}:
        rows = [
            make_indicator(indicator_type, float(h)).g_vector(Y_flat, float(y))
            for y, h in zip(Y_query, h_batch)
        ]
        return np.stack(rows, axis=0)

    G = np.empty((h_batch.size, Y_flat.size), dtype=float)
    np.subtract(Y_query[:, None], Y_flat[None, :], out=G)
    np.divide(G, h_batch[:, None], out=G)
    return _apply_indicator_transform(G, indicator_type, h_batch)


def _raw_point_scores_batched(
    model,
    X_pts: np.ndarray,
    Y_pts: np.ndarray,
    h_pts: np.ndarray,
    max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
) -> np.ndarray:
    """Evaluate raw point scores in memory-bounded query batches."""
    X_arr, h_arr = _validate_query_bandwidths(model, X_pts, h_pts)
    Y_arr = np.asarray(Y_pts, dtype=float).ravel()
    if X_arr.shape[0] != Y_arr.size:
        raise ValueError(
            "Y_pts must contain one response per query point; "
            f"got {Y_arr.size} responses for {X_arr.shape[0]} points"
        )
    if max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be positive")

    n_pts = Y_arr.size
    if n_pts == 0:
        return np.empty(0, dtype=float)

    Y_flat = np.asarray(model.Y, dtype=float).ravel()
    n_sites = int(model.n)
    r = int(getattr(model, "r", 1))
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_arr)
    bytes_per_query = (2 * Y_flat.size + n_sites) * _FLOAT_BYTES
    query_batch = max(1, max_batch_bytes // max(1, bytes_per_query))

    scores = np.empty(n_pts, dtype=float)
    for start in range(0, n_pts, query_batch):
        stop = min(start + query_batch, n_pts)
        G = _adaptive_point_indicator(
            model.indicator_type,
            Y_flat,
            Y_arr[start:stop],
            h_arr[start:stop],
        )
        if r > 1:
            G_site = G.reshape(stop - start, n_sites, r).mean(axis=2)
        else:
            G_site = G
        for local, point in enumerate(range(start, stop)):
            F_value = float(C[:, point] @ G_site[local])
            scores[point] = abs(float(np.clip(F_value, 0.0, 1.0)) - 0.5)
    return scores

def _projected_point_scores(
    model,
    X_pts: np.ndarray,
    Y_pts: np.ndarray,
    h_pts: np.ndarray,
    t_grid: np.ndarray | None,
) -> np.ndarray:
    """DCP scores |Ftilde(y|x) - 1/2| with per-point adaptive h.

    ``t_grid=None`` evaluates the locked guarantee-bearing raw point score.
    When ``t_grid`` is supplied, the historical Policy B score is retained:
    Ftilde(y) = clip(max(raw F at y, max_{t_k <= y} raw F(t_k)), 0, 1).
    """
    if t_grid is None:
        return _raw_point_scores_batched(model, X_pts, Y_pts, h_pts)

    X_pts, h_pts = _validate_query_bandwidths(model, X_pts, h_pts)
    Y_pts = np.asarray(Y_pts, dtype=float).ravel()
    if X_pts.shape[0] != Y_pts.size:
        raise ValueError(
            "Y_pts must contain one response per query point; "
            f"got {Y_pts.size} responses for {X_pts.shape[0]} points"
        )
    n_pts = len(Y_pts)

    Y_flat = model.Y.ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_pts)
    t_arr = np.asarray(t_grid, dtype=float).ravel()

    scores = np.empty(n_pts)
    for j in range(n_pts):
        ind_j = make_indicator(model.indicator_type, float(h_pts[j]))
        thresholds = np.append(t_arr, float(Y_pts[j]))
        G_j = ind_j.g_matrix(Y_flat, thresholds)
        if getattr(model, "r", 1) > 1:
            G_site = G_j.reshape(model.n, model.r, -1).mean(axis=1)
        else:
            G_site = G_j
        F_vals = C[:, j] @ G_site
        raw_at_y = F_vals[-1]
        pos = int(np.searchsorted(t_arr, float(Y_pts[j]), side="right"))
        grid_part = F_vals[:pos].max() if pos > 0 else -np.inf
        F_j = float(np.clip(max(raw_at_y, grid_part), 0.0, 1.0))
        scores[j] = abs(F_j - 0.5)
    return scores


def adaptive_point_scores(
    model,
    X_pts: np.ndarray,
    Y_pts: np.ndarray,
    h_pts: np.ndarray,
    *,
    max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
) -> np.ndarray:
    """Return raw DCP point scores with one adaptive bandwidth per point.

    These scores evaluate ``|clip(F(Y_i | X_i), 0, 1) - 1/2|`` directly at
    each observed response. They are the guarantee-bearing score layer in the
    locked protocol. Monotone projection is reserved for interval reporting.

    ``max_batch_bytes`` bounds temporary indicator arrays; the coefficient
    matrix and returned score vector are not included in that limit.
    """
    return _raw_point_scores_batched(
        model,
        X_pts,
        Y_pts,
        h_pts,
        max_batch_bytes=max_batch_bytes,
    )


def adaptive_recalibrate_q(
    model,
    X_cal: np.ndarray,
    Y_cal: np.ndarray,
    h_cal: np.ndarray,
    alpha: float,
    t_grid: np.ndarray | None = None,
) -> float:
    """Recompute split-CP q_hat using per-point adaptive h on calibration data.

    The default ``t_grid=None`` uses raw point scores, as required by the
    guarantee layer in the locked protocol. Passing ``t_grid`` retains the
    historical monotone-projected score path for reproducibility only.
    """
    Y_cal = np.asarray(Y_cal).ravel()
    n_cal = len(Y_cal)
    scores = _projected_point_scores(model, X_cal, Y_cal, h_cal, t_grid)
    k = int(np.ceil((1 - alpha) * (1 + n_cal)))
    if k > n_cal:
        return float("inf")  # split-CP edge case: predict the whole space
    return float(np.sort(scores)[k - 1])


def adaptive_predict_cdf(
    model,
    X_query: np.ndarray,
    h_query: np.ndarray,
    t_grid: np.ndarray,
    *,
    clip: bool = False,
    max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
) -> np.ndarray:
    """Predict adaptive-bandwidth CDF rows in memory-bounded batches.

    The CKME coefficient matrix is computed once. Query-specific indicator
    tensors are then evaluated in query batches when a full threshold grid
    fits in the temporary-memory budget, or in threshold chunks when even one
    full row would exceed it.

    Parameters
    ----------
    model
        Fitted ``CKMEModel`` or compatible object.
    X_query, h_query
        Query points and one adaptive bandwidth per point.
    t_grid
        Threshold grid for CDF evaluation.
    clip
        Clip the raw CKME CDF to ``[0, 1]`` when true. The default preserves
        the raw values used by the canonical monotone-projection helper.
    max_batch_bytes
        Approximate cap for temporary indicator arrays. The required returned
        CDF matrix and CKME coefficient matrix are not included.
    """
    X_arr, h_arr = _validate_query_bandwidths(model, X_query, h_query)
    t_arr = np.asarray(t_grid, dtype=float).ravel()
    if t_arr.size == 0:
        raise ValueError("t_grid must contain at least one threshold")
    if max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be positive")

    n_query = X_arr.shape[0]
    if n_query == 0:
        return np.empty((0, t_arr.size), dtype=float)

    Y_flat = np.asarray(model.Y, dtype=float).ravel()
    n_sites = int(model.n)
    r = int(getattr(model, "r", 1))
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_arr)
    F_all = np.empty((n_query, t_arr.size), dtype=float)

    if model.indicator_type == "step":
        # h has no effect for the exact step indicator, so share G across all
        # query points and only chunk the threshold axis.
        elements_per_threshold = Y_flat.size + n_sites
        threshold_batch = max(
            1,
            max_batch_bytes // max(1, elements_per_threshold * _FLOAT_BYTES),
        )
        for threshold_start in range(0, t_arr.size, threshold_batch):
            threshold_stop = min(
                threshold_start + threshold_batch,
                t_arr.size,
            )
            t_batch = t_arr[threshold_start:threshold_stop]
            G = (Y_flat[:, None] <= t_batch[None, :]).astype(float)
            if r > 1:
                G_site = G.reshape(n_sites, r, -1).mean(axis=1)
            else:
                G_site = G
            F_all[:, threshold_start:threshold_stop] = C.T @ G_site
    else:
        query_batch, threshold_batch = _temporary_batch_shape(
            n_query,
            Y_flat.size,
            n_sites,
            t_arr.size,
            max_batch_bytes,
        )
        for query_start in range(0, n_query, query_batch):
            query_stop = min(query_start + query_batch, n_query)
            for threshold_start in range(0, t_arr.size, threshold_batch):
                threshold_stop = min(
                    threshold_start + threshold_batch,
                    t_arr.size,
                )
                G = _adaptive_grid_indicator(
                    model.indicator_type,
                    Y_flat,
                    t_arr[threshold_start:threshold_stop],
                    h_arr[query_start:query_stop],
                )
                if r > 1:
                    G_site = G.reshape(
                        query_stop - query_start,
                        n_sites,
                        r,
                        threshold_stop - threshold_start,
                    ).mean(axis=2)
                else:
                    G_site = G
                for local, point in enumerate(range(query_start, query_stop)):
                    F_all[
                        point,
                        threshold_start:threshold_stop,
                    ] = C[:, point] @ G_site[local]

    if clip:
        np.clip(F_all, 0.0, 1.0, out=F_all)
    return F_all


def adaptive_predict_interval(
    model,
    X_query: np.ndarray,
    h_query: np.ndarray,
    t_grid: np.ndarray,
    q_hat: float,
    *,
    max_batch_bytes: int = _DEFAULT_MAX_BATCH_BYTES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict [L, U] with adaptive h and canonical CDF projection."""
    F_all = adaptive_predict_cdf(
        model,
        X_query,
        h_query,
        t_grid,
        clip=False,
        max_batch_bytes=max_batch_bytes,
    )
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

    The default ``t_grid=None`` uses raw point scores, as required by the
    guarantee layer in the locked protocol. Passing ``t_grid`` retains the
    historical monotone-projected score path for reproducibility only.
    """
    scores = _projected_point_scores(model, X_test, Y_test, h_test, t_grid)
    return (scores <= q_hat).astype(int)
