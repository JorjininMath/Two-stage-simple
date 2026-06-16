"""Metrics for the KME/CKME feasibility experiment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .ckme_cdf import monotonicity_violation
from .dgp import Scenario
from .features import rho_hat


QUANTILE_LEVELS = (0.1, 0.5, 0.9)
INPUT_SIZE_GROUPS = (20, 50, 200, 1000)


def empirical_cdf(values: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Empirical CDF evaluated on a fixed threshold grid."""

    values = np.asarray(values, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    return np.mean(values[:, None] <= t_grid[None, :], axis=0)


def cdf_quantiles(cdf_values: np.ndarray, t_grid: np.ndarray, levels: Sequence[float] = QUANTILE_LEVELS) -> np.ndarray:
    """Invert CDF curves on the supplied grid."""

    cdf_values = np.asarray(cdf_values, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    out = np.empty((cdf_values.shape[0], len(levels)), dtype=float)
    for i, curve in enumerate(cdf_values):
        for j, level in enumerate(levels):
            idx = int(np.searchsorted(curve, level, side="left"))
            idx = min(max(idx, 0), t_grid.size - 1)
            out[i, j] = t_grid[idx]
    return out


def standard_error(values: np.ndarray) -> float:
    """NaN-safe standard error."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def scenario_diagnostics(
    pred_cdf: np.ndarray,
    raw_cdf: np.ndarray,
    oracle_cdf: np.ndarray,
    t_grid: np.ndarray,
    scenarios: Sequence[Scenario],
) -> dict[str, np.ndarray]:
    """Compute per-scenario feasibility diagnostics."""

    ise = np.mean((pred_cdf - oracle_cdf) ** 2, axis=1)
    iae = np.mean(np.abs(pred_cdf - oracle_cdf), axis=1)
    pred_q = cdf_quantiles(pred_cdf, t_grid)
    oracle_q = cdf_quantiles(oracle_cdf, t_grid)
    qerr = np.abs(pred_q - oracle_q)
    return {
        "ise": ise,
        "iae": iae,
        "qerr_0.1": qerr[:, 0],
        "qerr_0.5": qerr[:, 1],
        "qerr_0.9": qerr[:, 2],
        "mono_violation": monotonicity_violation(raw_cdf),
        "n1": np.array([s.n_a for s in scenarios], dtype=int),
        "n2": np.array([s.n_s for s in scenarios], dtype=int),
        "rho_hat": np.array([rho_hat(s.inter_arrivals, s.services) for s in scenarios], dtype=float),
        "true_rho": np.array([s.rho for s in scenarios], dtype=float),
    }


def _add_metric_summary(row: dict[str, object], prefix: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    row[f"{prefix}_mean"] = float(np.mean(values)) if values.size else np.nan
    row[f"{prefix}_se"] = standard_error(values) if values.size else np.nan


def build_metrics_row(
    seed: int,
    dgp: str,
    n_fit: int,
    n_val: int,
    n_test: int,
    r_train: int,
    r_oracle: int,
    rff_dim: int,
    grid_size: int,
    selected_tau: float,
    selected_ridge: float,
    val_mse: float,
    diagnostics: dict[str, np.ndarray],
) -> dict[str, object]:
    """Build one wide metrics row for a seed."""

    row: dict[str, object] = {
        "seed": seed,
        "dgp": dgp,
        "n_fit": n_fit,
        "n_val": n_val,
        "n_test": n_test,
        "r_train": r_train,
        "r_oracle": r_oracle,
        "rff_dim": rff_dim,
        "grid_size": grid_size,
        "selected_tau": selected_tau,
        "selected_ridge": selected_ridge,
        "val_mse": val_mse,
    }
    for key in ("ise", "iae", "qerr_0.1", "qerr_0.5", "qerr_0.9", "mono_violation"):
        _add_metric_summary(row, key, diagnostics[key])

    n1 = diagnostics["n1"]
    true_rho = diagnostics["true_rho"]
    for size in INPUT_SIZE_GROUPS:
        mask = n1 == size
        row[f"count_n{size}"] = int(np.sum(mask))
        if np.any(mask):
            for key in ("ise", "iae", "qerr_0.1", "qerr_0.5", "qerr_0.9"):
                row[f"{key}_n{size}_mean"] = float(np.mean(diagnostics[key][mask]))
        else:
            for key in ("ise", "iae", "qerr_0.1", "qerr_0.5", "qerr_0.9"):
                row[f"{key}_n{size}_mean"] = np.nan

    traffic_masks = {
        "light": true_rho <= 0.75,
        "heavy": true_rho > 0.75,
    }
    for name, mask in traffic_masks.items():
        row[f"count_{name}"] = int(np.sum(mask))
        if np.any(mask):
            for key in ("ise", "iae", "qerr_0.1", "qerr_0.5", "qerr_0.9"):
                row[f"{key}_{name}_mean"] = float(np.mean(diagnostics[key][mask]))
        else:
            for key in ("ise", "iae", "qerr_0.1", "qerr_0.5", "qerr_0.9"):
                row[f"{key}_{name}_mean"] = np.nan
    return row
