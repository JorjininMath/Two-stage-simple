"""
run_homogeneity.py

Score homogeneity mechanism experiment for CKME-CP.

Tests whether adaptive bandwidth improves conditional coverage primarily
by homogenizing the conformal score distribution across x, rather than
by improving CDF/quantile estimation accuracy.

For each simulator x macrorep x bandwidth configuration:
  1. Train CKME (Stage 1) with shared (ell_x, lam)
  2. Collect Stage 2 calibration data, calibrate CP
  3. At each eval x_m:
     a. Compute score ECDF from B fresh MC draws
     b. Compute MC conditional coverage
     c. Compute pre-CP quantile estimates
  4. Compute homogeneity metrics from score ECDFs

Usage:
  python exp_score_homogeneity/run_homogeneity.py
  python exp_score_homogeneity/run_homogeneity.py --simulators exp2
  python exp_score_homogeneity/run_homogeneity.py --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm, gamma as _gamma

from Two_stage import run_stage1_train, run_stage2
from Two_stage.sim_functions import get_experiment_config
from Two_stage.sim_functions.exp2 import (
    exp2_true_function, exp2_noise_variance_function,
)
from Two_stage.sim_functions.sim_nongauss_B2 import (
    _sigma_tar as _B2_sigma_tar, nongauss_B2_noise_variance,
)
from CKME.parameters import Params
from CKME.coefficients import compute_ckme_coeffs
from CKME.indicators import make_indicator

_HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Mild heteroscedastic simulator (Phase 1)
# ---------------------------------------------------------------------------
# σ(x) = 0.3 * (1 + 0.5 * sin(x)),  x ∈ [0, 2π]
# σ ratio ≈ 3x (0.15 to 0.45), smooth, no near-zero regions
# Same f(x) as exp2: f(x) = exp(x/10) * sin(x)

_MILD_X_BOUNDS = (np.array([0.0]), np.array([2 * np.pi]))


def _mild_sigma(x: np.ndarray) -> np.ndarray:
    """σ(x) for mild heteroscedastic simulator."""
    return 0.3 * (1.0 + 0.5 * np.sin(x))


def _mild_variance(x: np.ndarray) -> np.ndarray:
    return _mild_sigma(np.asarray(x, dtype=float)) ** 2


def _mild_simulator(x, random_state=None):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    y_true = exp2_true_function(x)
    return y_true + _mild_sigma(x) * rng.standard_normal(x.shape)


def _register_mild_simulator():
    """Register mild simulator in the global registry (once)."""
    from Two_stage.sim_functions import _EXPERIMENT_REGISTRY
    if "mild" not in _EXPERIMENT_REGISTRY:
        _EXPERIMENT_REGISTRY["mild"] = {
            "simulator": _mild_simulator,
            "bounds": _MILD_X_BOUNDS,
            "d": 1,
        }


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------

def _mild_oracle_quantile(x: np.ndarray, tau: float) -> np.ndarray:
    mu = exp2_true_function(x)
    sigma = _mild_sigma(x)
    return mu + sigma * _norm.ppf(tau)


def _mild_oracle_cdf(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    mu = exp2_true_function(x)[:, None]
    sigma = _mild_sigma(x)[:, None]
    return _norm.cdf((t[None, :] - mu) / sigma)


def _exp2_oracle_quantile(x: np.ndarray, tau: float) -> np.ndarray:
    mu = exp2_true_function(x)
    sigma = np.sqrt(exp2_noise_variance_function(x))
    return mu + sigma * _norm.ppf(tau)


def _exp2_oracle_cdf(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Oracle CDF F(t|x) for exp2. Returns shape (len(x), len(t))."""
    mu = exp2_true_function(x)[:, None]
    sigma = np.sqrt(exp2_noise_variance_function(x))[:, None]
    return _norm.cdf((t[None, :] - mu) / sigma)


def _B2L_oracle_quantile(x: np.ndarray, tau: float, k: float = 2.0) -> np.ndarray:
    """Oracle quantile for centered Gamma (B2L)."""
    mu = exp2_true_function(x)
    sigma_tar = _B2_sigma_tar(x)
    theta = sigma_tar / np.sqrt(k)
    # Y = f(x) + Gamma(k, theta) - k*theta
    return mu + _gamma.ppf(tau, a=k, scale=theta) - k * theta


def _B2L_oracle_cdf(x: np.ndarray, t: np.ndarray, k: float = 2.0) -> np.ndarray:
    """Oracle CDF for centered Gamma (B2L). Returns shape (len(x), len(t))."""
    mu = exp2_true_function(x)[:, None]
    sigma_tar = _B2_sigma_tar(x)[:, None]
    theta = sigma_tar / np.sqrt(k)
    # Y = f(x) + G - k*theta, G~Gamma(k, theta)
    # F(t|x) = P(G <= t - f(x) + k*theta) = GammaCDF(t - f(x) + k*theta; k, theta)
    g_arg = t[None, :] - mu + k * theta
    return _gamma.cdf(np.maximum(g_arg, 0.0), a=k, scale=theta)


_ORACLE_QUANTILE = {
    "mild": _mild_oracle_quantile,
    "exp2": _exp2_oracle_quantile,
    "nongauss_B2L": _B2L_oracle_quantile,
}

_ORACLE_CDF = {
    "mild": _mild_oracle_cdf,
    "exp2": _exp2_oracle_cdf,
    "nongauss_B2L": _B2L_oracle_cdf,
}

# Oracle conditional variance (for adaptive h)
_ORACLE_VAR = {
    "mild": _mild_variance,
    "exp2": exp2_noise_variance_function,
    "nongauss_B2L": lambda x: nongauss_B2_noise_variance(x, k=2.0),
}


# ---------------------------------------------------------------------------
# Adaptive-h and CDF computation (adapted from run_consistency.py)
# ---------------------------------------------------------------------------

def _adaptive_h_oracle(x: np.ndarray, c_scale: float,
                       oracle_var_fn) -> np.ndarray:
    """h(x) = c_scale * sigma(x) using oracle variance."""
    sigma = np.sqrt(np.maximum(oracle_var_fn(x), 1e-8))
    return c_scale * sigma


def _batch_cdf_on_tgrid(model, t_grid: np.ndarray,
                         C_eval: np.ndarray) -> np.ndarray:
    """F̂(t|x_m) for all M eval points with model's fixed h. Returns (M, T)."""
    Y_flat = model.Y.ravel()
    T = len(t_grid)
    G_mat = model.indicator.g_matrix(Y_flat, t_grid)
    G_site = G_mat.reshape(model.n, model.r, T).mean(axis=1)
    return np.clip(C_eval.T @ G_site, 0.0, 1.0)


def _perpoint_cdf_on_tgrid(model, t_grid: np.ndarray, C_eval: np.ndarray,
                            h_vals: np.ndarray) -> np.ndarray:
    """F̂(t|x_m) with per-point bandwidth. Returns (M, T)."""
    Y_flat = model.Y.ravel()
    T = len(t_grid)
    M = C_eval.shape[1]
    F_all = np.empty((M, T))
    for m in range(M):
        ind_m = make_indicator(model.indicator_type, float(h_vals[m]))
        G_mat = ind_m.g_matrix(Y_flat, t_grid)
        G_site = G_mat.reshape(model.n, model.r, T).mean(axis=1)
        F_all[m] = np.clip(C_eval[:, m] @ G_site, 0.0, 1.0)
    return F_all


def _invert_cdf(F_all: np.ndarray, t_grid: np.ndarray,
                tau: float) -> np.ndarray:
    """q̂_τ(x_m) = inf{ t : F_all[m, :] >= tau }."""
    T = len(t_grid)
    M = F_all.shape[0]
    q_arr = np.empty(M)
    for m in range(M):
        idx = np.searchsorted(F_all[m], tau)
        q_arr[m] = t_grid[min(idx, T - 1)]
    return q_arr


def _calibrate_scores(model, X_cal, Y_cal, h_mode, h_val_or_fn, alpha):
    """Compute CP calibration quantile q_hat.

    h_val_or_fn: scalar h (fixed) or array of per-point h (adaptive).
    """
    X_cal = np.atleast_2d(X_cal)
    Y_cal = np.asarray(Y_cal).ravel()
    n_cal = len(Y_cal)

    C_cal = compute_ckme_coeffs(model.L, model.kx, model.X, X_cal)
    Y_flat = model.Y.ravel()

    scores = np.empty(n_cal)
    for j in range(n_cal):
        if h_mode == "adaptive":
            ind_j = make_indicator(model.indicator_type, float(h_val_or_fn[j]))
        else:
            ind_j = model.indicator
        g_j = ind_j.g_matrix(Y_flat, np.array([float(Y_cal[j])]))[:, 0]
        g_site = g_j.reshape(model.n, model.r).mean(axis=1)
        F_j = float(np.clip(C_cal[:, j] @ g_site, 0.0, 1.0))
        scores[j] = abs(F_j - 0.5)

    k = int(np.ceil((1 - alpha) * (1 + n_cal)))
    k = min(k, n_cal)
    q_hat = float(np.sort(scores)[k - 1])
    return q_hat


# ---------------------------------------------------------------------------
# Score ECDF computation
# ---------------------------------------------------------------------------

def _compute_score_ecdf_at_x(model, C_m, x_m_h, Y_mc, h_mode):
    """Compute B scores s = |F̂(Y_b|x_m) - 0.5| for MC draws at one x_m.

    C_m: shape (n,) coefficients for x_m
    x_m_h: scalar bandwidth at x_m
    Y_mc: shape (B,) MC draws of Y|x_m
    Returns: scores shape (B,)
    """
    Y_flat = model.Y.ravel()
    B = len(Y_mc)
    B_chunk = 200

    if h_mode == "adaptive":
        ind = make_indicator(model.indicator_type, float(x_m_h))
    else:
        ind = model.indicator

    F_b = np.empty(B)
    for b0 in range(0, B, B_chunk):
        Y_chunk = Y_mc[b0:b0 + B_chunk]
        G_chunk = ind.g_matrix(Y_flat, Y_chunk)
        G_bar = G_chunk.reshape(model.n, model.r, -1).mean(axis=1)
        F_b[b0:b0 + B_chunk] = C_m @ G_bar

    np.clip(F_b, 0.0, 1.0, out=F_b)
    return np.abs(F_b - 0.5)


def _ks_from_pooled(score_ecdfs: list[np.ndarray]) -> tuple[float, float, np.ndarray]:
    """Compute KS homogeneity metrics from per-x score samples.

    score_ecdfs: list of M arrays, each shape (B,) of score values.
    Returns: (ks_max, ks_mean, per_x_ks array of shape (M,))
    """
    # Pooled ECDF: combine all scores
    all_scores = np.concatenate(score_ecdfs)
    all_scores_sorted = np.sort(all_scores)
    N_pool = len(all_scores_sorted)

    M = len(score_ecdfs)
    per_x_ks = np.empty(M)

    for m in range(M):
        s_m = np.sort(score_ecdfs[m])
        B_m = len(s_m)
        # Evaluate both ECDFs at the union of all score values
        # For efficiency, use the KS statistic formula directly
        # KS = sup_t |F_m(t) - F_pool(t)|
        # Evaluate at the sorted values of s_m
        F_pool_at_sm = np.searchsorted(all_scores_sorted, s_m, side='right') / N_pool
        F_m_at_sm = np.arange(1, B_m + 1) / B_m
        ks1 = np.max(np.abs(F_m_at_sm - F_pool_at_sm))
        # Also check at left-continuous points
        F_m_left = np.arange(0, B_m) / B_m
        ks2 = np.max(np.abs(F_m_left - F_pool_at_sm))
        per_x_ks[m] = max(ks1, ks2)

    return float(per_x_ks.max()), float(per_x_ks.mean()), per_x_ks


# ---------------------------------------------------------------------------
# Core: one macrorep x one bandwidth config
# ---------------------------------------------------------------------------

def run_one_config(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    shared_params: Params,
    bw_label: str,
    h_fixed: float | None,
    c_scale: float | None,
    oracle_var_fn,
    oracle_quantile_fn,
    oracle_cdf_fn,
) -> tuple[list[dict], list[np.ndarray]]:
    """Run one macrorep for one bandwidth config.

    Returns:
      rows: list of per-eval-point dicts
      score_samples: list of M arrays, each (B,) scores at x_m
    """
    seed = base_seed + macrorep_id * 10000

    alpha = config["alpha"]
    n_0 = config["n_0"]
    r_0 = config["r_0"]
    n_1 = config["n_1"]
    r_1 = config["r_1"]
    M = config["M_eval"]
    B = config["B_test"]
    n_cand = config["n_cand"]
    method = config["method"]
    mixed_ratio = config["mixed_ratio"]
    t_grid_size = config["t_grid_size"]

    exp_cfg = get_experiment_config(simulator_func)
    bounds = exp_cfg["bounds"]
    sim_fn = exp_cfg["simulator"]
    x_lo = float(bounds[0][0])
    x_hi = float(bounds[1][0])

    x_eval = np.linspace(x_lo, x_hi, M)
    X_eval = x_eval.reshape(-1, 1)

    rng_seed = seed
    rng_cand = np.random.default_rng(rng_seed)
    n_cand_eff = max(n_cand, 2 * n_0)
    X_cand = rng_cand.uniform(x_lo, x_hi, size=(n_cand_eff, 1))

    # Determine h for training: use CV h for adaptive/fixed_cv, else use h_fixed
    if c_scale is not None:
        # Adaptive: train with shared_params (h is placeholder, not used for eval)
        train_params = shared_params
        h_mode = "adaptive"
    elif h_fixed is not None:
        train_params = Params(ell_x=shared_params.ell_x, lam=shared_params.lam,
                              h=h_fixed)
        h_mode = "fixed"
    else:
        # fixed_cv: use shared_params as-is
        train_params = shared_params
        h_mode = "fixed"

    # --- Stage 1 ---
    stage1 = run_stage1_train(
        n_0=n_0, r_0=r_0,
        simulator_func=simulator_func,
        params=train_params,
        t_grid_size=t_grid_size,
        random_state=rng_seed + 1,
    )

    # --- Stage 2: collect calibration data ---
    stage2 = run_stage2(
        stage1_result=stage1,
        X_cand=X_cand,
        n_1=n_1, r_1=r_1,
        simulator_func=simulator_func,
        method=method,
        alpha=alpha,
        mixed_ratio=mixed_ratio,
        random_state=rng_seed + 2,
    )

    model = stage2.model
    t_grid = stage1.t_grid

    # --- Bandwidth at eval and calibration points ---
    C_eval = compute_ckme_coeffs(model.L, model.kx, model.X, X_eval)

    if h_mode == "adaptive":
        h_vals = _adaptive_h_oracle(x_eval, c_scale, oracle_var_fn)
        # Re-calibrate CP with adaptive h at calibration points
        X_cal = np.atleast_2d(stage2.X_stage2)
        Y_cal = np.asarray(stage2.Y_stage2).ravel()
        h_cal = _adaptive_h_oracle(X_cal.ravel(), c_scale, oracle_var_fn)
        q_hat = _calibrate_scores(model, X_cal, Y_cal, "adaptive", h_cal, alpha)
    else:
        h_scalar = float(train_params.h)
        h_vals = np.full(M, h_scalar)
        q_hat = stage2.cp.q_hat

    # --- CDF on t_grid ---
    if h_mode == "adaptive":
        F_tgrid = _perpoint_cdf_on_tgrid(model, t_grid, C_eval, h_vals)
    else:
        F_tgrid = _batch_cdf_on_tgrid(model, t_grid, C_eval)

    # --- L1: pre-CP quantile estimates ---
    q_lo_hat = _invert_cdf(F_tgrid, t_grid, alpha / 2)
    q_hi_hat = _invert_cdf(F_tgrid, t_grid, 1 - alpha / 2)

    # --- Oracle quantiles ---
    q_lo_oracle = oracle_quantile_fn(x_eval, alpha / 2)
    q_hi_oracle = oracle_quantile_fn(x_eval, 1 - alpha / 2)

    # --- CDF error (mean over t_grid) ---
    F_oracle = oracle_cdf_fn(x_eval, t_grid)  # (M, T)
    cdf_err_per_x = np.mean(np.abs(F_tgrid - F_oracle), axis=1)  # (M,)

    # --- CP interval bounds ---
    tau_lo = float(np.clip(0.5 - q_hat, 0.0, 1.0))
    tau_hi = float(np.clip(0.5 + q_hat, 0.0, 1.0))
    L_arr = _invert_cdf(F_tgrid, t_grid, tau_lo)
    U_arr = _invert_cdf(F_tgrid, t_grid, tau_hi)

    # --- Score ECDF + MC coverage at each eval point ---
    score_samples = []
    cov_mc = np.empty(M)

    for m in range(M):
        Y_mc = sim_fn(np.full(B, x_eval[m]), random_state=rng_seed + 3 + m)
        scores_m = _compute_score_ecdf_at_x(
            model, C_eval[:, m], h_vals[m], Y_mc, h_mode
        )
        score_samples.append(scores_m)
        cov_mc[m] = np.mean(scores_m <= q_hat)

    # --- Homogeneity metrics ---
    ks_max, ks_mean, ks_per_x = _ks_from_pooled(score_samples)

    # --- Build row dicts ---
    rows = []
    for m in range(M):
        rows.append({
            "macrorep": macrorep_id,
            "bandwidth": bw_label,
            "x_eval": float(x_eval[m]),
            "h_at_x": float(h_vals[m]),
            "cov_mc": float(cov_mc[m]),
            "L": float(L_arr[m]),
            "U": float(U_arr[m]),
            "q_hat": float(q_hat),
            "q_lo_hat": float(q_lo_hat[m]),
            "q_hi_hat": float(q_hi_hat[m]),
            "q_lo_oracle": float(q_lo_oracle[m]),
            "q_hi_oracle": float(q_hi_oracle[m]),
            "ks_from_pooled": float(ks_per_x[m]),
            "cdf_err": float(cdf_err_per_x[m]),
        })

    return rows, score_samples


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    target = 1.0 - alpha
    records = []

    for bw, g_bw in df.groupby("bandwidth"):
        for macro_id, g_m in g_bw.groupby("macrorep"):
            vals = g_m.sort_values("x_eval")

            # Coverage gap
            delta_cov = np.abs(vals["cov_mc"].values - target)
            cov_gap_sup = delta_cov.max()
            cov_gap_mae = delta_cov.mean()

            # Width
            widths = vals["U"].values - vals["L"].values
            width_mean = widths.mean()
            width_std_x = widths.std(ddof=1) if len(widths) > 1 else 0.0

            # Quantile error
            err_lo = np.abs(vals["q_lo_hat"].values - vals["q_lo_oracle"].values)
            err_hi = np.abs(vals["q_hi_hat"].values - vals["q_hi_oracle"].values)
            q_err_sup = max(err_lo.max(), err_hi.max())
            q_err_mae_lo = err_lo.mean()
            q_err_mae_hi = err_hi.mean()

            # CDF error
            cdf_err_mean = vals["cdf_err"].values.mean()

            # Score homogeneity
            ks_max = vals["ks_from_pooled"].values.max()
            ks_mean_val = vals["ks_from_pooled"].values.mean()

            records.append({
                "bandwidth": bw,
                "macrorep": macro_id,
                "ks_max": ks_max,
                "ks_mean": ks_mean_val,
                "cov_gap_sup": cov_gap_sup,
                "cov_gap_mae": cov_gap_mae,
                "width_mean": width_mean,
                "width_std_x": width_std_x,
                "q_err_sup": q_err_sup,
                "q_err_mae_lo": q_err_mae_lo,
                "q_err_mae_hi": q_err_mae_hi,
                "cdf_err_mean": cdf_err_mean,
            })

    per_macro = pd.DataFrame(records)
    agg_rows = []
    for bw, g in per_macro.groupby("bandwidth"):
        row = {"bandwidth": bw, "n_macro": len(g)}
        for col in ["ks_max", "ks_mean",
                     "cov_gap_sup", "cov_gap_mae",
                     "width_mean", "width_std_x",
                     "q_err_sup", "q_err_mae_lo", "q_err_mae_hi",
                     "cdf_err_mean"]:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_sd"] = g[col].std(ddof=1) if len(g) > 1 else np.nan
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def get_config(cfg_path: Path | None = None) -> dict:
    path = cfg_path or (_HERE / "config.txt")
    raw = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        raw[k.strip()] = v.strip()

    return {
        "n_0":         int(raw["n_0"]),
        "r_0":         int(raw["r_0"]),
        "n_1":         int(raw["n_1"]),
        "r_1":         int(raw["r_1"]),
        "alpha":       float(raw["alpha"]),
        "t_grid_size": int(raw["t_grid_size"]),
        "method":      raw["method"],
        "mixed_ratio": float(raw["mixed_ratio"]),
        "n_cand":      int(raw["n_cand"]),
        "M_eval":      int(raw["M_eval"]),
        "B_test":      int(raw["B_test"]),
        "n_macro":     int(raw["n_macro"]),
        "n_jobs":      int(raw.get("n_jobs", "4")),
        "base_seed":   int(raw["base_seed"]),
        "simulators":  [s.strip() for s in raw["simulators"].split(",")],
        "h_fixed_small": float(raw["h_fixed_small"]),
        "h_fixed_large": float(raw["h_fixed_large"]),
        "c_values":    [float(v) for v in raw["c_values"].split(",")],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Score homogeneity mechanism experiment for CKME-CP")
    parser.add_argument("--simulators", type=str, default=None)
    parser.add_argument("--n_macro", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="Fast test: n_0=n_1=64, n_macro=2, B_test=200, M_eval=10")
    args = parser.parse_args()

    # Register mild simulator before anything else
    _register_mild_simulator()

    cfg_path = Path(args.config) if args.config else None
    config = get_config(cfg_path)

    if args.simulators is not None:
        config["simulators"] = [s.strip() for s in args.simulators.split(",")]
    if args.n_macro is not None:
        config["n_macro"] = args.n_macro
    if args.base_seed is not None:
        config["base_seed"] = args.base_seed

    if args.quick:
        config["n_0"] = 64
        config["n_1"] = 64
        config["n_macro"] = 2
        config["B_test"] = 200
        config["M_eval"] = 10
        config["c_values"] = [1.0, 2.0]
        print("Quick mode: n_0=n_1=64, n_macro=2, B_test=200, M_eval=10, c=[1.0,2.0]")

    n_macro = config["n_macro"]
    out_base = Path(args.output_dir) if args.output_dir else (_HERE / "output")
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"simulators : {config['simulators']}")
    print(f"n_0={config['n_0']}, r_0={config['r_0']}, "
          f"n_1={config['n_1']}, r_1={config['r_1']}, alpha={config['alpha']}")
    print(f"n_macro={n_macro}, M_eval={config['M_eval']}, B_test={config['B_test']}")
    print(f"h_fixed: small={config['h_fixed_small']}, large={config['h_fixed_large']}")
    print(f"c_values: {config['c_values']}")
    print(f"Output dir : {out_base}")

    for sim in config["simulators"]:
        print(f"\n{'='*60}")
        print(f"Simulator: {sim}")
        print(f"{'='*60}")

        # --- Load pretrained params (shared ell_x, lam; h used for fixed_cv) ---
        pretrained_path = Path(args.pretrained) if args.pretrained \
            else (_HERE / "pretrained_params.json")
        if not pretrained_path.exists():
            # Try experiment-specific pretrained params
            pretrained_path = _HERE.parent / "exp_nongauss" / "pretrained_params.json"

        if pretrained_path.exists():
            with open(pretrained_path) as f:
                cache = json.load(f)
            sim_cache = cache.get(sim, {})
            # Two formats: flat {"ell_x":..., "lam":..., "h":...}
            #           or nested {"512": {"ell_x":..., ...}, ...}
            if "ell_x" in sim_cache:
                # Flat format (exp_nongauss style)
                d = sim_cache
            else:
                # Nested by n (exp_conditional_coverage style)
                n_key = str(config["n_0"])
                if n_key in sim_cache:
                    d = sim_cache[n_key]
                elif sim_cache:
                    d = next(iter(sim_cache.values()))
                    print(f"  WARNING: no pretrained params for n={config['n_0']}, "
                          f"using {next(iter(sim_cache.keys()))}")
                else:
                    d = {"ell_x": 0.5, "lam": 1e-3, "h": 0.2}
                    print(f"  WARNING: no pretrained params for {sim}, using defaults")
            shared_params = Params(ell_x=d["ell_x"], lam=d["lam"], h=d["h"])
            print(f"  Shared params: ell_x={d['ell_x']}, lam={d['lam']}, h_cv={d['h']}")
        else:
            shared_params = Params(ell_x=0.5, lam=1e-3, h=0.2)
            print(f"  WARNING: no pretrained_params.json found, using defaults")

        # --- Build bandwidth configurations ---
        bw_configs = []
        # 3 fixed
        bw_configs.append(("fixed_small", config["h_fixed_small"], None))
        bw_configs.append(("fixed_cv", None, None))  # h from shared_params
        bw_configs.append(("fixed_large", config["h_fixed_large"], None))
        # c-scan adaptive
        for c in config["c_values"]:
            bw_configs.append((f"adaptive_c{c:.1f}", None, c))

        oracle_var_fn = _ORACLE_VAR.get(sim)
        oracle_q_fn = _ORACLE_QUANTILE.get(sim)
        oracle_cdf_fn = _ORACLE_CDF.get(sim)

        if oracle_q_fn is None or oracle_cdf_fn is None:
            print(f"  ERROR: no oracle functions for {sim}, skipping")
            continue
        if oracle_var_fn is None:
            print(f"  ERROR: no oracle variance for {sim}, skipping")
            continue

        # --- Run all macroreps x bandwidth configs ---
        all_rows = []
        t0 = time.time()

        for k in range(n_macro):
            for bw_label, h_fixed, c_scale in bw_configs:
                rows, _ = run_one_config(
                    macrorep_id=k,
                    base_seed=config["base_seed"],
                    config=config,
                    simulator_func=sim,
                    shared_params=shared_params,
                    bw_label=bw_label,
                    h_fixed=h_fixed,
                    c_scale=c_scale,
                    oracle_var_fn=oracle_var_fn,
                    oracle_quantile_fn=oracle_q_fn,
                    oracle_cdf_fn=oracle_cdf_fn,
                )
                all_rows.extend(rows)

            elapsed = time.time() - t0
            print(f"  macrorep {k+1}/{n_macro} done  "
                  f"({len(bw_configs)} configs, {elapsed:.0f}s elapsed)")

        # --- Save ---
        df = pd.DataFrame(all_rows)
        results_path = out_base / f"results_{sim}.csv"
        df.to_csv(results_path, index=False)
        print(f"  Saved results -> {results_path}")

        summary = compute_summary(df, config["alpha"])
        summary_path = out_base / f"summary_{sim}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Saved summary -> {summary_path}")

        # Print compact summary table
        cols_show = ["bandwidth",
                     "ks_max_mean", "cov_gap_sup_mean",
                     "q_err_sup_mean", "cdf_err_mean_mean",
                     "width_mean_mean"]
        cols_show = [c for c in cols_show if c in summary.columns]
        print(summary[cols_show].to_string(index=False))

    total = time.time() - t0
    print(f"\nDone. Total wall time: {total:.0f}s")


if __name__ == "__main__":
    main()
