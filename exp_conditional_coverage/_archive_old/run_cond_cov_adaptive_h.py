"""
run_cond_cov_adaptive_h.py

Adaptive bandwidth variant of the conditional coverage experiment.

Key modification vs run_cond_cov.py:
  - Instead of a globally fixed h (tuned by CV), use h(x) = c_scale * sigma_hat(x)
  - sigma_hat(x) = kernel-weighted average of per-site empirical std from Stage 1 data
  - Both calibration scores and test coverage check use the same adaptive h(x)

Motivation (from exp_cond_cov_summary.md, point 2):
  In exp2, sigma(x=pi) = 0.01 but CV-tuned h ~ 0.08-0.2, giving h/sigma ~ 8-20.
  This causes F_hat(Y|x=pi) ≈ 0.5 for all Y → coverage → 1.0 deterministically.
  Adaptive h(x) ∝ sigma(x) fixes h/sigma ratio to be approximately constant.

Usage:
  python exp_conditional_coverage/run_cond_cov_adaptive_h.py
  python exp_conditional_coverage/run_cond_cov_adaptive_h.py --c_scale 2.0 --n_macro 5
  python exp_conditional_coverage/run_cond_cov_adaptive_h.py --simulators exp2 --c_scale 1.5
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm, gamma as _gamma

from Two_stage import run_stage1_train, run_stage2
from Two_stage.sim_functions import get_experiment_config
from Two_stage.sim_functions.exp2 import exp2_true_function, exp2_noise_variance_function
from Two_stage.sim_functions.exp1 import exp1_true_function, exp1_noise_variance_function
from CKME.parameters import Params, ParamGrid
from CKME.coefficients import compute_ckme_coeffs
from CKME.indicators import make_indicator

_HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Config loader (reuse same config.txt as run_cond_cov.py)
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    cfg = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        cfg[key.strip()] = val.strip()
    return cfg


def get_config(cfg_path: Path | None = None) -> dict:
    path = cfg_path or (_HERE / "config.txt")
    raw = load_config(path)
    return {
        "n_vals":      [int(v) for v in raw["n_vals"].split(",")],
        "r_0":         int(raw["r_0"]),
        "r_1":         int(raw["r_1"]),
        "alpha":       float(raw["alpha"]),
        "t_grid_size": int(raw["t_grid_size"]),
        "method":      raw["method"],
        "mixed_ratio": float(raw["mixed_ratio"]),
        "n_cand":      int(raw["n_cand"]),
        "M_eval":      int(raw["M_eval"]),
        "B_test":      int(raw["B_test"]),
        "n_macro":     int(raw["n_macro"]),
        "base_seed":   int(raw["base_seed"]),
        "simulators":  [s.strip() for s in raw["simulators"].split(",")],
        "tune_params": raw.get("tune_params", "false").lower() == "true",
        "h_default":   float(raw.get("h_default", "0.2")),
        "ell_x_list":  [float(v) for v in raw.get("ell_x_list", "0.5,1.0,2.0").split(",")],
        "lam_list":    [float(v) for v in raw.get("lam_list", "0.001,0.01,0.1").split(",")],
        "h_list":      [float(v) for v in raw.get("h_list", "0.1,0.2,0.3").split(",")],
        "cv_folds":    int(raw.get("cv_folds", "5")),
        "n_jobs":      int(raw.get("n_jobs", "4")),
        "cv_max_n":    int(raw["cv_max_n"]) if "cv_max_n" in raw else None,
    }


# ---------------------------------------------------------------------------
# Oracle quantile helpers (same as run_cond_cov.py)
# ---------------------------------------------------------------------------

def oracle_quantiles_exp1(x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact oracle quantiles for exp1: MG1 queue with Gaussian heteroscedastic noise."""
    mu = exp1_true_function(x)
    sigma = np.sqrt(exp1_noise_variance_function(x))
    q_lo = mu + sigma * _norm.ppf(alpha / 2)
    q_hi = mu + sigma * _norm.ppf(1 - alpha / 2)
    return q_lo, q_hi


def oracle_quantiles_exp2(x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    mu = exp2_true_function(x)
    sigma = np.sqrt(exp2_noise_variance_function(x))
    q_lo = mu + sigma * _norm.ppf(alpha / 2)
    q_hi = mu + sigma * _norm.ppf(1 - alpha / 2)
    return q_lo, q_hi


def oracle_quantiles_nongauss_B2L(x: np.ndarray, alpha: float, k: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    from math import sqrt
    sigma_tar = 0.1 + 0.1 * (x - np.pi) ** 2
    theta = sigma_tar / sqrt(k)
    mu = exp2_true_function(x)
    q_lo = mu + _gamma.ppf(alpha / 2, a=k, scale=theta) - k * theta
    q_hi = mu + _gamma.ppf(1 - alpha / 2, a=k, scale=theta) - k * theta
    return q_lo, q_hi


_ORACLE_FN = {
    "exp1":          oracle_quantiles_exp1,
    "exp2":          oracle_quantiles_exp2,
    "nongauss_B2L":  oracle_quantiles_nongauss_B2L,
}

# ---------------------------------------------------------------------------
# True noise std functions for oracle adaptive bandwidth h(x) = c * sigma_true(x)
# ---------------------------------------------------------------------------

def _sigma_true_exp1(x: np.ndarray) -> np.ndarray:
    return np.sqrt(exp1_noise_variance_function(np.asarray(x)))


def _sigma_true_exp2(x: np.ndarray) -> np.ndarray:
    return np.sqrt(exp2_noise_variance_function(np.asarray(x)))


def _sigma_true_nongauss_B2L(x: np.ndarray) -> np.ndarray:
    # B2L: Y = f(x) + G - k*theta(x),  G ~ Gamma(k=2, theta(x))
    # Var(Y) = k * theta^2 = 2*(sigma_tar/sqrt(2))^2 = sigma_tar^2
    # => std(Y|x) = sigma_tar(x)
    return 0.1 + 0.1 * (np.asarray(x) - np.pi) ** 2


# Maps simulator name -> callable x (1D array) -> sigma_true (1D array)
_SIGMA_TRUE_FN: dict = {
    "exp1":         _sigma_true_exp1,
    "exp2":         _sigma_true_exp2,
    "nongauss_B2L": _sigma_true_nongauss_B2L,
}

_DEFAULT_PARAMS = Params(ell_x=0.5, lam=0.001, h=0.1)


# ---------------------------------------------------------------------------
# Two-stage hyperparameter tuning (identical to run_cond_cov.py)
# ---------------------------------------------------------------------------

def tune_params_for_n(
    simulator_func: str,
    n_0: int,
    r_0: int,
    t_grid_size: int,
    cv_seed: int,
    cv_folds: int,
    ell_x_list: list,
    lam_list: list,
    h_list: list,
    h_default: float,
    tune_h: bool = False,
) -> Params:
    """One-stage CV for adaptive-h: tune (ell_x, lam) only; h is set adaptively at runtime."""
    grid1 = ParamGrid(ell_x_list=ell_x_list, lam_list=lam_list, h_list=[h_default])
    result1 = run_stage1_train(
        n_0=n_0, r_0=r_0, simulator_func=simulator_func,
        param_grid=grid1, t_grid_size=t_grid_size,
        cv_folds=cv_folds, random_state=cv_seed,
    )
    if not tune_h:
        return result1.params

    best_ell_x = result1.params.ell_x
    best_lam   = result1.params.lam
    grid2 = ParamGrid(ell_x_list=[best_ell_x], lam_list=[best_lam], h_list=h_list)
    result2 = run_stage1_train(
        n_0=n_0, r_0=r_0, simulator_func=simulator_func,
        param_grid=grid2, t_grid_size=t_grid_size,
        cv_folds=cv_folds, random_state=cv_seed + 1,
    )
    return result2.params


# ---------------------------------------------------------------------------
# Adaptive bandwidth: estimate local sigma from Stage 1 training data
# ---------------------------------------------------------------------------

def estimate_local_sigma(
    model,
    X_query: np.ndarray,
    c_scale: float = 1.0,
    h_min: float = 1e-3,
    n_neighbors: int = 5,
) -> np.ndarray:
    """
    Estimate local noise std sigma_hat(x) at query points using k-nearest-neighbor
    average of per-site empirical stds from Stage 1 training replicates.

    Parameters
    ----------
    model : CKMEModel
        Trained CKME model with r > 1 (requires replicated training data).
    X_query : ndarray, shape (q, d)
        Query points.
    c_scale : float
        Multiplicative scaling: returned h(x) = c_scale * sigma_hat(x).
    h_min : float
        Minimum h to avoid numerical issues when sigma_hat(x) ~ 0.
    n_neighbors : int
        Number of nearest training sites to average for sigma estimation.
        k-NN gives a more local estimate than kernel-weighted average,
        which is important when sigma(x) has sharp minima (e.g., exp2 near x=pi).

    Returns
    -------
    h_adaptive : ndarray, shape (q,)
        Per-query adaptive bandwidth h(x) = max(c_scale * sigma_hat(x), h_min).

    Notes
    -----
    - model.Y has shape (n_sites, r) in distinct-sites mode (r > 1)
    - Per-site empirical std: s_j = std(Y[j, :], ddof=1)
    - k-NN estimate: sigma_hat(x) = mean of s_j for j in k nearest sites to x
    - k-NN is preferred over Nadaraya-Watson here because NW with CKME kernel
      bandwidth (ell_x) can be too wide and oversmooth sigma at local minima.
    """
    assert model.r > 1, "estimate_local_sigma requires r > 1 (replicated training data)"

    X_query = np.atleast_2d(X_query)
    q = X_query.shape[0]

    # Per-site empirical std from Stage 1 replicates: shape (n_sites,)
    s_train = np.sqrt(np.maximum(model.Y.var(axis=1, ddof=1), 0.0))

    # Pairwise squared distances: (q, n_sites)
    diff = X_query[:, None, :] - model.X[None, :, :]  # (q, n_sites, d)
    D2 = (diff ** 2).sum(axis=2)  # (q, n_sites)

    # k nearest neighbors for each query point
    k = min(n_neighbors, model.n)
    nn_idx = np.argpartition(D2, k, axis=1)[:, :k]  # (q, k) — indices into training sites

    # Average per-site std over k nearest neighbors: shape (q,)
    sigma_hat = s_train[nn_idx].mean(axis=1)  # (q,)

    # Apply scale and floor
    h_adaptive = np.maximum(c_scale * sigma_hat, h_min)
    return h_adaptive


# ---------------------------------------------------------------------------
# Adaptive calibration: compute q_hat with per-point h(x)
# ---------------------------------------------------------------------------

def calibrate_with_adaptive_h(
    model,
    X_1: np.ndarray,
    Y_stage2_matrix: np.ndarray,
    alpha: float,
    c_scale: float,
    h_min: float,
    n_neighbors: int = 5,
    n_cal_max: int | None = None,
    rng: np.random.Generator | None = None,
    sigma_fn=None,
) -> float:
    """
    Re-calibrate CP using adaptive h(x) = c_scale * sigma(x).

    Parameters
    ----------
    model : CKMEModel
        Trained CKME model (Stage 1).
    X_1 : ndarray, shape (n_1, d)
        Unique Stage 2 design sites.
    Y_stage2_matrix : ndarray, shape (n_1, r_1)
        Stage 2 Y values, reshaped so row j has r_1 replicates for site j.
    alpha : float
        Significance level.
    c_scale : float
        Bandwidth scale factor.
    h_min : float
        Minimum bandwidth.
    n_cal_max : int or None
        If set, subsample at most n_cal_max sites for calibration.
        Reduces compute_ckme_coeffs cost from O(n^2 * n_1) to O(n^2 * n_cal_max).
        CP quantile accuracy is O(1/sqrt(n_cal)) so 2000 sites is already very precise.
    rng : np.random.Generator or None
        RNG for subsampling (only used when n_cal_max is set).
    sigma_fn : callable or None
        If provided, use oracle h(x) = c_scale * sigma_fn(x[:, 0]).
        If None, estimate sigma via k-NN from Stage 1 replicates (requires model.r > 1).

    Returns
    -------
    q_hat : float
        Adaptive conformal quantile.
    """
    n_1, r_1 = Y_stage2_matrix.shape

    # Subsample calibration sites if n_cal_max is set and n_1 exceeds it
    if n_cal_max is not None and n_1 > n_cal_max:
        idx = (rng if rng is not None else np.random.default_rng()).choice(
            n_1, size=n_cal_max, replace=False
        )
        X_1 = X_1[idx]
        Y_stage2_matrix = Y_stage2_matrix[idx]
        n_1 = n_cal_max

    # Adaptive h per Stage 2 unique site: shape (n_1,)
    if sigma_fn is not None:
        h_sites = np.maximum(c_scale * sigma_fn(X_1[:, 0]), h_min)
    else:
        h_sites = estimate_local_sigma(model, X_1, c_scale=c_scale, h_min=h_min, n_neighbors=n_neighbors)

    # CKME coefficients for Stage 2 sites: shape (n_sites_train, n_1)
    # Cost: O(n_train^2 * n_1) — dominant term, reduced by subsampling above.
    C_1 = compute_ckme_coeffs(model.L, model.kx, model.X, X_1)

    # Training Y flattened: shape (n_sites_train * r_0,)
    Y_flat_train = model.Y.ravel()

    # Compute F_hat(y | x) for each (site j, rep k) calibration point
    # Vectorize over r_1 per site using g_matrix to eliminate inner loop.
    n_cal = n_1 * r_1
    F_cal = np.empty(n_cal)

    for j in range(n_1):
        h_j = float(h_sites[j])
        ind_j = make_indicator(model.indicator_type, h_j)
        c_j = C_1[:, j]  # (n_sites_train,)

        # g_matrix: (n_sites_train * r_0, r_1) — one call instead of r_1 g_vector calls
        G_all = ind_j.g_matrix(Y_flat_train, Y_stage2_matrix[j])
        G_bar = G_all.reshape(model.n, model.r, r_1).mean(axis=1)  # (n_sites_train, r_1)
        F_j = np.clip(c_j @ G_bar, 0.0, 1.0)  # (r_1,)
        F_cal[j * r_1:(j + 1) * r_1] = F_j

    # Nonconformity scores: |F_hat - 0.5|
    scores = np.abs(F_cal - 0.5)

    # Conformal quantile with finite-sample correction
    k_idx = int(np.ceil((1 - alpha) * (1 + n_cal)))
    k_idx = min(k_idx, n_cal)
    q_hat = float(np.sort(scores)[k_idx - 1])
    return q_hat


# ---------------------------------------------------------------------------
# Core: one macrorep with adaptive h
# ---------------------------------------------------------------------------

def run_one_macrorep_adaptive(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    params_by_n: dict,
    c_scale: float,
    h_min: float,
    n_neighbors: int = 5,
    dtype: Optional[type] = None,
    n_cal_max: int | None = None,
    stage_ratio: float = 1.0,
    sigma_fn=None,
) -> list[dict]:
    """Run one macro-rep with adaptive h across all n_vals.

    Parameters
    ----------
    stage_ratio : float
        Fraction of n to use for each stage. E.g. 0.5 means Stage 1 and Stage 2
        each receive n//2 sites (total budget = n, split evenly).
        Default=1.0 (original behavior: Stage 1 = Stage 2 = n).
    sigma_fn : callable or None
        If provided (oracle mode), h(x) = c_scale * sigma_fn(x).
        If None (knn mode), sigma is estimated from Stage 1 replicates via k-NN.
    """
    seed = base_seed + macrorep_id * 10000
    alpha = config["alpha"]
    n_vals = config["n_vals"]
    r_0 = config["r_0"]
    r_1 = config["r_1"]
    M = config["M_eval"]
    B = config["B_test"]
    n_cand = config["n_cand"]
    method = config["method"]
    mixed_ratio = config["mixed_ratio"]
    t_grid_size = config["t_grid_size"]

    exp_cfg = get_experiment_config(simulator_func)
    bounds = exp_cfg["bounds"]
    simulator = exp_cfg["simulator"]

    x_lo = float(bounds[0][0])
    x_hi = float(bounds[1][0])

    # Fixed evaluation grid
    x_eval = np.linspace(x_lo, x_hi, M)
    X_eval = x_eval.reshape(-1, 1)

    # Oracle quantiles
    oracle_fn = _ORACLE_FN.get(simulator_func)
    if oracle_fn is not None:
        q_lo_oracle, q_hi_oracle = oracle_fn(x_eval, alpha)
    else:
        q_lo_oracle = q_hi_oracle = np.full(M, np.nan)

    rows = []
    for idx_n, n_0 in enumerate(n_vals):
        rng_seed = seed + idx_n * 1000

        # Actual sites per stage: n_stage = floor(n_0 * stage_ratio), min 2 for sigma est.
        n_stage = max(2, int(n_0 * stage_ratio))

        # Candidate points for Stage 2 (must be >= n_stage for site selection)
        n_cand_eff = max(n_cand, 2 * n_stage)
        rng_cand = np.random.default_rng(rng_seed)
        X_cand = rng_cand.uniform(x_lo, x_hi, size=(n_cand_eff, 1))

        # Stage 1: train CKME model with n_stage sites
        stage1 = run_stage1_train(
            n_0=n_stage,
            r_0=r_0,
            simulator_func=simulator_func,
            params=params_by_n[n_0],
            t_grid_size=t_grid_size,
            random_state=rng_seed + 1,
            dtype=dtype,
        )

        # Stage 2: site selection + data collection with n_stage sites
        # (We ignore stage2.cp.q_hat and re-calibrate with adaptive h)
        stage2 = run_stage2(
            stage1_result=stage1,
            X_cand=X_cand,
            n_1=n_stage,
            r_1=r_1,
            simulator_func=simulator_func,
            method=method,
            alpha=alpha,
            mixed_ratio=mixed_ratio,
            random_state=rng_seed + 2,
        )

        model = stage2.model

        # Re-calibrate with adaptive h
        # X_1: unique Stage 2 sites, shape (n_stage, d)
        # Y_stage2: shape (n_stage * r_1,) -> reshape to (n_stage, r_1)
        X_1 = stage2.X_1                                    # (n_stage, d)
        Y_stage2_matrix = stage2.Y_stage2.reshape(n_stage, r_1)  # (n_stage, r_1)

        q_hat = calibrate_with_adaptive_h(
            model=model,
            X_1=X_1,
            Y_stage2_matrix=Y_stage2_matrix,
            alpha=alpha,
            c_scale=c_scale,
            h_min=h_min,
            n_neighbors=n_neighbors,
            n_cal_max=n_cal_max,
            rng=np.random.default_rng(rng_seed + 4),
            sigma_fn=sigma_fn,
        )

        # Compute adaptive h at eval points: shape (M,)
        if sigma_fn is not None:
            h_eval = np.maximum(c_scale * sigma_fn(x_eval), h_min)
        else:
            h_eval = estimate_local_sigma(model, X_eval, c_scale=c_scale, h_min=h_min, n_neighbors=n_neighbors)

        # Pre-compute CKME coefficients for eval points: (n_sites_train, M)
        C_eval = compute_ckme_coeffs(model.L, model.kx, model.X, X_eval)

        # Training Y flattened
        Y_flat_train = model.Y.ravel()

        # Generate fresh test draws: shape (M, B)
        rng_seed_y = rng_seed + 3
        X_mc = np.repeat(x_eval, B)
        Y_mc = simulator(X_mc, random_state=rng_seed_y)
        Y_mc = Y_mc.reshape(M, B)

        # Chunk B to bound peak memory: G_all shape = (n*r_0, B_chunk).
        # At n=16384, r_0=10, B=2000: full G_all ~ 1.3 GB (float32).
        # With B_chunk=200: ~130 MB per eval point, safe for parallel workers.
        B_chunk = min(200, B)

        cov_mc = np.empty(M)
        for m in range(M):
            c_m = C_eval[:, m]    # (n_sites_train,)
            Y_b = Y_mc[m]         # (B,)
            h_m = float(h_eval[m])
            ind_m = make_indicator(model.indicator_type, h_m)

            F_b = np.empty(B)
            for b0 in range(0, B, B_chunk):
                Y_chunk = Y_b[b0:b0 + B_chunk]
                # G matrix: (n_sites_train * r_0, B_chunk)
                G_chunk = ind_m.g_matrix(Y_flat_train, Y_chunk)
                # Average over r_0: (n_sites_train, B_chunk)
                G_bar = G_chunk.reshape(model.n, model.r, -1).mean(axis=1)
                F_b[b0:b0 + B_chunk] = c_m @ G_bar

            np.clip(F_b, 0.0, 1.0, out=F_b)
            score_b = np.abs(F_b - 0.5)
            cov_mc[m] = np.mean(score_b <= q_hat)

        for m in range(M):
            rows.append({
                "macrorep":    macrorep_id,
                "n_0":         n_0,
                "x_eval":      x_eval[m],
                "cov_mc":      float(cov_mc[m]),
                "q_hat":       float(q_hat),
                "q_lo_oracle": float(q_lo_oracle[m]),
                "q_hi_oracle": float(q_hi_oracle[m]),
                "h_at_x":      float(h_eval[m]),   # record adaptive h for diagnostics
            })

    return rows


# ---------------------------------------------------------------------------
# Summary: same as run_cond_cov.py
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    target = 1.0 - alpha
    records = []
    for n_0, g_n in df.groupby("n_0"):
        for macro_id, g_m in g_n.groupby("macrorep"):
            delta = np.abs(g_m["cov_mc"].values - target)
            records.append({
                "n_0":      n_0,
                "macrorep": macro_id,
                "mae_cov":  delta.mean(),
                "sup_err":  delta.max(),
            })

    per_macro = pd.DataFrame(records)
    rows_summary = []
    for n_0, g in per_macro.groupby("n_0"):
        rows_summary.append({
            "n_0":          n_0,
            "mae_cov_mean": g["mae_cov"].mean(),
            "mae_cov_sd":   g["mae_cov"].std(ddof=1),
            "sup_err_mean": g["sup_err"].mean(),
            "sup_err_sd":   g["sup_err"].std(ddof=1),
            "n_macro":      len(g),
        })
    return pd.DataFrame(rows_summary)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Conditional coverage experiment with adaptive h(x) = c_scale * sigma_hat(x)"
    )
    parser.add_argument("--n_macro",    type=int,   default=None)
    parser.add_argument("--n_jobs",     type=int,   default=None,
                        help="Parallel workers. Overrides config.txt n_jobs (default 4).")
    parser.add_argument("--simulators", type=str,   default=None)
    parser.add_argument("--base_seed",  type=int,   default=None)
    parser.add_argument("--output_dir", type=str,   default=None)
    parser.add_argument("--config",     type=str,   default=None)
    parser.add_argument("--n_vals",     type=str,   default=None,
                        help="Comma-separated n values. Overrides config.txt n_vals.")
    parser.add_argument("--c_scale",    type=float, default=2.0,
                        help="h(x) = c_scale * sigma_hat(x). Default=2.0.")
    parser.add_argument("--h_min",       type=float, default=1e-3,
                        help="Minimum h floor to avoid numerical issues. Default=1e-3.")
    parser.add_argument("--n_neighbors", type=int,   default=5,
                        help="k for k-NN sigma estimation. Default=5.")
    parser.add_argument("--dtype",       type=str,   default="float32",
                        choices=["float32", "float64"],
                        help="Working precision for CKME Cholesky. float32 halves memory "
                             "(critical for n>=2^12 with parallel macroreps). Default=float32.")
    parser.add_argument("--cv_max_n",    type=int,   default=None,
                        help="If set, only run CV tuning for n <= cv_max_n; for larger n "
                             "reuse the largest tuned n's params. Avoids running CV at huge n. "
                             "Default=None (CV at every n).")
    parser.add_argument("--n_cal_max",   type=int,   default=2000,
                        help="Max calibration sites for adaptive CP. Subsample Stage 2 sites "
                             "to this many when n_1 > n_cal_max. Reduces O(n^2 * n_1) cost to "
                             "O(n^2 * n_cal_max). Default=2000 (very precise for CP quantiles).")
    parser.add_argument("--stage_ratio", type=float, default=1.0,
                        help="Fraction of n used per stage. E.g. 0.5 means Stage1=Stage2=n//2 "
                             "(total budget n split evenly between stages). Default=1.0 "
                             "(original: both stages use n). Cholesky cost scales as "
                             "(stage_ratio*n)^3, so 0.5 reduces cost by 8x.")
    parser.add_argument("--sigma_mode",  type=str,   default="knn",
                        choices=["knn", "oracle"],
                        help="How to estimate sigma(x) for adaptive h(x) = c_scale * sigma(x). "
                             "'knn': k-NN from Stage 1 replicates (requires r_0 >= 2). "
                             "'oracle': use true sigma_true(x) from simulator (cleaner theory "
                             "experiment; r_0=1 is fine). Default='knn'.")
    parser.add_argument("--r_0",         type=int,   default=None,
                        help="Stage 1 replications per site. Overrides config.txt r_0. "
                             "With --sigma_mode oracle, r_0=1 is valid. "
                             "With --sigma_mode knn, r_0 >= 2 is required.")
    parser.add_argument("--r_1",         type=int,   default=None,
                        help="Stage 2 replications per site. Overrides config.txt r_1. "
                             "Set r_0=r_1=1 with --stage_ratio 0.5 for total obs = n.")
    parser.add_argument("--B_test",      type=int,   default=None,
                        help="Fresh test draws per eval point for MC coverage. Overrides config.txt B_test.")
    parser.add_argument("--no_tune",     action="store_true",
                        help="Skip CV tuning; use default params (ell_x=0.5, lam=0.001, h=0.1) "
                             "or cached params from JSON if available.")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    config = get_config(cfg_path)
    if args.no_tune:
        config["tune_params"] = False

    if args.n_macro is not None:
        config["n_macro"] = args.n_macro
    if args.base_seed is not None:
        config["base_seed"] = args.base_seed
    if args.simulators is not None:
        config["simulators"] = [s.strip() for s in args.simulators.split(",")]
    if args.r_0 is not None:
        config["r_0"] = args.r_0
    if args.r_1 is not None:
        config["r_1"] = args.r_1
    if args.B_test is not None:
        config["B_test"] = args.B_test

    if args.n_vals is not None:
        config["n_vals"] = [int(v) for v in args.n_vals.split(",")]

    stage_ratio = args.stage_ratio

    # Auto-scale n_cand so that n_cand >= max n_stage; site_selection requires this.
    # With stage_ratio < 1, n_stage = n * stage_ratio < n, so n_cand can be smaller.
    max_n_stage = max(max(2, int(n * stage_ratio)) for n in config["n_vals"])
    if config["n_cand"] < max_n_stage:
        new_n_cand = max(2 * max_n_stage, config["n_cand"])
        print(f"  Note: bumping n_cand {config['n_cand']} -> {new_n_cand} "
              f"to satisfy n_cand >= max_n_stage={max_n_stage} (stage_ratio={stage_ratio})")
        config["n_cand"] = new_n_cand

    c_scale     = args.c_scale
    h_min       = args.h_min
    n_neighbors = args.n_neighbors
    dtype       = np.float32 if args.dtype == "float32" else np.float64
    # cv_max_n: CLI overrides config; config overrides built-in default (None)
    cv_max_n    = args.cv_max_n if args.cv_max_n is not None else config.get("cv_max_n")
    n_cal_max   = args.n_cal_max if args.n_cal_max > 0 else None
    sigma_mode  = args.sigma_mode

    # Output directory: subdirectory tagged with c_scale
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = _HERE / f"output_adaptive_h_c{c_scale:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_macro   = config["n_macro"]
    base_seed = config["base_seed"]
    simulators = config["simulators"]
    n_jobs    = min(args.n_jobs if args.n_jobs is not None else config.get("n_jobs", 4), n_macro)

    # Guard: knn mode requires r_0 >= 2
    if sigma_mode == "knn" and config["r_0"] < 2:
        raise ValueError(
            f"--sigma_mode knn requires r_0 >= 2 (got r_0={config['r_0']}). "
            "Use --sigma_mode oracle to run with r_0=1."
        )

    n_stages_actual = [max(2, int(n * stage_ratio)) for n in config["n_vals"]]
    sigma_desc = f"oracle sigma_true(x)" if sigma_mode == "oracle" else f"k-NN (k={n_neighbors})"
    total_obs_per_n = [
        max(2, int(n * stage_ratio)) * config["r_0"] + max(2, int(n * stage_ratio)) * config["r_1"]
        for n in config["n_vals"]
    ]
    print(f"=== Adaptive h experiment ===")
    print(f"h(x) = {c_scale} * sigma(x)   [{sigma_desc}]  (h_min={h_min})")
    print(f"Simulators : {simulators}")
    print(f"n_vals     : {config['n_vals']}  (stage_ratio={stage_ratio})")
    print(f"n_stage    : {n_stages_actual}  (actual sites per stage, Stage1=Stage2)")
    print(f"r_0={config['r_0']}, r_1={config['r_1']}  ->  total obs per n: {total_obs_per_n}")
    print(f"n_macro    : {n_macro}  |  n_jobs: {n_jobs}")
    print(f"dtype      : {args.dtype}  |  cv_max_n: {cv_max_n}  |  n_cal_max: {n_cal_max}")
    print(f"M_eval={config['M_eval']}, B_test={config['B_test']}, alpha={config['alpha']}")
    print(f"Output dir : {out_dir}")

    for sim in simulators:
        print(f"\n=== Simulator: {sim} ===")

        # Hyperparameter tuning (same as run_cond_cov.py; load from fixed-h cache)
        params_by_n: dict = {}
        # Try to reuse cached params from the fixed-h experiment (same ell_x, lam, h)
        # h from cache is used only for ell_x, lam; adaptive h overrides the indicator h
        params_cache_path = _HERE / "output" / f"tuned_params_{sim}.json"
        adaptive_cache_path = out_dir / f"tuned_params_{sim}.json"

        if config["tune_params"]:
            cached: dict = {}
            # First try the adaptive output dir cache, then fall back to fixed-h cache
            for cache_path in [adaptive_cache_path, params_cache_path]:
                if cache_path.exists():
                    with open(cache_path) as f:
                        cached = json.load(f)
                    print(f"  Loaded tuned params from {cache_path}")
                    break

            # Decide which n's to actually CV-tune. Anything above cv_max_n
            # reuses the largest tuned n's params (CKME hyperparams typically
            # stabilize once n is moderate, so this avoids running expensive
            # CV at huge n on the laptop).
            tunable_ns = [
                n for n in config["n_vals"]
                if cv_max_n is None or n <= cv_max_n
            ]
            if not tunable_ns:
                # All n_vals exceed cv_max_n: tune at cv_max_n (cheap) and
                # reuse for all n_vals. Never fall back to min(n_vals) which
                # could be huge (e.g. 16384).
                ref_n = cv_max_n if cv_max_n is not None else min(config["n_vals"])
                tunable_ns = [ref_n]
            reuse_ns = [n for n in config["n_vals"] if n not in tunable_ns]
            needs_tune = [n for n in tunable_ns if str(n) not in cached]
            if needs_tune:
                print(f"  Tuning params for n={needs_tune}...")
            if reuse_ns:
                print(f"  Reusing largest-tuned params for n={reuse_ns} (cv_max_n={cv_max_n})")

            for idx_n, n_0 in enumerate(tunable_ns):
                # Tune at n_stage (actual sites used in Stage 1), not n_0 (budget label)
                n_stage_cv = max(2, int(n_0 * stage_ratio))
                if str(n_0) in cached:
                    d = cached[str(n_0)]
                    params_by_n[n_0] = Params(ell_x=d["ell_x"], lam=d["lam"], h=d["h"])
                    print(f"    n={n_0} (n_stage={n_stage_cv}): ell_x={d['ell_x']}, lam={d['lam']}, h={d['h']} (h overridden adaptively)")
                else:
                    cv_seed = base_seed + 999999 + idx_n * 100
                    p = tune_params_for_n(
                        simulator_func=sim, n_0=n_stage_cv, r_0=config["r_0"],
                        t_grid_size=config["t_grid_size"], cv_seed=cv_seed,
                        cv_folds=config["cv_folds"], ell_x_list=config["ell_x_list"],
                        lam_list=config["lam_list"], h_list=config["h_list"],
                        h_default=config["h_default"],
                    )
                    params_by_n[n_0] = p
                    cached[str(n_0)] = {"ell_x": p.ell_x, "lam": p.lam, "h": p.h}
                    print(f"    n={n_0} (n_stage={n_stage_cv}): tuned -> ell_x={p.ell_x}, lam={p.lam}, h={p.h}")

            # Fill in larger n's with the largest tuned params (ell_x, lam, h
            # all copied — adaptive h overrides the indicator h at runtime).
            if reuse_ns:
                src_n = max(tunable_ns)
                src_p = params_by_n[src_n]
                for n_0 in reuse_ns:
                    params_by_n[n_0] = src_p
                    print(f"    n={n_0}: reused from n={src_n} -> "
                          f"ell_x={src_p.ell_x}, lam={src_p.lam}, h={src_p.h}")

            with open(adaptive_cache_path, "w") as f:
                json.dump(cached, f, indent=2)
        else:
            for n_0 in config["n_vals"]:
                params_by_n[n_0] = _DEFAULT_PARAMS

        all_rows: list[dict] = []

        # Resolve sigma_fn for this simulator
        if sigma_mode == "oracle":
            sigma_fn = _SIGMA_TRUE_FN.get(sim)
            if sigma_fn is None:
                print(f"  WARNING: oracle sigma not defined for {sim}; falling back to k-NN.")
        else:
            sigma_fn = None

        if n_jobs <= 1:
            for k in range(n_macro):
                rows = run_one_macrorep_adaptive(
                    k, base_seed, config, sim, params_by_n, c_scale, h_min, n_neighbors,
                    dtype=dtype, n_cal_max=n_cal_max, stage_ratio=stage_ratio,
                    sigma_fn=sigma_fn,
                )
                all_rows.extend(rows)
                print(f"  macrorep {k+1}/{n_macro} done")
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                for k in range(n_macro):
                    f = ex.submit(
                        run_one_macrorep_adaptive,
                        k, base_seed, config, sim, params_by_n, c_scale, h_min, n_neighbors,
                        dtype, n_cal_max, stage_ratio, sigma_fn,
                    )
                    futures[f] = k
                done = 0
                for f in as_completed(futures):
                    k = futures[f]
                    try:
                        rows = f.result()
                        all_rows.extend(rows)
                    except Exception as e:
                        print(f"  macrorep {k} FAILED: {e}")
                    done += 1
                    print(f"  macrorep {k+1} done ({done}/{n_macro})")

        df = pd.DataFrame(all_rows)
        results_path = out_dir / f"results_{sim}.csv"
        df.to_csv(results_path, index=False)
        print(f"  Saved results -> {results_path}")

        summary = compute_summary(df, config["alpha"])
        summary_path = out_dir / f"summary_{sim}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Saved summary -> {summary_path}")
        print(summary[["n_0", "mae_cov_mean", "mae_cov_sd", "sup_err_mean", "sup_err_sd"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
