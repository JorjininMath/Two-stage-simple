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
    }


# ---------------------------------------------------------------------------
# Oracle quantile helpers (same as run_cond_cov.py)
# ---------------------------------------------------------------------------

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
    "exp2":          oracle_quantiles_exp2,
    "nongauss_B2L":  oracle_quantiles_nongauss_B2L,
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
) -> Params:
    """Two-stage CV: tune (ell_x, lam) then h."""
    grid1 = ParamGrid(ell_x_list=ell_x_list, lam_list=lam_list, h_list=[h_default])
    result1 = run_stage1_train(
        n_0=n_0, r_0=r_0, simulator_func=simulator_func,
        param_grid=grid1, t_grid_size=t_grid_size,
        cv_folds=cv_folds, random_state=cv_seed,
    )
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
) -> float:
    """
    Re-calibrate CP using adaptive h(x) = c_scale * sigma_hat(x).

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

    Returns
    -------
    q_hat : float
        Adaptive conformal quantile.
    """
    n_1, r_1 = Y_stage2_matrix.shape

    # Adaptive h per Stage 2 unique site: shape (n_1,)
    h_sites = estimate_local_sigma(model, X_1, c_scale=c_scale, h_min=h_min, n_neighbors=n_neighbors)

    # CKME coefficients for Stage 2 sites: shape (n_sites_train, n_1)
    C_1 = compute_ckme_coeffs(model.L, model.kx, model.X, X_1)

    # Training Y flattened: shape (n_sites_train * r_0,)
    Y_flat_train = model.Y.ravel()

    # Compute F_hat(y | x) for each (site j, rep k) calibration point
    n_cal = n_1 * r_1
    F_cal = np.empty(n_cal)

    for j in range(n_1):
        h_j = float(h_sites[j])
        ind_j = make_indicator(model.indicator_type, h_j)
        c_j = C_1[:, j]  # (n_sites_train,)

        for k in range(r_1):
            y_jk = float(Y_stage2_matrix[j, k])
            # g_vector shape: (n_sites_train * r_0,)
            g_all = ind_j.g_vector(Y_flat_train, y_jk)
            # Average over replicates: (n_sites_train,)
            g_bar = g_all.reshape(model.n, model.r).mean(axis=1)
            F_jk = float(np.clip(c_j @ g_bar, 0.0, 1.0))
            F_cal[j * r_1 + k] = F_jk

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
) -> list[dict]:
    """Run one macro-rep with adaptive h across all n_vals."""
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

        # Candidate points for Stage 2
        rng_cand = np.random.default_rng(rng_seed)
        X_cand = rng_cand.uniform(x_lo, x_hi, size=(n_cand, 1))

        # Stage 1: train CKME model
        stage1 = run_stage1_train(
            n_0=n_0,
            r_0=r_0,
            simulator_func=simulator_func,
            params=params_by_n[n_0],
            t_grid_size=t_grid_size,
            random_state=rng_seed + 1,
            dtype=dtype,
        )

        # Stage 2: site selection + data collection
        # (We ignore stage2.cp.q_hat and re-calibrate with adaptive h)
        stage2 = run_stage2(
            stage1_result=stage1,
            X_cand=X_cand,
            n_1=n_0,
            r_1=r_1,
            simulator_func=simulator_func,
            method=method,
            alpha=alpha,
            mixed_ratio=mixed_ratio,
            random_state=rng_seed + 2,
        )

        model = stage2.model

        # Re-calibrate with adaptive h
        # X_1: unique Stage 2 sites, shape (n_1, d)
        # Y_stage2: shape (n_1 * r_1,) -> reshape to (n_1, r_1)
        X_1 = stage2.X_1                              # (n_1, d)
        Y_stage2_matrix = stage2.Y_stage2.reshape(n_0, r_1)  # (n_1, r_1)

        q_hat = calibrate_with_adaptive_h(
            model=model,
            X_1=X_1,
            Y_stage2_matrix=Y_stage2_matrix,
            alpha=alpha,
            c_scale=c_scale,
            h_min=h_min,
            n_neighbors=n_neighbors,
        )

        # Compute adaptive h at eval points: shape (M,)
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

        cov_mc = np.empty(M)
        for m in range(M):
            c_m = C_eval[:, m]    # (n_sites_train,)
            Y_b = Y_mc[m]         # (B,)
            h_m = float(h_eval[m])
            ind_m = make_indicator(model.indicator_type, h_m)

            # G matrix with adaptive h_m: shape (n_sites_train * r_0, B)
            G_all = ind_m.g_matrix(Y_flat_train, Y_b)
            # Average over r_0 replicates: (n_sites_train, B)
            G_b = G_all.reshape(model.n, model.r, B).mean(axis=1)

            # F_hat(Y_b | x_m) with adaptive h: shape (B,)
            F_b = c_m @ G_b
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
    parser.add_argument("--n_jobs",     type=int,   default=4)
    parser.add_argument("--simulators", type=str,   default=None)
    parser.add_argument("--base_seed",  type=int,   default=None)
    parser.add_argument("--output_dir", type=str,   default=None)
    parser.add_argument("--config",     type=str,   default=None)
    parser.add_argument("--n_vals",     type=str,   default="64,256,1024,4096,16384",
                        help="Comma-separated n values. Default=2^6,2^8,2^10,2^12,2^14.")
    parser.add_argument("--c_scale",    type=float, default=2.0,
                        help="h(x) = c_scale * sigma_hat(x). Default=2.0.")
    parser.add_argument("--h_min",       type=float, default=1e-3,
                        help="Minimum h floor to avoid numerical issues. Default=1e-3.")
    parser.add_argument("--n_neighbors", type=int,   default=5,
                        help="k for k-NN sigma estimation. Default=5.")
    parser.add_argument("--dtype",       type=str,   default="float64",
                        choices=["float32", "float64"],
                        help="Working precision for CKME Cholesky. float32 halves memory "
                             "(critical for n>=2^13 with parallel macroreps). Default=float64.")
    parser.add_argument("--cv_max_n",    type=int,   default=None,
                        help="If set, only run CV tuning for n <= cv_max_n; for larger n "
                             "reuse the largest tuned n's params. Avoids running CV at huge n. "
                             "Default=None (CV at every n).")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    config = get_config(cfg_path)

    if args.n_macro is not None:
        config["n_macro"] = args.n_macro
    if args.base_seed is not None:
        config["base_seed"] = args.base_seed
    if args.simulators is not None:
        config["simulators"] = [s.strip() for s in args.simulators.split(",")]

    # Override n_vals (adaptive-h experiment uses larger powers-of-2 by default)
    config["n_vals"] = [int(v) for v in args.n_vals.split(",")]

    # Auto-scale n_cand so that n_cand >= max(n_vals); site_selection requires this
    max_n = max(config["n_vals"])
    if config["n_cand"] < max_n:
        new_n_cand = max(2 * max_n, config["n_cand"])
        print(f"  Note: bumping n_cand {config['n_cand']} -> {new_n_cand} "
              f"to satisfy n_cand >= max(n_vals)={max_n}")
        config["n_cand"] = new_n_cand

    c_scale     = args.c_scale
    h_min       = args.h_min
    n_neighbors = args.n_neighbors
    dtype       = np.float32 if args.dtype == "float32" else np.float64
    cv_max_n    = args.cv_max_n

    # Output directory: subdirectory tagged with c_scale
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = _HERE / f"output_adaptive_h_c{c_scale:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_macro   = config["n_macro"]
    base_seed = config["base_seed"]
    simulators = config["simulators"]
    n_jobs    = min(args.n_jobs, n_macro)

    print(f"=== Adaptive h experiment ===")
    print(f"h(x) = {c_scale} * sigma_hat(x)   (h_min={h_min}, k-NN={n_neighbors})")
    print(f"Simulators : {simulators}")
    print(f"n_vals     : {config['n_vals']}")
    print(f"n_macro    : {n_macro}  |  n_jobs: {n_jobs}")
    print(f"dtype      : {args.dtype}  |  cv_max_n: {cv_max_n}")
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
                tunable_ns = [min(config["n_vals"])]
            reuse_ns = [n for n in config["n_vals"] if n not in tunable_ns]
            needs_tune = [n for n in tunable_ns if str(n) not in cached]
            if needs_tune:
                print(f"  Tuning params for n={needs_tune}...")
            if reuse_ns:
                print(f"  Reusing largest-tuned params for n={reuse_ns} (cv_max_n={cv_max_n})")

            for idx_n, n_0 in enumerate(tunable_ns):
                if str(n_0) in cached:
                    d = cached[str(n_0)]
                    params_by_n[n_0] = Params(ell_x=d["ell_x"], lam=d["lam"], h=d["h"])
                    print(f"    n={n_0}: ell_x={d['ell_x']}, lam={d['lam']}, h={d['h']} (h overridden adaptively)")
                else:
                    cv_seed = base_seed + 999999 + idx_n * 100
                    p = tune_params_for_n(
                        simulator_func=sim, n_0=n_0, r_0=config["r_0"],
                        t_grid_size=config["t_grid_size"], cv_seed=cv_seed,
                        cv_folds=config["cv_folds"], ell_x_list=config["ell_x_list"],
                        lam_list=config["lam_list"], h_list=config["h_list"],
                        h_default=config["h_default"],
                    )
                    params_by_n[n_0] = p
                    cached[str(n_0)] = {"ell_x": p.ell_x, "lam": p.lam, "h": p.h}
                    print(f"    n={n_0}: tuned -> ell_x={p.ell_x}, lam={p.lam}, h={p.h}")

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

        if n_jobs <= 1:
            for k in range(n_macro):
                rows = run_one_macrorep_adaptive(
                    k, base_seed, config, sim, params_by_n, c_scale, h_min, n_neighbors,
                    dtype=dtype,
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
                        dtype,
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
