"""
run_consistency.py

Asymptotic consistency experiment for CKME-CP.

Tests three hierarchical levels as n grows:
  L1: quantile consistency  -- q̂_τ(x) → q_τ(x) (pre-CP, estimation level)
  L2: endpoint consistency  -- L_n(x) → q_{α/2}(x), U_n(x) → q_{1-α/2}(x)
  L3: coverage consistency  -- Cov_n(x) → 1-α

Supports fixed-h (CV-tuned scalar) and adaptive-h (h(x) = c_scale * sigma_hat(x)).

Usage (from project root):
  python exp_conditional_coverage/run_consistency.py
  python exp_conditional_coverage/run_consistency.py --h_mode adaptive --c_scale 2.0
  python exp_conditional_coverage/run_consistency.py --simulators exp1,exp2 --n_macro 10
  python exp_conditional_coverage/run_consistency.py --quick   # fast end-to-end test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

from Two_stage import run_stage1_train, run_stage2
from Two_stage.sim_functions import get_experiment_config
from Two_stage.sim_functions.exp1 import exp1_true_function, exp1_noise_variance_function
from Two_stage.sim_functions.exp2 import exp2_true_function, exp2_noise_variance_function
from CKME.parameters import Params, ParamGrid
from CKME.coefficients import compute_ckme_coeffs
from CKME.indicators import make_indicator

_HERE   = Path(__file__).parent
_ALPHA  = 0.1   # overridden by config / CLI

SIMULATORS = ["exp1", "exp2"]

# ---------------------------------------------------------------------------
# Oracle quantile helpers (closed-form, Gaussian noise)
# ---------------------------------------------------------------------------

def _gaussian_oracle(mu_fn, var_fn, x, alpha):
    mu    = mu_fn(x)
    sigma = np.sqrt(var_fn(x))
    return mu + sigma * _norm.ppf(alpha / 2), mu + sigma * _norm.ppf(1 - alpha / 2)


_ORACLE = {
    "exp1": lambda x, a: _gaussian_oracle(exp1_true_function, exp1_noise_variance_function, x, a),
    "exp2": lambda x, a: _gaussian_oracle(exp2_true_function, exp2_noise_variance_function, x, a),
}

# Oracle conditional variance functions (used for adaptive-h sigma)
_ORACLE_VAR = {
    "exp1": exp1_noise_variance_function,
    "exp2": exp2_noise_variance_function,
}

# ---------------------------------------------------------------------------
# Adaptive-h: kernel-weighted local sigma
# ---------------------------------------------------------------------------

def _adaptive_h(model, x_query: np.ndarray, c_scale: float,
                oracle_var_fn=None) -> np.ndarray:
    """
    Compute h(x) = c_scale * sigma(x) for each query point.

    Uses oracle conditional std when oracle_var_fn is provided (preferred for
    experiments where the true noise variance is known analytically).
    oracle_var_fn : callable x -> Var[Y|x], shape (n,) -> (n,)
    """
    x_1d = x_query.ravel() if x_query.ndim > 1 else x_query
    if oracle_var_fn is not None:
        sigma = np.sqrt(np.maximum(oracle_var_fn(x_1d), 1e-8))
        return c_scale * sigma
    # Fallback: kernel-weighted local variance (estimated, works with r_0=1)
    X_q    = x_query.reshape(-1, 1) if x_query.ndim == 1 else x_query
    C      = compute_ckme_coeffs(model.L, model.kx, model.X, X_q)   # (n, M)
    Y_site = model.Y.mean(axis=1) if model.Y.ndim == 2 else model.Y.ravel()
    h_vals = np.empty(X_q.shape[0])
    for m in range(X_q.shape[0]):
        c_m     = C[:, m]
        mu_hat  = float(c_m @ Y_site)
        var_hat = float(c_m @ (Y_site - mu_hat) ** 2)
        h_vals[m] = c_scale * np.sqrt(max(var_hat, 1e-8))
    return h_vals


# ---------------------------------------------------------------------------
# Quantile inversion from CKME CDF (L1 metric)
# ---------------------------------------------------------------------------

def _batch_cdf_on_tgrid(model, t_grid: np.ndarray, C_eval: np.ndarray) -> np.ndarray:
    """
    Compute F̂(t|x_m) for all M eval points with the model's fixed h.
    Returns F_all : (M, T), clipped to [0, 1].
    G_t_mat is computed once, saving ~200x redundant work at n=8192.
    """
    Y_flat = model.Y.ravel()                                        # (n*r,)
    T      = len(t_grid)
    G_mat  = model.indicator.g_matrix(Y_flat, t_grid)              # (n*r, T)
    G_site = G_mat.reshape(model.n, model.r, T).mean(axis=1)       # (n, T)
    F_all  = np.clip(C_eval.T @ G_site, 0.0, 1.0)                  # (M, T)
    return F_all


def _perpoint_cdf_on_tgrid(model, t_grid: np.ndarray, C_eval: np.ndarray,
                            h_vals: np.ndarray) -> np.ndarray:
    """
    Compute F̂(t|x_m) using per-point bandwidth h_vals[m].
    Used for adaptive-h mode. Each eval point gets its own indicator.
    Returns F_all : (M, T), clipped to [0, 1].
    """
    Y_flat = model.Y.ravel()
    T      = len(t_grid)
    M      = C_eval.shape[1]
    F_all  = np.empty((M, T))
    for m in range(M):
        ind_m  = make_indicator(model.indicator_type, float(h_vals[m]))
        G_mat  = ind_m.g_matrix(Y_flat, t_grid)                         # (n*r, T)
        G_site = G_mat.reshape(model.n, model.r, T).mean(axis=1)        # (n, T)
        F_all[m] = np.clip(C_eval[:, m] @ G_site, 0.0, 1.0)
    return F_all


def _recalibrate_adaptive_cp(model, stage2, c_scale: float, alpha: float,
                             oracle_var_fn=None) -> float:
    """
    Recompute CP calibration scores using per-point adaptive h for Stage 2 data.
    For each calibration point (x_j, y_j), the score is |F̂_{h(x_j)}(y_j|x_j) - 0.5|
    where h(x_j) = c_scale * sigma(x_j)  (oracle if available, else estimated).
    Returns new q_hat.
    """
    X_cal = np.atleast_2d(stage2.X_stage2)   # (n_cal, d)
    Y_cal = np.asarray(stage2.Y_stage2).ravel()
    n_cal = len(Y_cal)

    # Per-point adaptive h for calibration data
    x_cal_1d = X_cal.ravel() if X_cal.shape[1] == 1 else X_cal
    h_cal = _adaptive_h(model, x_cal_1d, c_scale, oracle_var_fn)   # (n_cal,)

    Y_flat  = model.Y.ravel()                        # (n*r,)
    C_cal   = compute_ckme_coeffs(model.L, model.kx, model.X, X_cal)  # (n, n_cal)

    scores = np.empty(n_cal)
    for j in range(n_cal):
        ind_j  = make_indicator(model.indicator_type, float(h_cal[j]))
        g_j    = ind_j.g_matrix(Y_flat, np.array([float(Y_cal[j])]))[:, 0]  # (n*r,)
        g_site = g_j.reshape(model.n, model.r).mean(axis=1)                  # (n,)
        F_j    = float(np.clip(C_cal[:, j] @ g_site, 0.0, 1.0))
        scores[j] = abs(F_j - 0.5)

    # Same formula as CP/calibration.py
    k = int(np.ceil((1 - alpha) * (1 + n_cal)))
    k = min(k, n_cal)
    q_hat = float(np.sort(scores)[k - 1])
    return q_hat


def _invert_cdf(F_all: np.ndarray, t_grid: np.ndarray, tau: float) -> np.ndarray:
    """Invert CDF rows: q̂_τ(x_m) = inf{ t : F_all[m, :] >= tau }."""
    T     = len(t_grid)
    M     = F_all.shape[0]
    q_arr = np.empty(M)
    for m in range(M):
        idx      = np.searchsorted(F_all[m], tau)
        q_arr[m] = t_grid[min(idx, T - 1)]
    return q_arr


# ---------------------------------------------------------------------------
# Core: one macrorep
# ---------------------------------------------------------------------------

def run_one_macrorep(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    params_by_n: dict,
    h_mode: str,
    c_scale: float,
    dtype,
) -> list[dict]:
    """Run one macrorep across all n_vals. Returns list of row dicts."""
    sim_idx = SIMULATORS.index(simulator_func) if simulator_func in SIMULATORS else 0
    seed    = base_seed + macrorep_id * 10000 + sim_idx * 100

    alpha      = config["alpha"]
    n_vals     = config["n_vals"]
    r_0        = config["r_0"]
    r_1        = config["r_1"]
    M          = config["M_eval"]
    B          = config["B_test"]
    n_cand     = config["n_cand"]
    method     = config["method"]
    mixed_ratio = config["mixed_ratio"]
    t_grid_size = config["t_grid_size"]

    exp_cfg = get_experiment_config(simulator_func)
    bounds  = exp_cfg["bounds"]
    sim_fn  = exp_cfg["simulator"]

    x_lo = float(bounds[0][0])
    x_hi = float(bounds[1][0])

    x_eval = np.linspace(x_lo, x_hi, M)
    X_eval = x_eval.reshape(-1, 1)

    oracle_fn = _ORACLE.get(simulator_func)
    if oracle_fn is not None:
        q_lo_oracle, q_hi_oracle = oracle_fn(x_eval, alpha)
    else:
        q_lo_oracle = q_hi_oracle = np.full(M, np.nan)

    rows = []
    for idx_n, n_0 in enumerate(n_vals):
        rng_seed    = seed + idx_n * 1000
        n_cand_eff  = max(n_cand, 2 * n_0)
        rng_cand    = np.random.default_rng(rng_seed)
        X_cand      = rng_cand.uniform(x_lo, x_hi, size=(n_cand_eff, 1))

        # --- Stage 1: train CKME ---
        stage1 = run_stage1_train(
            n_0=n_0, r_0=r_0,
            simulator_func=simulator_func,
            params=params_by_n[n_0],
            t_grid_size=t_grid_size,
            random_state=rng_seed + 1,
            dtype=dtype,
        )

        # --- Stage 2: CP calibration ---
        stage2 = run_stage2(
            stage1_result=stage1,
            X_cand=X_cand,
            n_1=n_0, r_1=r_1,
            simulator_func=simulator_func,
            method=method,
            alpha=alpha,
            mixed_ratio=mixed_ratio,
            random_state=rng_seed + 2,
        )

        model  = stage2.model
        t_grid = stage1.t_grid
        q_hat  = stage2.cp.q_hat

        # Pre-compute CKME coefficients for all M eval points
        C_eval = compute_ckme_coeffs(model.L, model.kx, model.X, X_eval)  # (n, M)

        # --- Bandwidth per eval point ---
        oracle_var_fn = _ORACLE_VAR.get(simulator_func)
        if h_mode == "adaptive":
            h_vals = _adaptive_h(model, x_eval, c_scale, oracle_var_fn)
            # Recompute CP calibration scores using adaptive h for Stage 2 data
            q_hat = _recalibrate_adaptive_cp(model, stage2, c_scale, alpha,
                                             oracle_var_fn)
        else:
            h_scalar = float(model.params.h)
            h_vals   = np.full(M, h_scalar)

        # --- CDF on t_grid (used for L1 and L2) ---
        if h_mode == "adaptive":
            F_tgrid = _perpoint_cdf_on_tgrid(model, t_grid, C_eval, h_vals)  # (M, T)
        else:
            F_tgrid = _batch_cdf_on_tgrid(model, t_grid, C_eval)             # (M, T)

        # --- L1: pre-CP quantile estimates (estimation level, before CP calibration) ---
        q_lo_hat = _invert_cdf(F_tgrid, t_grid, alpha / 2)
        q_hi_hat = _invert_cdf(F_tgrid, t_grid, 1 - alpha / 2)

        # --- L2: CP interval bounds [L(x), U(x)] via CDF inversion ---
        # {y : |F̂(y|x) - 0.5| <= q_hat} = {y : F̂(y|x) ∈ [0.5-q_hat, 0.5+q_hat]}
        tau_lo = float(np.clip(0.5 - q_hat, 0.0, 1.0))
        tau_hi = float(np.clip(0.5 + q_hat, 0.0, 1.0))
        L_arr  = _invert_cdf(F_tgrid, t_grid, tau_lo)
        U_arr  = _invert_cdf(F_tgrid, t_grid, tau_hi)

        # --- L3: MC conditional coverage ---
        Y_flat = model.Y.ravel()

        X_mc = np.repeat(x_eval, B)
        Y_mc = sim_fn(X_mc, random_state=rng_seed + 3).reshape(M, B)

        B_chunk = min(200, B)
        cov_mc  = np.empty(M)

        for m in range(M):
            c_m = C_eval[:, m]
            Y_b = Y_mc[m]
            # adaptive: use per-point h; fixed: use model's h
            ind_m = (make_indicator(model.indicator_type, float(h_vals[m]))
                     if h_mode == "adaptive" else model.indicator)

            F_b = np.empty(B)
            for b0 in range(0, B, B_chunk):
                Y_chunk = Y_b[b0:b0 + B_chunk]
                G_chunk = ind_m.g_matrix(Y_flat, Y_chunk)
                G_bar   = G_chunk.reshape(model.n, model.r, -1).mean(axis=1)
                F_b[b0:b0 + B_chunk] = c_m @ G_bar

            np.clip(F_b, 0.0, 1.0, out=F_b)
            score_b   = np.abs(F_b - 0.5)
            cov_mc[m] = np.mean(score_b <= q_hat)

        for m in range(M):
            rows.append({
                "macrorep":    macrorep_id,
                "n_0":         n_0,
                "x_eval":      float(x_eval[m]),
                "L":           float(L_arr[m]),
                "U":           float(U_arr[m]),
                "q_hat":       float(q_hat),
                "cov_mc":      float(cov_mc[m]),
                "q_lo_hat":    float(q_lo_hat[m]),
                "q_hi_hat":    float(q_hi_hat[m]),
                "q_lo_oracle": float(q_lo_oracle[m]),
                "q_hi_oracle": float(q_hi_oracle[m]),
                "h_at_x":      float(h_vals[m]),
            })

    return rows


# top-level worker for ProcessPoolExecutor pickling
def _run_task(args_tuple):
    k, base_seed, config, sim, params_by_n, h_mode, c_scale, dtype = args_tuple
    return run_one_macrorep(k, base_seed, config, sim, params_by_n, h_mode, c_scale, dtype)


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    target = 1.0 - alpha
    records = []
    for n_0, g_n in df.groupby("n_0"):
        for macro_id, g_m in g_n.groupby("macrorep"):
            delta_cov = np.abs(g_m["cov_mc"].values - target)
            # L2 endpoint errors (require oracle cols)
            has_oracle = g_m["q_lo_oracle"].notna().all()
            if has_oracle:
                err_L = np.abs(g_m["L"].values   - g_m["q_lo_oracle"].values)
                err_U = np.abs(g_m["U"].values    - g_m["q_hi_oracle"].values)
                err_qlo = np.abs(g_m["q_lo_hat"].values - g_m["q_lo_oracle"].values)
                err_qhi = np.abs(g_m["q_hi_hat"].values - g_m["q_hi_oracle"].values)
            else:
                nan_M = np.full(len(g_m), np.nan)
                err_L = err_U = err_qlo = err_qhi = nan_M

            records.append({
                "n_0":           n_0,
                "macrorep":      macro_id,
                # L3
                "mae_cov":       delta_cov.mean(),
                "sup_err_cov":   delta_cov.max(),
                # L2
                "mae_ep_L":      err_L.mean(),
                "mae_ep_U":      err_U.mean(),
                "sup_ep":        np.maximum(err_L, err_U).max(),
                # L1
                "mae_q_lo":      err_qlo.mean(),
                "mae_q_hi":      err_qhi.mean(),
                "sup_q":         np.maximum(err_qlo, err_qhi).max(),
            })

    per_macro = pd.DataFrame(records)
    agg_rows = []
    for n_0, g in per_macro.groupby("n_0"):
        row = {"n_0": n_0, "n_macro": len(g)}
        for col in ["mae_cov", "sup_err_cov",
                    "mae_ep_L", "mae_ep_U", "sup_ep",
                    "mae_q_lo", "mae_q_hi", "sup_q"]:
            row[f"{col}_mean"] = g[col].mean()
            row[f"{col}_sd"]   = g[col].std(ddof=1) if len(g) > 1 else np.nan
        agg_rows.append(row)
    return pd.DataFrame(agg_rows)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def get_config(cfg_path: Path | None = None) -> dict:
    path = cfg_path or (_HERE / "config.txt")
    raw  = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        raw[k.strip()] = v.strip()

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
        "n_jobs":      int(raw.get("n_jobs", "4")),
        "cv_max_n":    int(raw["cv_max_n"]) if "cv_max_n" in raw else None,
        "base_seed":   int(raw["base_seed"]),
        "simulators":  [s.strip() for s in raw["simulators"].split(",")],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Asymptotic consistency experiment for CKME-CP")
    parser.add_argument("--h_mode",     type=str, default="fixed",
                        choices=["fixed", "adaptive"],
                        help="Bandwidth mode: fixed (CV-tuned scalar) or adaptive h(x)=c*sigma_hat(x)")
    parser.add_argument("--c_scale",    type=float, default=2.0,
                        help="Scale factor for adaptive-h (ignored for fixed mode)")
    parser.add_argument("--simulators", type=str, default=None,
                        help="Comma-separated simulator list (overrides config)")
    parser.add_argument("--n_vals",     type=str, default=None,
                        help="Comma-separated n values (overrides config)")
    parser.add_argument("--n_macro",    type=int, default=None)
    parser.add_argument("--n_jobs",     type=int, default=None)
    parser.add_argument("--base_seed",  type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: auto from h_mode)")
    parser.add_argument("--config",     type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained_params.json (default: exp_conditional_coverage/pretrained_params.json)")
    parser.add_argument("--cv_max_n",   type=int, default=None,
                        help="Only tune CV for n <= cv_max_n (overrides config)")
    parser.add_argument("--dtype",      type=str, default="float32",
                        choices=["float32", "float64"])
    parser.add_argument("--quick",      action="store_true",
                        help="Scale down for fast end-to-end test: n_vals=[32,64], n_macro=2, B_test=100")
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else None
    config   = get_config(cfg_path)

    # CLI overrides
    if args.n_vals     is not None: config["n_vals"]     = [int(v) for v in args.n_vals.split(",")]
    if args.n_macro    is not None: config["n_macro"]    = args.n_macro
    if args.base_seed  is not None: config["base_seed"]  = args.base_seed
    if args.simulators is not None: config["simulators"] = [s.strip() for s in args.simulators.split(",")]

    # Quick mode: scaled-down sizes for pipeline testing
    if args.quick:
        config["n_vals"]  = [32, 64]
        config["n_macro"] = 2
        config["B_test"]  = 100
        config["M_eval"]  = 20
        print("Quick mode: n_vals=[32,64], n_macro=2, B_test=100, M_eval=20")

    cv_max_n = args.cv_max_n if args.cv_max_n is not None else config.get("cv_max_n")
    n_macro  = config["n_macro"]
    n_jobs   = min(args.n_jobs if args.n_jobs is not None else config.get("n_jobs", 4), n_macro)
    dtype    = np.float32 if args.dtype == "float32" else np.float64
    h_mode   = args.h_mode
    c_scale  = args.c_scale

    # Output directory
    if args.output_dir:
        out_base = Path(args.output_dir)
    elif h_mode == "adaptive":
        out_base = _HERE / f"output_consistency_adaptive_c{c_scale:.2f}"
    else:
        out_base = _HERE / "output_consistency_fixed"
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"h_mode     : {h_mode}" + (f" (c_scale={c_scale})" if h_mode == "adaptive" else ""))
    print(f"simulators : {config['simulators']}")
    print(f"n_vals     : {config['n_vals']}")
    print(f"r_0={config['r_0']}, r_1={config['r_1']}, alpha={config['alpha']}")
    print(f"n_macro={n_macro}, n_jobs={n_jobs}, dtype={args.dtype}, cv_max_n={cv_max_n}")
    print(f"M_eval={config['M_eval']}, B_test={config['B_test']}")
    print(f"Output dir : {out_base}")

    for sim in config["simulators"]:
        print(f"\n=== Simulator: {sim} ===")

        # --- Load pretrained hyperparameters ---
        pretrained_path = Path(args.pretrained) if args.pretrained \
            else (_HERE / "pretrained_params.json")
        params_by_n: dict[int, Params] = {}

        if pretrained_path.exists():
            with open(pretrained_path) as f:
                cache = json.load(f)
            sim_cache = cache.get(sim, {})

            tunable_ns = [n for n in config["n_vals"] if cv_max_n is None or n <= cv_max_n]
            reuse_ns   = [n for n in config["n_vals"] if n not in tunable_ns]

            for n in tunable_ns:
                key = str(n)
                if key in sim_cache:
                    d = sim_cache[key]
                    params_by_n[n] = Params(ell_x=d["ell_x"], lam=d["lam"], h=d["h"])
                    print(f"  n={n}: loaded  -> ell_x={d['ell_x']}, lam={d['lam']}, h={d['h']}")
                else:
                    print(f"  n={n}: WARNING pretrained_params.json has no entry for {sim}/n={n}. "
                          "Run pretrain_params.py first. Using default params.")
                    params_by_n[n] = Params(ell_x=0.5, lam=1e-3, h=0.1)

            if reuse_ns and tunable_ns:
                src_n = max(tunable_ns)
                src_p = params_by_n.get(src_n, Params(ell_x=0.5, lam=1e-3, h=0.1))
                for n in reuse_ns:
                    params_by_n[n] = src_p
                    print(f"  n={n}: reused  from n={src_n}")
        else:
            print(f"  WARNING: {pretrained_path} not found. Using default params for all n. "
                  "Run pretrain_params.py first.")
            for n in config["n_vals"]:
                params_by_n[n] = Params(ell_x=0.5, lam=1e-3, h=0.1)

        # --- Run macroreps ---
        all_rows: list[dict] = []
        t0 = time.time()

        if n_jobs <= 1:
            for k in range(n_macro):
                rows = run_one_macrorep(k, config["base_seed"], config, sim,
                                        params_by_n, h_mode, c_scale, dtype)
                all_rows.extend(rows)
                elapsed = time.time() - t0
                print(f"  macrorep {k+1}/{n_macro} done  ({elapsed:.0f}s elapsed)")
        else:
            task_args = [
                (k, config["base_seed"], config, sim, params_by_n, h_mode, c_scale, dtype)
                for k in range(n_macro)
            ]
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = {pool.submit(_run_task, t): t[0] for t in task_args}
                done = 0
                for fut in as_completed(futures):
                    k = futures[fut]
                    try:
                        rows = fut.result()
                        all_rows.extend(rows)
                    except Exception as e:
                        print(f"  macrorep {k} FAILED: {e}")
                    done += 1
                    elapsed = time.time() - t0
                    print(f"  macrorep {k+1} done ({done}/{n_macro}, {elapsed:.0f}s elapsed)")

        # --- Save results ---
        df = pd.DataFrame(all_rows)
        results_path = out_base / f"results_{sim}.csv"
        df.to_csv(results_path, index=False)
        print(f"  Saved raw results -> {results_path}")

        summary = compute_summary(df, config["alpha"])
        summary_path = out_base / f"summary_{sim}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Saved summary     -> {summary_path}")

        # Print compact table
        cols_show = ["n_0",
                     "mae_cov_mean", "sup_err_cov_mean",
                     "mae_ep_L_mean", "mae_ep_U_mean",
                     "mae_q_lo_mean", "mae_q_hi_mean"]
        cols_show = [c for c in cols_show if c in summary.columns]
        print(summary[cols_show].to_string(index=False))

    total = time.time() - t0
    print(f"\nDone. Total wall time: {total:.0f}s")


if __name__ == "__main__":
    main()
