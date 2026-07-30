"""
run_cond_cov.py

Asymptotic conditional coverage experiment.

For each n in n_vals, fit CKME-CP on n_0=n sites (r_0 reps each), then calibrate
with Stage 2 (n_1=n, r_1 reps). On a fixed evaluation grid of M_eval x-points,
estimate conditional coverage using B_test fresh draws at each x:

    Cov_n(x) = (1/B) * sum_b 1{Y_b(x) in [L_n(x), U_n(x)]}

Metrics saved:
  results_{sim}.csv   -- per (macrorep, n, x_eval) row: L, U, cov_mc, q_lo_oracle, q_hi_oracle
  summary_{sim}.csv   -- per n: MAE-Cov, SupErr, EndpointErr_L, EndpointErr_U (mean ± SD)

Usage (from project root):
  python exp_conditional_coverage/run_cond_cov.py
  python exp_conditional_coverage/run_cond_cov.py --n_macro 5 --n_jobs 4
  python exp_conditional_coverage/run_cond_cov.py --simulators exp2 --n_macro 2
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm, gamma as _gamma

from Two_stage import run_stage1_train, run_stage2
from Two_stage.sim_functions import get_experiment_config
from Two_stage.sim_functions.exp1 import exp1_true_function, exp1_noise_variance_function
from Two_stage.sim_functions.exp2 import exp2_true_function, exp2_noise_variance_function
from CKME.parameters import Params, ParamGrid
from CKME.coefficients import compute_ckme_coeffs

_HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Config loader
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
        "n_vals":     [int(v) for v in raw["n_vals"].split(",")],
        "r_0":        int(raw["r_0"]),
        "r_1":        int(raw["r_1"]),
        "alpha":      float(raw["alpha"]),
        "t_grid_size": int(raw["t_grid_size"]),
        "method":     raw["method"],
        "mixed_ratio": float(raw["mixed_ratio"]),
        "n_cand":     int(raw["n_cand"]),
        "M_eval":     int(raw["M_eval"]),
        "B_test":     int(raw["B_test"]),
        "n_macro":    int(raw["n_macro"]),
        "base_seed":  int(raw["base_seed"]),
        "simulators": [s.strip() for s in raw["simulators"].split(",")],
        # Tuning config (with defaults if not present)
        "tune_params": raw.get("tune_params", "false").lower() == "true",
        "h_default":   float(raw.get("h_default", "0.2")),
        "ell_x_list":  [float(v) for v in raw.get("ell_x_list", "0.5,1.0,2.0").split(",")],
        "lam_list":    [float(v) for v in raw.get("lam_list", "0.001,0.01,0.1").split(",")],
        "h_list":      [float(v) for v in raw.get("h_list", "0.1,0.2,0.3").split(",")],
        "cv_folds":    int(raw.get("cv_folds", "5")),
    }


# ---------------------------------------------------------------------------
# Oracle quantile helpers
# ---------------------------------------------------------------------------

def oracle_quantiles_exp1(x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact oracle quantiles for exp1: MG1 queue with Gaussian heteroscedastic noise."""
    mu = exp1_true_function(x)
    sigma = np.sqrt(exp1_noise_variance_function(x))
    q_lo = mu + sigma * _norm.ppf(alpha / 2)
    q_hi = mu + sigma * _norm.ppf(1 - alpha / 2)
    return q_lo, q_hi


def oracle_quantiles_exp2(x: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact oracle quantiles for exp2: Gaussian heteroscedastic."""
    mu = exp2_true_function(x)
    sigma = np.sqrt(exp2_noise_variance_function(x))
    q_lo = mu + sigma * _norm.ppf(alpha / 2)
    q_hi = mu + sigma * _norm.ppf(1 - alpha / 2)
    return q_lo, q_hi


def oracle_quantiles_nongauss_B2L(x: np.ndarray, alpha: float, k: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Oracle quantiles for nongauss_B2L: centered Gamma noise, k=2."""
    from math import sqrt
    sigma_tar = 0.1 + 0.1 * (x - np.pi) ** 2
    theta = sigma_tar / sqrt(k)          # scale of Gamma(k, theta)
    mu = exp2_true_function(x)
    # ε = Gamma(k, theta) - k*theta  =>  q_tau(ε) = gamma.ppf(tau, a=k, scale=theta) - k*theta
    q_lo = mu + _gamma.ppf(alpha / 2, a=k, scale=theta) - k * theta
    q_hi = mu + _gamma.ppf(1 - alpha / 2, a=k, scale=theta) - k * theta
    return q_lo, q_hi


_ORACLE_FN = {
    "exp1": oracle_quantiles_exp1,
    "exp2": oracle_quantiles_exp2,
    "nongauss_B2L": oracle_quantiles_nongauss_B2L,
}

# Fixed hyperparams fallback
_DEFAULT_PARAMS = Params(ell_x=0.5, lam=0.001, h=0.1)


# ---------------------------------------------------------------------------
# Two-stage hyperparameter tuning (per simulator × n_val)
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
    """
    Two-stage CV:
      Stage 1 — grid search over (ell_x, lam) with fixed h=h_default
      Stage 2 — grid search over h with the best (ell_x, lam)
    Uses a dedicated pilot dataset (cv_seed) separate from macrorep data.
    """
    # Stage 1: tune ell_x and lam
    grid1 = ParamGrid(ell_x_list=ell_x_list, lam_list=lam_list, h_list=[h_default])
    result1 = run_stage1_train(
        n_0=n_0,
        r_0=r_0,
        simulator_func=simulator_func,
        param_grid=grid1,
        t_grid_size=t_grid_size,
        cv_folds=cv_folds,
        random_state=cv_seed,
    )
    best_ell_x = result1.params.ell_x
    best_lam   = result1.params.lam

    # Stage 2: tune h
    grid2 = ParamGrid(ell_x_list=[best_ell_x], lam_list=[best_lam], h_list=h_list)
    result2 = run_stage1_train(
        n_0=n_0,
        r_0=r_0,
        simulator_func=simulator_func,
        param_grid=grid2,
        t_grid_size=t_grid_size,
        cv_folds=cv_folds,
        random_state=cv_seed + 1,
    )
    return result2.params


# ---------------------------------------------------------------------------
# Core: one macrorep
# ---------------------------------------------------------------------------

def run_one_macrorep(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    params_by_n: dict,          # {n_0: Params}
    dtype=None,
    stage_ratio: float = 1.0,
) -> list[dict]:
    """Run one macro-rep across all n_vals. Returns list of row dicts.

    Parameters
    ----------
    stage_ratio : float
        Fraction of n used per stage. E.g. 0.5 means Stage1=Stage2=n//2.
        Default=1.0 (original: both stages use n).
    dtype : numpy dtype or None
        Working precision for CKME Cholesky (float32 halves memory). Default=None (float64).
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

    # Fixed evaluation grid (same across macroreps)
    x_eval = np.linspace(x_lo, x_hi, M)  # (M,)
    X_eval = x_eval.reshape(-1, 1)         # (M, 1)

    # Oracle quantiles at eval points (None if not available)
    oracle_fn = _ORACLE_FN.get(simulator_func)
    if oracle_fn is not None:
        q_lo_oracle, q_hi_oracle = oracle_fn(x_eval, alpha)
    else:
        q_lo_oracle = q_hi_oracle = np.full(M, np.nan)

    rows = []
    for idx_n, n_0 in enumerate(n_vals):
        rng_seed = seed + idx_n * 1000

        # Actual sites per stage
        n_stage = max(2, int(n_0 * stage_ratio))

        # Candidate points for Stage 2 (must be >= n_stage)
        n_cand_eff = max(n_cand, 2 * n_stage)
        rng_cand = np.random.default_rng(rng_seed)
        X_cand = rng_cand.uniform(x_lo, x_hi, size=(n_cand_eff, 1))

        # Stage 1: train CKME model
        stage1 = run_stage1_train(
            n_0=n_stage,
            r_0=r_0,
            simulator_func=simulator_func,
            params=params_by_n[n_0],
            t_grid_size=t_grid_size,
            random_state=rng_seed + 1,
            dtype=dtype,
        )

        # Stage 2: site selection + CP calibration
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

        model  = stage2.model
        q_hat  = stage2.cp.q_hat

        # Pre-compute CKME coefficients for all M eval points: (n_sites, M)
        C_eval = compute_ckme_coeffs(model.L, model.kx, model.X, X_eval)

        # Training Y flattened: (n_sites * r_0,)
        Y_flat_train = model.Y.ravel()

        # Generate all B*M fresh test draws at once
        rng_seed_y = rng_seed + 3
        X_mc = np.repeat(x_eval, B)                        # (M*B,)
        Y_mc = simulator(X_mc, random_state=rng_seed_y)    # (M*B,)
        Y_mc = Y_mc.reshape(M, B)                          # (M, B)

        # Fixed indicator (same h everywhere, from CV-tuned params)
        indicator = model.indicator

        # Chunk B to bound peak memory: G_all shape = (n*r_0, B_chunk).
        B_chunk = min(200, B)

        h_fixed = float(model.params.h)   # scalar, same for all eval points
        cov_mc = np.empty(M)
        for m in range(M):
            c_m = C_eval[:, m]   # (n_sites,)
            Y_b = Y_mc[m]        # (B,)

            F_b = np.empty(B)
            for b0 in range(0, B, B_chunk):
                Y_chunk = Y_b[b0:b0 + B_chunk]
                # G matrix: (n_sites * r_0, B_chunk)
                G_chunk = indicator.g_matrix(Y_flat_train, Y_chunk)
                # Average over r_0: (n_sites, B_chunk)
                G_bar = G_chunk.reshape(model.n, model.r, -1).mean(axis=1)
                F_b[b0:b0 + B_chunk] = c_m @ G_bar

            np.clip(F_b, 0.0, 1.0, out=F_b)

            # Nonconformity score: |F̂ - 0.5|, covered if <= q_hat
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
                "h_at_x":      h_fixed,   # constant; saved for comparability with adaptive-h
            })

    return rows


# ---------------------------------------------------------------------------
# Summary: aggregate across macroreps
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    For each n_0, compute per-macrorep metrics then report mean ± SD.
    Metrics: MAE-Cov, SupErr.
    """
    target = 1.0 - alpha
    records = []
    for n_0, g_n in df.groupby("n_0"):
        for macro_id, g_m in g_n.groupby("macrorep"):
            delta = np.abs(g_m["cov_mc"].values - target)
            records.append({
                "n_0":     n_0,
                "macrorep": macro_id,
                "mae_cov": delta.mean(),
                "sup_err": delta.max(),
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
    parser = argparse.ArgumentParser(description="Asymptotic conditional coverage experiment")
    parser.add_argument("--n_macro",    type=int,   default=None, help="Override n_macro in config")
    parser.add_argument("--n_jobs",     type=int,   default=4,    help="Parallel workers")
    parser.add_argument("--simulators", type=str,   default=None, help="Comma-separated simulator list")
    parser.add_argument("--base_seed",  type=int,   default=None)
    parser.add_argument("--output_dir", type=str,   default=None)
    parser.add_argument("--config",     type=str,   default=None)
    parser.add_argument("--n_vals",     type=str,   default=None,
                        help="Comma-separated n values. Overrides config.txt n_vals.")
    parser.add_argument("--r_0",        type=int,   default=None,
                        help="Stage 1 replications per site. Overrides config.txt r_0.")
    parser.add_argument("--r_1",        type=int,   default=None,
                        help="Stage 2 replications per site. Overrides config.txt r_1.")
    parser.add_argument("--B_test",     type=int,   default=None,
                        help="Fresh test draws per eval point for MC coverage. Overrides config.txt B_test.")
    parser.add_argument("--stage_ratio", type=float, default=1.0,
                        help="Fraction of n used per stage. E.g. 0.5 means Stage1=Stage2=n//2 "
                             "(total budget n split evenly). Default=1.0 (both stages use n).")
    parser.add_argument("--cv_max_n",   type=int,   default=None,
                        help="Only run CV tuning for n <= cv_max_n; reuse largest tuned params "
                             "for larger n. Avoids expensive CV at huge n. Default=None (CV at every n).")
    parser.add_argument("--dtype",      type=str,   default="float32",
                        choices=["float32", "float64"],
                        help="Working precision for CKME Cholesky. float32 halves memory. Default=float32.")
    parser.add_argument("--no_tune",    action="store_true",
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
    if args.n_vals is not None:
        config["n_vals"] = [int(v) for v in args.n_vals.split(",")]
    if args.r_0 is not None:
        config["r_0"] = args.r_0
    if args.r_1 is not None:
        config["r_1"] = args.r_1
    if args.B_test is not None:
        config["B_test"] = args.B_test

    stage_ratio = args.stage_ratio
    cv_max_n    = args.cv_max_n
    dtype       = np.float32 if args.dtype == "float32" else np.float64

    # Auto-scale n_cand so that n_cand >= max n_stage (same logic as adaptive-h script)
    max_n_stage = max(max(2, int(n * stage_ratio)) for n in config["n_vals"])
    if config["n_cand"] < max_n_stage:
        new_n_cand = max(2 * max_n_stage, config["n_cand"])
        print(f"  Note: bumping n_cand {config['n_cand']} -> {new_n_cand} "
              f"to satisfy n_cand >= max_n_stage={max_n_stage} (stage_ratio={stage_ratio})")
        config["n_cand"] = new_n_cand

    out_dir = Path(args.output_dir) if args.output_dir else (_HERE / "output")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_macro   = config["n_macro"]
    base_seed = config["base_seed"]
    simulators = config["simulators"]
    n_jobs    = min(args.n_jobs, n_macro)

    n_stages_actual = [max(2, int(n * stage_ratio)) for n in config["n_vals"]]
    total_obs_per_n = [
        max(2, int(n * stage_ratio)) * config["r_0"] + max(2, int(n * stage_ratio)) * config["r_1"]
        for n in config["n_vals"]
    ]
    print(f"Simulators : {simulators}")
    print(f"n_vals     : {config['n_vals']}  (stage_ratio={stage_ratio})")
    print(f"n_stage    : {n_stages_actual}  (actual sites per stage, Stage1=Stage2)")
    print(f"r_0={config['r_0']}, r_1={config['r_1']}  ->  total obs per n: {total_obs_per_n}")
    print(f"n_macro    : {n_macro}  |  n_jobs: {n_jobs}")
    print(f"dtype      : {args.dtype}  |  cv_max_n: {cv_max_n}")
    print(f"M_eval={config['M_eval']}, B_test={config['B_test']}, alpha={config['alpha']}")
    print(f"Output dir : {out_dir}")

    for sim in simulators:
        print(f"\n=== Simulator: {sim} ===")

        # --- Per-n hyperparameter tuning (done once, shared across macroreps) ---
        params_by_n: dict = {}
        params_cache_path = out_dir / f"tuned_params_{sim}.json"

        if config["tune_params"]:
            cached: dict = {}
            if params_cache_path.exists():
                with open(params_cache_path) as f:
                    cached = json.load(f)
                print(f"  Loaded tuned params from {params_cache_path}")

            # cv_max_n: only tune for n <= cv_max_n, reuse for larger n
            tunable_ns = [n for n in config["n_vals"] if cv_max_n is None or n <= cv_max_n]
            if not tunable_ns:
                ref_n = cv_max_n if cv_max_n is not None else min(config["n_vals"])
                tunable_ns = [ref_n]
            reuse_ns = [n for n in config["n_vals"] if n not in tunable_ns]
            needs_tune = [n for n in tunable_ns if str(n) not in cached]
            if needs_tune:
                print(f"  Tuning params for n={needs_tune} (two-stage CV)...")
            if reuse_ns:
                print(f"  Reusing largest-tuned params for n={reuse_ns} (cv_max_n={cv_max_n})")

            for idx_n, n_0 in enumerate(tunable_ns):
                # Tune at n_stage (actual sites), not n_0 (budget label)
                n_stage_cv = max(2, int(n_0 * stage_ratio))
                if str(n_0) in cached:
                    d = cached[str(n_0)]
                    params_by_n[n_0] = Params(ell_x=d["ell_x"], lam=d["lam"], h=d["h"])
                    print(f"    n={n_0} (n_stage={n_stage_cv}): loaded -> ell_x={d['ell_x']}, lam={d['lam']}, h={d['h']}")
                else:
                    cv_seed = base_seed + 999999 + idx_n * 100
                    p = tune_params_for_n(
                        simulator_func=sim,
                        n_0=n_stage_cv,
                        r_0=config["r_0"],
                        t_grid_size=config["t_grid_size"],
                        cv_seed=cv_seed,
                        cv_folds=config["cv_folds"],
                        ell_x_list=config["ell_x_list"],
                        lam_list=config["lam_list"],
                        h_list=config["h_list"],
                        h_default=config["h_default"],
                    )
                    params_by_n[n_0] = p
                    cached[str(n_0)] = {"ell_x": p.ell_x, "lam": p.lam, "h": p.h}
                    print(f"    n={n_0} (n_stage={n_stage_cv}): tuned -> ell_x={p.ell_x}, lam={p.lam}, h={p.h}")

            # Fill reuse_ns with largest tuned params
            if reuse_ns:
                src_n = max(tunable_ns)
                src_p = params_by_n[src_n]
                for n_0 in reuse_ns:
                    params_by_n[n_0] = src_p
                    print(f"    n={n_0}: reused from n={src_n} -> ell_x={src_p.ell_x}, lam={src_p.lam}, h={src_p.h}")

            with open(params_cache_path, "w") as f:
                json.dump(cached, f, indent=2)
            print(f"  Params cache saved -> {params_cache_path}")
        else:
            for n_0 in config["n_vals"]:
                params_by_n[n_0] = _DEFAULT_PARAMS

        all_rows: list[dict] = []

        if n_jobs <= 1:
            for k in range(n_macro):
                rows = run_one_macrorep(k, base_seed, config, sim, params_by_n,
                                        dtype=dtype, stage_ratio=stage_ratio)
                all_rows.extend(rows)
                print(f"  macrorep {k+1}/{n_macro} done")
        else:
            futures = {}
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                for k in range(n_macro):
                    f = ex.submit(run_one_macrorep, k, base_seed, config, sim, params_by_n,
                                  dtype, stage_ratio)
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
        print(f"  Saved raw results -> {results_path}")

        summary = compute_summary(df, config["alpha"])
        summary_path = out_dir / f"summary_{sim}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Saved summary     -> {summary_path}")
        print(summary[["n_0", "mae_cov_mean", "mae_cov_sd", "sup_err_mean", "sup_err_sd"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
