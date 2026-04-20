"""
run_plugin.py

Plug-in sigma variant of the score homogeneity experiment.

Instead of oracle sigma(x), we use sigma_hat(x) built from sample std across
the r_0 replications at each Stage 1 site, interpolated to arbitrary x via
kernel smoothing with bandwidth h_sigma.

Adds a per-simulator diagnostic measuring sigma_hat RMSE against sigma_true.

Usage:
  python exp_score_homogeneity_plugin/run_plugin.py
  python exp_score_homogeneity_plugin/run_plugin.py --simulators exp2
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

from Two_stage import run_stage1_train, run_stage2
from Two_stage.sim_functions import get_experiment_config
from CKME.parameters import Params
from CKME.coefficients import compute_ckme_coeffs

# Reuse all helpers from the oracle experiment.
from exp_score_homogeneity import run_homogeneity as oracle_mod
from exp_score_homogeneity.run_homogeneity import (
    _register_mild_simulator,
    _batch_cdf_on_tgrid,
    _perpoint_cdf_on_tgrid,
    _invert_cdf,
    _calibrate_scores,
    _compute_score_ecdf_at_x,
    _ks_from_pooled,
    compute_summary,
    _ORACLE_VAR,
    _ORACLE_QUANTILE,
    _ORACLE_CDF,
)

_HERE = Path(__file__).parent


# ---------------------------------------------------------------------------
# Plug-in sigma_hat: sample std per site + kernel smoothing in X-space
# ---------------------------------------------------------------------------

def _site_sample_std(Y_site: np.ndarray) -> np.ndarray:
    """Y_site shape (n, r) -> sample std per site, shape (n,)."""
    return Y_site.std(axis=1, ddof=1)


def _kernel_smooth_sigma(X_sites: np.ndarray, sigma_sites: np.ndarray,
                          h_sigma: float):
    """Return closure x -> sigma_hat(x) via Nadaraya-Watson smoothing.

    X_sites: shape (n, d) or (n,); sigma_sites: shape (n,).
    Uses Gaussian kernel with bandwidth h_sigma in X-space.
    """
    X_sites = np.atleast_2d(X_sites)
    if X_sites.shape[0] == 1 and X_sites.shape[1] != 1:
        X_sites = X_sites.T  # make (n, d)
    n, d = X_sites.shape
    s = np.asarray(sigma_sites, dtype=float)

    def sigma_hat(x):
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        if x_arr.shape[0] == 1 and x_arr.shape[1] != d:
            x_arr = x_arr.T
        # pairwise sq distances (m, n)
        diff = x_arr[:, None, :] - X_sites[None, :, :]
        sq = np.sum(diff * diff, axis=-1)
        w = np.exp(-0.5 * sq / (h_sigma * h_sigma))
        w_sum = w.sum(axis=1)
        w_sum = np.where(w_sum < 1e-300, 1e-300, w_sum)
        return (w @ s) / w_sum

    return sigma_hat


def _compute_h_sigma(x_lo: float, x_hi: float, n_0: int,
                      factor: float) -> float:
    """h_sigma = factor * range(X) * n_0^(-1/5)."""
    return float(factor) * (x_hi - x_lo) * (n_0 ** (-1.0 / 5.0))


# ---------------------------------------------------------------------------
# One macrorep x one bandwidth config (plug-in sigma variant)
# ---------------------------------------------------------------------------

def run_one_config_plugin(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    shared_params: Params,
    bw_label: str,
    h_fixed: float | None,
    c_scale: float | None,
    oracle_var_fn,          # used ONLY for sigma_hat RMSE diagnostic
    oracle_quantile_fn,
    oracle_cdf_fn,
    h_sigma: float,
):
    """Run one macrorep for one bandwidth config using plug-in sigma_hat."""
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

    if c_scale is not None:
        train_params = shared_params
        h_mode = "adaptive"
    elif h_fixed is not None:
        train_params = Params(ell_x=shared_params.ell_x, lam=shared_params.lam,
                              h=h_fixed)
        h_mode = "fixed"
    else:
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

    # --- Build sigma_hat from Stage 1 replications ---
    # stage1.model.Y has shape (n_0, r_0); stage1.model.X has shape (n_0, d).
    Y_site = stage1.model.Y  # (n_0, r_0)
    sigma_sites = _site_sample_std(Y_site)  # (n_0,)
    sigma_hat_fn = _kernel_smooth_sigma(stage1.model.X, sigma_sites, h_sigma)

    def _var_plugin(x):
        sh = sigma_hat_fn(x)
        return np.maximum(sh, 1e-8) ** 2

    # --- sigma_hat diagnostic vs oracle (only needs oracle_var_fn here) ---
    sigma_true_eval = np.sqrt(oracle_var_fn(x_eval))
    sigma_hat_eval = sigma_hat_fn(x_eval)
    sigma_rmse = float(np.sqrt(np.mean((sigma_hat_eval - sigma_true_eval) ** 2)))
    sigma_max_err = float(np.max(np.abs(sigma_hat_eval - sigma_true_eval)))

    # --- Stage 2 ---
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

    C_eval = compute_ckme_coeffs(model.L, model.kx, model.X, X_eval)

    if h_mode == "adaptive":
        # Plug-in: h(x) = c_scale * sigma_hat(x)
        h_vals = c_scale * np.sqrt(np.maximum(_var_plugin(x_eval), 1e-8))
        X_cal = np.atleast_2d(stage2.X_stage2)
        Y_cal = np.asarray(stage2.Y_stage2).ravel()
        h_cal = c_scale * np.sqrt(np.maximum(_var_plugin(X_cal.ravel()), 1e-8))
        q_hat = _calibrate_scores(model, X_cal, Y_cal, "adaptive", h_cal, alpha)
    else:
        h_scalar = float(train_params.h)
        h_vals = np.full(M, h_scalar)
        q_hat = stage2.cp.q_hat

    if h_mode == "adaptive":
        F_tgrid = _perpoint_cdf_on_tgrid(model, t_grid, C_eval, h_vals)
    else:
        F_tgrid = _batch_cdf_on_tgrid(model, t_grid, C_eval)

    q_lo_hat = _invert_cdf(F_tgrid, t_grid, alpha / 2)
    q_hi_hat = _invert_cdf(F_tgrid, t_grid, 1 - alpha / 2)

    q_lo_oracle = oracle_quantile_fn(x_eval, alpha / 2)
    q_hi_oracle = oracle_quantile_fn(x_eval, 1 - alpha / 2)

    F_oracle = oracle_cdf_fn(x_eval, t_grid)
    cdf_err_per_x = np.mean(np.abs(F_tgrid - F_oracle), axis=1)

    tau_lo = float(np.clip(0.5 - q_hat, 0.0, 1.0))
    tau_hi = float(np.clip(0.5 + q_hat, 0.0, 1.0))
    L_arr = _invert_cdf(F_tgrid, t_grid, tau_lo)
    U_arr = _invert_cdf(F_tgrid, t_grid, tau_hi)

    score_samples = []
    cov_mc = np.empty(M)
    for m in range(M):
        Y_mc = sim_fn(np.full(B, x_eval[m]), random_state=rng_seed + 3 + m)
        scores_m = _compute_score_ecdf_at_x(
            model, C_eval[:, m], h_vals[m], Y_mc, h_mode
        )
        score_samples.append(scores_m)
        cov_mc[m] = np.mean(scores_m <= q_hat)

    ks_max, ks_mean, ks_per_x = _ks_from_pooled(score_samples)

    rows = []
    for m in range(M):
        rows.append({
            "macrorep": macrorep_id,
            "bandwidth": bw_label,
            "x_eval": float(x_eval[m]),
            "h_at_x": float(h_vals[m]),
            "sigma_hat": float(sigma_hat_eval[m]),
            "sigma_true": float(sigma_true_eval[m]),
            "sigma_rmse_macro": sigma_rmse,
            "sigma_max_err_macro": sigma_max_err,
            "h_sigma": float(h_sigma),
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
# Config loader (adds h_sigma_factor)
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
        "h_sigma_factor": float(raw.get("h_sigma_factor", "0.5")),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plug-in sigma score homogeneity experiment")
    parser.add_argument("--simulators", type=str, default=None)
    parser.add_argument("--n_macro", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--h_sigma_factor", type=float, default=None)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    _register_mild_simulator()

    cfg_path = Path(args.config) if args.config else None
    config = get_config(cfg_path)

    if args.simulators is not None:
        config["simulators"] = [s.strip() for s in args.simulators.split(",")]
    if args.n_macro is not None:
        config["n_macro"] = args.n_macro
    if args.base_seed is not None:
        config["base_seed"] = args.base_seed
    if args.h_sigma_factor is not None:
        config["h_sigma_factor"] = args.h_sigma_factor

    if args.quick:
        config["n_0"] = 64
        config["n_1"] = 64
        config["n_macro"] = 2
        config["B_test"] = 200
        config["M_eval"] = 10
        config["c_values"] = [1.0, 2.0]
        print("Quick mode: n_0=n_1=64, n_macro=2")

    n_macro = config["n_macro"]
    out_base = Path(args.output_dir) if args.output_dir else (_HERE / "output")
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"simulators : {config['simulators']}")
    print(f"n_0={config['n_0']}, r_0={config['r_0']}, "
          f"n_1={config['n_1']}, r_1={config['r_1']}, alpha={config['alpha']}")
    print(f"n_macro={n_macro}, h_sigma_factor={config['h_sigma_factor']}")
    print(f"Output dir : {out_base}")

    for sim in config["simulators"]:
        print(f"\n{'='*60}")
        print(f"Simulator: {sim}")
        print(f"{'='*60}")

        pretrained_path = Path(args.pretrained) if args.pretrained \
            else (_HERE / "pretrained_params.json")
        if not pretrained_path.exists():
            pretrained_path = _HERE.parent / "exp_score_homogeneity" / "pretrained_params.json"
        if not pretrained_path.exists():
            pretrained_path = _HERE.parent / "exp_nongauss" / "pretrained_params.json"

        if pretrained_path.exists():
            with open(pretrained_path) as f:
                cache = json.load(f)
            sim_cache = cache.get(sim, {})
            if "ell_x" in sim_cache:
                d = sim_cache
            else:
                n_key = str(config["n_0"])
                if n_key in sim_cache:
                    d = sim_cache[n_key]
                elif sim_cache:
                    d = next(iter(sim_cache.values()))
                else:
                    d = {"ell_x": 0.5, "lam": 1e-3, "h": 0.2}
            shared_params = Params(ell_x=d["ell_x"], lam=d["lam"], h=d["h"])
            print(f"  Shared params: ell_x={d['ell_x']}, lam={d['lam']}, h_cv={d['h']}")
        else:
            shared_params = Params(ell_x=0.5, lam=1e-3, h=0.2)
            print("  WARNING: using default params")

        bw_configs = []
        bw_configs.append(("fixed_small", config["h_fixed_small"], None))
        bw_configs.append(("fixed_cv", None, None))
        bw_configs.append(("fixed_large", config["h_fixed_large"], None))
        for c in config["c_values"]:
            bw_configs.append((f"adaptive_c{c:.1f}", None, c))

        oracle_var_fn = _ORACLE_VAR.get(sim)
        oracle_q_fn = _ORACLE_QUANTILE.get(sim)
        oracle_cdf_fn = _ORACLE_CDF.get(sim)
        if oracle_q_fn is None or oracle_cdf_fn is None or oracle_var_fn is None:
            print(f"  ERROR: missing oracle functions for {sim}")
            continue

        # Determine x range from a simulator config call for h_sigma
        exp_cfg = get_experiment_config(sim)
        x_lo = float(exp_cfg["bounds"][0][0])
        x_hi = float(exp_cfg["bounds"][1][0])
        h_sigma = _compute_h_sigma(x_lo, x_hi, config["n_0"],
                                    config["h_sigma_factor"])
        print(f"  h_sigma = {h_sigma:.4f} "
              f"(factor={config['h_sigma_factor']}, range={x_hi-x_lo:.3f}, "
              f"n_0={config['n_0']})")

        all_rows = []
        t0 = time.time()
        for k in range(n_macro):
            for bw_label, h_fixed, c_scale in bw_configs:
                rows, _ = run_one_config_plugin(
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
                    h_sigma=h_sigma,
                )
                all_rows.extend(rows)
            elapsed = time.time() - t0
            print(f"  macrorep {k+1}/{n_macro} done  "
                  f"({len(bw_configs)} configs, {elapsed:.0f}s elapsed)")

        df = pd.DataFrame(all_rows)
        results_path = out_base / f"results_{sim}.csv"
        df.to_csv(results_path, index=False)
        print(f"  Saved results -> {results_path}")

        summary = compute_summary(df, config["alpha"])
        # Add sigma_hat diagnostic columns (mean over macroreps)
        sigma_diag = (df.groupby("bandwidth")
                        .agg(sigma_rmse_mean=("sigma_rmse_macro", "mean"),
                             sigma_rmse_sd=("sigma_rmse_macro", "std"),
                             sigma_max_err_mean=("sigma_max_err_macro", "mean"))
                        .reset_index())
        summary = summary.merge(sigma_diag, on="bandwidth", how="left")

        summary_path = out_base / f"summary_{sim}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"  Saved summary -> {summary_path}")

        cols_show = ["bandwidth", "ks_max_mean", "cov_gap_sup_mean",
                      "q_err_sup_mean", "cdf_err_mean_mean",
                      "width_mean_mean", "sigma_rmse_mean"]
        cols_show = [c for c in cols_show if c in summary.columns]
        print(summary[cols_show].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
