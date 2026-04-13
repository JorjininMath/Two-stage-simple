"""
run_gibbs_compare.py

Compare CKME-CP against Gibbs et al. (RLCP) DGPs.

DGPs (location-scale, X in [-3, 3]):
  gibbs_s1: Y = 0.5*x + |sin(x)| * N(0,1)   -- sigma zeros at x=k*pi
  gibbs_s2: Y = 0.5*x + 2*dnorm(x,0,1.5) * N(0,1) -- bell-shaped sigma

Evaluation (matching Gibbs RLCP_Gibbs_comparison.R):
  - Local coverage at 21 centers in [-2.5, 2.5] with radius=0.4
  - Overall marginal coverage and interval width
  - X_test drawn from N(0,1) truncated to [-3, 3]

Usage (from project root):
  python exp_gibbs_compare/run_gibbs_compare.py
  python exp_gibbs_compare/run_gibbs_compare.py --n_macro 20 --n_workers 4
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

from Two_stage import run_stage1_train, run_stage2
from Two_stage.data_collection import collect_stage2_data
from Two_stage.config_utils import load_config_from_file, get_config, get_x_cand
from CKME.parameters import Params
from CKME.coefficients import compute_ckme_coeffs
from CKME.indicators import make_indicator

# Gibbs local-coverage evaluation grid (same as RLCP_Gibbs_comparison.R)
_CENTERS = np.arange(-2.5, 2.5 + 1e-9, 0.25)   # 21 points
_RADIUS  = 0.4

SIMULATORS = ["gibbs_s1", "gibbs_s2"]
CONFIG_PATH = _root / "exp_gibbs_compare" / "config.txt"


# ---------------------------------------------------------------------------
# X_test from N(0,1) truncated to [-3, 3]
# ---------------------------------------------------------------------------

def _draw_gaussian_test(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n points from N(0,1), reject outside [-3, 3]."""
    result = []
    while len(result) < n:
        batch = rng.standard_normal(n * 2)
        batch = batch[(batch >= -3.0) & (batch <= 3.0)]
        result.extend(batch.tolist())
    return np.array(result[:n], dtype=float).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Local coverage at Gibbs grid
# ---------------------------------------------------------------------------

def _local_coverage(x_test: np.ndarray, covered: np.ndarray) -> pd.DataFrame:
    """
    Compute empirical local coverage at each center in _CENTERS.

    Parameters
    ----------
    x_test : (n,) array of 1-D test inputs
    covered : (n,) binary array (1 = interval covers y)

    Returns
    -------
    DataFrame with columns: center, local_coverage, ball_size
    """
    rows = []
    for c in _CENTERS:
        mask = np.abs(x_test - c) <= _RADIUS
        n_ball = mask.sum()
        cov = covered[mask].mean() if n_ball > 0 else np.nan
        rows.append({"center": float(c), "local_coverage": cov, "ball_size": int(n_ball)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Adaptive h helpers
# ---------------------------------------------------------------------------

def _estimate_local_sigma(
    model, X_query: np.ndarray, c_scale: float = 2.0,
    h_min: float = 1e-3, n_neighbors: int = 5,
) -> np.ndarray:
    """k-NN estimate of sigma_hat(x), return h(x) = max(c_scale * sigma_hat, h_min)."""
    X_query = np.atleast_2d(X_query)
    s_train = np.sqrt(np.maximum(model.Y.var(axis=1, ddof=1), 0.0))
    diff = X_query[:, None, :] - model.X[None, :, :]
    D2 = (diff ** 2).sum(axis=2)
    k = min(n_neighbors, model.n)
    nn_idx = np.argpartition(D2, k, axis=1)[:, :k]
    sigma_hat = s_train[nn_idx].mean(axis=1)
    return np.maximum(c_scale * sigma_hat, h_min)


def _calibrate_adaptive(model, X_1, Y_stage2_matrix, alpha, c_scale, h_min=1e-3):
    """Re-calibrate CP with per-site adaptive h(x) = c_scale * sigma_hat(x)."""
    n_1, r_1 = Y_stage2_matrix.shape
    h_sites = _estimate_local_sigma(model, X_1, c_scale=c_scale, h_min=h_min)
    C_1 = compute_ckme_coeffs(model.L, model.kx, model.X, X_1)
    Y_flat = model.Y.ravel()
    n_cal = n_1 * r_1
    F_cal = np.empty(n_cal)
    for j in range(n_1):
        ind_j = make_indicator(model.indicator_type, float(h_sites[j]))
        G_all = ind_j.g_matrix(Y_flat, Y_stage2_matrix[j])
        G_bar = G_all.reshape(model.n, model.r, r_1).mean(axis=1)
        F_cal[j * r_1:(j + 1) * r_1] = np.clip(C_1[:, j] @ G_bar, 0.0, 1.0)
    scores = np.abs(F_cal - 0.5)
    k_idx = min(int(np.ceil((1 - alpha) * (1 + n_cal))), n_cal)
    return float(np.sort(scores)[k_idx - 1])


def _adaptive_predict_interval(model, X_query, t_grid, q_hat, c_scale, h_min=1e-3):
    """Predict intervals using per-point adaptive h(x)."""
    X_query = np.atleast_2d(X_query)
    q = X_query.shape[0]
    h_vals = _estimate_local_sigma(model, X_query, c_scale=c_scale, h_min=h_min)
    C_q = compute_ckme_coeffs(model.L, model.kx, model.X, X_query)
    Y_flat = model.Y.ravel()
    L = np.empty(q)
    U = np.empty(q)
    for i in range(q):
        ind_i = make_indicator(model.indicator_type, float(h_vals[i]))
        G_all = ind_i.g_matrix(Y_flat, t_grid)
        G_bar = G_all.reshape(model.n, model.r, -1).mean(axis=1)
        F_i = np.clip(C_q[:, i] @ G_bar, 0.0, 1.0)
        scores_i = np.abs(F_i - 0.5)
        mask = scores_i <= q_hat
        if mask.any():
            idx = np.where(mask)[0]
            L[i] = t_grid[idx[0]]
            U[i] = t_grid[idx[-1]]
        else:
            L[i] = t_grid[0]
            U[i] = t_grid[-1]
    return L, U


# ---------------------------------------------------------------------------
# Per-macrorep runner
# ---------------------------------------------------------------------------

def run_one_macrorep(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    out_dir: Path,
    method: str,
    params: Params | None = None,
    h_mode: str = "fixed",
    c_scale: float = 2.0,
) -> dict:
    seed = base_seed + macrorep_id * 10000
    rng  = np.random.default_rng(seed)

    n_0  = config["n_0"]
    r_0  = config["r_0"]
    n_1  = config["n_1"]
    r_1  = config["r_1"]
    n_t  = config["n_test"]
    alpha = config["alpha"]
    n_g  = config.get("t_grid_size", 500)
    if params is None:
        params = config["params"]

    X_cand = get_x_cand(simulator_func, config["n_cand"], random_state=seed + 1)

    stage1 = run_stage1_train(
        n_0=n_0, r_0=r_0,
        simulator_func=simulator_func,
        params=params,
        t_grid_size=n_g,
        random_state=seed + 2,
        verbose=False,
    )

    stage2 = run_stage2(
        stage1_result=stage1,
        X_cand=X_cand,
        n_1=n_1, r_1=r_1,
        simulator_func=simulator_func,
        method=method,
        alpha=alpha,
        random_state=seed + 3,
        verbose=False,
    )

    # --- Test data: X from N(0,1) truncated to [-3,3], Y from simulator ---
    X_test_sites = _draw_gaussian_test(n_t, rng)   # (n_test, 1)
    X_test, Y_test = collect_stage2_data(
        X_1=X_test_sites, r_1=1,
        simulator_func=simulator_func,
        random_state=seed + 4,
    )
    x_test_1d = X_test.ravel()

    # --- Predict intervals (fixed h) ---
    cp = stage2.cp
    L, U = cp.predict_interval(X_test, stage2.t_grid)
    covered = ((Y_test >= L) & (Y_test <= U)).astype(int)
    width   = U - L

    result = {
        "coverage": float(covered.mean()),
        "width":    float(width.mean()),
    }

    # --- Adaptive h ---
    if h_mode == "adaptive":
        model = stage2.model
        X_1 = stage2.X_1
        Y_s2_mat = stage2.Y_stage2.reshape(len(X_1), r_1)
        q_hat_adap = _calibrate_adaptive(model, X_1, Y_s2_mat, alpha, c_scale)
        L_a, U_a = _adaptive_predict_interval(
            model, X_test, stage2.t_grid, q_hat_adap, c_scale,
        )
        cov_a = ((Y_test >= L_a) & (Y_test <= U_a)).astype(int)
        width_a = U_a - L_a
        result["coverage_adaptive"] = float(cov_a.mean())
        result["width_adaptive"]    = float(width_a.mean())

    # --- Save per-point results ---
    case_name = f"{simulator_func}_{method}"
    case_dir  = out_dir / f"macrorep_{macrorep_id}" / f"case_{case_name}"
    case_dir.mkdir(parents=True, exist_ok=True)

    per_point = pd.DataFrame({
        "x": x_test_1d,
        "y": Y_test,
        "L": L,
        "U": U,
        "covered": covered,
        "width": width,
    })
    if h_mode == "adaptive":
        per_point["L_adaptive"] = L_a
        per_point["U_adaptive"] = U_a
        per_point["covered_adaptive"] = cov_a
        per_point["width_adaptive"] = width_a
    per_point.to_csv(case_dir / "per_point.csv", index=False)

    # --- Local coverage ---
    lc_df = _local_coverage(x_test_1d, covered)
    lc_df.to_csv(case_dir / "local_coverage.csv", index=False)
    result["local_coverage"] = lc_df["local_coverage"].values

    if h_mode == "adaptive":
        lc_adap = _local_coverage(x_test_1d, cov_a)
        lc_adap.to_csv(case_dir / "local_coverage_adaptive.csv", index=False)
        result["local_coverage_adaptive"] = lc_adap["local_coverage"].values

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CKME-CP vs Gibbs DGPs (local coverage comparison)")
    parser.add_argument("--config",      type=str, default=str(CONFIG_PATH))
    parser.add_argument("--output_dir",  type=str, default=None)
    parser.add_argument("--n_macro",     type=int, default=10)
    parser.add_argument("--base_seed",   type=int, default=42)
    parser.add_argument("--method",      type=str, default="lhs", choices=("lhs", "sampling", "mixed"))
    parser.add_argument("--n_workers",   type=int, default=1)
    parser.add_argument("--h_mode",      type=str, default="fixed", choices=("fixed", "adaptive"))
    parser.add_argument("--c_scale",     type=float, default=2.0)
    parser.add_argument("--macrorep_id", type=int, default=None,
                        help="If set, run only this single macrorep and exit (for SLURM array jobs)")
    args = parser.parse_args()

    config  = get_config(load_config_from_file(Path(args.config)), quick=False)
    if args.h_mode == "adaptive" and args.output_dir is None:
        out_dir = _root / "exp_gibbs_compare" / f"output_adaptive_c{args.c_scale:.2f}"
    else:
        out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_gibbs_compare" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained params if available
    pretrained_path = _root / "exp_gibbs_compare" / "pretrained_params.json"
    pretrained: dict[str, Params] = {}
    if pretrained_path.exists():
        with open(pretrained_path) as f:
            raw = json.load(f)
        pretrained = {sim: Params(**raw[sim]) for sim in raw if sim in SIMULATORS}
        print(f"Loaded pretrained params from {pretrained_path}")

    # --- SLURM single-task mode ---
    if args.macrorep_id is not None:
        k = args.macrorep_id
        print(f"[ARC] macrorep_id={k}, h_mode={args.h_mode}")
        for sim in SIMULATORS:
            print(f"  simulator: {sim}")
            one = run_one_macrorep(
                macrorep_id=k,
                base_seed=args.base_seed,
                config=config,
                simulator_func=sim,
                out_dir=out_dir,
                method=args.method,
                params=pretrained.get(sim),
                h_mode=args.h_mode,
                c_scale=args.c_scale,
            )
            msg = f"  cov={one['coverage']:.3f}, width={one['width']:.3f}"
            if "coverage_adaptive" in one:
                msg += f" | adap cov={one['coverage_adaptive']:.3f}, width={one['width_adaptive']:.3f}"
            print(f"  [{sim}] {msg}")
        print(f"[ARC] macrorep {k} done.")
        return

    all_rows = []
    for sim in SIMULATORS:
        print(f"\n--- Simulator: {sim} ---")

        if args.n_workers > 1:
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                futs = {
                    pool.submit(
                        run_one_macrorep,
                        k, args.base_seed, config, sim,
                        out_dir, args.method, pretrained.get(sim),
                        args.h_mode, args.c_scale,
                    ): k
                    for k in range(args.n_macro)
                }
                results: dict[int, dict] = {}
                for fut in as_completed(futs):
                    k = futs[fut]
                    results[k] = fut.result()
                    print(f"  macrorep {k} done")
            macrorep_results = [results[k] for k in range(args.n_macro)]
        else:
            macrorep_results = []
            for k in range(args.n_macro):
                one = run_one_macrorep(
                    macrorep_id=k,
                    base_seed=args.base_seed,
                    config=config,
                    simulator_func=sim,
                    out_dir=out_dir,
                    method=args.method,
                    params=pretrained.get(sim),
                    h_mode=args.h_mode,
                    c_scale=args.c_scale,
                )
                macrorep_results.append(one)
                msg = f"  macrorep {k}: cov={one['coverage']:.3f}, width={one['width']:.3f}"
                if "coverage_adaptive" in one:
                    msg += f" | adap cov={one['coverage_adaptive']:.3f}, width={one['width_adaptive']:.3f}"
                print(msg)

        cov_list   = [r["coverage"] for r in macrorep_results]
        width_list = [r["width"]    for r in macrorep_results]
        lc_mat     = np.array([r["local_coverage"] for r in macrorep_results])  # (n_macro, 21)

        print(f"  [fixed]    coverage: {np.mean(cov_list):.3f} ± {np.std(cov_list, ddof=1):.3f}  "
              f"width: {np.mean(width_list):.3f} ± {np.std(width_list, ddof=1):.3f}")

        row = {
            "simulator":       sim,
            "method":          "CKME-CP",
            "mean_coverage":   np.mean(cov_list),
            "sd_coverage":     np.std(cov_list, ddof=1) if len(cov_list) > 1 else np.nan,
            "mean_width":      np.mean(width_list),
            "sd_width":        np.std(width_list, ddof=1) if len(width_list) > 1 else np.nan,
            "n_macroreps":     len(cov_list),
        }

        if "coverage_adaptive" in macrorep_results[0]:
            cov_a_list   = [r["coverage_adaptive"] for r in macrorep_results]
            width_a_list = [r["width_adaptive"]    for r in macrorep_results]
            print(f"  [adaptive] coverage: {np.mean(cov_a_list):.3f} ± {np.std(cov_a_list, ddof=1):.3f}  "
                  f"width: {np.mean(width_a_list):.3f} ± {np.std(width_a_list, ddof=1):.3f}")
            row["mean_coverage_adaptive"] = np.mean(cov_a_list)
            row["sd_coverage_adaptive"]   = np.std(cov_a_list, ddof=1) if len(cov_a_list) > 1 else np.nan
            row["mean_width_adaptive"]    = np.mean(width_a_list)
            row["sd_width_adaptive"]      = np.std(width_a_list, ddof=1) if len(width_a_list) > 1 else np.nan

        # Save aggregated local coverage
        lc_agg = pd.DataFrame({
            "center":           _CENTERS,
            "mean_local_cov":   np.nanmean(lc_mat, axis=0),
            "sd_local_cov":     np.nanstd(lc_mat, axis=0, ddof=1),
        })
        if "local_coverage_adaptive" in macrorep_results[0]:
            lc_a_mat = np.array([r["local_coverage_adaptive"] for r in macrorep_results])
            lc_agg["mean_local_cov_adaptive"] = np.nanmean(lc_a_mat, axis=0)
            lc_agg["sd_local_cov_adaptive"]   = np.nanstd(lc_a_mat, axis=0, ddof=1)
        lc_agg.to_csv(out_dir / f"local_coverage_{sim}.csv", index=False)

        all_rows.append(row)

    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "gibbs_compare_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
