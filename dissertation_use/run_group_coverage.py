"""
run_group_coverage.py

Dissertation experiment: group conditional coverage comparison.

Four methods on two DGPs from WSC paper:
  - Exp1 (exp2):        Gaussian noise,    sigma(x) = 0.01 + 0.2*(x-pi)^2
  - Exp2 (nongauss_A1L): Student-t nu=3,   scale   = 0.01 + 0.2*(x-pi)^2

Methods:
  1. CKME fixed-h  (CV-tuned scalar bandwidth)
  2. CKME adaptive-h  (h(x) = c * sigma(x), oracle sigma)
  3. DCP-DR  (R)
  4. hetGP   (R)

Output: per_point.csv with coverage columns for all 4 methods, then
analyze with plot_group_coverage.py.

Usage:
  python dissertation_use/run_group_coverage.py
  python dissertation_use/run_group_coverage.py --n_macro 10 --c_scale 2.0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from Two_stage import run_stage1_train, run_stage2
from Two_stage.evaluation import evaluate_per_point
from Two_stage.test_data import generate_test_data
from Two_stage.config_utils import load_config_from_file, get_config, get_x_cand
from CKME.coefficients import compute_ckme_coeffs
from CKME.indicators import make_indicator
from CKME.parameters import Params
from CP.evaluation import compute_interval_score

R_SCRIPT = _root / "run_benchmarks_one_case.R"

SIMULATORS = ["exp2", "nongauss_A1L"]
MIXED_RATIO = 0.7
_PI = np.pi


def _estimate_sigma(model, x_query: np.ndarray) -> np.ndarray:
    """Estimate local noise std sigma_hat(x) using within-site variance + CKME weighting."""
    X_q = x_query.reshape(-1, 1) if x_query.ndim == 1 else np.atleast_2d(x_query)
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_q)  # (n, M)
    # Within-site sample variance: s_i^2 = Var(Y_{i,1}, ..., Y_{i,r})
    if model.Y.ndim == 2 and model.Y.shape[1] > 1:
        site_var = model.Y.var(axis=1)  # (n,)
    else:
        # Fallback: r=1, no within-site variance available
        Y_site = model.Y.mean(axis=1) if model.Y.ndim == 2 else model.Y.ravel()
        sigma_hat = np.empty(X_q.shape[0])
        for m in range(X_q.shape[0]):
            c_m = C[:, m]
            mu_hat = float(c_m @ Y_site)
            var_hat = float(c_m @ (Y_site - mu_hat) ** 2)
            sigma_hat[m] = np.sqrt(max(var_hat, 1e-8))
        return sigma_hat
    # Kernel-weighted average of within-site variances
    sigma_hat = np.empty(X_q.shape[0])
    for m in range(X_q.shape[0]):
        var_hat = float(C[:, m] @ site_var)
        sigma_hat[m] = np.sqrt(max(var_hat, 1e-8))
    return sigma_hat


def _adaptive_h_vals(model, x_query: np.ndarray, c_scale: float) -> np.ndarray:
    return c_scale * _estimate_sigma(model, x_query.ravel())


def _recalibrate_adaptive_cp(model, stage2, c_scale: float, alpha: float) -> float:
    """Recompute CP q_hat using per-point adaptive h on calibration data."""
    X_cal = np.atleast_2d(stage2.X_stage2)
    Y_cal = np.asarray(stage2.Y_stage2).ravel()
    n_cal = len(Y_cal)

    h_cal = _adaptive_h_vals(model, X_cal, c_scale)
    Y_flat = model.Y.ravel()
    C_cal = compute_ckme_coeffs(model.L, model.kx, model.X, X_cal)

    scores = np.empty(n_cal)
    for j in range(n_cal):
        ind_j = make_indicator(model.indicator_type, float(h_cal[j]))
        g_j = ind_j.g_matrix(Y_flat, np.array([float(Y_cal[j])]))[:, 0]
        g_site = g_j.reshape(model.n, model.r).mean(axis=1)
        F_j = float(np.clip(C_cal[:, j] @ g_site, 0.0, 1.0))
        scores[j] = abs(F_j - 0.5)

    k = int(np.ceil((1 - alpha) * (1 + n_cal)))
    k = min(k, n_cal)
    return float(np.sort(scores)[k - 1])


def _adaptive_coverage_and_intervals(
    model, X_test, Y_test, t_grid, q_hat, c_scale,
):
    """Compute adaptive-h coverage, intervals, width, and IS."""
    X_test = np.atleast_2d(X_test)
    Y_test = np.asarray(Y_test).ravel()
    n_test = len(Y_test)

    h_vals = _adaptive_h_vals(model, X_test, c_scale)
    Y_flat = model.Y.ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_test)
    T = len(t_grid)

    tau_lo = float(np.clip(0.5 - q_hat, 0.0, 1.0))
    tau_hi = float(np.clip(0.5 + q_hat, 0.0, 1.0))

    scores = np.empty(n_test)
    L_arr = np.empty(n_test)
    U_arr = np.empty(n_test)

    for i in range(n_test):
        ind_i = make_indicator(model.indicator_type, float(h_vals[i]))
        # Score
        g_s = ind_i.g_matrix(Y_flat, np.array([float(Y_test[i])]))[:, 0]
        g_site_s = g_s.reshape(model.n, model.r).mean(axis=1)
        F_s = float(np.clip(C[:, i] @ g_site_s, 0.0, 1.0))
        scores[i] = abs(F_s - 0.5)
        # Interval
        G_mat = ind_i.g_matrix(Y_flat, t_grid)
        G_site = G_mat.reshape(model.n, model.r, T).mean(axis=1)
        F_m = np.clip(C[:, i] @ G_site, 0.0, 1.0)
        L_arr[i] = t_grid[min(np.searchsorted(F_m, tau_lo), T - 1)]
        U_arr[i] = t_grid[min(np.searchsorted(F_m, tau_hi), T - 1)]

    covered = (scores <= q_hat).astype(int)
    width = U_arr - L_arr
    return covered, L_arr, U_arr, width


def _run_r_benchmarks(case_dir: Path, output_csv: Path, alpha: float, n_grid: int) -> pd.DataFrame:
    cmd = ["Rscript", str(R_SCRIPT), str(case_dir), str(output_csv), str(alpha), str(n_grid)]
    result = subprocess.run(cmd, cwd=str(_root), capture_output=True, text=True, check=False)
    if not output_csv.exists():
        raise RuntimeError(
            f"R script did not produce {output_csv}.\n"
            f"stderr: {result.stderr or 'none'}\nstdout: {result.stdout or 'none'}"
        )
    return pd.read_csv(output_csv)


def run_one_macrorep(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    out_dir: Path,
    method: str,
    n_grid: int,
    params=None,
    c_scale: float = 2.0,
) -> dict:
    seed = base_seed + macrorep_id * 10000
    alpha = config["alpha"]
    if params is None:
        params = config["params"]

    X_cand = get_x_cand(simulator_func, config["n_cand"], random_state=seed + 1)

    stage1 = run_stage1_train(
        n_0=config["n_0"], r_0=config["r_0"],
        simulator_func=simulator_func,
        params=params,
        t_grid_size=n_grid,
        random_state=seed + 2,
        verbose=False,
    )

    stage2 = run_stage2(
        stage1_result=stage1,
        X_cand=X_cand,
        n_1=config["n_1"], r_1=config["r_1"],
        simulator_func=simulator_func,
        method=method,
        alpha=alpha,
        mixed_ratio=MIXED_RATIO,
        random_state=seed + 3,
        verbose=False,
    )

    X_test, Y_test = generate_test_data(
        stage2_result=stage2,
        n_test=config["n_test"],
        r_test=config["r_test"],
        X_cand=X_cand,
        simulator_func=simulator_func,
        random_state=seed + 4,
    )

    # CKME fixed-h results
    eval_result = evaluate_per_point(stage2, X_test, Y_test)
    rows = eval_result["rows"]

    # CKME adaptive-h results
    model = stage2.model
    q_hat_adap = _recalibrate_adaptive_cp(model, stage2, c_scale, alpha)
    cov_adap, L_adap, U_adap, w_adap = _adaptive_coverage_and_intervals(
        model, X_test, Y_test, stage2.t_grid, q_hat_adap, c_scale,
    )
    is_adap, _ = compute_interval_score(Y_test.ravel(), L_adap, U_adap, alpha)
    for i, row in enumerate(rows):
        row["covered_adaptive"] = int(cov_adap[i])
        row["L_adaptive"] = float(L_adap[i])
        row["U_adaptive"] = float(U_adap[i])
        row["width_adaptive"] = float(w_adap[i])
        row["interval_score_adaptive"] = float(is_adap[i])

    # Save data for R benchmarks
    case_name = f"{simulator_func}_{method}"
    case_dir = out_dir / f"macrorep_{macrorep_id}" / f"case_{case_name}"
    rep0_dir = case_dir / "macrorep_0"
    rep0_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(rep0_dir / "X0.csv", stage1.X_all, delimiter=",")
    np.savetxt(rep0_dir / "Y0.csv", stage1.Y_all, delimiter=",")
    np.savetxt(rep0_dir / "X1.csv", stage2.X_stage2, delimiter=",")
    np.savetxt(rep0_dir / "Y1.csv", stage2.Y_stage2, delimiter=",")
    np.savetxt(rep0_dir / "X_test.csv", X_test, delimiter=",")
    np.savetxt(rep0_dir / "Y_test.csv", Y_test, delimiter=",")

    # R benchmarks (DCP-DR + hetGP)
    bench_csv = case_dir / "benchmarks.csv"
    try:
        bench_df = _run_r_benchmarks(case_dir, bench_csv, alpha, n_grid)
        for i, row in enumerate(rows):
            for col in bench_df.columns:
                row[col] = bench_df.iloc[i][col]
    except RuntimeError as e:
        print(f"  Warning: R benchmarks failed for {simulator_func}; DCP-DR/hetGP will be NaN.\n  {e}",
              file=sys.stderr)

    df = pd.DataFrame(rows)
    df.to_csv(case_dir / "per_point.csv", index=False)

    # Summary stats
    out = {"simulator": simulator_func, "macrorep": macrorep_id}
    for label, col in [("CKME_fixed", "covered_score"), ("CKME_adaptive", "covered_adaptive"),
                       ("DCP-DR", "covered_score_dr"), ("DCP-QR", "covered_score_qr"),
                       ("hetGP", "covered_interval_hetgp")]:
        out[f"{label}_cov"] = df[col].mean() if col in df.columns else np.nan
    for label, col in [("CKME_fixed", "width"), ("CKME_adaptive", "width_adaptive"),
                       ("DCP-DR", "width_dr"), ("DCP-QR", "width_qr"),
                       ("hetGP", "width_hetgp")]:
        out[f"{label}_width"] = df[col].mean() if col in df.columns else np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description="Dissertation: group conditional coverage")
    parser.add_argument("--config", type=str, default="dissertation_use/config.txt")
    parser.add_argument("--output_dir", type=str, default="dissertation_use/output")
    parser.add_argument("--n_macro", type=int, default=10)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="lhs", choices=("lhs", "sampling", "mixed"))
    parser.add_argument("--c_scale", type=float, default=2.0)
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Number of parallel workers (default: 1 = serial)")
    args = parser.parse_args()

    config = get_config(load_config_from_file(_root / args.config), quick=False)
    n_grid = config.get("t_grid_size", 500)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"n_macro={args.n_macro}, method={args.method}, c_scale={args.c_scale}")
    print(f"output_dir={out_dir}")

    # Load pretrained params if available
    pretrained_path = _root / "dissertation_use" / "pretrained_params.json"
    pretrained: dict = {}
    if pretrained_path.exists():
        with open(pretrained_path) as f:
            raw = json.load(f)
        pretrained = {sim: Params(**raw[sim]) for sim in raw if sim in SIMULATORS}
        print(f"Loaded pretrained params from {pretrained_path}")
    else:
        print(f"No pretrained params found at {pretrained_path}; using config defaults.",
              file=sys.stderr)

    # Build job list: (sim, macrorep_id)
    jobs = [(sim, k) for sim in SIMULATORS for k in range(args.n_macro)]

    all_rows = []
    if args.n_workers <= 1:
        # Serial
        for sim, k in jobs:
            result = run_one_macrorep(
                macrorep_id=k, base_seed=args.base_seed, config=config,
                simulator_func=sim, out_dir=out_dir, method=args.method,
                n_grid=n_grid, params=pretrained.get(sim), c_scale=args.c_scale,
            )
            all_rows.append(result)
            print(f"  [{sim}] macrorep {k}: CKME_fix={result['CKME_fixed_cov']:.3f}  "
                  f"CKME_adap={result['CKME_adaptive_cov']:.3f}")
    else:
        # Parallel
        print(f"Running {len(jobs)} jobs with {args.n_workers} workers...")
        futures = {}
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            for sim, k in jobs:
                fut = pool.submit(
                    run_one_macrorep,
                    macrorep_id=k, base_seed=args.base_seed, config=config,
                    simulator_func=sim, out_dir=out_dir, method=args.method,
                    n_grid=n_grid, params=pretrained.get(sim), c_scale=args.c_scale,
                )
                futures[fut] = (sim, k)
            for fut in as_completed(futures):
                sim, k = futures[fut]
                result = fut.result()
                all_rows.append(result)
                print(f"  [{sim}] macrorep {k}: CKME_fix={result['CKME_fixed_cov']:.3f}  "
                      f"CKME_adap={result['CKME_adaptive_cov']:.3f}")

    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
