"""
exp_nongauss: Compare CKME, DCP-DR, hetGP on six non-Gaussian noise DGPs.

All six cases share the exp2 true function f(x) = exp(x/10)*sin(x), x in [0, 2*pi]
and target scale sigma_tar(x), sigma_tar(x) = 0.01 + 0.2*(x-pi)^2.

  Small (light non-Gaussianity):
    nongauss_A1S : Student-t,      nu=10  (light tails)
    nongauss_B2S : Centered Gamma, k=9    (mild skew)
    nongauss_C1S : Gaussian mix,   pi=0.02 (light contamination)

  Large (strong non-Gaussianity):
    nongauss_A1L : Student-t,      nu=3   (heavy tails)
    nongauss_B2L : Centered Gamma, k=2    (strong skew)
    nongauss_C1L : Gaussian mix,   pi=0.10 (heavy contamination)

Supports adaptive bandwidth h(x) = c * sigma_tar(x) for theory-aligned
group coverage validation (T-G1/T-G2).

Usage (from project root):
  python exp_nongauss/run_nongauss_compare.py
  python exp_nongauss/run_nongauss_compare.py --n_macro 10 --method mixed
  python exp_nongauss/run_nongauss_compare.py --h_mode adaptive --c_scale 2.0
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import json

import numpy as np
import pandas as pd
from exp_nongauss.config_utils import load_config_from_file, get_config, get_x_cand
from Two_stage import run_stage1_train, run_stage2
from Two_stage.evaluation import evaluate_per_point
from Two_stage.test_data import generate_test_data
from CKME.coefficients import compute_ckme_coeffs
from CKME.indicators import make_indicator
from CP.scores import score_from_cdf

R_SCRIPT = _root / "run_benchmarks_one_case.R"

SIMULATORS = [
    "nongauss_A1S", "nongauss_B2S", "nongauss_C1S",   # Small (light non-Gaussianity)
    "nongauss_A1L", "nongauss_B2L", "nongauss_C1L",   # Large (strong non-Gaussianity)
]
MIXED_RATIO = 0.7

METHOD_COV   = {"CKME": "covered_score", "DCP-DR": "covered_score_dr", "hetGP": "covered_interval_hetgp"}
METHOD_WIDTH = {"CKME": "width",            "DCP-DR": "width_dr",             "hetGP": "width_hetgp"}
METHOD_SCORE = {"CKME": "interval_score",   "DCP-DR": "interval_score_dr",    "hetGP": "interval_score_hetgp"}

_PI = np.pi


def _nongauss_oracle_var(x: np.ndarray) -> np.ndarray:
    """Oracle scale^2 for all nongauss sims (matches sim_nongauss_A1._sigma_tar)."""
    return (0.01 + 0.2 * (x - _PI) ** 2) ** 2


def _adaptive_h_vals(model, x_query: np.ndarray, c_scale: float) -> np.ndarray:
    """h(x) = c_scale * sigma_tar(x) for each query point."""
    x_1d = x_query.ravel()
    sigma = np.sqrt(np.maximum(_nongauss_oracle_var(x_1d), 1e-8))
    return c_scale * sigma


def _recalibrate_adaptive_cp(model, stage2, c_scale: float, alpha: float) -> float:
    """Recompute CP q_hat using per-point adaptive h for calibration data."""
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


def _adaptive_score_coverage(
    model, X_test: np.ndarray, Y_test: np.ndarray, q_hat: float, c_scale: float,
) -> np.ndarray:
    """Compute score-based coverage using per-point adaptive h."""
    X_test = np.atleast_2d(X_test)
    Y_test = np.asarray(Y_test).ravel()
    n_test = len(Y_test)

    h_vals = _adaptive_h_vals(model, X_test, c_scale)
    Y_flat = model.Y.ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_test)

    scores = np.empty(n_test)
    for i in range(n_test):
        ind_i = make_indicator(model.indicator_type, float(h_vals[i]))
        g_i = ind_i.g_matrix(Y_flat, np.array([float(Y_test[i])]))[:, 0]
        g_site = g_i.reshape(model.n, model.r).mean(axis=1)
        F_i = float(np.clip(C[:, i] @ g_site, 0.0, 1.0))
        scores[i] = abs(F_i - 0.5)

    return (scores <= q_hat).astype(int)


def _adaptive_predict_interval(
    model, X_query: np.ndarray, t_grid: np.ndarray, q_hat: float, c_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict intervals using per-point adaptive h."""
    X_query = np.atleast_2d(X_query)
    M = X_query.shape[0]
    T = len(t_grid)

    h_vals = _adaptive_h_vals(model, X_query, c_scale)
    Y_flat = model.Y.ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_query)

    tau_lo = float(np.clip(0.5 - q_hat, 0.0, 1.0))
    tau_hi = float(np.clip(0.5 + q_hat, 0.0, 1.0))

    L_arr = np.empty(M)
    U_arr = np.empty(M)
    for m in range(M):
        ind_m = make_indicator(model.indicator_type, float(h_vals[m]))
        G_mat = ind_m.g_matrix(Y_flat, t_grid)
        G_site = G_mat.reshape(model.n, model.r, T).mean(axis=1)
        F_m = np.clip(C[:, m] @ G_site, 0.0, 1.0)
        L_arr[m] = t_grid[min(np.searchsorted(F_m, tau_lo), T - 1)]
        U_arr[m] = t_grid[min(np.searchsorted(F_m, tau_hi), T - 1)]

    return L_arr, U_arr


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
    h_mode: str = "fixed",
    c_scale: float = 2.0,
) -> dict:
    seed = base_seed + macrorep_id * 10000
    # np.random.seed is intentionally omitted: all simulators and design
    # functions use np.random.default_rng(random_state), so per-call seeds
    # below provide full reproducibility without a global legacy seed.

    n_0    = config.get("n_0", 200)
    r_0    = config.get("r_0", 5)
    n_1    = config.get("n_1", 100)
    r_1    = config.get("r_1", 5)
    alpha  = config["alpha"]
    if params is None:
        params = config["params"]
    X_cand = get_x_cand(simulator_func, config["n_cand"], random_state=seed + 1)

    stage1 = run_stage1_train(
        n_0=n_0, r_0=r_0,
        simulator_func=simulator_func,
        params=params,
        t_grid_size=n_grid,
        random_state=seed + 2,
        verbose=False,
    )
    X0, Y0 = stage1.X_all, stage1.Y_all

    stage2 = run_stage2(
        stage1_result=stage1,
        X_cand=X_cand,
        n_1=n_1, r_1=r_1,
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

    eval_result = evaluate_per_point(stage2, X_test, Y_test)
    rows_ckme = eval_result["rows"]

    # Adaptive h: recalibrate CP and recompute coverage/intervals
    if h_mode == "adaptive":
        model = stage2.model
        q_hat_adap = _recalibrate_adaptive_cp(model, stage2, c_scale, alpha)
        cov_adap = _adaptive_score_coverage(model, X_test, Y_test, q_hat_adap, c_scale)
        L_adap, U_adap = _adaptive_predict_interval(
            model, X_test, stage2.t_grid, q_hat_adap, c_scale,
        )
        width_adap = U_adap - L_adap
        from CP.evaluation import compute_interval_score
        is_adap, _ = compute_interval_score(Y_test.ravel(), L_adap, U_adap, alpha)
        for i, row in enumerate(rows_ckme):
            row["covered_score_adaptive"] = int(cov_adap[i])
            row["L_adaptive"] = float(L_adap[i])
            row["U_adaptive"] = float(U_adap[i])
            row["width_adaptive"] = float(width_adap[i])
            row["interval_score_adaptive"] = float(is_adap[i])

    # Save data for R benchmarks
    case_name = f"{simulator_func}_{method}"
    case_dir  = out_dir / f"macrorep_{macrorep_id}" / f"case_{case_name}"
    rep0_dir  = case_dir / "macrorep_0"
    rep0_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(rep0_dir / "X0.csv",     X0,              delimiter=",")
    np.savetxt(rep0_dir / "Y0.csv",     Y0,              delimiter=",")
    np.savetxt(rep0_dir / "X1.csv",     stage2.X_stage2, delimiter=",")
    np.savetxt(rep0_dir / "Y1.csv",     stage2.Y_stage2, delimiter=",")
    np.savetxt(rep0_dir / "X_test.csv", X_test,          delimiter=",")
    np.savetxt(rep0_dir / "Y_test.csv", Y_test,          delimiter=",")

    bench_csv = case_dir / "benchmarks.csv"
    try:
        bench_df = _run_r_benchmarks(case_dir, bench_csv, alpha, n_grid)
        for i, row in enumerate(rows_ckme):
            for col in bench_df.columns:
                row[col] = bench_df.iloc[i][col]
    except RuntimeError as e:
        print(f"  Warning: R benchmarks failed for {simulator_func}; DCP-DR/hetGP will be NaN.\n  {e}",
              file=sys.stderr)
    df = pd.DataFrame(rows_ckme)
    df.to_csv(case_dir / "per_point.csv", index=False)

    out = {}
    for name, col in METHOD_COV.items():
        if col in df.columns:
            out[f"{name}_coverage"] = df[col].mean()
    for name, col in METHOD_WIDTH.items():
        if col in df.columns:
            out[f"{name}_width"] = df[col].mean()
    for name, col in METHOD_SCORE.items():
        if col in df.columns:
            out[f"{name}_interval_score"] = df[col].mean()
    if "covered_score_adaptive" in df.columns:
        out["CKME_adaptive_coverage"] = df["covered_score_adaptive"].mean()
        out["CKME_adaptive_width"] = df["width_adaptive"].mean()
        out["CKME_adaptive_interval_score"] = df["interval_score_adaptive"].mean()
    return out


def main():
    parser = argparse.ArgumentParser(description="Non-Gaussian comparison: CKME vs DCP-DR vs hetGP")
    parser.add_argument("--config",     type=str, default="exp_nongauss/config.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_macro",    type=int, default=5)
    parser.add_argument("--base_seed",  type=int, default=42)
    parser.add_argument("--method",     type=str, default="lhs", choices=("lhs", "sampling", "mixed"))
    parser.add_argument("--h_mode",     type=str, default="fixed", choices=("fixed", "adaptive"),
                        help="Bandwidth mode: fixed (CV-tuned) or adaptive h(x)=c*sigma_tar(x)")
    parser.add_argument("--c_scale",    type=float, default=2.0,
                        help="Scale factor for adaptive-h (default: 2.0)")
    parser.add_argument("--n_workers",  type=int, default=1,
                        help="Number of parallel worker processes for macroreps (default: 1 = sequential)")
    args = parser.parse_args()

    config    = get_config(load_config_from_file(_root / args.config), quick=False)
    n_grid    = config.get("t_grid_size", 500)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    elif args.h_mode == "adaptive":
        out_dir = _root / "exp_nongauss" / f"output_adaptive_c{args.c_scale:.2f}"
    else:
        out_dir = _root / "exp_nongauss" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"h_mode={args.h_mode}" + (f" (c_scale={args.c_scale})" if args.h_mode == "adaptive" else ""))
    print(f"n_macro={args.n_macro}, method={args.method}, output_dir={out_dir}")

    # Load per-simulator pretrained hyperparameters (produced by pretrain_params.py).
    from CKME.parameters import Params
    pretrained_path = _root / "exp_nongauss" / "pretrained_params.json"
    pretrained: dict = {}
    if pretrained_path.exists():
        with open(pretrained_path) as f:
            raw = json.load(f)
        pretrained = {sim: Params(**raw[sim]) for sim in raw if sim in SIMULATORS}
        print(f"Loaded pretrained params from {pretrained_path}")
    else:
        print(
            f"Warning: {pretrained_path} not found; using config params for all simulators.\n"
            "Run 'python exp_nongauss/pretrain_params.py' first to improve accuracy.",
            file=sys.stderr,
        )

    if not R_SCRIPT.exists():
        print(f"Warning: R script not found at {R_SCRIPT}; DCP-DR and hetGP will be missing.",
              file=sys.stderr)

    all_rows = []
    for sim in SIMULATORS:
        print(f"\n--- Simulator: {sim} ---")

        # Collect one result dict per macrorep (sequential or parallel)
        if args.n_workers > 1:
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                futs = {
                    pool.submit(
                        run_one_macrorep,
                        k, args.base_seed, config, sim,
                        out_dir, args.method, n_grid, pretrained.get(sim),
                        args.h_mode, args.c_scale,
                    ): k
                    for k in range(args.n_macro)
                }
                result_map: dict[int, dict] = {}
                for fut in as_completed(futs):
                    k = futs[fut]
                    result_map[k] = fut.result()
                    print(f"  macrorep {k} done")
            macrorep_results = [result_map[k] for k in range(args.n_macro)]
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
                    n_grid=n_grid,
                    params=pretrained.get(sim),
                    h_mode=args.h_mode,
                    c_scale=args.c_scale,
                )
                macrorep_results.append(one)

        list_cov   = {m: [] for m in METHOD_COV}
        list_width = {m: [] for m in METHOD_WIDTH}
        list_score = {m: [] for m in METHOD_SCORE}
        for one in macrorep_results:
            for m in METHOD_COV:
                if f"{m}_coverage" in one:
                    list_cov[m].append(one[f"{m}_coverage"])
            for m in METHOD_WIDTH:
                if f"{m}_width" in one:
                    list_width[m].append(one[f"{m}_width"])
            for m in METHOD_SCORE:
                if f"{m}_interval_score" in one:
                    list_score[m].append(one[f"{m}_interval_score"])

        for m in list(METHOD_COV.keys()):
            cov_v = list_cov.get(m, [])
            w_v   = list_width.get(m, [])
            s_v   = list_score.get(m, [])
            all_rows.append({
                "simulator":            sim,
                "method":               m,
                "mean_coverage":        np.mean(cov_v) if cov_v else np.nan,
                "sd_coverage":          np.std(cov_v, ddof=1) if len(cov_v) > 1 else np.nan,
                "mean_width":           np.mean(w_v) if w_v else np.nan,
                "sd_width":             np.std(w_v, ddof=1) if len(w_v) > 1 else np.nan,
                "mean_interval_score":  np.mean(s_v) if s_v else np.nan,
                "sd_interval_score":    np.std(s_v, ddof=1) if len(s_v) > 1 else np.nan,
                "n_macroreps":          len(cov_v),
            })

    summary = pd.DataFrame(all_rows)
    summary_path = out_dir / "nongauss_compare_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
