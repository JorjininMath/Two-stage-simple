"""
exp_adaptive_h / Exp2: Fixed h vs Oracle adaptive h(x) = c * s(x) on 4 DGPs.

For each macrorep, both arms share Stage 1 + Stage 2 data (paired comparison):
  - "fixed" arm:  standard CP pipeline with scalar h from CV (Exp1 baseline)
  - "oracle" arm: same trained model, but at eval time use adaptive
                  h(x) = c * s(x) where s(x) is the per-DGP oracle scale.
                  CP is recalibrated with adaptive h on the same Stage 2 data.

Outputs:
    exp_adaptive_h/output_exp2/
        macrorep_{k}/case_{sim}_fixed/per_point.csv
        macrorep_{k}/case_{sim}_oracle/per_point.csv

Usage (from project root):
    python exp_adaptive_h/run_exp2_oracle.py
    python exp_adaptive_h/run_exp2_oracle.py --n_macro 50 --n_workers 6
    python exp_adaptive_h/run_exp2_oracle.py --c_scale 0.5
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

from exp_adaptive_h.config_utils import load_config_from_file, get_config, get_x_cand
from exp_adaptive_h.adaptive_h_utils import (
    get_oracle_h,
    adaptive_recalibrate_q,
    adaptive_predict_interval,
    adaptive_score_coverage,
)
from CKME.parameters import Params
from CP.evaluation import compute_interval_score
from Two_stage import run_stage1_train, run_stage2
from Two_stage.evaluation import evaluate_per_point
from Two_stage.test_data import generate_test_data

SIMULATORS = [
    "wsc_gauss",
    "gibbs_s1",
    "exp1",
    "nongauss_A1L",
]

STAGE1_DESIGN = "grid"
STAGE2_METHOD = "lhs"

# Per-DGP t_grid margin (must match run_exp1_baseline.py).
T_GRID_MARGIN = {
    "wsc_gauss":    0.30,
    "gibbs_s1":     0.30,
    "exp1":         0.50,
    "nongauss_A1L": 2.00,
}


def _add_truncation_cols(df: pd.DataFrame, Y_test: np.ndarray, t_grid: np.ndarray) -> None:
    t_lo = float(t_grid[0])
    t_hi = float(t_grid[-1])
    Y_arr = np.asarray(Y_test, dtype=float).ravel()
    L_arr = df["L"].to_numpy()
    U_arr = df["U"].to_numpy()
    df["y_in_grid"]    = ((Y_arr >= t_lo) & (Y_arr <= t_hi)).astype(int)
    df["L_at_grid_lo"] = (L_arr <= t_lo + 1e-9).astype(int)
    df["U_at_grid_hi"] = (U_arr >= t_hi - 1e-9).astype(int)
    df["t_grid_lo"]    = t_lo
    df["t_grid_hi"]    = t_hi


def _evaluate_oracle_arm(
    stage2,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    simulator_func: str,
    c_scale: float,
    alpha: float,
) -> pd.DataFrame:
    """Run adaptive-h evaluation: recalibrate q, compute L/U, coverage, IS."""
    model  = stage2.model
    t_grid = stage2.t_grid

    # Recalibrate q_hat with adaptive h on Stage 2 data
    h_cal = get_oracle_h(simulator_func, stage2.X_stage2, c_scale)
    q_oracle = adaptive_recalibrate_q(
        model, stage2.X_stage2, stage2.Y_stage2, h_cal, alpha
    )

    # Predict L, U with adaptive h on test points
    h_test = get_oracle_h(simulator_func, X_test, c_scale)
    L, U = adaptive_predict_interval(model, X_test, h_test, t_grid, q_oracle)
    width = U - L

    # Score-based coverage with adaptive h
    cov_score = adaptive_score_coverage(model, X_test, Y_test, h_test, q_oracle)
    cov_interval = ((Y_test >= L) & (Y_test <= U)).astype(int)
    is_vals, _ = compute_interval_score(Y_test, L, U, alpha)
    status = np.where(Y_test < L, "below", np.where(Y_test > U, "above", "in"))

    rows = {
        "y":                 Y_test.astype(float),
        "L":                 L.astype(float),
        "U":                 U.astype(float),
        "covered_interval":  cov_interval.astype(int),
        "covered_score":     cov_score.astype(int),
        "width":             width.astype(float),
        "interval_score":    is_vals.astype(float),
        "status":            status.astype(str),
        "h_query":           h_test.astype(float),
    }
    df = pd.DataFrame(rows)
    X_2d = np.atleast_2d(X_test)
    for j in range(X_2d.shape[1]):
        df[f"x{j}"] = X_2d[:, j]
    df.attrs["q_hat"] = q_oracle
    return df


def run_one_macrorep(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    out_dir: Path,
    n_grid: int,
    params: Params,
    c_scale: float,
) -> dict:
    seed = base_seed + macrorep_id * 10000

    n_0   = config["n_0"]
    r_0   = config["r_0"]
    n_1   = config["n_1"]
    r_1   = config["r_1"]
    alpha = config["alpha"]

    X_cand = get_x_cand(simulator_func, config["n_cand"], random_state=seed + 1)

    stage1 = run_stage1_train(
        n_0=n_0, r_0=r_0,
        simulator_func=simulator_func,
        params=params,
        design_method=STAGE1_DESIGN,
        t_grid_size=n_grid,
        t_grid_margin=T_GRID_MARGIN.get(simulator_func),
        random_state=seed + 2,
        verbose=False,
    )

    stage2 = run_stage2(
        stage1_result=stage1,
        X_cand=X_cand,
        n_1=n_1, r_1=r_1,
        simulator_func=simulator_func,
        method=STAGE2_METHOD,
        alpha=alpha,
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

    # Fixed-h arm
    fixed_eval = evaluate_per_point(stage2, X_test, Y_test)
    df_fixed = pd.DataFrame(fixed_eval["rows"])
    _add_truncation_cols(df_fixed, Y_test, stage2.t_grid)
    case_dir_fix = out_dir / f"macrorep_{macrorep_id}" / f"case_{simulator_func}_fixed"
    case_dir_fix.mkdir(parents=True, exist_ok=True)
    df_fixed.to_csv(case_dir_fix / "per_point.csv", index=False)

    # Oracle adaptive-h arm
    df_oracle = _evaluate_oracle_arm(
        stage2, X_test, Y_test, simulator_func, c_scale, alpha
    )
    _add_truncation_cols(df_oracle, Y_test, stage2.t_grid)
    case_dir_or = out_dir / f"macrorep_{macrorep_id}" / f"case_{simulator_func}_oracle"
    case_dir_or.mkdir(parents=True, exist_ok=True)
    df_oracle.to_csv(case_dir_or / "per_point.csv", index=False)

    return {
        "macrorep":          macrorep_id,
        "simulator":         simulator_func,
        "cov_fixed":         float(df_fixed["covered_score"].mean()),
        "cov_oracle":        float(df_oracle["covered_score"].mean()),
        "width_fixed":       float(df_fixed["width"].mean()),
        "width_oracle":      float(df_oracle["width"].mean()),
        "is_fixed":          float(df_fixed["interval_score"].mean()),
        "is_oracle":         float(df_oracle["interval_score"].mean()),
        "q_fixed":           float(stage2.cp.q_hat),
        "q_oracle":          float(df_oracle.attrs["q_hat"]),
    }


def _worker(args_tuple):
    return run_one_macrorep(*args_tuple)


def main():
    parser = argparse.ArgumentParser(description="Exp2: fixed h vs oracle adaptive h(x)")
    parser.add_argument("--config",     type=str, default="exp_adaptive_h/config.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_macro",    type=int, default=50)
    parser.add_argument("--base_seed",  type=int, default=20260501)
    parser.add_argument("--c_scale",    type=float, default=1.0,
                        help="Multiplier in oracle h(x) = c * s(x). Default 1.0.")
    parser.add_argument("--n_workers",  type=int, default=1)
    args = parser.parse_args()

    config = get_config(load_config_from_file(_root / args.config), quick=False)
    n_grid = config.get("t_grid_size", 1000)

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_adaptive_h" / "output_exp2"
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrained_path = _root / "exp_adaptive_h" / "pretrained_params.json"
    if not pretrained_path.exists():
        print(
            f"ERROR: {pretrained_path} not found.\n"
            "Run 'python exp_adaptive_h/pretrain_params.py' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    raw = json.loads(pretrained_path.read_text())
    pretrained = {sim: Params(**raw[sim]) for sim in SIMULATORS if sim in raw}
    missing = [s for s in SIMULATORS if s not in pretrained]
    if missing:
        print(f"ERROR: pretrained params missing for: {missing}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded pretrained params: {pretrained_path}")
    for sim, p in pretrained.items():
        print(f"  {sim:14s}  ell_x={p.ell_x}, lam={p.lam}, h_fixed={p.h}")

    print(
        f"\nn_macro={args.n_macro}, n_workers={args.n_workers}, c_scale={args.c_scale}, "
        f"stage1={STAGE1_DESIGN}, stage2={STAGE2_METHOD}, output_dir={out_dir}"
    )

    rep_rows: list[dict] = []
    for sim in SIMULATORS:
        print(f"\n--- Simulator: {sim} ---")
        params = pretrained[sim]
        jobs = [
            (k, args.base_seed, config, sim, out_dir, n_grid, params, args.c_scale)
            for k in range(args.n_macro)
        ]

        if args.n_workers > 1:
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                fut_to_k = {pool.submit(_worker, j): j[0] for j in jobs}
                for fut in as_completed(fut_to_k):
                    k = fut_to_k[fut]
                    one = fut.result()
                    rep_rows.append(one)
                    print(
                        f"  macrorep {k:3d}  "
                        f"cov: fix={one['cov_fixed']:.3f} or={one['cov_oracle']:.3f}  "
                        f"width: fix={one['width_fixed']:.3f} or={one['width_oracle']:.3f}  "
                        f"IS: fix={one['is_fixed']:.3f} or={one['is_oracle']:.3f}"
                    )
        else:
            for j in jobs:
                one = _worker(j)
                rep_rows.append(one)
                k = j[0]
                print(
                    f"  macrorep {k:3d}  "
                    f"cov: fix={one['cov_fixed']:.3f} or={one['cov_oracle']:.3f}  "
                    f"width: fix={one['width_fixed']:.3f} or={one['width_oracle']:.3f}  "
                    f"IS: fix={one['is_fixed']:.3f} or={one['is_oracle']:.3f}"
                )

    rep_df = pd.DataFrame(rep_rows).sort_values(["simulator", "macrorep"]).reset_index(drop=True)
    rep_df.to_csv(out_dir / "exp2_per_macrorep.csv", index=False)

    summary = (
        rep_df.groupby("simulator")
        .agg(
            mean_cov_fixed=("cov_fixed", "mean"),
            mean_cov_oracle=("cov_oracle", "mean"),
            mean_width_fixed=("width_fixed", "mean"),
            mean_width_oracle=("width_oracle", "mean"),
            mean_is_fixed=("is_fixed", "mean"),
            mean_is_oracle=("is_oracle", "mean"),
            n_macroreps=("cov_fixed", "count"),
        )
        .reindex(SIMULATORS)
        .reset_index()
    )
    summary_path = out_dir / "exp2_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
