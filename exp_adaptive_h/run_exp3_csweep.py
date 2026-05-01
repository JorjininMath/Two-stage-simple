"""
exp_adaptive_h / Exp3: c sensitivity sweep on nongauss_A1L (oracle regime).

Per macrorep, share Stage 1 + Stage 2 data across:
  - one "fixed" arm (CV-tuned scalar h, baseline reference)
  - K "oracle" arms with h(x) = c * s(x) for c in C_LIST

This is a paired comparison: same data, only h differs.

Outputs:
    exp_adaptive_h/output_exp3/
        macrorep_{k}/case_nongauss_A1L_fixed/per_point.csv
        macrorep_{k}/case_nongauss_A1L_c{c}/per_point.csv      (one per c)
        exp3_per_macrorep.csv  (cov / width / IS for each (k, arm))
        exp3_summary.csv

Usage (from project root):
    python exp_adaptive_h/run_exp3_csweep.py --n_macro 50 --n_workers 6
    python exp_adaptive_h/run_exp3_csweep.py --c_list 0.3,0.5,1.0,2.0
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

SIMULATOR = "nongauss_A1L"
STAGE1_DESIGN = "grid"
STAGE2_METHOD = "lhs"
T_GRID_MARGIN = 2.00  # match run_exp1/2 for nongauss_A1L

DEFAULT_C_LIST = [0.3, 0.5, 1.0, 2.0]


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
    model  = stage2.model
    t_grid = stage2.t_grid

    h_cal = get_oracle_h(simulator_func, stage2.X_stage2, c_scale)
    q_oracle = adaptive_recalibrate_q(
        model, stage2.X_stage2, stage2.Y_stage2, h_cal, alpha
    )

    h_test = get_oracle_h(simulator_func, X_test, c_scale)
    L, U = adaptive_predict_interval(model, X_test, h_test, t_grid, q_oracle)
    width = U - L

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
    out_dir: Path,
    n_grid: int,
    params: Params,
    c_list: list[float],
) -> list[dict]:
    seed = base_seed + macrorep_id * 10000

    n_0   = config["n_0"]
    r_0   = config["r_0"]
    n_1   = config["n_1"]
    r_1   = config["r_1"]
    alpha = config["alpha"]

    X_cand = get_x_cand(SIMULATOR, config["n_cand"], random_state=seed + 1)

    stage1 = run_stage1_train(
        n_0=n_0, r_0=r_0,
        simulator_func=SIMULATOR,
        params=params,
        design_method=STAGE1_DESIGN,
        t_grid_size=n_grid,
        t_grid_margin=T_GRID_MARGIN,
        random_state=seed + 2,
        verbose=False,
    )

    stage2 = run_stage2(
        stage1_result=stage1,
        X_cand=X_cand,
        n_1=n_1, r_1=r_1,
        simulator_func=SIMULATOR,
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
        simulator_func=SIMULATOR,
        random_state=seed + 4,
    )

    arm_rows: list[dict] = []

    # Fixed-h arm (one per macrorep)
    fixed_eval = evaluate_per_point(stage2, X_test, Y_test)
    df_fixed = pd.DataFrame(fixed_eval["rows"])
    _add_truncation_cols(df_fixed, Y_test, stage2.t_grid)
    case_dir_fix = out_dir / f"macrorep_{macrorep_id}" / f"case_{SIMULATOR}_fixed"
    case_dir_fix.mkdir(parents=True, exist_ok=True)
    df_fixed.to_csv(case_dir_fix / "per_point.csv", index=False)
    arm_rows.append({
        "macrorep":       macrorep_id,
        "arm":            "fixed",
        "c":              float("nan"),
        "coverage":       float(df_fixed["covered_score"].mean()),
        "coverage_intv":  float(df_fixed["covered_interval"].mean()),
        "width":          float(df_fixed["width"].mean()),
        "interval_score": float(df_fixed["interval_score"].mean()),
        "q_hat":          float(stage2.cp.q_hat),
    })

    # Oracle arms, one per c
    for c in c_list:
        df_or = _evaluate_oracle_arm(stage2, X_test, Y_test, SIMULATOR, c, alpha)
        _add_truncation_cols(df_or, Y_test, stage2.t_grid)
        case_dir_or = out_dir / f"macrorep_{macrorep_id}" / f"case_{SIMULATOR}_c{c:g}"
        case_dir_or.mkdir(parents=True, exist_ok=True)
        df_or.to_csv(case_dir_or / "per_point.csv", index=False)
        arm_rows.append({
            "macrorep":       macrorep_id,
            "arm":            f"oracle_c{c:g}",
            "c":              float(c),
            "coverage":       float(df_or["covered_score"].mean()),
            "coverage_intv":  float(df_or["covered_interval"].mean()),
            "width":          float(df_or["width"].mean()),
            "interval_score": float(df_or["interval_score"].mean()),
            "q_hat":          float(df_or.attrs["q_hat"]),
        })

    return arm_rows


def _worker(args_tuple):
    return run_one_macrorep(*args_tuple)


def _parse_c_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Exp3: c-sweep on nongauss_A1L (oracle)")
    parser.add_argument("--config",     type=str, default="exp_adaptive_h/config.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_macro",    type=int, default=50)
    parser.add_argument("--base_seed",  type=int, default=20260501)
    parser.add_argument("--c_list",     type=str, default=",".join(str(c) for c in DEFAULT_C_LIST),
                        help="Comma-separated c values for oracle h(x)=c*s(x). "
                             f"Default: {DEFAULT_C_LIST}")
    parser.add_argument("--n_workers",  type=int, default=1)
    args = parser.parse_args()

    c_list = _parse_c_list(args.c_list)

    config = get_config(load_config_from_file(_root / args.config), quick=False)
    n_grid = config.get("t_grid_size", 1000)

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_adaptive_h" / "output_exp3"
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
    if SIMULATOR not in raw:
        print(f"ERROR: pretrained params missing for {SIMULATOR}", file=sys.stderr)
        sys.exit(1)
    params = Params(**raw[SIMULATOR])
    print(f"Loaded pretrained params: {pretrained_path}")
    print(f"  {SIMULATOR:14s}  ell_x={params.ell_x}, lam={params.lam}, h_fixed={params.h}")

    print(
        f"\nn_macro={args.n_macro}, n_workers={args.n_workers}, "
        f"c_list={c_list}, output_dir={out_dir}"
    )

    rep_rows: list[dict] = []
    jobs = [
        (k, args.base_seed, config, out_dir, n_grid, params, c_list)
        for k in range(args.n_macro)
    ]

    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            fut_to_k = {pool.submit(_worker, j): j[0] for j in jobs}
            for fut in as_completed(fut_to_k):
                k = fut_to_k[fut]
                arm_rows = fut.result()
                rep_rows.extend(arm_rows)
                fixed = [r for r in arm_rows if r["arm"] == "fixed"][0]
                summary_str = " | ".join(
                    f"c={r['c']:g}: cov={r['coverage']:.3f} IS={r['interval_score']:.2f}"
                    for r in arm_rows if r["arm"] != "fixed"
                )
                print(
                    f"  macrorep {k:3d}  fix: cov={fixed['coverage']:.3f} "
                    f"IS={fixed['interval_score']:.2f} || {summary_str}"
                )
    else:
        for j in jobs:
            arm_rows = _worker(j)
            rep_rows.extend(arm_rows)
            k = j[0]
            fixed = [r for r in arm_rows if r["arm"] == "fixed"][0]
            summary_str = " | ".join(
                f"c={r['c']:g}: cov={r['coverage']:.3f} IS={r['interval_score']:.2f}"
                for r in arm_rows if r["arm"] != "fixed"
            )
            print(
                f"  macrorep {k:3d}  fix: cov={fixed['coverage']:.3f} "
                f"IS={fixed['interval_score']:.2f} || {summary_str}"
            )

    rep_df = pd.DataFrame(rep_rows).sort_values(["macrorep", "arm"]).reset_index(drop=True)
    rep_df.to_csv(out_dir / "exp3_per_macrorep.csv", index=False)

    summary = (
        rep_df.groupby("arm", sort=False)
        .agg(
            c=("c", "first"),
            mean_coverage=("coverage", "mean"),
            sd_coverage=("coverage", "std"),
            mean_width=("width", "mean"),
            sd_width=("width", "std"),
            mean_interval_score=("interval_score", "mean"),
            sd_interval_score=("interval_score", "std"),
            n_macroreps=("coverage", "count"),
        )
        .reset_index()
    )
    summary_path = out_dir / "exp3_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
