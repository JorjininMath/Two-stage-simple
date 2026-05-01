"""
exp_adaptive_h / Exp4: Plug-in adaptive h(x) vs fixed-h vs oracle adaptive h(x).

For each (macrorep, simulator, Stage1 budget B): train Stage 1 once, run Stage 2
once, then evaluate THREE arms on the same X_test / Y_test:
  - "fixed"   : standard CP pipeline with scalar h from pretrained CV.
  - "plugin"  : adaptive h(x) = c * sigma_hat(x) where sigma_hat(x) is built
                from Stage 1 raw data via per-site std + Nadaraya-Watson smoothing.
  - "oracle"  : adaptive h(x) = c * s(x) where s(x) is the per-DGP oracle scale.

We sweep n_0 * r_0 in {50, 100, 250, 500} with r_0 fixed at 10 (so
n_0 in {5, 10, 25, 50}). All other configs (n_1, r_1, n_test, alpha) are taken
from config.txt.

Outputs:
    exp_adaptive_h/output_exp4/
        macrorep_{k}/budget_{B}/case_{sim}_{arm}/per_point.csv
        exp4_per_arm.csv     (one row per (k, sim, B, arm))
        exp4_summary.csv     (aggregated by (sim, B, arm))

Usage (from project root):
    python exp_adaptive_h/run_exp4_plugin.py --n_macro 50 --n_workers 6
    python exp_adaptive_h/run_exp4_plugin.py --budgets 50,100,250,500 --r0_fixed 10
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
from exp_adaptive_h.plugin_sigma import PluginSigma
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

T_GRID_MARGIN = {
    "wsc_gauss":    0.30,
    "gibbs_s1":     0.30,
    "exp1":         0.50,
    "nongauss_A1L": 2.00,
}

DEFAULT_BUDGETS = [50, 100, 250, 500]
DEFAULT_R0_FIXED = 10


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


def _evaluate_adaptive_arm(
    stage2,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    h_cal: np.ndarray,
    h_test: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    """Run adaptive-h evaluation given precomputed h_cal (Stage2 sites) and h_test."""
    model  = stage2.model
    t_grid = stage2.t_grid

    q_hat = adaptive_recalibrate_q(model, stage2.X_stage2, stage2.Y_stage2, h_cal, alpha)

    L, U = adaptive_predict_interval(model, X_test, h_test, t_grid, q_hat)
    width = U - L

    cov_score = adaptive_score_coverage(model, X_test, Y_test, h_test, q_hat)
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
    df.attrs["q_hat"] = q_hat
    return df


def run_one_macrorep(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    budget: int,
    r0_fixed: int,
    out_dir: Path,
    n_grid: int,
    params: Params,
    c_scale: float,
) -> list[dict]:
    """Train one (macrorep, simulator, budget) combo and evaluate 3 arms."""
    if budget % r0_fixed != 0:
        raise ValueError(f"budget={budget} is not divisible by r0_fixed={r0_fixed}")
    n_0 = budget // r0_fixed
    r_0 = r0_fixed

    n_1   = config["n_1"]
    r_1   = config["r_1"]
    alpha = config["alpha"]

    sim_idx = SIMULATORS.index(simulator_func)
    seed = (
        base_seed
        + macrorep_id * 100000
        + sim_idx * 10000
        + DEFAULT_BUDGETS.index(budget) * 1000
    )

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

    arm_rows: list[dict] = []
    base_macrorep_dir = out_dir / f"macrorep_{macrorep_id}" / f"budget_{budget}"

    # --- Arm 1: fixed h ---
    fixed_eval = evaluate_per_point(stage2, X_test, Y_test)
    df_fixed = pd.DataFrame(fixed_eval["rows"])
    df_fixed["h_query"] = float(params.h)
    _add_truncation_cols(df_fixed, Y_test, stage2.t_grid)
    case_dir_fix = base_macrorep_dir / f"case_{simulator_func}_fixed"
    case_dir_fix.mkdir(parents=True, exist_ok=True)
    df_fixed.to_csv(case_dir_fix / "per_point.csv", index=False)
    arm_rows.append({
        "macrorep":       macrorep_id,
        "simulator":      simulator_func,
        "budget":         budget,
        "n_0":            n_0,
        "r_0":            r_0,
        "arm":            "fixed",
        "coverage":       float(df_fixed["covered_score"].mean()),
        "coverage_intv":  float(df_fixed["covered_interval"].mean()),
        "width":          float(df_fixed["width"].mean()),
        "interval_score": float(df_fixed["interval_score"].mean()),
        "q_hat":          float(stage2.cp.q_hat),
        "mean_h_query":   float(params.h),
    })

    # --- Arm 2: plug-in adaptive h(x) = c * sigma_hat(x) ---
    plugin = PluginSigma.fit(stage1.X_all, stage1.Y_all, n_0, r_0)
    h_cal_pl  = plugin.get_h(stage2.X_stage2, c_scale)
    h_test_pl = plugin.get_h(X_test, c_scale)
    df_plugin = _evaluate_adaptive_arm(
        stage2, X_test, Y_test, h_cal_pl, h_test_pl, alpha
    )
    _add_truncation_cols(df_plugin, Y_test, stage2.t_grid)
    case_dir_pl = base_macrorep_dir / f"case_{simulator_func}_plugin"
    case_dir_pl.mkdir(parents=True, exist_ok=True)
    df_plugin.to_csv(case_dir_pl / "per_point.csv", index=False)
    arm_rows.append({
        "macrorep":       macrorep_id,
        "simulator":      simulator_func,
        "budget":         budget,
        "n_0":            n_0,
        "r_0":            r_0,
        "arm":            "plugin",
        "coverage":       float(df_plugin["covered_score"].mean()),
        "coverage_intv":  float(df_plugin["covered_interval"].mean()),
        "width":          float(df_plugin["width"].mean()),
        "interval_score": float(df_plugin["interval_score"].mean()),
        "q_hat":          float(df_plugin.attrs["q_hat"]),
        "mean_h_query":   float(np.mean(h_test_pl)),
    })

    # --- Arm 3: oracle adaptive h(x) = c * s(x) ---
    h_cal_or  = get_oracle_h(simulator_func, stage2.X_stage2, c_scale)
    h_test_or = get_oracle_h(simulator_func, X_test, c_scale)
    df_oracle = _evaluate_adaptive_arm(
        stage2, X_test, Y_test, h_cal_or, h_test_or, alpha
    )
    _add_truncation_cols(df_oracle, Y_test, stage2.t_grid)
    case_dir_or = base_macrorep_dir / f"case_{simulator_func}_oracle"
    case_dir_or.mkdir(parents=True, exist_ok=True)
    df_oracle.to_csv(case_dir_or / "per_point.csv", index=False)
    arm_rows.append({
        "macrorep":       macrorep_id,
        "simulator":      simulator_func,
        "budget":         budget,
        "n_0":            n_0,
        "r_0":            r_0,
        "arm":            "oracle",
        "coverage":       float(df_oracle["covered_score"].mean()),
        "coverage_intv":  float(df_oracle["covered_interval"].mean()),
        "width":          float(df_oracle["width"].mean()),
        "interval_score": float(df_oracle["interval_score"].mean()),
        "q_hat":          float(df_oracle.attrs["q_hat"]),
        "mean_h_query":   float(np.mean(h_test_or)),
    })

    return arm_rows


def _worker(args_tuple):
    return run_one_macrorep(*args_tuple)


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Exp4: plug-in vs fixed vs oracle adaptive h")
    parser.add_argument("--config",     type=str, default="exp_adaptive_h/config.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_macro",    type=int, default=50)
    parser.add_argument("--base_seed",  type=int, default=20260501)
    parser.add_argument("--c_scale",    type=float, default=1.0,
                        help="Multiplier in adaptive h(x) = c * scale. Default 1.0.")
    parser.add_argument("--budgets",    type=str,
                        default=",".join(str(b) for b in DEFAULT_BUDGETS),
                        help=f"Comma-separated Stage 1 budgets n_0*r_0. "
                             f"Default {DEFAULT_BUDGETS}.")
    parser.add_argument("--r0_fixed",   type=int, default=DEFAULT_R0_FIXED,
                        help="Fixed r_0 across budgets (n_0 = budget / r0_fixed).")
    parser.add_argument("--n_workers",  type=int, default=1)
    args = parser.parse_args()

    budgets = _parse_int_list(args.budgets)
    for b in budgets:
        if b % args.r0_fixed != 0:
            print(f"ERROR: budget {b} not divisible by r0_fixed={args.r0_fixed}",
                  file=sys.stderr)
            sys.exit(1)

    config = get_config(load_config_from_file(_root / args.config), quick=False)
    n_grid = config.get("t_grid_size", 1000)

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_adaptive_h" / "output_exp4"
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
        f"budgets={budgets}, r0_fixed={args.r0_fixed}\n"
        f"output_dir={out_dir}"
    )

    rep_rows: list[dict] = []
    for sim in SIMULATORS:
        params = pretrained[sim]
        for B in budgets:
            print(f"\n--- {sim}  budget={B} (n_0={B // args.r0_fixed}, r_0={args.r0_fixed}) ---")
            jobs = [
                (k, args.base_seed, config, sim, B, args.r0_fixed,
                 out_dir, n_grid, params, args.c_scale)
                for k in range(args.n_macro)
            ]
            if args.n_workers > 1:
                with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                    fut_to_k = {pool.submit(_worker, j): j[0] for j in jobs}
                    for fut in as_completed(fut_to_k):
                        k = fut_to_k[fut]
                        arm_rows = fut.result()
                        rep_rows.extend(arm_rows)
                        by_arm = {r["arm"]: r for r in arm_rows}
                        print(
                            f"  k={k:3d}  "
                            f"fix:cov={by_arm['fixed']['coverage']:.3f}/IS={by_arm['fixed']['interval_score']:.2f}  "
                            f"plug:cov={by_arm['plugin']['coverage']:.3f}/IS={by_arm['plugin']['interval_score']:.2f}  "
                            f"orcl:cov={by_arm['oracle']['coverage']:.3f}/IS={by_arm['oracle']['interval_score']:.2f}"
                        )
            else:
                for j in jobs:
                    arm_rows = _worker(j)
                    rep_rows.extend(arm_rows)
                    k = j[0]
                    by_arm = {r["arm"]: r for r in arm_rows}
                    print(
                        f"  k={k:3d}  "
                        f"fix:cov={by_arm['fixed']['coverage']:.3f}/IS={by_arm['fixed']['interval_score']:.2f}  "
                        f"plug:cov={by_arm['plugin']['coverage']:.3f}/IS={by_arm['plugin']['interval_score']:.2f}  "
                        f"orcl:cov={by_arm['oracle']['coverage']:.3f}/IS={by_arm['oracle']['interval_score']:.2f}"
                    )

    rep_df = (
        pd.DataFrame(rep_rows)
        .sort_values(["simulator", "budget", "macrorep", "arm"])
        .reset_index(drop=True)
    )
    rep_df.to_csv(out_dir / "exp4_per_arm.csv", index=False)

    summary = (
        rep_df.groupby(["simulator", "budget", "arm"], sort=False)
        .agg(
            mean_coverage=("coverage", "mean"),
            sd_coverage=("coverage", "std"),
            mean_width=("width", "mean"),
            sd_width=("width", "std"),
            mean_interval_score=("interval_score", "mean"),
            sd_interval_score=("interval_score", "std"),
            mean_q_hat=("q_hat", "mean"),
            mean_h=("mean_h_query", "mean"),
            n_macroreps=("coverage", "count"),
        )
        .reset_index()
    )
    summary_path = out_dir / "exp4_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
