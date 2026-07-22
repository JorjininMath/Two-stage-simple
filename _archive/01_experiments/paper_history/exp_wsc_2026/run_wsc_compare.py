"""
run_wsc_compare.py

Reproduces Tables 2-3 of the WSC 2026 paper:
  "Two-Stage Adaptive Design for Conditional Distribution Estimation"

Experiment setup:
  DGPs   : wsc_gauss (Exp 1, Gaussian) and nongauss_A1L (Exp 2, Student-t nu=3)
           Both use f(x) = exp(x/10)*sin(x),  sigma(x) = 0.01 + 0.2*(x-pi)^2
  Stage 1: n_0=500, r_0=10 (5000 samples, fixed)
  Stage 2: (n_1, r_1) in {(100,50), (200,25), (500,10)} x method in {lhs, sampling, mixed}
  Metrics: marginal coverage, width, interval score (Winkler)
  Default: 50 macroreps, alpha=0.1

Usage (from project root):
    # Quick test (1 macrorep)
    python exp_wsc/run_wsc_compare.py --n_macro 1

    # Full run, sequential
    python exp_wsc/run_wsc_compare.py --n_macro 50

    # Full run, parallel (8 workers)
    python exp_wsc/run_wsc_compare.py --n_macro 50 --n_workers 8

    # Then generate formatted tables
    python exp_wsc/make_tables.py
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

from exp_wsc.config_utils import load_config_from_file, get_config, get_x_cand
from Two_stage import run_stage1_train, run_stage2
from Two_stage.evaluation import evaluate_per_point
from Two_stage.test_data import generate_test_data
from CKME.parameters import Params

R_SCRIPT  = _root / "run_benchmarks_one_case.R"
MIXED_RATIO = 0.7

# DGPs: (registry name, paper label)
DGPS = [
    ("wsc_gauss",    "Exp1_Gaussian"),
    ("nongauss_A1L", "Exp2_Student_t3"),
]

# Stage 2 budget allocations from Table 2-3 column headers
STAGE2_CASES = [
    (100, 50),
    (200, 25),
    (500, 10),
]

METHODS = ["lhs", "sampling", "mixed"]

METHOD_COV   = {"CKME": "covered_score",    "DCP-DR": "covered_score_dr",    "hetGP": "covered_interval_hetgp"}
METHOD_WIDTH = {"CKME": "width",             "DCP-DR": "width_dr",             "hetGP": "width_hetgp"}
METHOD_SCORE = {"CKME": "interval_score",    "DCP-DR": "interval_score_dr",    "hetGP": "interval_score_hetgp"}


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
    out_dir: Path,
    n_grid: int,
    pretrained: dict,
) -> list[dict]:
    """Run all DGP × budget × method combinations for one macrorep.

    Returns a list of summary dicts (one per combination).
    """
    seed = base_seed + macrorep_id * 10000

    rows = []
    for sim_func, dgp_label in DGPS:
        params: Params | None = pretrained.get(sim_func)

        X_cand = get_x_cand(sim_func, config["n_cand"], random_state=seed + 1)

        stage1 = run_stage1_train(
            n_0=config["n_0"],
            r_0=config["r_0"],
            simulator_func=sim_func,
            params=params,
            t_grid_size=n_grid,
            random_state=seed + 2,
            verbose=False,
        )
        X0, Y0 = stage1.X_all, stage1.Y_all

        for case_idx, (n_1, r_1) in enumerate(STAGE2_CASES):
            for method_idx, method in enumerate(METHODS):
                offset = case_idx * 100 + method_idx * 10
                stage2 = run_stage2(
                    stage1_result=stage1,
                    X_cand=X_cand,
                    n_1=n_1,
                    r_1=r_1,
                    simulator_func=sim_func,
                    method=method,
                    alpha=config["alpha"],
                    mixed_ratio=MIXED_RATIO,
                    random_state=seed + 3 + offset,
                    verbose=False,
                )

                X_test, Y_test = generate_test_data(
                    stage2_result=stage2,
                    n_test=config["n_test"],
                    r_test=config["r_test"],
                    X_cand=X_cand,
                    simulator_func=sim_func,
                    random_state=seed + 4 + offset,
                )

                eval_result = evaluate_per_point(stage2, X_test, Y_test)
                rows_ckme = eval_result["rows"]

                # Save raw data for R benchmarks
                case_name = f"{sim_func}_n{n_1}_r{r_1}_{method}"
                case_dir  = out_dir / f"macrorep_{macrorep_id}" / case_name
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
                    bench_df = _run_r_benchmarks(case_dir, bench_csv, config["alpha"], n_grid)
                    for i, row in enumerate(rows_ckme):
                        for col in bench_df.columns:
                            row[col] = bench_df.iloc[i][col]
                except RuntimeError as e:
                    print(
                        f"  Warning: R benchmarks failed for {case_name}; "
                        f"DCP-DR/hetGP will be NaN.\n  {e}",
                        file=sys.stderr,
                    )

                df = pd.DataFrame(rows_ckme)
                df.to_csv(case_dir / "per_point.csv", index=False)

                summary = {
                    "dgp": dgp_label,
                    "sim": sim_func,
                    "n_1": n_1,
                    "r_1": r_1,
                    "method": method,
                    "macrorep": macrorep_id,
                }
                for name, col in METHOD_COV.items():
                    summary[f"{name}_coverage"] = df[col].mean() if col in df.columns else np.nan
                for name, col in METHOD_WIDTH.items():
                    summary[f"{name}_width"] = df[col].mean() if col in df.columns else np.nan
                for name, col in METHOD_SCORE.items():
                    summary[f"{name}_interval_score"] = df[col].mean() if col in df.columns else np.nan

                rows.append(summary)

    return rows


def main():
    parser = argparse.ArgumentParser(description="WSC 2026: CKME vs DCP-DR vs hetGP")
    parser.add_argument("--config",      type=str, default="exp_wsc/config.txt")
    parser.add_argument("--output_dir",  type=str, default=None)
    parser.add_argument("--n_macro",     type=int, default=50)
    parser.add_argument("--base_seed",   type=int, default=42)
    parser.add_argument("--n_workers",   type=int, default=1,
                        help="Parallel worker processes (default: 1 = sequential)")
    args = parser.parse_args()

    config = get_config(load_config_from_file(_root / args.config), quick=False)
    n_grid = config.get("t_grid_size", 500)
    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_wsc" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained hyperparameters
    pretrained: dict = {}
    pretrained_path = _root / "exp_wsc" / "pretrained_params.json"
    if pretrained_path.exists():
        with open(pretrained_path) as f:
            raw = json.load(f)
        pretrained = {sim: Params(**raw[sim]) for sim in raw}
        print(f"Loaded pretrained params from {pretrained_path}")
    else:
        print(
            f"Warning: {pretrained_path} not found; using config fallback params.\n"
            "Run 'python exp_wsc/pretrain_params.py' first for better accuracy.",
            file=sys.stderr,
        )

    if not R_SCRIPT.exists():
        print(f"Warning: R script not found at {R_SCRIPT}; benchmarks will be NaN.", file=sys.stderr)

    print(f"n_macro={args.n_macro}, n_workers={args.n_workers}, output_dir={out_dir}")

    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futs = {
                pool.submit(
                    run_one_macrorep, k, args.base_seed, config, out_dir, n_grid, pretrained
                ): k
                for k in range(args.n_macro)
            }
            result_map: dict[int, list] = {}
            for fut in as_completed(futs):
                k = futs[fut]
                result_map[k] = fut.result()
                print(f"  macrorep {k} done")
        all_rows = [row for k in range(args.n_macro) for row in result_map[k]]
    else:
        all_rows = []
        for k in range(args.n_macro):
            print(f"  macrorep {k} ...")
            all_rows.extend(run_one_macrorep(k, args.base_seed, config, out_dir, n_grid, pretrained))

    per_rep = pd.DataFrame(all_rows)
    per_rep.to_csv(out_dir / "wsc_per_macrorep.csv", index=False)

    # Aggregate across macroreps
    group_cols = ["dgp", "sim", "n_1", "r_1", "method"]
    metric_cols = [c for c in per_rep.columns if c not in group_cols + ["macrorep"]]

    agg = per_rep.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg["n_macroreps"] = per_rep.groupby(group_cols)["macrorep"].count().values
    agg.to_csv(out_dir / "wsc_summary.csv", index=False)

    print(f"\nWrote {out_dir / 'wsc_summary.csv'}")
    print("Run 'python exp_wsc/make_tables.py' to format Tables 2-3.")


if __name__ == "__main__":
    main()
