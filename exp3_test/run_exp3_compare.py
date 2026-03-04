"""
exp3_test: Compare CKME, DCP-DR, hetGP (exp_test or exp3: Branin–Hoo).
Runs Stage1 + one Stage2 case; n_0, r_0, n_1, r_1 from config.txt.
Output: summary table (mean cov, width, score over test points and over macroreps).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from exp3_test.config_utils import load_config_from_file, get_config, get_x_cand
from Two_stage import run_stage1_train, run_stage2
from Two_stage.evaluation import evaluate_per_point
from Two_stage.test_data import generate_test_data

# Reuse R script from exp_stage2_impact_arc (same interface: data_dir, output_csv, alpha, n_grid)
R_SCRIPT = _root / "exp_stage2_impact_arc" / "run_benchmarks_one_case.R"

MIXED_RATIO = 0.7

METHOD_COV = {"CKME": "covered_interval", "DCP-DR": "covered_interval_dr", "hetGP": "covered_interval_hetgp"}
METHOD_WIDTH = {"CKME": "width", "DCP-DR": "width_dr", "hetGP": "width_hetgp"}
METHOD_SCORE = {"CKME": "interval_score", "DCP-DR": "interval_score_dr", "hetGP": "interval_score_hetgp"}


def _run_benchmarks_one_case(case_dir: Path, output_csv: Path, alpha: float, n_grid: int) -> pd.DataFrame:
    cmd = [
        "Rscript",
        str(R_SCRIPT),
        str(case_dir),
        str(output_csv),
        str(alpha),
        str(n_grid),
    ]
    result = subprocess.run(cmd, cwd=str(_root), capture_output=True, text=True, check=False)
    if not output_csv.exists():
        raise RuntimeError(
            f"R script did not produce {output_csv}. stderr: {result.stderr or 'none'}; stdout: {result.stdout or 'none'}"
        )
    return pd.read_csv(output_csv)


def run_one_macrorep(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    out_dir: Path,
    method: str,
    n_grid: int,
) -> dict:
    """Run one macrorep: Stage1 + Stage2 (one case), return per-method mean coverage, width, interval_score."""
    seed = base_seed + macrorep_id * 10000
    np.random.seed(seed)
    simulator_func = config["simulator_func"]
    n_0 = config.get("n_0", 100)
    r_0 = config.get("r_0", 10)
    n_1 = config.get("n_1", 200)
    r_1 = config.get("r_1", 5)
    X_cand = get_x_cand(simulator_func, config["n_cand"], random_state=seed + 1)
    alpha = config["alpha"]

    stage1 = run_stage1_train(
        n_0=n_0,
        r_0=r_0,
        simulator_func=simulator_func,
        params=config["params"],
        t_grid_size=n_grid,
        random_state=seed + 2,
        verbose=False,
    )
    X0, Y0 = stage1.X_all, stage1.Y_all

    stage2 = run_stage2(
        stage1_result=stage1,
        X_cand=X_cand,
        n_1=n_1,
        r_1=r_1,
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

    case_name = f"n{n_1}_r{r_1}_{method}"
    case_dir = out_dir / f"macrorep_{macrorep_id}" / f"case_{case_name}"
    rep0_dir = case_dir / "macrorep_0"
    rep0_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(rep0_dir / "X0.csv", X0, delimiter=",")
    np.savetxt(rep0_dir / "Y0.csv", Y0, delimiter=",")
    np.savetxt(rep0_dir / "X1.csv", stage2.X_stage2, delimiter=",")
    np.savetxt(rep0_dir / "Y1.csv", stage2.Y_stage2, delimiter=",")
    np.savetxt(rep0_dir / "X_test.csv", X_test, delimiter=",")
    np.savetxt(rep0_dir / "Y_test.csv", Y_test, delimiter=",")
    bench_csv = case_dir / "benchmarks.csv"
    bench_df = _run_benchmarks_one_case(case_dir, bench_csv, alpha, n_grid)

    # Merge bench into rows so we have one DataFrame with all methods
    for i, row in enumerate(rows_ckme):
        for col in bench_df.columns:
            row[col] = bench_df.iloc[i][col]
    df = pd.DataFrame(rows_ckme)

    out = {}
    for name, cov_col in METHOD_COV.items():
        if cov_col in df.columns:
            out[f"{name}_coverage"] = df[cov_col].mean()
    for name, w_col in METHOD_WIDTH.items():
        if w_col in df.columns:
            out[f"{name}_width"] = df[w_col].mean()
    for name, s_col in METHOD_SCORE.items():
        if s_col in df.columns:
            out[f"{name}_interval_score"] = df[s_col].mean()
    return out


def main():
    parser = argparse.ArgumentParser(description="exp3: compare CKME, DCP-DR, hetGP (cov, width, score)")
    parser.add_argument("--config", type=str, default="exp3_test/config.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_macro", type=int, default=5, help="Number of macroreps to average")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="lhs", choices=("lhs", "sampling", "mixed"))
    args = parser.parse_args()

    config_path = _root / args.config
    config = load_config_from_file(config_path)
    config = get_config(config, quick=False)
    n_grid = config.get("t_grid_size", 500)

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp3_test" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not R_SCRIPT.exists():
        print(f"Warning: R script not found at {R_SCRIPT}; DCP-DR and hetGP will be missing.", file=sys.stderr)

    # Collect per-macrorep means
    list_cov = {m: [] for m in METHOD_COV}
    list_width = {m: [] for m in METHOD_WIDTH}
    list_score = {m: [] for m in METHOD_SCORE}

    for k in range(args.n_macro):
        one = run_one_macrorep(
            macrorep_id=k,
            base_seed=args.base_seed,
            config=config,
            out_dir=out_dir,
            method=args.method,
            n_grid=n_grid,
        )
        for m in METHOD_COV:
            key = f"{m}_coverage"
            if key in one:
                list_cov[m].append(one[key])
        for m in METHOD_WIDTH:
            key = f"{m}_width"
            if key in one:
                list_width[m].append(one[key])
        for m in METHOD_SCORE:
            key = f"{m}_interval_score"
            if key in one:
                list_score[m].append(one[key])

    # Summary table
    rows = []
    for m in list(METHOD_COV.keys()):
        cov_vals = list_cov.get(m, [])
        w_vals = list_width.get(m, [])
        s_vals = list_score.get(m, [])
        rows.append({
            "method": m,
            "mean_coverage": np.mean(cov_vals) if cov_vals else np.nan,
            "sd_coverage": np.std(cov_vals, ddof=1) if len(cov_vals) > 1 else np.nan,
            "mean_width": np.mean(w_vals) if w_vals else np.nan,
            "sd_width": np.std(w_vals, ddof=1) if len(w_vals) > 1 else np.nan,
            "mean_interval_score": np.mean(s_vals) if s_vals else np.nan,
            "sd_interval_score": np.std(s_vals, ddof=1) if len(s_vals) > 1 else np.nan,
            "n_macroreps": len(cov_vals),
        })
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "exp3_compare_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
