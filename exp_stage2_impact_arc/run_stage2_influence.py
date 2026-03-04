"""Stage 2 impact: 9 cases, CKME + DCP-DR + hetGP. Log only."""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
from exp_stage2_impact_arc.config_utils import (
    load_config_from_file,
    get_config,
    get_x_cand,
)
from Two_stage import run_stage1_train, run_stage2
from Two_stage.evaluation import evaluate_per_point
from Two_stage.test_data import generate_test_data

R_SCRIPT_ONE_CASE = _root / "exp_stage2_impact_arc" / "run_benchmarks_one_case.R"

N_0 = 500
R_0 = 10
T_GRID_SIZE = 1000
N_MACRO = 50
MIXED_RATIO = 0.7

STAGE2_CASES = [
    (1000, 5, "lhs"),
    (1000, 5, "sampling"),
    (1000, 5, "mixed"),
    (250, 20, "lhs"),
    (250, 20, "sampling"),
    (250, 20, "mixed"),
    (100, 50, "lhs"),
    (100, 50, "sampling"),
    (100, 50, "mixed"),
]


def _sheet_name(n_1: int, r_1: int, method: str) -> str:
    return f"n{n_1}_r{r_1}_{method}"


def _run_benchmarks_one_case(
    case_dir: Path,
    output_csv: Path,
    alpha: float,
    n_grid: int,
) -> pd.DataFrame:
    cmd = [
        "Rscript",
        str(R_SCRIPT_ONE_CASE),
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
    n_grid: int,
    run_benchmarks: bool = True,
) -> list[tuple[str, list[dict]]]:
    seed = base_seed + macrorep_id * 10000
    np.random.seed(seed)

    simulator_func = config["simulator_func"]
    X_cand = get_x_cand(simulator_func, config["n_cand"], random_state=seed + 1)
    alpha = config["alpha"]

    stage1 = run_stage1_train(
        n_0=N_0,
        r_0=R_0,
        simulator_func=simulator_func,
        params=config["params"],
        t_grid_size=T_GRID_SIZE,
        random_state=seed + 2,
        verbose=False,
    )
    X0, Y0 = stage1.X_all, stage1.Y_all

    results = []
    macrorep_dir = out_dir / f"macrorep_{macrorep_id}"
    for case_idx, (n_1, r_1, method) in enumerate(STAGE2_CASES):
        stage2 = run_stage2(
            stage1_result=stage1,
            X_cand=X_cand,
            n_1=n_1,
            r_1=r_1,
            simulator_func=simulator_func,
            method=method,
            alpha=alpha,
            mixed_ratio=MIXED_RATIO,
            random_state=seed + 3 + case_idx * 1000,
            verbose=False,
        )
        X_test, Y_test = generate_test_data(
            stage2_result=stage2,
            n_test=config["n_test"],
            r_test=config["r_test"],
            X_cand=X_cand,
            simulator_func=simulator_func,
            random_state=seed + 4 + case_idx * 1000,
        )
        eval_result = evaluate_per_point(stage2, X_test, Y_test)
        rows_ckme = eval_result["rows"]
        sheet_name = _sheet_name(n_1, r_1, method)

        if run_benchmarks and R_SCRIPT_ONE_CASE.exists():
            case_dir = macrorep_dir / f"case_{sheet_name}"
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
            for i, row in enumerate(rows_ckme):
                for col in bench_df.columns:
                    row[col] = bench_df.iloc[i][col]

        results.append((sheet_name, rows_ckme))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="exp_stage2_impact_arc/config.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_macro", type=int, default=N_MACRO)
    parser.add_argument("--macrorep_id", type=int, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--no_run_benchmarks", action="store_true")
    args = parser.parse_args()
    args.run_benchmarks = not args.no_run_benchmarks

    config_path = _root / args.config
    config = load_config_from_file(config_path)
    config = get_config(config, quick=False)

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_stage2_impact_arc" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_grid = config.get("t_grid_size", T_GRID_SIZE)

    if args.macrorep_id is not None:
        # ARC mode: run only one macrorep, log to log_macrorep_<id>.txt
        macrorep_id = args.macrorep_id
        log_path = out_dir / f"log_macrorep_{macrorep_id}.txt"
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"start: {datetime.now().isoformat()}\n")
            f.write(f"config: {config_path}\n")
            f.write(f"output_dir: {out_dir}\n")
            f.write(f"macrorep_id: {macrorep_id}, run_benchmarks: {args.run_benchmarks}\n")
        results = run_one_macrorep(
            macrorep_id=macrorep_id,
            base_seed=args.base_seed,
            config=config,
            out_dir=out_dir,
            n_grid=n_grid,
            run_benchmarks=args.run_benchmarks,
        )
        excel_path = out_dir / f"macrorep_{macrorep_id}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, rows in results:
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] macrorep {macrorep_id} done\n")
        return

    # Local mode: run all macroreps, one timestamped log
    n_macro = args.n_macro
    log_path = out_dir / f"stage2_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"start: {datetime.now().isoformat()}\n")
        f.write(f"config: {config_path}\n")
        f.write(f"output_dir: {out_dir}\n")
        f.write(f"n_macro: {n_macro}, run_benchmarks: {args.run_benchmarks}\n")

    for macrorep_id in range(n_macro):
        results = run_one_macrorep(
            macrorep_id=macrorep_id,
            base_seed=args.base_seed,
            config=config,
            out_dir=out_dir,
            n_grid=n_grid,
            run_benchmarks=args.run_benchmarks,
        )
        excel_path = out_dir / f"macrorep_{macrorep_id}.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            for sheet_name, rows in results:
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] macrorep {macrorep_id} done\n")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().strftime('%H:%M:%S')}] done\n")


if __name__ == "__main__":
    main()
