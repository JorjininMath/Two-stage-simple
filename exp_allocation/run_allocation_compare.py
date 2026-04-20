"""
exp_allocation: Stage-2 budget allocation and site selection comparison.

Paper Experiment 1 (Section 5.2).
Compares CKME-CP, DCP-DR, and hetGP across:
  - Two simulators: exp1, exp2
  - Three Stage-2 budget allocations per simulator (fixed total B2 = 5000)
  - Three site selection methods: lhs, sampling, mixed

Usage (from project root):
  python exp_allocation/run_allocation_compare.py
  python exp_allocation/run_allocation_compare.py --n_macro 10 --n_workers 4
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

from Two_stage.config_utils import load_config_from_file, get_config, get_x_cand
from Two_stage import run_stage1_train, run_stage2
from Two_stage.evaluation import evaluate_per_point
from Two_stage.test_data import generate_test_data
from CKME.parameters import Params, ParamGrid

# ---------------------------------------------------------------------------
# Experiment configuration (matches paper Section 5.2 exactly)
# ---------------------------------------------------------------------------

R_SCRIPT = _root / "run_benchmarks_one_case.R"

MIXED_RATIO = 0.7
SITE_METHODS = ["lhs", "sampling", "mixed"]

# Stage-1 settings per simulator
STAGE1_CFG = {
    "exp3_alloc": {"n_0": 250,  "r_0": 20, "t_grid_size": 500},
    "exp2":       {"n_0": 250,  "r_0": 20, "t_grid_size": 500},
}

# Stage-2 budget allocations per simulator (n_1, r_1), total = 5000
ALLOCATIONS = {
    "exp3_alloc": [(100, 50), (200, 25), (500, 10)],
    "exp2":       [(100, 50), (200, 25), (500, 10)],
}

# CV param grid for Stage-1 hyperparameter tuning
PARAM_GRID = ParamGrid(
    ell_x_list=[0.05, 0.1, 0.2, 0.5],
    lam_list=[1e-5, 1e-3, 1e-1],
    h_list=[0.05, 0.1],
)

# Coverage/width/IS column names
#   CKME:   score-based coverage (revised paper definition)
#   DCP-DR: score-based coverage (native to DCP-DR)
#   hetGP:  interval-based coverage (Option A; no conformal score)
#           covered_score_hetgp is also recorded for future Option B switch
METHOD_COV   = {"CKME": "covered_score",    "DCP-DR": "covered_score_dr",    "hetGP": "covered_interval_hetgp"}
METHOD_WIDTH = {"CKME": "width",            "DCP-DR": "width_dr",            "hetGP": "width_hetgp"}
METHOD_IS    = {"CKME": "interval_score",   "DCP-DR": "interval_score_dr",   "hetGP": "interval_score_hetgp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_r_benchmarks(case_dir: Path, output_csv: Path, alpha: float, n_grid: int) -> pd.DataFrame:
    cmd = ["Rscript", str(R_SCRIPT), str(case_dir), str(output_csv), str(alpha), str(n_grid)]
    result = subprocess.run(cmd, cwd=str(_root), capture_output=True, text=True, check=False)
    if not output_csv.exists():
        raise RuntimeError(
            f"R script did not produce {output_csv}.\n"
            f"stderr: {result.stderr or 'none'}\nstdout: {result.stdout or 'none'}"
        )
    return pd.read_csv(output_csv)


# ---------------------------------------------------------------------------
# Core: one macrorep × one simulator
# Stage-1 is run once, then Stage-2 is run for each allocation × method.
# ---------------------------------------------------------------------------

def run_one_sim_macrorep(
    macrorep_id: int,
    base_seed: int,
    sim: str,
    config: dict,
    out_dir: Path,
    params: Params | None,
    s1cfg: dict | None = None,
    sim_allocations: list | None = None,
) -> list[dict]:
    seed = base_seed + macrorep_id * 10000
    if s1cfg is None:
        s1cfg = STAGE1_CFG[sim]
    if sim_allocations is None:
        sim_allocations = ALLOCATIONS[sim]
    n_0, r_0 = s1cfg["n_0"], s1cfg["r_0"]
    n_grid = s1cfg["t_grid_size"]
    alpha = config["alpha"]

    X_cand = get_x_cand(sim, config["n_cand"], random_state=seed + 1)

    # Stage 1: train CKME once per (macrorep, simulator)
    stage1 = run_stage1_train(
        n_0=n_0, r_0=r_0,
        simulator_func=sim,
        params=params,
        param_grid=None if params is not None else PARAM_GRID,
        cv_folds=5 if params is None else None,
        t_grid_size=n_grid,
        random_state=seed + 2,
        verbose=False,
    )
    X0, Y0 = stage1.X_all, stage1.Y_all

    rows = []
    for (n_1, r_1) in sim_allocations:
        for method in SITE_METHODS:
            case_name = f"{sim}_n1{n_1}_r1{r_1}_{method}"
            case_dir = out_dir / f"macrorep_{macrorep_id}" / f"case_{case_name}"
            rep0_dir = case_dir / "macrorep_0"
            rep0_dir.mkdir(parents=True, exist_ok=True)

            stage2 = run_stage2(
                stage1_result=stage1,
                X_cand=X_cand,
                n_1=n_1, r_1=r_1,
                simulator_func=sim,
                method=method,
                alpha=alpha,
                mixed_ratio=MIXED_RATIO,
                random_state=seed + 3 + SITE_METHODS.index(method),
                verbose=False,
            )

            X_test, Y_test = generate_test_data(
                stage2_result=stage2,
                n_test=config["n_test"],
                r_test=config["r_test"],
                X_cand=X_cand,
                simulator_func=sim,
                random_state=seed + 7,
            )

            eval_result = evaluate_per_point(stage2, X_test, Y_test)
            point_rows = eval_result["rows"]

            # Save CSV data for R benchmarks
            np.savetxt(rep0_dir / "X0.csv",     X0,              delimiter=",")
            np.savetxt(rep0_dir / "Y0.csv",     Y0,              delimiter=",")
            np.savetxt(rep0_dir / "X1.csv",     stage2.X_stage2, delimiter=",")
            np.savetxt(rep0_dir / "Y1.csv",     stage2.Y_stage2, delimiter=",")
            np.savetxt(rep0_dir / "X_test.csv", X_test,          delimiter=",")
            np.savetxt(rep0_dir / "Y_test.csv", Y_test,          delimiter=",")

            bench_csv = case_dir / "benchmarks.csv"
            try:
                bench_df = _run_r_benchmarks(case_dir, bench_csv, alpha, n_grid)
                for i, row in enumerate(point_rows):
                    for col in bench_df.columns:
                        row[col] = bench_df.iloc[i][col]
            except RuntimeError as e:
                print(
                    f"  Warning: R benchmarks failed for {case_name}; "
                    f"DCP-DR/hetGP will be NaN.\n  {e}",
                    file=sys.stderr,
                )

            df = pd.DataFrame(point_rows)
            df.to_csv(case_dir / "per_point.csv", index=False)

            # Aggregate metrics
            summary = {
                "simulator": sim, "n_1": n_1, "r_1": r_1,
                "method": method, "macrorep": macrorep_id,
            }
            for mname, col in METHOD_COV.items():
                summary[f"{mname}_coverage"] = df[col].mean() if col in df.columns else float("nan")
            for mname, col in METHOD_WIDTH.items():
                summary[f"{mname}_width"] = df[col].mean() if col in df.columns else float("nan")
            for mname, col in METHOD_IS.items():
                summary[f"{mname}_IS"] = df[col].mean() if col in df.columns else float("nan")
            rows.append(summary)

    return rows


# ---------------------------------------------------------------------------
# Module-level worker (must be top-level for ProcessPoolExecutor pickle)
# ---------------------------------------------------------------------------

def _run_task(args_tuple):
    m, sim, base_seed, config, base_out_dir, pretrained, s1cfg, sim_allocations = args_tuple
    sim_out_dir = base_out_dir / sim
    sim_out_dir.mkdir(parents=True, exist_ok=True)
    params = pretrained.get(sim, None)
    return run_one_sim_macrorep(
        m, base_seed, sim, config, sim_out_dir, params,
        s1cfg=s1cfg, sim_allocations=sim_allocations,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Allocation comparison: CKME vs DCP-DR vs hetGP")
    parser.add_argument("--config",     type=str, default="exp_allocation/config.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_macro",    type=int, default=5)
    parser.add_argument("--base_seed",  type=int, default=42)
    parser.add_argument("--n_workers",  type=int, default=1,
                        help="Parallel worker processes (default: 1 = sequential)")
    parser.add_argument("--sims", type=str, default=None,
                        help="Comma-separated simulators to run (default: all). E.g. --sims exp1")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test mode: tiny data sizes, fixed params, no CV, skips R")
    args = parser.parse_args()

    config = load_config_from_file(_root / args.config)
    base_out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_allocation" / "output"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # Load pretrained hyperparameters (produced by pretrain_params.py if it exists)
    pretrained: dict[str, Params] = {}
    pretrained_path = _root / "exp_allocation" / "pretrained_params.json"
    if pretrained_path.exists():
        with open(pretrained_path) as f:
            raw = json.load(f)
        pretrained = {sim: Params(**raw[sim]) for sim in raw if sim in STAGE1_CFG}
        print(f"Loaded pretrained params from {pretrained_path}")
    else:
        print(
            f"No pretrained_params.json found; will run CV tuning inline (slower).\n"
            "Run 'python exp_allocation/pretrain_params.py' first to speed things up.",
            file=sys.stderr,
        )

    if not R_SCRIPT.exists():
        print(f"Warning: R script not found at {R_SCRIPT}; DCP-DR/hetGP will be missing.",
              file=sys.stderr)

    # Quick mode: small sizes for pipeline testing
    # Filter simulators
    all_sims = list(STAGE1_CFG.keys())
    sims = (
        [s.strip() for s in args.sims.split(",")]
        if args.sims else all_sims
    )
    unknown = [s for s in sims if s not in STAGE1_CFG]
    if unknown:
        print(f"Error: unknown simulators {unknown}. Choose from {all_sims}.", file=sys.stderr)
        sys.exit(1)

    if args.quick:
        print("Quick mode: using tiny data sizes, fixed params, skipping R benchmarks.")
        quick_params = Params(ell_x=0.5, lam=1e-3, h=0.05)
        pretrained = {s: quick_params for s in sims}
        for s in sims:
            STAGE1_CFG[s]["n_0"] = 30
            STAGE1_CFG[s]["r_0"] = 3
            STAGE1_CFG[s]["t_grid_size"] = 50
            ALLOCATIONS[s] = [(10, 5)]
        config["n_test"] = 20
        config["r_test"] = 1
        config["n_cand"] = 50
        def _no_r(*a, **kw):
            raise RuntimeError("R skipped in quick mode")
        subprocess.run = _no_r  # type: ignore

    # Build task list: (macrorep_id, sim)
    tasks = [(m, sim) for m in range(args.n_macro) for sim in sims]

    all_rows: list[dict] = []

    # Snapshot STAGE1_CFG and ALLOCATIONS now (may have been modified by quick mode)
    s1cfg_snap = {s: dict(STAGE1_CFG[s]) for s in sims}
    alloc_snap  = {s: list(ALLOCATIONS[s]) for s in sims}

    if args.n_workers > 1:
        task_args = [
            (m, sim, args.base_seed, config, base_out_dir, pretrained,
             s1cfg_snap[sim], alloc_snap[sim])
            for m, sim in tasks
        ]
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {pool.submit(_run_task, t): (t[0], t[1]) for t in task_args}
            for fut in as_completed(futures):
                m, sim = futures[fut]
                try:
                    all_rows.extend(fut.result())
                    print(f"  Done: macrorep={m}, sim={sim}")
                except Exception as exc:
                    print(f"  Error macrorep={m}, sim={sim}: {exc}", file=sys.stderr)
    else:
        for m, sim in tasks:
            print(f"\n--- macrorep={m}, sim={sim} ---")
            all_rows.extend(_run_task(
                (m, sim, args.base_seed, config, base_out_dir, pretrained,
                 s1cfg_snap[sim], alloc_snap[sim])
            ))

    # Save per-simulator summary CSVs
    if not all_rows:
        print("No results collected (all tasks failed).", file=sys.stderr)
        return
    df_all = pd.DataFrame(all_rows)
    metric_cols = [c for c in df_all.columns if c not in ("simulator", "n_1", "r_1", "method", "macrorep")]

    for sim in sims:
        sim_out_dir = base_out_dir / sim
        df_sim = df_all[df_all["simulator"] == sim]
        df_sim.to_csv(sim_out_dir / "allocation_all_macroreps.csv", index=False)
        agg = df_sim.groupby(["n_1", "r_1", "method"])[metric_cols].agg(["mean", "std"])
        agg.columns = ["_".join(c) for c in agg.columns]
        agg.reset_index().to_csv(sim_out_dir / "allocation_summary.csv", index=False)
        print(f"\n[{sim}] summary saved to {sim_out_dir}/allocation_summary.csv")

    # Combined summary if multiple simulators were run
    if len(sims) > 1:
        agg_all = df_all.groupby(["simulator", "n_1", "r_1", "method"])[metric_cols].agg(["mean", "std"])
        agg_all.columns = ["_".join(c) for c in agg_all.columns]
        agg_all.reset_index().to_csv(base_out_dir / "allocation_summary_combined.csv", index=False)
        print(f"\nCombined summary saved to {base_out_dir}/allocation_summary_combined.csv")

    print(f"\nDone. Results saved to {base_out_dir}/")


if __name__ == "__main__":
    main()
