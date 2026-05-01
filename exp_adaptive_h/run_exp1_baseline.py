"""
exp_adaptive_h / Exp1: Fixed-h baseline (CV-tuned scalar h) on 4 DGPs.

DGPs (all 1D):
    wsc_gauss     - Gaussian, smooth U sigma(x) = 0.01 + 0.20*(x-pi)^2
    gibbs_s1      - Gaussian, interior-zero sigma(x) = |sin(x)|, x in [-3, 3]
    exp1          - Gaussian (MG1), boundary explosion sigma -> infty as x -> 0.9
    nongauss_A1L  - Student-t nu=3, smooth U scale s(x) = 0.01 + 0.20*(x-pi)^2

Sampling scheme (set in config.txt):
    Stage 1 (D_0, fits CKME):  grid (1D, equal-spaced; clean sigma_hat for later)
    Stage 2 (D_1, CP calib):   LHS  (CP exchangeability)
    Test    (X_test):          LHS  (matches stage2 method via test_data.py)

Per macrorep k: seed = base_seed + k * 10000. Fresh draws of (D_0, D_1, X_test) per
macrorep so each macrorep is an independent CP realisation. Within a macrorep, all
4 DGPs share the same seed offset structure but draw their own data.

Outputs:
    exp_adaptive_h/output_exp1/
        macrorep_{k}/case_{sim}/per_point.csv
        exp1_summary.csv        (mean coverage / width / interval-score per DGP)

Usage (from project root):
    python exp_adaptive_h/run_exp1_baseline.py
    python exp_adaptive_h/run_exp1_baseline.py --n_macro 50 --n_workers 8
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
from CKME.parameters import Params
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

# Per-DGP t_grid margin (fraction of [P0.5, P99.5] range).
# Default is 0.10 (in stage1_train.py); heavy-tailed DGPs need larger margin
# to avoid right-tail truncation in CDF inversion. See Idea_buttal.md.
T_GRID_MARGIN = {
    "wsc_gauss":    0.30,
    "gibbs_s1":     0.30,
    "exp1":         0.50,  # boundary explosion sigma -> infty as x -> 0.9
    "nongauss_A1L": 2.00,  # Student-t nu=3, heavy right tail
}


def run_one_macrorep(
    macrorep_id: int,
    base_seed: int,
    config: dict,
    simulator_func: str,
    out_dir: Path,
    n_grid: int,
    params: Params,
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

    eval_result = evaluate_per_point(stage2, X_test, Y_test)
    rows = eval_result["rows"]
    df = pd.DataFrame(rows)

    # t_grid truncation diagnostics (Method 1: see Idea_buttal.md)
    t_lo = float(stage2.t_grid[0])
    t_hi = float(stage2.t_grid[-1])
    Y_arr = np.asarray(Y_test, dtype=float).ravel()
    L_arr = df["L"].to_numpy()
    U_arr = df["U"].to_numpy()
    df["y_in_grid"]    = ((Y_arr >= t_lo) & (Y_arr <= t_hi)).astype(int)
    df["L_at_grid_lo"] = (L_arr <= t_lo + 1e-9).astype(int)
    df["U_at_grid_hi"] = (U_arr >= t_hi - 1e-9).astype(int)
    df["t_grid_lo"]    = t_lo
    df["t_grid_hi"]    = t_hi

    case_dir = out_dir / f"macrorep_{macrorep_id}" / f"case_{simulator_func}"
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(case_dir / "per_point.csv", index=False)

    return {
        "macrorep":       macrorep_id,
        "simulator":      simulator_func,
        "coverage":       float(df["covered_score"].mean()),
        "coverage_intv":  float(df["covered_interval"].mean()),
        "width":          float(df["width"].mean()),
        "interval_score": float(df["interval_score"].mean()),
        "n_test":         len(df),
    }


def _worker(args_tuple):
    return run_one_macrorep(*args_tuple)


def main():
    parser = argparse.ArgumentParser(description="Exp1: fixed-h baseline on 4 DGPs")
    parser.add_argument("--config",     type=str, default="exp_adaptive_h/config.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_macro",    type=int, default=50)
    parser.add_argument("--base_seed",  type=int, default=20260501)
    parser.add_argument("--n_workers",  type=int, default=1,
                        help="Parallel processes for macroreps (default: 1 = sequential)")
    args = parser.parse_args()

    config = get_config(load_config_from_file(_root / args.config), quick=False)
    n_grid = config.get("t_grid_size", 500)

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_adaptive_h" / "output_exp1"
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
        print(f"  {sim:14s}  ell_x={p.ell_x}, lam={p.lam}, h={p.h}")

    print(
        f"\nn_macro={args.n_macro}, n_workers={args.n_workers}, "
        f"stage1={STAGE1_DESIGN}, stage2={STAGE2_METHOD}, "
        f"output_dir={out_dir}"
    )

    rep_rows: list[dict] = []
    for sim in SIMULATORS:
        print(f"\n--- Simulator: {sim} ---")
        params = pretrained[sim]
        jobs = [
            (k, args.base_seed, config, sim, out_dir, n_grid, params)
            for k in range(args.n_macro)
        ]

        if args.n_workers > 1:
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                fut_to_k = {pool.submit(_worker, j): j[0] for j in jobs}
                for fut in as_completed(fut_to_k):
                    k = fut_to_k[fut]
                    one = fut.result()
                    rep_rows.append(one)
                    print(f"  macrorep {k:3d}  cov={one['coverage']:.3f}  "
                          f"width={one['width']:.3f}  IS={one['interval_score']:.3f}")
        else:
            for j in jobs:
                one = _worker(j)
                rep_rows.append(one)
                k = j[0]
                print(f"  macrorep {k:3d}  cov={one['coverage']:.3f}  "
                      f"width={one['width']:.3f}  IS={one['interval_score']:.3f}")

    rep_df = pd.DataFrame(rep_rows).sort_values(["simulator", "macrorep"]).reset_index(drop=True)
    rep_df.to_csv(out_dir / "exp1_per_macrorep.csv", index=False)

    summary = (
        rep_df.groupby("simulator")
        .agg(
            mean_coverage=("coverage", "mean"),
            sd_coverage=("coverage", "std"),
            mean_coverage_intv=("coverage_intv", "mean"),
            mean_width=("width", "mean"),
            sd_width=("width", "std"),
            mean_interval_score=("interval_score", "mean"),
            sd_interval_score=("interval_score", "std"),
            n_macroreps=("coverage", "count"),
        )
        .reindex(SIMULATORS)
        .reset_index()
    )
    summary_path = out_dir / "exp1_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
