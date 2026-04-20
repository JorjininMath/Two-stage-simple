"""
pretrain_params.py

One-time CV hyperparameter tuning for exp_conditional_coverage.

Runs two-stage CV (ell_x+lam first, then h) at each n in n_vals, saves results to
pretrained_params.json. Run once before the main experiment; the main script
(run_consistency.py) loads this file at startup.

Usage (from project root):
  python exp_conditional_coverage/pretrain_params.py
  python exp_conditional_coverage/pretrain_params.py --simulators exp1
  python exp_conditional_coverage/pretrain_params.py --n_vals 64,128,512,2048 --cv_max_n 512
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from Two_stage import run_stage1_train
from CKME.parameters import ParamGrid, Params

_HERE = Path(__file__).parent

SIMULATORS = ["exp1", "exp2"]

# CV grid (not read from config; defined here for explicitness)
ELL_X_LIST = [0.1, 0.5, 1.0, 2.0, 3.0]
LAM_LIST   = [1e-4, 1e-3, 1e-2, 1e-1]
H_LIST     = [0.05, 0.1, 0.2, 0.3, 0.5]
H_DEFAULT  = 0.2    # fixed h for ell_x/lam stage

CV_FOLDS   = 5
PRETRAIN_SEED = 999999   # dedicated pilot seed, separate from macrorep seeds


def tune_one(
    simulator_func: str,
    n_stage: int,
    r_0: int,
    t_grid_size: int,
    seed: int,
    cv_folds: int,
    verbose: bool = True,
) -> Params:
    """Two-stage CV: (1) tune ell_x+lam with fixed h; (2) tune h with best ell_x+lam."""
    # Stage 1: tune ell_x and lam
    grid1 = ParamGrid(ell_x_list=ELL_X_LIST, lam_list=LAM_LIST, h_list=[H_DEFAULT])
    r1 = run_stage1_train(
        n_0=n_stage, r_0=r_0,
        simulator_func=simulator_func,
        param_grid=grid1,
        t_grid_size=t_grid_size,
        cv_folds=cv_folds,
        random_state=seed,
        verbose=False,
    )
    best_ell_x, best_lam = r1.params.ell_x, r1.params.lam
    if verbose:
        print(f"      stage-1 CV: ell_x={best_ell_x}, lam={best_lam}")

    # Stage 2: tune h
    grid2 = ParamGrid(ell_x_list=[best_ell_x], lam_list=[best_lam], h_list=H_LIST)
    r2 = run_stage1_train(
        n_0=n_stage, r_0=r_0,
        simulator_func=simulator_func,
        param_grid=grid2,
        t_grid_size=t_grid_size,
        cv_folds=cv_folds,
        random_state=seed + 1,
        verbose=False,
    )
    best_h = r2.params.h
    if verbose:
        print(f"      stage-2 CV: h={best_h}")

    return Params(ell_x=best_ell_x, lam=best_lam, h=best_h)


def main():
    parser = argparse.ArgumentParser(description="Pre-tune CKME hyperparameters for consistency experiment")
    parser.add_argument("--simulators", type=str, default=",".join(SIMULATORS),
                        help="Comma-separated simulators to tune")
    parser.add_argument("--n_vals",    type=str, default="64,128,512,2048",
                        help="Comma-separated n values to tune at")
    parser.add_argument("--cv_max_n", type=int, default=512,
                        help="Only tune for n <= cv_max_n; reuse largest-tuned params for larger n")
    parser.add_argument("--r_0",       type=int, default=1)
    parser.add_argument("--t_grid_size", type=int, default=500)
    parser.add_argument("--output",    type=str, default=None,
                        help="Output JSON path (default: exp_conditional_coverage/pretrained_params.json)")
    parser.add_argument("--force",     action="store_true",
                        help="Re-tune even if entry already exists in output JSON")
    args = parser.parse_args()

    sims   = [s.strip() for s in args.simulators.split(",")]
    n_vals = [int(v) for v in args.n_vals.split(",")]
    out_path = Path(args.output) if args.output else (_HERE / "pretrained_params.json")

    # Load existing cache
    cache: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            cache = json.load(f)
        print(f"Loaded existing cache from {out_path}")

    # Determine which n values actually need tuning
    tunable_ns = [n for n in n_vals if args.cv_max_n is None or n <= args.cv_max_n]
    reuse_ns   = [n for n in n_vals if n not in tunable_ns]
    if reuse_ns:
        print(f"  cv_max_n={args.cv_max_n}: will tune n={tunable_ns}, reuse for n={reuse_ns}")

    for sim in sims:
        print(f"\n=== {sim} ===")
        if sim not in cache:
            cache[sim] = {}

        # Tune at each tunable n
        for idx_n, n in enumerate(tunable_ns):
            key = str(n)
            if key in cache[sim] and not args.force:
                p = cache[sim][key]
                print(f"  n={n}: cached -> ell_x={p['ell_x']}, lam={p['lam']}, h={p['h']}")
                continue

            print(f"  n={n}: tuning (two-stage CV, cv_folds={args.cv_folds if hasattr(args, 'cv_folds') else CV_FOLDS})...")
            seed = PRETRAIN_SEED + sims.index(sim) * 10000 + idx_n * 100
            params = tune_one(
                simulator_func=sim,
                n_stage=n,
                r_0=args.r_0,
                t_grid_size=args.t_grid_size,
                seed=seed,
                cv_folds=CV_FOLDS,
                verbose=True,
            )
            cache[sim][key] = {"ell_x": params.ell_x, "lam": params.lam, "h": params.h}
            print(f"  n={n}: tuned  -> ell_x={params.ell_x}, lam={params.lam}, h={params.h}")

            # Save after each n in case of interruption
            with open(out_path, "w") as f:
                json.dump(cache, f, indent=2)

        # Fill reuse_ns from largest tuned n
        if tunable_ns:
            src_n  = max(tunable_ns)
            src_key = str(src_n)
            if src_key in cache[sim]:
                src_p = cache[sim][src_key]
                for n in reuse_ns:
                    cache[sim][str(n)] = dict(src_p)   # copy, not alias
                    print(f"  n={n}: reused from n={src_n} -> "
                          f"ell_x={src_p['ell_x']}, lam={src_p['lam']}, h={src_p['h']}")

        with open(out_path, "w") as f:
            json.dump(cache, f, indent=2)

    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
