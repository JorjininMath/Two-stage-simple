"""
pretrain_params.py

Run once before run_allocation_compare.py to find optimal CKME hyperparameters
for exp1 and exp2 via k-fold cross-validation on a pilot dataset.

The best params are saved to exp_allocation/pretrained_params.json and loaded
automatically by run_allocation_compare.py.

Usage (from project root):
    python exp_allocation/pretrain_params.py
    python exp_allocation/pretrain_params.py --cv_folds 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from CKME.parameters import ParamGrid
from Two_stage import run_stage1_train

# Use Stage-1 sizes from the paper for tuning
PILOT_CFG = {
    "exp3_alloc": {"n_0": 250, "r_0": 20, "t_grid_size": 500},
    "exp2":       {"n_0": 250, "r_0": 20, "t_grid_size": 500},
}

PARAM_GRID = ParamGrid(
    ell_x_list=[0.05, 0.1, 0.2, 0.5],
    lam_list=[1e-5, 1e-3, 1e-1],
    h_list=[0.05, 0.1],
)


def pretrain_one(sim: str, cv_folds: int, seed: int) -> dict:
    cfg = PILOT_CFG[sim]
    result = run_stage1_train(
        n_0=cfg["n_0"],
        r_0=cfg["r_0"],
        simulator_func=sim,
        param_grid=PARAM_GRID,
        t_grid_size=cfg["t_grid_size"],
        cv_folds=cv_folds,
        random_state=seed,
        verbose=True,
    )
    return result.params.as_dict()


def main():
    parser = argparse.ArgumentParser(description="Pre-train CKME hyperparameters via CV")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--output",   type=str, default=None)
    parser.add_argument("--sims",     type=str, default=None,
                        help="Comma-separated subset to retune (default: all). E.g. --sims exp1")
    args = parser.parse_args()

    out_path = (
        Path(args.output) if args.output
        else _root / "exp_allocation" / "pretrained_params.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_params: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            best_params = json.load(f)

    sims_to_run = (
        [s.strip() for s in args.sims.split(",")]
        if args.sims else list(PILOT_CFG.keys())
    )

    for sim in sims_to_run:
        print(f"\n--- CV for {sim} ---")
        params = pretrain_one(sim, args.cv_folds, args.seed)
        best_params[sim] = params
        print(f"  Best params: {params}")

    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
