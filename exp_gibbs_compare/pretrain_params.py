"""
pretrain_params.py

Run once before run_gibbs_compare.py to CV-tune CKME hyperparameters
for the Gibbs DGPs.  Results saved to exp_gibbs_compare/pretrained_params.json.

Usage (from project root):
    python exp_gibbs_compare/pretrain_params.py
    python exp_gibbs_compare/pretrain_params.py --n_pilot 120 --r_pilot 8
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

SIMULATORS = ["gibbs_s1", "gibbs_s2"]

# Search grid tuned for [-3, 3] domain.
PARAM_GRID = ParamGrid(
    ell_x_list=[0.3, 0.5, 1.0, 2.0],
    lam_list=[1e-3, 1e-2, 1e-1],
    h_list=[0.1, 0.2, 0.4],
)


def pretrain_one(sim, n_pilot, r_pilot, cv_folds, t_grid_size, seed):
    result = run_stage1_train(
        n_0=n_pilot, r_0=r_pilot,
        simulator_func=sim,
        param_grid=PARAM_GRID,
        t_grid_size=t_grid_size,
        cv_folds=cv_folds,
        random_state=seed,
        verbose=True,
    )
    return result.params.as_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pilot",     type=int, default=120)
    parser.add_argument("--r_pilot",     type=int, default=8)
    parser.add_argument("--cv_folds",    type=int, default=5)
    parser.add_argument("--t_grid_size", type=int, default=200)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--output",      type=str, default=None)
    args = parser.parse_args()

    out_path = (
        Path(args.output) if args.output
        else _root / "exp_gibbs_compare" / "pretrained_params.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            best = json.load(f)

    for sim in SIMULATORS:
        print(f"\n--- CV for {sim} ---")
        params = pretrain_one(sim, args.n_pilot, args.r_pilot,
                              args.cv_folds, args.t_grid_size, args.seed)
        best[sim] = params
        print(f"  Best: {params}")

    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
