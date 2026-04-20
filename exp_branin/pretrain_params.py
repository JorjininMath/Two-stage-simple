"""
pretrain_params.py

Run once before run_branin_compare.py to find optimal CKME hyperparameters
for each Branin-Hoo simulator via k-fold cross-validation on a pilot dataset.

The best params are saved to exp_branin/pretrained_params.json and loaded
automatically by run_branin_compare.py.

Usage (from project root):
    python exp_branin/pretrain_params.py
    python exp_branin/pretrain_params.py --n_pilot 150 --r_pilot 5 --cv_folds 5
    python exp_branin/pretrain_params.py --sims branin_student
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

SIMULATORS = ["branin_gauss", "branin_student"]

# Search grid tuned for the Branin-Hoo domain: x1 in [-5,10], x2 in [0,15].
# Domain spans ~15 units per dimension, so ell_x should be larger than
# 1D experiments (domain ~6.3). Range [1.0, 3.0, 5.0, 8.0] covers
# fine-grained to coarse-grained length scales for this domain.
PARAM_GRID = ParamGrid(
    ell_x_list=[1.0, 3.0, 5.0, 8.0],
    lam_list=[1e-3, 1e-2, 1e-1],
    h_list=[0.1, 0.3, 0.5],
)


def pretrain_one(
    simulator_func: str,
    n_pilot: int,
    r_pilot: int,
    cv_folds: int,
    t_grid_size: int,
    random_state: int,
) -> dict:
    result = run_stage1_train(
        n_0=n_pilot,
        r_0=r_pilot,
        simulator_func=simulator_func,
        param_grid=PARAM_GRID,
        t_grid_size=t_grid_size,
        cv_folds=cv_folds,
        random_state=random_state,
        verbose=True,
    )
    return result.params.as_dict()


def main():
    parser = argparse.ArgumentParser(description="Pre-train CKME hyperparameters for Branin-Hoo")
    parser.add_argument("--n_pilot",     type=int, default=150,
                        help="Number of pilot design sites (default: 150)")
    parser.add_argument("--r_pilot",     type=int, default=5,
                        help="Replications per site in pilot (default: 5)")
    parser.add_argument("--cv_folds",    type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--t_grid_size", type=int, default=200,
                        help="Threshold grid size for CV (default: 200)")
    parser.add_argument("--seed",        type=int, default=0,
                        help="Random seed for pilot data (default: 0)")
    parser.add_argument("--output",      type=str, default=None,
                        help="Output JSON path (default: exp_branin/pretrained_params.json)")
    parser.add_argument("--sims",        type=str, default=None,
                        help="Comma-separated subset of simulators to retune (default: all). "
                             "E.g. --sims branin_student")
    args = parser.parse_args()

    out_path = (
        Path(args.output) if args.output
        else _root / "exp_branin" / "pretrained_params.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing params so we can update only the requested simulators
    best_params: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            best_params = json.load(f)

    sims_to_run = (
        [s.strip() for s in args.sims.split(",")]
        if args.sims else SIMULATORS
    )

    for sim in sims_to_run:
        print(f"\n--- CV for {sim} (n_pilot={args.n_pilot}, r_pilot={args.r_pilot}) ---")
        params = pretrain_one(
            simulator_func=sim,
            n_pilot=args.n_pilot,
            r_pilot=args.r_pilot,
            cv_folds=args.cv_folds,
            t_grid_size=args.t_grid_size,
            random_state=args.seed,
        )
        best_params[sim] = params
        print(f"  Best params: {params}")

    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved pretrained params to {out_path}")
    print("Run run_branin_compare.py to use these params.")


if __name__ == "__main__":
    main()
