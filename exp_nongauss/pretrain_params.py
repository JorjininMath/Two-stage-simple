"""
pretrain_params.py

Run once before run_nongauss_compare.py to find optimal CKME hyperparameters
for each non-Gaussian simulator via k-fold cross-validation on a pilot dataset.

The best params are saved to exp_nongauss/pretrained_params.json and loaded
automatically by run_nongauss_compare.py.

Usage (from project root):
    python exp_nongauss/pretrain_params.py
    python exp_nongauss/pretrain_params.py --n_pilot 150 --r_pilot 10 --cv_folds 5
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

SIMULATORS = [
    "nongauss_A1S", "nongauss_B2S", "nongauss_C1S",   # Small (light non-Gaussianity)
    "nongauss_A1L", "nongauss_B2L", "nongauss_C1L",   # Large (strong non-Gaussianity)
]

# Search grid tuned for the [0, 2*pi] domain.
PARAM_GRID = ParamGrid(
    ell_x_list=[0.5, 1.0, 2.0, 3.0],
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
    parser = argparse.ArgumentParser(description="Pre-train CKME hyperparameters via CV")
    parser.add_argument("--n_pilot",     type=int, default=100,
                        help="Number of pilot design sites (default: 100)")
    parser.add_argument("--r_pilot",     type=int, default=10,
                        help="Replications per site in pilot (default: 10)")
    parser.add_argument("--cv_folds",    type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--t_grid_size", type=int, default=200,
                        help="Threshold grid size for CV (default: 200)")
    parser.add_argument("--seed",        type=int, default=0,
                        help="Random seed for pilot data (default: 0)")
    parser.add_argument("--output",      type=str, default=None,
                        help="Output JSON path (default: exp_nongauss/pretrained_params.json)")
    parser.add_argument("--sims",        type=str, default=None,
                        help="Comma-separated subset of simulators to retune (default: all). "
                             "E.g. --sims nongauss_C1S,nongauss_C1L")
    args = parser.parse_args()

    out_path = (
        Path(args.output) if args.output
        else _root / "exp_nongauss" / "pretrained_params.json"
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
    print("Run run_nongauss_compare.py to use these params.")


if __name__ == "__main__":
    main()
