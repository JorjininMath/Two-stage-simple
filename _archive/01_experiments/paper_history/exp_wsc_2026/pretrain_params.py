"""
pretrain_params.py

Run once before run_wsc_compare.py to CV-tune CKME hyperparameters for each
WSC DGP on a pilot dataset.  Results are saved to pretrained_params.json and
loaded automatically by run_wsc_compare.py.

Usage (from project root):
    python exp_wsc/pretrain_params.py
    python exp_wsc/pretrain_params.py --n_pilot 150 --r_pilot 10 --cv_folds 5
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

# Two DGPs used in the WSC paper.
SIMULATORS = [
    "wsc_gauss",     # Exp 1: Gaussian noise,    sigma(x) = 0.01 + 0.2*(x-pi)^2
    "nongauss_A1L",  # Exp 2: Student-t (nu=3),  same sigma(x)
]

# Search grid suited for the [0, 2*pi] domain.
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
    parser = argparse.ArgumentParser(description="Pre-tune CKME hyperparameters for WSC DGPs")
    parser.add_argument("--n_pilot",     type=int, default=100)
    parser.add_argument("--r_pilot",     type=int, default=10)
    parser.add_argument("--cv_folds",    type=int, default=5)
    parser.add_argument("--t_grid_size", type=int, default=200)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--output",      type=str, default=None,
                        help="Output JSON path (default: exp_wsc/pretrained_params.json)")
    parser.add_argument("--sims",        type=str, default=None,
                        help="Comma-separated subset to retune, e.g. --sims wsc_gauss")
    args = parser.parse_args()

    out_path = (
        Path(args.output) if args.output
        else _root / "exp_wsc" / "pretrained_params.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_params: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            best_params = json.load(f)

    sims_to_run = (
        [s.strip() for s in args.sims.split(",")] if args.sims else SIMULATORS
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
        print(f"  Best: {params}")

    with open(out_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nSaved to {out_path}")
    print("Now run: python exp_wsc/run_wsc_compare.py")


if __name__ == "__main__":
    main()
