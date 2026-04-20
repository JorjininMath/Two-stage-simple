"""
pretrain_params.py

CV-tune CKME hyperparameters using round-1 D0 data (earliest ~25% of the
full sample). A random subsample of n_pilot observations is used to keep
Cholesky O(n^3) tractable.

Run once before run_returns.py:
    python exp_returns/pretrain_params.py
    python exp_returns/pretrain_params.py --n_pilot 600 --cv_folds 5

Output: exp_returns/pretrained_params.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from CKME.ckme import CKMEModel
from CKME.parameters import ParamGrid, Params
from exp_returns.preprocess import load_returns_data, get_round_splits, split_d0_d1

# Search grid tuned for lrealvol domain (~0.2-10%) and returns Y domain (~-25 to +15%)
PARAM_GRID = ParamGrid(
    ell_x_list=[0.3, 0.5, 1.0, 2.0, 4.0],   # scale of lrealvol (%)
    lam_list  =[1e-3, 1e-2, 1e-1],
    h_list    =[0.5, 1.0, 2.0, 3.0],          # scale of returns (%)
)


def pretrain(
    n_pilot: int,
    cv_folds: int,
    t_grid_size: int,
    split_ratio: float,
    holdout_frac: float,
    n_rounds: int,
    seed: int,
) -> dict:
    Y, X = load_returns_data()
    splits = get_round_splits(len(Y), n_rounds=n_rounds, holdout_frac=holdout_frac)
    if not splits:
        raise RuntimeError("No valid splits generated — check data length and holdout_frac.")

    # Use round 1 D0 (chronologically oldest portion)
    ind_cp, _ = splits[0]
    ind_d0, _ = split_d0_d1(ind_cp, split_ratio)
    X0, Y0 = X[ind_d0], Y[ind_d0]

    print(f"Round-1 D0 size: {len(Y0)} observations")

    # Subsample for speed if D0 is large
    rng = np.random.default_rng(seed)
    if len(Y0) > n_pilot:
        idx = rng.choice(len(Y0), size=n_pilot, replace=False)
        idx.sort()                  # preserve rough temporal order
        X0, Y0 = X0[idx], Y0[idx]
        print(f"  Subsampled to n_pilot={n_pilot}")

    # t_grid based on pilot data Y range
    t_grid = np.linspace(np.quantile(Y0, 0.001), np.quantile(Y0, 0.999), t_grid_size)

    n_combos = (len(PARAM_GRID.ell_x_list) * len(PARAM_GRID.lam_list) * len(PARAM_GRID.h_list))
    print(f"Starting CV over {n_combos} param combos × {cv_folds} folds ...")
    model = CKMEModel(indicator_type="logistic")
    model.fit(
        X0, Y0,
        param_grid=PARAM_GRID,
        t_grid=t_grid,
        cv_folds=cv_folds,
        random_state=seed,
        verbose=True,
    )
    return model.params.as_dict()


def main():
    parser = argparse.ArgumentParser(description="Pretrain CKME params for exp_returns")
    parser.add_argument("--n_pilot",     type=int,   default=600,
                        help="Max observations from D0 used for CV (default: 600)")
    parser.add_argument("--cv_folds",    type=int,   default=5)
    parser.add_argument("--t_grid_size", type=int,   default=200)
    parser.add_argument("--split_ratio", type=float, default=0.5)
    parser.add_argument("--holdout_frac",type=float, default=0.10)
    parser.add_argument("--n_rounds",    type=int,   default=5)
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--output",      type=str,   default=None)
    args = parser.parse_args()

    out_path = (
        Path(args.output) if args.output
        else _root / "exp_returns" / "pretrained_params.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best = pretrain(
        n_pilot     =args.n_pilot,
        cv_folds    =args.cv_folds,
        t_grid_size =args.t_grid_size,
        split_ratio =args.split_ratio,
        holdout_frac=args.holdout_frac,
        n_rounds    =args.n_rounds,
        seed        =args.seed,
    )
    print(f"\nBest params: {best}")
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved to {out_path}")
    print("Now run: python exp_returns/run_returns.py")


if __name__ == "__main__":
    main()
