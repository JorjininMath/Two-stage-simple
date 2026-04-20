"""
pretrain_params.py

Run once before run_onesided_compare.py to find optimal CKME hyperparameters
for each simulator via k-fold CV on a pilot dataset.

Uses one-sided pinball loss (averaged over tau=0.05 and tau=0.95) as the
tuning loss, directly targeting the conditional quantiles used in the
one-sided CP scores.

Best params are saved to exp_onesided/pretrained_params.json and loaded
automatically by run_onesided_compare.py.

Usage (from project root):
    python exp_onesided/pretrain_params.py
    python exp_onesided/pretrain_params.py --n_pilot 200 --r_pilot 5 --cv_folds 5
    python exp_onesided/pretrain_params.py --sims exp2,nongauss_B2L
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

from CKME.parameters import ParamGrid
from CKME.tuning import cross_validate_ckme
from CKME.loss_functions import OneSidedPinballLoss, HybridCRPSPinballLoss
from Two_stage.sim_functions import get_experiment_config

SIMULATORS = ["exp1", "exp2", "nongauss_B2L", "nongauss_A1S", "nongauss_A1L"]

# Search grid for 1D simulators on [0, 2*pi] domain.
PARAM_GRID = ParamGrid(
    ell_x_list=[0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0],
    lam_list=[1e-3, 1e-2, 1e-1],
    h_list=[0.1, 0.3, 0.5],
)

# Primary loss: one-sided pinball averaged over tau=0.05 and tau=0.95.
# Directly targets the conditional quantiles used in one-sided CP scores.
LOSS_FN = OneSidedPinballLoss(taus=[0.05, 0.95])

# Alternative: hybrid CRPS + pinball (more stable, less task-focused).
# Uncomment to use instead of the pinball loss above.
# LOSS_FN = HybridCRPSPinballLoss(taus=[0.05, 0.95], lam=0.5)


def pretrain_one(
    simulator_func: str,
    n_pilot: int,
    r_pilot: int,
    cv_folds: int,
    t_grid_size: int,
    random_state: int,
) -> dict:
    """CV-tune CKME params for one simulator using tail-weighted CRPS."""
    rng = np.random.default_rng(random_state)

    sim_cfg = get_experiment_config(simulator_func)
    simulator = sim_cfg["simulator"]
    x_lo = float(sim_cfg["bounds"][0].item())
    x_hi = float(sim_cfg["bounds"][1].item())

    # Pilot data: n_pilot fixed sites x r_pilot reps (X and Y correctly paired)
    X_sites = rng.uniform(x_lo, x_hi, size=(n_pilot, 1))
    Y_reps = [
        simulator(X_sites.ravel(), random_state=int(rng.integers(0, 2**31)))
        for _ in range(r_pilot)
    ]
    X_pilot = np.tile(X_sites, (r_pilot, 1))   # (n_pilot * r_pilot, 1)
    Y_pilot = np.concatenate(Y_reps)            # (n_pilot * r_pilot,)

    # t_grid covering pilot Y range with 10% margin
    y_margin = 0.1 * (Y_pilot.max() - Y_pilot.min())
    t_grid = np.linspace(Y_pilot.min() - y_margin, Y_pilot.max() + y_margin, t_grid_size)

    results = cross_validate_ckme(
        X_pilot, Y_pilot,
        param_grid=PARAM_GRID,
        t_grid=t_grid,
        loss_fn=LOSS_FN,
        cv_folds=cv_folds,
        random_state=random_state,
        verbose=True,
    )
    return results.best_params.as_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train CKME hyperparameters for exp_onesided via CV"
    )
    parser.add_argument("--n_pilot",     type=int, default=200,
                        help="Number of pilot design sites (default: 200)")
    parser.add_argument("--r_pilot",     type=int, default=5,
                        help="Replications per site in pilot (default: 5)")
    parser.add_argument("--cv_folds",    type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--t_grid_size", type=int, default=200,
                        help="Threshold grid size for CV (default: 200)")
    parser.add_argument("--seed",        type=int, default=0,
                        help="Random seed for pilot data (default: 0)")
    parser.add_argument("--output",      type=str, default=None,
                        help="Output JSON path (default: exp_onesided/pretrained_params.json)")
    parser.add_argument("--sims",        type=str, default=None,
                        help="Comma-separated subset of simulators to retune (default: all). "
                             "E.g. --sims exp2,nongauss_B2L")
    args = parser.parse_args()

    out_path = (
        Path(args.output) if args.output
        else _root / "exp_onesided" / "pretrained_params.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing so we can update only requested simulators
    best_params: dict = {}
    if out_path.exists():
        with open(out_path) as f:
            best_params = json.load(f)

    sims_to_run = (
        [s.strip() for s in args.sims.split(",")]
        if args.sims else SIMULATORS
    )

    for sim in sims_to_run:
        print(f"\n--- CV for {sim} (n_pilot={args.n_pilot}, r_pilot={args.r_pilot}, "
              f"loss=OneSidedPinball) ---")
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
    print("Run run_onesided_compare.py to use these params.")


if __name__ == "__main__":
    main()
