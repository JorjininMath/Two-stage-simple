"""
Pretrain CKME hyperparameters for exp_design DGPs.

Usage:
  python exp_design/pretrain_params.py --dgp exp2_gauss_low exp2_gauss_high
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

from CKME import CKMEModel
from CKME.parameters import ParamGrid
from CKME.tuning import tune_ckme_params
from Two_stage.design import generate_space_filling_design


def pretrain_one(dgp_name: str, dgp_config: dict, n_0: int = 250,
                 r_0: int = 20, seed: int = 2026) -> dict:
    """CV-tune params for one DGP. Returns dict with ell_x, lam, h."""
    simulator = dgp_config["simulator"]
    bounds = dgp_config["bounds"]
    d = dgp_config["d"]

    X_0 = generate_space_filling_design(
        n=n_0, d=d, method="lhs", bounds=bounds, random_state=seed,
    )

    X_all_list, Y_all_list = [], []
    for i in range(n_0):
        xi = X_0[i]
        xi_rep = np.tile(xi, (r_0, 1)) if d > 1 else np.full(r_0, xi.item())
        yi = simulator(xi_rep, random_state=seed + 100 + i)
        X_all_list.append(xi_rep.reshape(r_0, -1) if d > 1 else xi_rep.reshape(r_0, 1))
        Y_all_list.append(yi.ravel())

    X_all = np.vstack(X_all_list)
    Y_all = np.concatenate(Y_all_list)

    Y_lo = np.percentile(Y_all, 0.5)
    Y_hi = np.percentile(Y_all, 99.5)
    y_margin = 0.10 * (Y_hi - Y_lo)
    t_grid = np.linspace(Y_lo - y_margin, Y_hi + y_margin, 500)

    param_grid = ParamGrid(
        ell_x_list=[0.05, 0.1, 0.2, 0.5, 1.0],
        lam_list=[1e-5, 1e-3, 1e-1],
        h_list=[0.05, 0.1, 0.2],
    )

    best_params, _ = tune_ckme_params(
        X_all, Y_all, param_grid=param_grid, t_grid=t_grid,
        cv_folds=5, random_state=seed,
    )

    return {"ell_x": best_params.ell_x, "lam": best_params.lam, "h": best_params.h}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp", nargs="+", default=["exp2_gauss_low", "exp2_gauss_high"])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    from exp_design.run_design_compare import load_dgp

    out_file = Path(__file__).parent / "pretrained_params.json"
    if out_file.exists():
        with open(out_file) as f:
            all_params = json.load(f)
    else:
        all_params = {}

    for name in args.dgp:
        print(f"Tuning {name} ...", flush=True)
        config = load_dgp(name)
        result = pretrain_one(name, config, seed=args.seed)
        all_params[name] = result
        print(f"  {name}: {result}")

    with open(out_file, "w") as f:
        json.dump(all_params, f, indent=2)
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
