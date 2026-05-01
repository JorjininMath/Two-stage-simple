"""
pretrain_params.py

Run once before Exp1 (fixed-h baseline) to find optimal CKME hyperparameters
for each of the 4 DGPs via k-fold cross-validation on a pilot dataset.

Best params are saved to exp_adaptive_h/pretrained_params.json and loaded by
the run scripts. Per-DGP CV is needed because the 4 DGPs span very different
x-domains (e.g. exp1 in [0.1, 0.9] vs gibbs_s1 in [-3, 3]).

Usage (from project root):
    python exp_adaptive_h/pretrain_params.py
    python exp_adaptive_h/pretrain_params.py --n_pilot 200 --r_pilot 10 --cv_folds 5
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
    "wsc_gauss",      # Gaussian, smooth U
    "gibbs_s1",       # Gaussian, interior zero (|sin(x)|, x in [-3, 3])
    "exp1",           # Gaussian, boundary explosion (MG1, x in [0.1, 0.9])
    "nongauss_A1L",   # Student-t nu=3, smooth U
]

# Search grid covers small (exp1) and large (~6-wide) domains.
PARAM_GRID = ParamGrid(
    ell_x_list=[0.1, 0.3, 0.5, 1.0, 2.0, 3.0],
    lam_list=[1e-3, 1e-2, 1e-1],
    h_list=[0.05, 0.1, 0.3, 0.5, 1.0],
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
        cv_folds=cv_folds,
        t_grid_size=t_grid_size,
        random_state=random_state,
    )
    p = result.params
    return {"ell_x": float(p.ell_x), "lam": float(p.lam), "h": float(p.h)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pilot", type=int, default=200)
    ap.add_argument("--r_pilot", type=int, default=10)
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--t_grid_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).parent / "pretrained_params.json"),
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out: dict[str, dict] = {}
    if out_path.exists():
        try:
            out = json.loads(out_path.read_text())
        except Exception:
            out = {}

    for sim in SIMULATORS:
        print(f"\n=== CV pretraining: {sim} ===")
        best = pretrain_one(
            simulator_func=sim,
            n_pilot=args.n_pilot,
            r_pilot=args.r_pilot,
            cv_folds=args.cv_folds,
            t_grid_size=args.t_grid_size,
            random_state=args.seed,
        )
        print(f"  best: ell_x={best['ell_x']}, lam={best['lam']}, h={best['h']}")
        out[sim] = best

    out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
