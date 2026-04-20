"""
pretrain_params.py

CV-tune hyperparameters for the two dissertation DGPs.
Run once before run_group_coverage.py.

Usage:
  python dissertation_use/pretrain_params.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
from Two_stage import run_stage1_train
from Two_stage.config_utils import load_config_from_file, get_config
from CKME.parameters import ParamGrid

SIMULATORS = ["exp2", "nongauss_A1L"]

PARAM_GRID = ParamGrid(
    ell_x_list=[0.3, 0.5, 1.0, 2.0],
    lam_list=[1e-3, 1e-2, 1e-1],
    h_list=[0.1, 0.2, 0.3, 0.5],
)

PILOT_SEED_OFFSET = 999999


def main():
    config = get_config(load_config_from_file(_root / "dissertation_use" / "config.txt"), quick=False)
    out_path = _root / "dissertation_use" / "pretrained_params.json"

    results = {}
    for sim in SIMULATORS:
        print(f"--- CV tuning: {sim} ---")
        stage1 = run_stage1_train(
            n_0=config["n_0"],
            r_0=config["r_0"],
            simulator_func=sim,
            param_grid=PARAM_GRID,
            cv_folds=5,
            t_grid_size=config.get("t_grid_size", 500),
            random_state=PILOT_SEED_OFFSET,
            verbose=True,
        )
        p = stage1.params
        results[sim] = {"ell_x": p.ell_x, "lam": p.lam, "h": p.h}
        print(f"  Best: ell_x={p.ell_x}, lam={p.lam}, h={p.h}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
