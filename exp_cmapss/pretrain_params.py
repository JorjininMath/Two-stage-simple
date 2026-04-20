"""
pretrain_params.py

CV hyperparameter tuning for the C-MAPSS CKME model.
Run once before run_cmapss.py.

Because C-MAPSS has ~16k training samples, we subsample for CV speed.
Best params are saved to exp_cmapss/pretrained_params.json and
automatically picked up by run_cmapss.py.

Usage
-----
python exp_cmapss/pretrain_params.py [--config exp_cmapss/config.txt] [--n_subsample 3000]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from CKME import CKMEModel
from CKME.parameters import ParamGrid
from exp_cmapss.preprocess import load_and_preprocess


def load_config(config_path: str) -> dict:
    cfg = {}
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, val = line.partition("=")
            cfg[key.strip()] = val.split("#")[0].strip()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="exp_cmapss/config.txt")
    parser.add_argument(
        "--n_subsample", type=int, default=3000,
        help="Max training samples to use for CV (random subsample for speed).",
    )
    parser.add_argument("--cv_folds", type=int, default=3)
    parser.add_argument("--n_jobs",   type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir       = cfg.get("data_dir", "exp_cmapss/data")
    rul_cap        = int(cfg.get("rul_cap", 125))
    window         = int(cfg.get("window_size", 20))
    train_frac     = float(cfg.get("train_frac", 0.8))
    late_stage_rul = int(cfg.get("late_stage_rul", 30))
    random_state   = int(cfg.get("random_state", 42))
    t_grid_size    = int(cfg.get("t_grid_size", 300))
    output_json    = Path(cfg.get("output_dir", "exp_cmapss/output")).parent / "pretrained_params.json"

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    print("Loading and preprocessing FD001...")
    data = load_and_preprocess(
        data_dir=data_dir,
        rul_cap=rul_cap,
        window=window,
        train_frac=train_frac,
        late_stage_rul=late_stage_rul,
        random_state=random_state,
        verbose=True,
    )

    X_train = data["X_train"]
    Y_train = data["Y_train"]

    # -----------------------------------------------------------------------
    # Subsample for CV speed
    # -----------------------------------------------------------------------
    n_total = X_train.shape[0]
    if n_total > args.n_subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_total, size=args.n_subsample, replace=False)
        X_cv = X_train[idx]
        Y_cv = Y_train[idx]
        print(f"Subsampled {args.n_subsample} / {n_total} training samples for CV.")
    else:
        X_cv = X_train
        Y_cv = Y_train
        print(f"Using all {n_total} training samples for CV.")

    # -----------------------------------------------------------------------
    # t_grid for CV
    # -----------------------------------------------------------------------
    t_min, t_max = Y_cv.min(), Y_cv.max()
    margin = 0.05 * (t_max - t_min)
    t_grid = np.linspace(t_min - margin, t_max + margin, t_grid_size)

    # -----------------------------------------------------------------------
    # Parameter grid
    # (kept moderate — C-MAPSS is high-dim so smaller ell_x often better)
    # -----------------------------------------------------------------------
    param_grid = ParamGrid(
        ell_x_list=[0.3, 0.5, 1.0, 2.0, 5.0],
        lam_list=[1e-3, 1e-2, 1e-1],
        h_list=[0.1, 0.3, 0.5],
    )
    n_combos = len(param_grid.ell_x_list) * len(param_grid.lam_list) * len(param_grid.h_list)
    print(f"Grid: {n_combos} combinations, {args.cv_folds}-fold CV, n_jobs={args.n_jobs}")

    # -----------------------------------------------------------------------
    # CV tuning
    # -----------------------------------------------------------------------
    model = CKMEModel(indicator_type="logistic")
    model.fit(
        X=X_cv,
        Y=Y_cv,
        param_grid=param_grid,
        t_grid=t_grid,
        loss_type="crps",
        cv_folds=args.cv_folds,
        n_jobs=args.n_jobs,
        random_state=random_state,
        verbose=True,
    )

    best = model.params
    print(f"\nBest params: ell_x={best.ell_x}, lam={best.lam}, h={best.h}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    output_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "ell_x": best.ell_x,
        "lam": best.lam,
        "h": best.h,
        "cv_folds": args.cv_folds,
        "n_subsample": args.n_subsample,
        "best_cv_loss": float(model.tuning_results.best_loss),
    }
    output_json.write_text(json.dumps(result, indent=2))
    print(f"Saved to {output_json}")


if __name__ == "__main__":
    main()
