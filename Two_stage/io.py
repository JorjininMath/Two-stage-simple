"""
io.py

Save/load for Stage 1 train result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from .stage1_train import Stage1TrainResult


def save_stage1_train_result(result: Stage1TrainResult, path: Union[str, Path]) -> None:
    """
    Save Stage1TrainResult to disk.

    Saves model (via CKMEModel.save) and metadata. Also writes human-readable
    .json and .csv files for inspection.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Binary (for load)
    result.model.save(path / "model.npz")
    np.save(path / "t_grid.npy", result.t_grid)
    np.save(path / "X_all.npy", result.X_all)
    np.save(path / "Y_all.npy", result.Y_all)
    np.save(path / "X_0.npy", result.X_0)
    np.save(
        path / "params.npy",
        np.array([result.params.ell_x, result.params.lam, result.params.h]),
    )
    np.save(
        path / "meta.npy",
        np.array([result.n_0, result.r_0, result.d], dtype=int),
    )

    # Human-readable
    (path / "meta.json").write_text(json.dumps({
        "n_0": result.n_0,
        "r_0": result.r_0,
        "d": result.d,
    }, indent=2))
    (path / "params.json").write_text(json.dumps({
        "ell_x": result.params.ell_x,
        "lam": result.params.lam,
        "h": result.params.h,
    }, indent=2))
    (path / "model_info.json").write_text(json.dumps({
        "indicator_type": result.model.indicator_type,
        "ell_x": result.model.params.ell_x,
        "lam": result.model.params.lam,
        "h": result.model.params.h,
        "n": int(result.model.n),
        "d": int(result.model.d),
    }, indent=2))
    np.savetxt(path / "t_grid.csv", result.t_grid, delimiter=",")
    np.savetxt(path / "X_0.csv", result.X_0, delimiter=",")
    np.savetxt(path / "X_all.csv", result.X_all, delimiter=",")
    np.savetxt(path / "Y_all.csv", result.Y_all, delimiter=",")
    (path / "README.txt").write_text("""Stage 1 saved model

Readable files:
  meta.json     - n_0, r_0, d
  params.json   - ell_x, lam, h
  model_info.json - model metadata
  t_grid.csv    - threshold grid
  X_0.csv       - unique design sites
  X_all.csv     - all inputs
  Y_all.csv     - all outputs

Binary (for load_stage1_train_result):
  model.npz, *.npy
""")


def save_s0_scores(
    X_cand: np.ndarray,
    s0: np.ndarray,
    path: Union[str, Path],
) -> None:
    """
    Save S^0 scores and corresponding X_cand to disk.

    path: directory to write files into.
    Writes human-readable CSV files:
      s0_X_cand.csv - candidate points
      s0_scores.csv - S^0 scores (same order as X_cand)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savetxt(path / "s0_X_cand.csv", X_cand, delimiter=",")
    np.savetxt(path / "s0_scores.csv", s0, delimiter=",")


def load_stage1_train_result(path: Union[str, Path]) -> Stage1TrainResult:
    """
    Load Stage1TrainResult from disk.
    """
    from CKME import CKMEModel
    from CKME.parameters import Params

    path = Path(path)

    model = CKMEModel.load(path / "model.npz")
    t_grid = np.load(path / "t_grid.npy")
    X_all = np.load(path / "X_all.npy")
    Y_all = np.load(path / "Y_all.npy")
    X_0 = np.load(path / "X_0.npy")
    p = np.load(path / "params.npy")
    n_0, r_0, d = np.load(path / "meta.npy")

    params = Params(ell_x=float(p[0]), lam=float(p[1]), h=float(p[2]))

    return Stage1TrainResult(
        model=model,
        t_grid=t_grid,
        X_all=X_all,
        Y_all=Y_all,
        X_0=X_0,
        params=params,
        n_0=int(n_0),
        r_0=int(r_0),
        d=int(d),
    )
