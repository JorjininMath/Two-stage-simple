"""
stage2.py

Stage 2: site selection, data collection, CP calibration.
Includes save/load for Stage2Result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from CKME import CKMEModel
    from CP.cp import CP

import numpy as np

from .data_collection import collect_stage2_data
from .io import load_stage1_train_result
from .s0_score import compute_s0
from .site_selection import select_stage2_sites
from .stage2_cp import stage2_cp_calibrate
from .stage1_train import Stage1TrainResult
from .sim_functions import get_experiment_config

ArrayLike = np.ndarray


@dataclass
class Stage2Result:
    """
    Result from Stage 2: selected sites, D_1, calibrated CP.

    Attributes
    ----------
    model : CKMEModel
        Stage 1 model (reused).
    t_grid : ndarray
        Threshold grid.
    X_1 : ndarray
        Selected design sites, shape (n_1, d).
    X_stage2 : ndarray
        D_1 inputs, shape (n_1 * r_1, d).
    Y_stage2 : ndarray
        D_1 outputs, shape (n_1 * r_1,).
    cp : CP
        Calibrated conformal predictor.
    n_1 : int
        Number of selected sites.
    r_1 : int
        Replications per site.
    selection_method : str
        "sampling", "lhs", or "mixed".
    alpha : float
        Significance level.
    """

    model: "CKMEModel"
    t_grid: np.ndarray
    X_1: np.ndarray
    X_stage2: np.ndarray
    Y_stage2: np.ndarray
    cp: "CP"
    n_1: int
    r_1: int
    selection_method: str
    alpha: float

    def predict_interval(self, X_query: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """Predict intervals for X_query. Returns (L, U)."""
        return self.cp.predict_interval(X_query, self.t_grid)


def run_stage2(
    stage1_result: Union[Stage1TrainResult, str, Path],
    X_cand: ArrayLike,
    n_1: int,
    r_1: int,
    simulator_func: str = "exp1",
    method: Literal["sampling", "lhs", "mixed"] = "sampling",
    alpha: float = 0.1,
    X_bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    mixed_ratio: float = 0.7,
    random_state: Optional[int] = None,
    verbose: bool = False,
    s0_score_type: str = "tail",
) -> Stage2Result:
    """
    Run Stage 2: select sites, collect D_1, calibrate CP.

    Parameters
    ----------
    stage1_result : Stage1TrainResult or str or Path
        Stage 1 result. If str/Path, load from disk.
    X_cand : array-like, shape (n_cand, d)
        Candidate points for site selection.
    n_1 : int
        Number of sites to select.
    r_1 : int
        Replications per site.
    simulator_func : str, default="exp1"
        Simulator name.
    method : {"sampling", "lhs", "mixed"}, default="sampling"
        Site selection method.
    alpha : float, default=0.1
        CP significance level.
    X_bounds : tuple, optional
        (lower, upper) for lhs/mixed. From experiment config if None.
    mixed_ratio : float, default=0.7
        γ for mixed method.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Print progress.
    s0_score_type : str, default="tail"
        S⁰ score variant: "tail" (quantile interval width) or "epistemic"
        (bootstrap variance of CDF estimate).

    Returns
    -------
    Stage2Result
    """
    if isinstance(stage1_result, (str, Path)):
        stage1_result = load_stage1_train_result(stage1_result)

    res = stage1_result
    X_cand = np.atleast_2d(np.asarray(X_cand, dtype=float))

    if X_bounds is None:
        exp_config = get_experiment_config(simulator_func)
        X_bounds = exp_config["bounds"]

    # S^0
    s0 = compute_s0(
        res, X_cand, alpha=alpha, score_type=s0_score_type,
        random_state=random_state,
    )
    if verbose:
        print(f"Stage 2: S^0 range [{s0.min():.4f}, {s0.max():.4f}]")

    # Select sites
    X_1 = select_stage2_sites(
        X_cand=X_cand,
        scores=s0,
        n_1=n_1,
        method=method,
        X_bounds=X_bounds,
        random_state=random_state,
        mixed_ratio=mixed_ratio,
    )
    if verbose:
        print(f"  Selected {n_1} sites (method={method})")

    # Collect D_1
    X_stage2, Y_stage2 = collect_stage2_data(
        X_1=X_1,
        r_1=r_1,
        simulator_func=simulator_func,
        random_state=random_state,
    )
    if verbose:
        print(f"  Collected D_1: {X_stage2.shape[0]} points ({n_1} × {r_1})")

    # CP
    cp = stage2_cp_calibrate(
        model=res.model,
        X_stage2=X_stage2,
        Y_stage2=Y_stage2,
        alpha=alpha,
        verbose=verbose,
    )

    return Stage2Result(
        model=res.model,
        t_grid=res.t_grid,
        X_1=X_1,
        X_stage2=X_stage2,
        Y_stage2=Y_stage2,
        cp=cp,
        n_1=n_1,
        r_1=r_1,
        selection_method=method,
        alpha=alpha,
    )


def save_stage2_result(result: Stage2Result, path: Union[str, Path]) -> None:
    """
    Save Stage2Result to disk. Self-contained (includes model, t_grid).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    result.model.save(path / "model.npz")
    np.save(path / "t_grid.npy", result.t_grid)
    np.save(path / "X_1.npy", result.X_1)
    np.save(path / "X_stage2.npy", result.X_stage2)
    np.save(path / "Y_stage2.npy", result.Y_stage2)
    meta = {
        "n_1": result.n_1,
        "r_1": result.r_1,
        "selection_method": result.selection_method,
        "alpha": result.alpha,
        "q_hat": result.cp.q_hat,
    }
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    np.savetxt(path / "X_1.csv", result.X_1, delimiter=",")
    np.savetxt(path / "X_stage2.csv", result.X_stage2, delimiter=",")
    np.savetxt(path / "Y_stage2.csv", result.Y_stage2, delimiter=",")


def load_stage2_result(path: Union[str, Path]) -> Stage2Result:
    """
    Load Stage2Result from disk. Re-calibrates CP from saved D_1.
    """
    from CKME import CKMEModel

    path = Path(path)
    model = CKMEModel.load(path / "model.npz")
    t_grid = np.load(path / "t_grid.npy")
    X_1 = np.load(path / "X_1.npy")
    X_stage2 = np.load(path / "X_stage2.npy")
    Y_stage2 = np.load(path / "Y_stage2.npy")
    meta = json.loads((path / "meta.json").read_text())
    cp = stage2_cp_calibrate(
        model=model,
        X_stage2=X_stage2,
        Y_stage2=Y_stage2,
        alpha=meta["alpha"],
        verbose=False,
    )
    return Stage2Result(
        model=model,
        t_grid=t_grid,
        X_1=X_1,
        X_stage2=X_stage2,
        Y_stage2=Y_stage2,
        cp=cp,
        n_1=int(meta["n_1"]),
        r_1=int(meta["r_1"]),
        selection_method=meta["selection_method"],
        alpha=float(meta["alpha"]),
    )
