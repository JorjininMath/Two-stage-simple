"""
stage1_train.py

Step 1: Train CKME model and store. No S^0, no CP calibration.

This module provides a clean interface for the first step of the two-stage pipeline:
1. Collect Stage 1 data D_0
2. Train CKME model with hyperparameter tuning
3. Return result object (model, t_grid, metadata) ready for saving
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from CKME import CKMEModel
from CKME.parameters import ParamGrid, Params

from .data_collection import collect_stage1_data
from .sim_functions import get_experiment_config

ArrayLike = np.ndarray

# t_grid construction: clip heavy-tail outliers then add a symmetric margin
_T_GRID_LO_PCT: float = 0.5    # lower percentile of Y for range estimation
_T_GRID_HI_PCT: float = 99.5   # upper percentile of Y for range estimation
_T_GRID_MARGIN: float = 0.10   # fractional margin beyond the clipped range


@dataclass
class Stage1TrainResult:
    """
    Result from Step 1: trained model and metadata. No CP, no S^0.

    Attributes
    ----------
    model : CKMEModel
        Trained CKME model.
    t_grid : ndarray
        Threshold grid for CDF evaluation.
    X_all : ndarray
        Stage 1 input data D_0.
    Y_all : ndarray
        Stage 1 output data D_0.
    X_0 : ndarray
        Unique design sites.
    params : Params
        Optimal hyperparameters.
    n_0 : int
        Number of design sites.
    r_0 : int
        Replications per site.
    d : int
        Input dimension.
    """

    model: CKMEModel
    t_grid: np.ndarray
    X_all: np.ndarray
    Y_all: np.ndarray
    X_0: np.ndarray
    params: Params
    n_0: int
    r_0: int
    d: int


def run_stage1_train(
    n_0: int,
    r_0: int = 5,
    simulator_func: str = "exp1",
    param_grid: Optional[ParamGrid] = None,
    params: Optional[Params] = None,
    X_bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    d: Optional[int] = None,
    design_method: str = "lhs",
    indicator_type: str = "logistic",
    loss_type: str = "crps",
    cv_folds: int = 5,
    n_jobs: int = 1,
    t_grid_size: int = 100,
    t_grid_margin: Optional[float] = None,
    random_state: Optional[int] = None,
    verbose: bool = False,
    dtype: Optional[type] = None,
) -> Stage1TrainResult:
    """
    Step 1: Train CKME model. No S^0, no CP calibration.

    Parameters
    ----------
    n_0 : int
        Number of Stage 1 design sites.
    r_0 : int, default=5
        Replications per site.
    simulator_func : str, default="exp1"
        Experiment name: "exp1", "exp2", "exp3", "exp4", "expA".
    params : Params, optional
        Fixed hyperparameters. If provided, no CV tuning (fast).
    param_grid : ParamGrid, optional
        Hyperparameter search grid. Required if params not provided.
    X_bounds : tuple, optional
        (lower_bounds, upper_bounds) for design. From experiment config if None.
    d : int, optional
        Input dimension. From experiment config if None.
    design_method : str, default="lhs"
        "lhs" or "grid".
    indicator_type : str, default="logistic"
        Smooth indicator type.
    loss_type : str, default="crps"
        Loss for CV tuning.
    cv_folds : int, default=5
        Cross-validation folds.
    n_jobs : int, default=1
        Parallel jobs for CV.
    t_grid_size : int, default=100
        Points in threshold grid.
    t_grid_margin : float, optional
        Fractional margin beyond [P{LO}, P{HI}] of Y for t_grid bounds.
        If None, uses module default (0.10). Set larger (e.g. 1.0-2.0) for
        heavy-tailed DGPs where the right tail extends far beyond P99.5.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    Stage1TrainResult
        Trained model and metadata. Use save_stage1_train_result() to persist.
    """
    if params is None and (param_grid is None or param_grid.is_empty()):
        raise ValueError("Provide either params (fixed) or param_grid (tuning)")

    exp_config = get_experiment_config(simulator_func)
    if X_bounds is None:
        X_bounds = exp_config["bounds"]
    if d is None:
        d = exp_config["d"]

    if verbose:
        print("Step 1: Train CKME model")
        print("  Collecting data...")

    X_all, Y_all = collect_stage1_data(
        n_0=n_0,
        d=d,
        r_0=r_0,
        simulator_func=simulator_func,
        X_bounds=X_bounds,
        design_method=design_method,
        random_state=random_state,
    )

    if verbose:
        print(f"  Collected {X_all.shape[0]} points ({n_0} sites × {r_0} reps)")

    # Use percentile-based bounds to prevent heavy-tail outliers from dominating
    # the t_grid and causing CDF inversion to clip at the boundary for high τ.
    Y_lo = np.percentile(Y_all, _T_GRID_LO_PCT)
    Y_hi = np.percentile(Y_all, _T_GRID_HI_PCT)
    margin_frac = _T_GRID_MARGIN if t_grid_margin is None else float(t_grid_margin)
    y_margin = margin_frac * (Y_hi - Y_lo)
    t_grid = np.linspace(Y_lo - y_margin, Y_hi + y_margin, t_grid_size)

    if verbose:
        print("  Fitting model...")

    model = CKMEModel(indicator_type=indicator_type)
    if params is not None:
        model.fit(X=X_all, Y=Y_all, params=params, r=r_0, verbose=verbose, dtype=dtype)
    else:
        model.fit(
            X=X_all,
            Y=Y_all,
            param_grid=param_grid,
            t_grid=t_grid,
            loss_type=loss_type,
            cv_folds=cv_folds,
            n_jobs=n_jobs,
            random_state=random_state,
            r=r_0,
            verbose=verbose,
        )

    X_0 = X_all[::r_0]

    if verbose:
        print(f"  Done. ell_x={model.params.ell_x:.4f}, lam={model.params.lam:.4f}, h={model.params.h:.4f}")

    return Stage1TrainResult(
        model=model,
        t_grid=t_grid,
        X_all=X_all,
        Y_all=Y_all,
        X_0=X_0,
        params=model.params,
        n_0=n_0,
        r_0=r_0,
        d=d,
    )
