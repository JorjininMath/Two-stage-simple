"""
stage2.py

Stage 2: site selection, data collection, CP calibration.
Includes save/load for Stage2Result.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple, Union

if TYPE_CHECKING:
    from CKME import CKMEModel
    from CP.cp import CP

import numpy as np

from .data_collection import collect_stage2_data
from .design import sample_iid_qx
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
        Calibration inputs (method="iid") or selected design sites
        (legacy methods), shape (n_1, d).
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
        "iid" (protocol default), or legacy "sampling"/"lhs"/"mixed".
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
    X_cand: Optional[ArrayLike],
    n_1: int,
    r_1: int,
    simulator_func: str = "exp1",
    method: Literal["iid", "sampling", "lhs", "mixed"] = "iid",
    alpha: float = 0.1,
    X_bounds: Optional[Tuple[ArrayLike, ArrayLike]] = None,
    mixed_ratio: float = 0.7,
    random_state: Optional[int] = None,
    verbose: bool = False,
    s0_score_type: str = "tail",
    qx_sampler: Optional[Callable[[int, np.random.Generator], ArrayLike]] = None,
) -> Stage2Result:
    """
    Run Stage 2: build the calibration set, collect D_1, calibrate CP.

    Default protocol (method="iid"): calibration inputs are drawn iid from
    the target law q_X (uniform over X_bounds unless qx_sampler is given)
    with exactly ONE fresh simulator output each (r_1 = 1). With the Stage-1
    model frozen beforehand, calibration pairs and a fresh test pair are iid
    from q_X x F(.|x), so the split-CP finite-sample coverage guarantee
    holds exactly. Replications belong in Stage-1 TRAINING (scale learning,
    per-site ECDFs, distinct-site compression), not in calibration.

    Legacy modes ("sampling", "lhs", "mixed") select design sites from
    X_cand and replicate r_1 times per site. They are kept only to
    reproduce older experiments: clustered, non-q_X calibration data breaks
    exchangeability with a single-draw test point, so the exact guarantee
    does not apply — a UserWarning is emitted.

    Parameters
    ----------
    stage1_result : Stage1TrainResult or str or Path
        Stage 1 result. If str/Path, load from disk.
    X_cand : array-like, shape (n_cand, d), or None
        Candidate points for legacy site selection. Ignored (may be None)
        when method="iid".
    n_1 : int
        Calibration size (method="iid") or number of sites (legacy).
    r_1 : int
        Replications per site. Must be 1 when method="iid".
    simulator_func : str, default="exp1"
        Simulator name.
    method : {"iid", "sampling", "lhs", "mixed"}, default="iid"
        "iid" = iid q_X calibration (protocol default); others = legacy
        site selection.
    alpha : float, default=0.1
        CP significance level.
    X_bounds : tuple, optional
        (lower, upper) box bounds. From experiment config if None.
    mixed_ratio : float, default=0.7
        γ for the legacy mixed method.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Print progress.
    s0_score_type : str, default="tail"
        S⁰ score variant for legacy score-driven selection.
    qx_sampler : callable, optional
        Custom q_X sampler for method="iid": qx_sampler(n, rng) -> (n, d).
        Default is uniform over X_bounds.

    Returns
    -------
    Stage2Result
    """
    if isinstance(stage1_result, (str, Path)):
        stage1_result = load_stage1_train_result(stage1_result)

    res = stage1_result

    if X_bounds is None:
        exp_config = get_experiment_config(simulator_func)
        X_bounds = exp_config["bounds"]

    if method == "iid":
        # Protocol default: iid q_X calibration with one draw per input.
        if r_1 != 1:
            raise ValueError(
                f"method='iid' requires r_1=1 (got r_1={r_1}): replicated "
                "calibration scores are clustered and not exchangeable with "
                "a single-draw test point. Spend the same budget on "
                f"n_1={n_1 * r_1} distinct iid calibration points instead, "
                "or use replications in Stage-1 training."
            )
        X_1 = sample_iid_qx(
            n=n_1, d=res.d, bounds=X_bounds,
            random_state=random_state, qx_sampler=qx_sampler,
        )
        if verbose:
            print(f"Stage 2: {n_1} iid calibration inputs ~ q_X")
    else:
        # Legacy site-selection modes (kept to reproduce older experiments).
        warnings.warn(
            f"run_stage2(method='{method}', r_1={r_1}): legacy calibration "
            "uses design-selected sites"
            + (f" with {r_1} replications each" if r_1 > 1 else "")
            + "; calibration data is then not an iid sample from q_X, so "
            "the exact split-CP coverage guarantee does not apply. Use "
            "method='iid' (r_1=1) for the guarantee-bearing protocol.",
            UserWarning,
            stacklevel=2,
        )
        if X_cand is None:
            raise ValueError("X_cand is required for legacy methods "
                             "('sampling', 'lhs', 'mixed')")
        X_cand = np.atleast_2d(np.asarray(X_cand, dtype=float))

        # S^0 is only needed for score-driven site selection. Plain LHS
        # ignores scores, so skip this potentially expensive computation.
        if method == "lhs":
            s0 = np.zeros(X_cand.shape[0], dtype=float)
        else:
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

    # CP: default = raw point-evaluated scores (guarantee-bearing convention;
    # grid-free). Pass t_grid=res.t_grid to switch to the optional projected-
    # score variant (Policy B); see notes/planning/cdf_legality_policy.md §6.
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
