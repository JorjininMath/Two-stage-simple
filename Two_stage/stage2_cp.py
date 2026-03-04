"""
stage2_cp.py

Stage 2 conformal prediction. Uses Stage 1 model + D_1 only.
Score type: abs_median.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from CP import CP

if TYPE_CHECKING:
    from CKME.ckme import CKMEModel

ArrayLike = np.ndarray


def stage2_cp_calibrate(
    model: "CKMEModel",
    X_stage2: ArrayLike,
    Y_stage2: ArrayLike,
    alpha: float = 0.1,
    verbose: bool = False,
) -> CP:
    """
    Calibrate CP using Stage 2 data D_1 only.

    Uses score_type="abs_median". Model is the Stage 1 model (not refitted).

    Parameters
    ----------
    model : CKMEModel
        Stage 1 trained model.
    X_stage2 : array-like, shape (n, d)
        Stage 2 inputs D_1.
    Y_stage2 : array-like, shape (n,)
        Stage 2 outputs D_1.
    alpha : float, default=0.1
        Significance level. Coverage ≈ 1 - alpha.
    verbose : bool, default=False
        Print calibration info.

    Returns
    -------
    cp : CP
        Calibrated CP. Use cp.predict_interval(X_query, t_grid).
    """
    X_stage2 = np.atleast_2d(np.asarray(X_stage2, dtype=float))
    Y_stage2 = np.asarray(Y_stage2, dtype=float).ravel()
    if X_stage2.shape[0] != Y_stage2.shape[0]:
        raise ValueError(
            f"X_stage2 and Y_stage2 must have same length. "
            f"Got {X_stage2.shape[0]} vs {Y_stage2.shape[0]}"
        )
    cp = CP(model=model, alpha=alpha, score_type="abs_median")
    cp.calibrate(X_stage2, Y_stage2, verbose=verbose)
    if verbose:
        print(f"Stage 2 CP calibrated: q̂ = {cp.q_hat:.6f}")
    return cp
