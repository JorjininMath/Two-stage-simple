"""
cp.py

Conformal Prediction wrapper for CKME models.

This module implements the CP class, which provides:
- Nonconformity score computation (modular design for easy extension)
- Calibration using calibration set to compute conformal quantile q̂
- Prediction interval construction via level set search

Key idea
--------
Conformal Prediction provides distribution-free uncertainty quantification.
Given a trained CKME model and a calibration set, we:
1. Compute nonconformity scores A(x, y) for calibration data
2. Compute conformal quantile q̂ = (1-α) quantile of calibration scores
3. For new x, construct prediction set: C_{1-α}(x) = {y : A(x, y) ≤ q̂}

The score layer is designed to be modular, allowing easy extension to
different score types (e.g., MMD-based scores) without changing the main CP logic.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

# Import CKMEModel for type hints
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CKME.ckme import CKMEModel

# Import wrappers
from .calibration import calibrate as _calibrate
from .interval import predict_interval as _predict_interval
from .scores import VALID_SCORE_TYPES

ArrayLike = np.ndarray


class CP:
    """
    Conformal Prediction wrapper for CKME models.

    This class provides a modular interface for conformal prediction with CKME,
    separating score computation, calibration, and interval construction.

    Usage
    -----
    # Train CKME model
    model = CKMEModel(indicator_type="logistic")
    model.fit(X_train, Y_train, params=params)

    # Create CP wrapper
    cp = CP(model=model, alpha=0.1, score_type="abs_median")

    # Calibrate using calibration set
    cp.calibrate(X_cal, Y_cal)

    # Predict intervals for new queries
    t_grid = np.linspace(Y_min, Y_max, 100)
    L, U = cp.predict_interval(X_query, t_grid)
    # L, U are arrays of shape (n_query,) - left and right bounds
    """

    def __init__(
        self,
        model: CKMEModel,
        alpha: float = 0.1,
        score_type: str = "abs_median",
    ) -> None:
        """
        Initialize the CP wrapper.

        Parameters
        ----------
        model : CKMEModel
            Trained CKME model. Must be fitted before use.

        alpha : float, default=0.1
            Significance level. The prediction intervals will have coverage
            approximately (1 - alpha). For example, alpha=0.1 gives 90% coverage.

        score_type : str, default="abs_median"
            Type of nonconformity score: "abs_median" = |F(y | x) - 0.5|

        Raises
        ------
        ValueError
            If alpha is not in (0, 1), or if score_type is not supported.
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        if score_type not in VALID_SCORE_TYPES:
            raise ValueError(
                f"score_type must be one of {VALID_SCORE_TYPES}, "
                f"got {score_type}. "
                "Other score types (e.g., 'mmd') can be added in the future."
            )

        self.model = model
        self.alpha = alpha
        self.score_type = score_type

        # Calibration state (set by calibrate())
        self.q_hat: Optional[float] = None
        self.calibration_scores: Optional[ArrayLike] = None


    # ---------------------------------------------------------------------------
    # Calibration Layer (wrapper)
    # ---------------------------------------------------------------------------

    def calibrate(
        self,
        X_cal: ArrayLike,
        Y_cal: ArrayLike,
        verbose: bool = False,
    ) -> None:
        """
        Calibrate the CP model using a calibration set.

        This method wraps the calibration module to:
        1. Compute nonconformity scores for all (x, y) pairs in the calibration set
        2. Compute the conformal quantile q̂ = (1-α) quantile of calibration scores
        3. Store q̂ for use in prediction interval construction

        Parameters
        ----------
        X_cal : array-like, shape (n_cal, d)
            Calibration input points.

        Y_cal : array-like, shape (n_cal,)
            Calibration output values.

        verbose : bool, default=False
            If True, print calibration information.

        Notes
        -----
        The conformal quantile is computed as:
            q̂ = quantile(scores, (1-α) * (n_cal + 1) / n_cal)

        This ensures finite-sample coverage guarantees.
        """
        self.q_hat, self.calibration_scores = _calibrate(
            self.model,
            X_cal,
            Y_cal,
            self.alpha,
            score_type=self.score_type,
            verbose=verbose,
        )

    # ---------------------------------------------------------------------------
    # Interval Construction Layer (wrapper)
    # ---------------------------------------------------------------------------

    def predict_interval(
        self,
        X_query: ArrayLike,
        t_grid: ArrayLike,
        q_hat_scale: float = 1.0,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Construct prediction intervals for query inputs using level set search.

        This method wraps the interval construction module to:
        1. Compute pseudo-scores A(x, t) for all t in t_grid
        2. Find the leftmost and rightmost t where A(x, t) ≤ q̂ * q_hat_scale
        3. Return intervals [L(x), U(x)] for each query point

        The prediction set is defined as:
            C_{1-α}(x) = {y : A(x, y) ≤ q̂}  (or q̂ * q_hat_scale when scaling is used)

        Parameters
        ----------
        X_query : array-like, shape (n_query, d) or (d,)
            Query input points. Can be a single point (1D) or multiple points (2D).

        t_grid : array-like, shape (M,)
            Dense grid of Y values for level set search. Should cover the
            range of possible Y values.

        q_hat_scale : float, default=1.0
            Scale factor applied to the conformal quantile: effective threshold
            is q̂ * q_hat_scale. Use e.g. 1.05 to widen intervals (higher coverage).

        Returns
        -------
        L : ndarray, shape (n_query,)
            Left bounds of prediction intervals.

        U : ndarray, shape (n_query,)
            Right bounds of prediction intervals.

        Raises
        ------
        ValueError
            If the model has not been calibrated (calibrate() has not been called).
        """
        if self.q_hat is None:
            raise ValueError(
                "Model has not been calibrated. Call calibrate() first."
            )

        effective_q_hat = self.q_hat * q_hat_scale
        return _predict_interval(
            self.model,
            X_query,
            t_grid,
            effective_q_hat,
            score_type=self.score_type,
        )

