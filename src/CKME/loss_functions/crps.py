"""
crps.py

Implementation of Continuous Ranked Probability Score (CRPS) loss function.

CRPS measures the difference between predicted CDF and empirical CDF
of the true value. It is defined as:

    CRPS = ∫ [F_pred(t) - 1{Y_true <= t}]² dt

When approximated on a grid t_1 < ... < t_M, it becomes:

    CRPS_i(θ) = Σ_{m=1}^{M} w_m (F̂_θ(t_m | x_i) - 1{y_i ≤ t_m})²

where w_m are weights for numerical integration.

Weight options (as per documentation):
- Uniform spacing: w_m = Δt (constant spacing)
- Quantile-based spacing: w_m ≈ 1/M (equal probability mass)
- Adaptive tail weighting: w_m ∝ 1 / √(τ_m(1 - τ_m))

Lower CRPS values indicate better predictions.

References
----------
Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules,
prediction, and estimation. Journal of the American Statistical Association.
"""

from __future__ import annotations

from typing import Literal, List, Optional

import numpy as np

ArrayLike = np.ndarray

WeightScheme = Literal["uniform", "quantile", "adaptive_tail"]


class CRPSLoss:
    """
    Continuous Ranked Probability Score (CRPS) loss function.

    This class provides methods to compute CRPS for single batches and
    multiple batches (e.g., CV folds).

    The implementation follows the quadrature rule:
        CRPS_i(θ) = Σ_{m=1}^{M} w_m (F̂_θ(t_m | x_i) - 1{y_i ≤ t_m})²

    where weights w_m can be chosen according to different schemes.
    """

    def __init__(self, weight_scheme: WeightScheme = "uniform"):
        """
        Initialize CRPS loss function.

        Parameters
        ----------
        weight_scheme : {"uniform", "quantile", "adaptive_tail"}, default="uniform"
            Weight scheme for numerical integration:
            - "uniform": w_m = Δt (for evenly spaced grids)
            - "quantile": w_m ≈ 1/M (for quantile-based grids)
            - "adaptive_tail": w_m ∝ 1 / √(τ_m(1 - τ_m)) (emphasizes tails)
        """
        self.weight_scheme = weight_scheme

    def _compute_weights(
        self,
        t_grid: ArrayLike,
        quantile_levels: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Compute weights w_m for numerical integration.

        Parameters
        ----------
        t_grid : ndarray, shape (M,)
            Threshold grid.

        quantile_levels : ndarray, optional, shape (M,)
            Quantile levels corresponding to t_grid (for quantile/adaptive schemes).

        Returns
        -------
        weights : ndarray, shape (M,)
            Integration weights.
        """
        M = t_grid.shape[0]

        if self.weight_scheme == "uniform":
            # Uniform spacing: w_m = Δt
            # For evenly spaced grid, Δt is constant
            if M == 1:
                return np.array([1.0])
            dt = np.diff(t_grid)
            # Use average spacing for all points (simple approach)
            # For more accuracy, could use: w[0] = dt[0]/2, w[1:-1] = (dt[:-1] + dt[1:])/2, w[-1] = dt[-1]/2
            # But for simplicity, use constant Δt as per documentation
            avg_dt = np.mean(dt)
            return np.full(M, avg_dt)

        elif self.weight_scheme == "quantile":
            # Quantile-based spacing: w_m ≈ 1/M
            return np.full(M, 1.0 / M)

        elif self.weight_scheme == "adaptive_tail":
            # Adaptive tail weighting: w_m ∝ 1 / √(τ_m(1 - τ_m))
            if quantile_levels is None:
                # If quantile levels not provided, assume uniform quantiles
                quantile_levels = np.linspace(0, 1, M)
            else:
                quantile_levels = np.asarray(quantile_levels, dtype=float).ravel()
                if quantile_levels.shape[0] != M:
                    raise ValueError(
                        f"quantile_levels must have length {M}, got {quantile_levels.shape[0]}"
                    )

            # Compute weights: w_m ∝ 1 / √(τ_m(1 - τ_m))
            tau = quantile_levels
            # Avoid division by zero at boundaries
            weights = 1.0 / np.sqrt(np.maximum(tau * (1 - tau), 1e-10))
            # Normalize so that sum approximates the integral range
            # For proper normalization, we'd need the range, but for relative weighting this is fine
            weights = weights / np.sum(weights) * (t_grid[-1] - t_grid[0])
            return weights

        else:
            raise ValueError(f"Unknown weight_scheme: {self.weight_scheme}")

    def compute(
        self,
        F_pred: ArrayLike,  # shape (n, M) - predicted CDF values
        Y_true: ArrayLike,  # shape (n,) - true Y values
        t_grid: ArrayLike,  # shape (M,) - threshold grid
        quantile_levels: Optional[ArrayLike] = None,
    ) -> float:
        """
        Compute CRPS loss.

        Parameters
        ----------
        F_pred : ndarray, shape (n, M)
            Predicted CDF values F(t_m | x_i) for each sample and threshold.
            Each row corresponds to one sample, each column to one threshold.

        Y_true : ndarray, shape (n,)
            True Y values for each sample.

        t_grid : ndarray, shape (M,)
            Threshold grid used for CDF evaluation. Must be sorted.

        quantile_levels : ndarray, optional, shape (M,)
            Quantile levels corresponding to t_grid. Required for
            "adaptive_tail" weight scheme, optional for others.

        Returns
        -------
        crps : float
            Average CRPS across all samples.

        Notes
        -----
        The implementation follows the quadrature rule:
            CRPS_i(θ) = Σ_{m=1}^{M} w_m (F̂_θ(t_m | x_i) - 1{y_i ≤ t_m})²
        """
        F_pred = np.asarray(F_pred, dtype=float)
        Y_true = np.asarray(Y_true, dtype=float).ravel()
        t_grid = np.asarray(t_grid, dtype=float).ravel()

        n, M = F_pred.shape
        if Y_true.shape[0] != n:
            raise ValueError(
                f"F_pred and Y_true must have same number of samples, "
                f"got {n} and {Y_true.shape[0]}"
            )
        if t_grid.shape[0] != M:
            raise ValueError(
                f"F_pred and t_grid must have same number of thresholds, "
                f"got {M} and {t_grid.shape[0]}"
            )

        # Empirical CDF: 1{Y_true <= t} for each sample and threshold
        # Shape: (n, M)
        empirical_cdf = (Y_true[:, np.newaxis] <= t_grid[np.newaxis, :]).astype(float)

        # Squared difference: [F_pred(t) - 1{Y <= t}]²
        # Shape: (n, M)
        squared_diff = (F_pred - empirical_cdf) ** 2

        # Compute weights according to the chosen scheme
        # Shape: (M,)
        weights = self._compute_weights(t_grid, quantile_levels)

        # Apply quadrature rule: Σ_m w_m * squared_diff
        # Shape: (n,)
        crps_per_sample = np.sum(weights[np.newaxis, :] * squared_diff, axis=1)

        # Return average CRPS
        return float(np.mean(crps_per_sample))

    def compute_batch(
        self,
        F_pred_list: List[ArrayLike],  # List of (n_i, M) arrays
        Y_true_list: List[ArrayLike],  # List of (n_i,) arrays
        t_grid: ArrayLike,
        quantile_levels: Optional[ArrayLike] = None,
    ) -> float:
        """
        Compute CRPS loss over multiple batches (e.g., CV folds).

        This is a convenience method for cross-validation where data is
        split into multiple folds. It concatenates all batches and
        computes the overall CRPS.

        Parameters
        ----------
        F_pred_list : list of ndarray
            List of predicted CDF matrices, one per batch/fold.
            Each matrix has shape (n_i, M) where n_i is the number of
            samples in that batch.

        Y_true_list : list of ndarray
            List of true Y arrays, one per batch/fold.
            Each array has shape (n_i,).

        t_grid : ndarray, shape (M,)
            Threshold grid (same for all batches).

        quantile_levels : ndarray, optional, shape (M,)
            Quantile levels corresponding to t_grid. Required for
            "adaptive_tail" weight scheme.

        Returns
        -------
        crps : float
            Average CRPS across all batches.
        """
        F_all = np.vstack(F_pred_list)
        Y_all = np.concatenate(Y_true_list)
        return self.compute(F_all, Y_all, t_grid, quantile_levels=quantile_levels)

