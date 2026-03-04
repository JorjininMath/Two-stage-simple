#!/usr/bin/env python3
"""
CRPS (Continuous Ranked Probability Score) Loss Function

CRPS measures the distance between predicted CDF and true step function:
    CRPS(F̂(·|x), y) = ∫ [F̂(t|x) - 1{y ≤ t}]² dt
    ≈ Σ_{m=1}^M w_m [F̂(t_m|x) - 1{y ≤ t_m}]²

Using weighted empirical CDF:
    F̂(t|x) = Σⱼ wⱼ(x) · 1{yⱼ ≤ t}

Following PDF implementation: use quantile grid with appropriate weights.
"""

import numpy as np
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from core.models.ckme_weights import CKMEModel


class CRPSLoss:
    """
    CRPS Loss Function for CKME parameter optimization
    
    Similar structure to MMDLoss, following PDF implementation guidelines.
    
    Formula:
        CRPS(F̂(·|x_i), y_i) = ∫ [F̂(t|x_i) - 1{y_i ≤ t}]² dt
        ≈ Σ_{m=1}^M w_m [F̂(t_m|x_i) - 1{y_i ≤ t_m}]²
        
    where:
        - t_m are grid points (sample quantiles)
        - F̂(t_m|x_i) = Σⱼ wⱼ(x_i) · 1{yⱼ ≤ t_m} (weighted ECDF)
        - w_m are integration weights
    """
    
    def __init__(self, 
                 grid_size: int = 100,
                 weight_type: Literal['uniform', 'quantile', 'tail_weighted'] = 'quantile'):
        """
        Initialize CRPS loss
        
        Args:
            grid_size: Number of grid points M (default: 100)
            weight_type: 
                - 'uniform': w_m = Δt (for evenly spaced grid)
                - 'quantile': w_m ≈ 1/M (for quantile-based grid, default)
                - 'tail_weighted': w_m ∝ 1/√(τ_m(1-τ_m)) (adaptive tail weighting)
        """
        self.grid_size = grid_size
        self.weight_type = weight_type
    
    def _create_quantile_grid(self, Y: np.ndarray) -> tuple:
        """
        Create grid using sample quantiles (as suggested in PDF)
        
        Args:
            Y: Training Y values [n]
            
        Returns:
            (t_grid, weights) where:
            - t_grid: [M] quantile values
            - weights: [M] integration weights
        """
        # Create quantile levels: avoid 0 and 1 to prevent numerical issues
        # Use (0.5/M, 1.5/M, ..., (M-0.5)/M) instead of [0, 1/M, 2/M, ..., 1]
        m = self.grid_size
        quantile_levels = (np.arange(m) + 0.5) / m  # (0.5/M, 1.5/M, ..., (M-0.5)/M)
        t_grid = np.quantile(Y, quantile_levels)  # [M]
        
        # Compute weights based on weight_type
        if self.weight_type == 'quantile':
            # Quantile-based: w_m ≈ 1/M (empirical probability mass between quantiles)
            weights = np.ones(self.grid_size) / self.grid_size
        elif self.weight_type == 'tail_weighted':
            # Adaptive tail weighting: w_m ∝ 1/√(τ_m(1-τ_m))
            # Now tau is in (0, 1) so no extreme values at boundaries
            tau = quantile_levels
            weights = 1.0 / np.sqrt(tau * (1 - tau) + 1e-10)
            weights = weights / weights.sum()  # Normalize
        else:  # uniform
            # Uniform spacing: simplified to 1/M (constant rescaling doesn't affect optimization)
            weights = np.ones(self.grid_size) / self.grid_size
        
        return t_grid, weights
    
    def compute(self, model: 'CKMEModel') -> float:
        """
        Compute CRPS loss for the given model
        
        Args:
            model: CKMEModel instance
            
        Returns:
            Average CRPS loss across all training points (non-negative float)
        """
        try:
            n = model.n
            Y = model.Y  # Training Y values [n]
            
            # Create quantile grid (following PDF recommendation)
            t_grid, omega = self._create_quantile_grid(Y)  # [M], [M]
            M = len(t_grid)
            
            # Compute weights for all training points
            # W[i, j] = weight of training point j for query point i (x_i)
            W = model.compute_weights(model.X)  # [n, n]
            
            # Sanity check: ensure W has correct shape [n_train, n_query]
            # where n_train = len(Y) and n_query = len(model.X)
            n_train = Y.shape[0]
            n_query = model.X.shape[0]
            assert W.shape == (n_train, n_query), f"W should be [n_train={n_train}, n_query={n_query}], got {W.shape}"
            
            # Ensure weights are normalized (safety check)
            W = W / (W.sum(axis=0, keepdims=True) + 1e-12)
            
            # Compute F̂(t_m | x_i) for all i, m
            # F̂(t_m | x_i) = Σⱼ wⱼ(x_i) · 1{yⱼ ≤ t_m}
            # indicator_matrix[j, m] = 1 if Y[j] <= t_grid[m], else 0
            indicator_matrix = (Y[:, np.newaxis] <= t_grid[np.newaxis, :])  # [n, M]
            
            # F_hat[i, m] = Σⱼ W[j, i] · indicator_matrix[j, m]
            # = (W.T @ indicator_matrix)[i, m]
            F_hat = W.T @ indicator_matrix  # [n, M]
            # F_hat[i, m] = F̂(t_grid[m] | x_i)
            
            # Compute 1{y_i ≤ t_m} for all i, m
            # Y_indicator[i, m] = 1 if Y[i] <= t_grid[m], else 0
            Y_indicator = (Y[:, np.newaxis] <= t_grid[np.newaxis, :])  # [n, M]
            
            # Compute squared differences: [F̂(t_m|x_i) - 1{y_i ≤ t_m}]²
            diff_squared = (F_hat - Y_indicator) ** 2  # [n, M]
            
            # Weighted sum over m (integration): Σ_m w_m [F̂(t_m|x_i) - 1{y_i ≤ t_m}]²
            crps_per_sample = diff_squared @ omega  # [n]
            
            # Average over all training points
            crps = np.mean(crps_per_sample)
            
            return float(max(crps, 0.0))  # Ensure non-negative
            
        except Exception:
            return float('inf')
    
    def compute_for_validation(self, X_train: np.ndarray, Y_train: np.ndarray,
                              X_val: np.ndarray, Y_val: np.ndarray,
                              params) -> float:
        """
        Compute CRPS loss on validation set
        
        This is used during K-Fold CV optimization.
        For validation points, we compute F̂(t|x_val) using training data,
        and evaluate CRPS using validation Y values.
        
        Args:
            X_train: Training inputs [n_train]
            Y_train: Training outputs [n_train]
            X_val: Validation inputs [n_val]
            Y_val: Validation outputs [n_val]
            params: Model parameters (must have ell_x, lam, sigma_y attributes)
            
        Returns:
            Average CRPS loss on validation set
        """
        try:
            from core.models import CKMEModel
            
            # Ensure X_train and X_val are properly shaped
            X_train = np.asarray(X_train)
            if X_train.ndim == 1:
                X_train = X_train.reshape(-1, 1)
            
            X_val = np.asarray(X_val)
            if X_val.ndim == 1:
                X_val = X_val.reshape(-1, 1)
            
            # Create model with training data
            model = CKMEModel(X_train, Y_train, params)
            
            # Use training Y values to create quantile grid
            t_grid, omega = self._create_quantile_grid(Y_train)
            
            # Compute weights for validation points
            W_val = model.compute_weights(X_val)  # [n_train, n_val]
            
            # Sanity check: ensure W_val has correct shape [n_train, n_val]
            n_train = Y_train.shape[0]
            n_val = X_val.shape[0]  # X_val is now [n_val, d_x]
            assert W_val.shape == (n_train, n_val), f"W_val should be [n_train={n_train}, n_val={n_val}], got {W_val.shape}"
            
            W_val = W_val / (W_val.sum(axis=0, keepdims=True) + 1e-12)
            
            # Compute F̂(t_m | x_val_i) for all validation points i and grid points m
            # F̂(t_m | x_val_i) = Σⱼ wⱼ(x_val_i) · 1{y_train_j ≤ t_m}
            indicator_matrix = (Y_train[:, np.newaxis] <= t_grid[np.newaxis, :])  # [n_train, M]
            F_hat = W_val.T @ indicator_matrix  # [n_val, M]
            # F_hat[i, m] = F̂(t_grid[m] | X_val[i])
            
            # Compute 1{y_val_i ≤ t_m} for all validation points i and grid points m
            Y_val_indicator = (Y_val[:, np.newaxis] <= t_grid[np.newaxis, :])  # [n_val, M]
            
            # Compute squared differences: [F̂(t_m|x_val_i) - 1{y_val_i ≤ t_m}]²
            diff_squared = (F_hat - Y_val_indicator) ** 2  # [n_val, M]
            
            # Weighted sum over m (integration): Σ_m w_m [F̂(t_m|x_val_i) - 1{y_val_i ≤ t_m}]²
            crps_per_sample = diff_squared @ omega  # [n_val]
            
            # Average over validation points
            crps = np.mean(crps_per_sample)
            
            return float(max(crps, 0.0))  # Ensure non-negative
            
        except Exception:
            return float('inf')

