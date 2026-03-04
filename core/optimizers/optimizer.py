#!/usr/bin/env python3
"""
Parameter Optimizer using K-Fold Cross-Validation with MMD or CRPS Loss
"""

import numpy as np
from typing import List, Tuple, Literal
from core.models import CKMEModel
from core.loss import MMDLoss, CRPSLoss
from utils.kernels import silverman_bandwidth


class ParameterOptimizer:
    """
    Parameter optimizer using K-Fold Cross-Validation with MMD or CRPS loss
    
    Performs grid search over parameter combinations:
    - For MMD: (ell_x, lam, sigma_y) - 3 parameters
    - For CRPS: (ell_x, lam) - 2 parameters (no sigma_y needed)
    
    Uses GroupKFold to split by unique x values (sites) to prevent information leakage.
    """
    
    def __init__(self, config, k: int = 5, loss_type: Literal['mmd', 'crps'] = 'mmd'):
        """
        Initialize parameter optimizer
        
        Args:
            config: Configuration object (must have optimize_params attribute)
            k: Number of folds for cross-validation (default: 5)
            loss_type: 'mmd' or 'crps' (default: 'mmd')
        """
        self.config = config
        self.k = k
        self.loss_type = loss_type
        
        # Initialize loss function
        if loss_type == 'crps':
            self.loss_fn = CRPSLoss(grid_size=100, weight_type='quantile')
        else:  # mmd
            self.loss_fn = MMDLoss()
        
        # Parameter grids
        # ell_x: 3 values around 0.1 (10^-1)
        self.ell_x_candidates = [10**-1.5, 10**-1, 10**-0.5]  # [0.0316, 0.1, 0.316]
        # lam: [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        self.lam_candidates = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
        # sigma_y: Fixed to Silverman bandwidth (not optimized)
        # For MMD, we still use multipliers but only 1.0 (fixed)
        self.sigma_y_multipliers = [1.0]
    
    def _split_k_folds(self, X: np.ndarray, Y: np.ndarray, seed: int = None) -> List[Tuple]:
        """
        Split data into k folds using GroupKFold (by unique x values/sites)
        
        Args:
            X: Training inputs [n, d_x]
            Y: Training outputs [n]
            seed: Random seed for reproducibility
            
        Returns:
            List of (X_train_fold, Y_train_fold, X_val_fold, Y_val_fold) tuples
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_obs = len(X)
        
        # Find unique sites (handle both 1D and 2D input)
        if X.ndim == 1 or X.shape[1] == 1:
            # 1D case
            X_flat = X.ravel() if X.ndim > 1 else X
            unique_sites = np.unique(X_flat)
            n_sites = len(unique_sites)
            
            # Create site_id for each observation
            site_id = np.zeros(n_obs, dtype=int)
            for idx, site in enumerate(unique_sites):
                site_id[np.abs(X_flat - site) < 1e-10] = idx
        else:
            # 2D case: find unique (x1, x2) pairs
            unique_sites = []
            seen = []
            site_id = np.zeros(n_obs, dtype=int)
            site_idx = 0
            
            for i, x_point in enumerate(X):
                is_new = True
                for j, seen_point in enumerate(seen):
                    if np.allclose(x_point, seen_point, atol=1e-10):
                        site_id[i] = j
                        is_new = False
                        break
                if is_new:
                    unique_sites.append(x_point)
                    seen.append(x_point)
                    site_id[i] = site_idx
                    site_idx += 1
            
            unique_sites = np.array(unique_sites)
            n_sites = len(unique_sites)
        
        # Shuffle and split sites (not observations)
        site_indices = np.arange(n_sites)
        np.random.shuffle(site_indices)
        site_folds = np.array_split(site_indices, self.k)
        
        folds = []
        for i in range(self.k):
            val_sites = set(site_folds[i])
            val_mask = np.array([site_id[j] in val_sites for j in range(n_obs)])
            train_mask = ~val_mask
            
            folds.append((X[train_mask], Y[train_mask], X[val_mask], Y[val_mask]))
        
        return folds
    
    def _compute_loss_on_validation(self, X_train: np.ndarray, Y_train: np.ndarray,
                                    X_val: np.ndarray, Y_val: np.ndarray,
                                    params) -> float:
        """
        Compute loss on validation set (MMD or CRPS)
        
        Args:
            X_train: Training inputs for this fold
            Y_train: Training outputs for this fold
            X_val: Validation inputs for this fold
            Y_val: Validation outputs for this fold
            params: Parameter object (must have ell_x, lam, and sigma_y if MMD)
            
        Returns:
            Average loss on validation set
        """
        try:
            if self.loss_type == 'mmd':
                # Use MMDLoss module for validation set computation
                return self.loss_fn.compute_for_validation(
                    X_train, Y_train, X_val, Y_val, params
                )
            
            else:  # crps
                # Use CRPSLoss module for validation set computation
                return self.loss_fn.compute_for_validation(
                    X_train, Y_train, X_val, Y_val, params
                )
                
        except Exception:
            return float('inf')
    
    def optimize(self, X: np.ndarray, Y: np.ndarray, params_class, seed: int = None):
        """
        Optimize parameters using K-Fold Cross-Validation with MMD or CRPS loss
        
        Args:
            X: Training inputs [n, d_x]
            Y: Training outputs [n]
            params_class: Class for creating parameter objects
                - For MMD: must have ell_x, lam, sigma_y attributes
                - For CRPS: must have ell_x, lam attributes (sigma_y not needed)
            seed: Random seed for fold splitting (default: None)
            
        Returns:
            Optimized parameters object
        """
        if not self.config.optimize_params:
            # Use default parameters
            if self.loss_type == 'mmd':
                return params_class(ell_x=0.01, lam=1e-5, sigma_y=silverman_bandwidth(Y))
            else:  # crps
                return params_class(ell_x=0.01, lam=1e-5, sigma_y=silverman_bandwidth(Y))  # sigma_y still needed for model, but not optimized
        
        # Split data into k folds
        folds = self._split_k_folds(X, Y, seed)
        sigma_y_base = silverman_bandwidth(Y)
        best_params = None
        best_avg_loss = float('inf')
        
        if self.loss_type == 'mmd':
            # Grid search over parameter combinations: ell_x × lam × sigma_y_multiplier
            for ell_x in self.ell_x_candidates:
                for lam in self.lam_candidates:
                    for sigma_y_mult in self.sigma_y_multipliers:
                        fold_losses = []
                        for X_train_fold, Y_train_fold, X_val_fold, Y_val_fold in folds:
                            # Compute sigma_y for this fold from training data
                            sigma_y_fold = sigma_y_mult * silverman_bandwidth(Y_train_fold)
                            params = params_class(ell_x=ell_x, lam=lam, sigma_y=sigma_y_fold)
                            fold_loss = self._compute_loss_on_validation(
                                X_train_fold, Y_train_fold, X_val_fold, Y_val_fold, params
                            )
                            fold_losses.append(fold_loss)
                        
                        avg_loss = np.mean(fold_losses)
                        if avg_loss < best_avg_loss:
                            best_avg_loss = avg_loss
                            # Use full data sigma_y for final params
                            best_params = params_class(ell_x=ell_x, lam=lam, sigma_y=sigma_y_mult * sigma_y_base)
        else:  # crps
            # Grid search over parameter combinations: ell_x × lam (no sigma_y)
            # For CRPS, sigma_y is not optimized, but still needed for model initialization
            # We use a fixed sigma_y (e.g., Silverman bandwidth) for model creation
            for ell_x in self.ell_x_candidates:
                for lam in self.lam_candidates:
                    fold_losses = []
                    for X_train_fold, Y_train_fold, X_val_fold, Y_val_fold in folds:
                        # Use fixed sigma_y (not optimized for CRPS)
                        sigma_y_fold = silverman_bandwidth(Y_train_fold)
                        params = params_class(ell_x=ell_x, lam=lam, sigma_y=sigma_y_fold)
                        fold_loss = self._compute_loss_on_validation(
                            X_train_fold, Y_train_fold, X_val_fold, Y_val_fold, params
                        )
                        fold_losses.append(fold_loss)
                    
                    avg_loss = np.mean(fold_losses)
                    if avg_loss < best_avg_loss:
                        best_avg_loss = avg_loss
                        # Use full data sigma_y for final params (fixed, not optimized)
                        best_params = params_class(ell_x=ell_x, lam=lam, sigma_y=sigma_y_base)
        
        if best_params is None:
            # Fallback to default parameters
            if self.loss_type == 'mmd':
                best_params = params_class(ell_x=0.01, lam=1e-5, sigma_y=sigma_y_base)
            else:  # crps
                best_params = params_class(ell_x=0.01, lam=1e-5, sigma_y=sigma_y_base)
        
        return best_params

