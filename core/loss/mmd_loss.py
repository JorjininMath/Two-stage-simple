#!/usr/bin/env python3
"""
MMD (Maximum Mean Discrepancy) Loss Function
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.models.ckme_weights import CKMEModel


class MMDLoss:
    """
    MMD Loss: ||μ̂_Y|X=xi - k_Y(·, yi)||²_HY
    
    Computes the squared distance between the empirical conditional kernel mean embedding
    and the true kernel embedding for each training point, averaged over all points.
    
    Formula:
        MMD² = (1/n) * Σᵢ [k_Y(yi,yi) - 2*w(xi)^T*k_Y(Y,yi) + w(xi)^T*K_YY*w(xi)]
    """
    
    def compute(self, model: 'CKMEModel') -> float:
        """
        Compute MMD loss for the given model
        
        Args:
            model: CKMEModel instance
            
        Returns:
            MMD loss value (non-negative float)
        """
        try:
            from utils.kernels import rbf_kernel
            
            # Compute weight matrix W = A^(-1) K_X for all training points
            K_x = rbf_kernel(model.X, model.X, model.ell_x)
            
            A = K_x + model.lam * model.n * np.eye(model.n)
            A_inv = np.linalg.inv(A + 1e-10 * np.eye(model.n))
            W = A_inv @ K_x  # (n, n) where column i is w(xi)
            
            # Y-space kernel matrix
            K_YY = rbf_kernel(model.Y.reshape(-1, 1), model.Y.reshape(-1, 1), model.sigma_y)
            
            # Compute MMD loss terms
            # term1: k_Y(yi, yi) diagonal terms (for RBF kernel, k_Y(y,y)=1)
            term1 = np.diag(K_YY).mean()
            
            # term2: -2 * w(xi)^T * k_Y(Y, yi) averaged over all i
            k_Y_Y_yi = rbf_kernel(model.Y.reshape(-1, 1), model.Y.reshape(-1, 1), model.sigma_y)  # (n, n)
            term2 = -2.0 * np.mean([W[:, i].T @ k_Y_Y_yi[:, i] for i in range(model.n)])
            
            # term3: w(xi)^T * K_YY * w(xi) averaged over all i
            term3 = np.mean([W[:, i].T @ K_YY @ W[:, i] for i in range(model.n)])
            
            mmd_loss = term1 + term2 + term3
            return float(max(mmd_loss, 0.0))  # Ensure non-negative
            
        except Exception:
            return float('inf')
    
    def compute_for_validation(self, X_train: np.ndarray, Y_train: np.ndarray,
                              X_val: np.ndarray, Y_val: np.ndarray,
                              params) -> float:
        """
        Compute MMD loss on validation set
        
        This is used during K-Fold CV optimization.
        For validation points, we compute weights using training data,
        and evaluate MMD loss using validation Y values.
        
        Args:
            X_train: Training inputs [n_train]
            Y_train: Training outputs [n_train]
            X_val: Validation inputs [n_val]
            Y_val: Validation outputs [n_val]
            params: Model parameters (must have ell_x, lam, sigma_y attributes)
            
        Returns:
            Average MMD loss on validation set
        """
        try:
            from utils.kernels import rbf_kernel
            from core.models import CKMEModel
            
            # Create model with training data
            model = CKMEModel(X_train, Y_train, params)
            
            # Compute weights for validation points
            weights_val = model.compute_weights(X_val)  # (n_train, n_val)
            
            # Numerical stability: ensure non-negative and normalized
            weights_val = np.maximum(weights_val, 0.0)
            weights_val = weights_val / (np.sum(weights_val, axis=0, keepdims=True) + 1e-12)
            
            # Compute Y-space kernels
            K_Y_val = rbf_kernel(Y_train.reshape(-1, 1), Y_val.reshape(-1, 1), params.sigma_y)
            G_Y = rbf_kernel(Y_train.reshape(-1, 1), Y_train.reshape(-1, 1), params.sigma_y)
            
            # MMD² for each validation point
            mmd_losses = []
            for j in range(len(X_val)):
                w_j = weights_val[:, j]
                term1 = 1.0  # k_Y(y_j, y_j) = 1 for RBF
                term2 = -2.0 * np.sum(w_j * K_Y_val[:, j])
                term3 = np.sum(w_j[:, np.newaxis] * G_Y * w_j[np.newaxis, :])
                mmd_losses.append(max(term1 + term2 + term3, 0.0))
            
            return float(np.mean(mmd_losses))
            
        except Exception:
            return float('inf')

