#!/usr/bin/env python3
"""
CKME Weights Computation
Computes conditional kernel mean embedding weights for query points
"""

import numpy as np
from typing import Union


class CKMEModel:
    """
    Conditional Kernel Mean Embedding Model
    Computes weights for conditional distribution estimation
    """
    
    def __init__(self, X: np.ndarray, Y: np.ndarray, params):
        """
        Initialize CKME model
        
        Args:
            X: Training inputs [n, d_x]
            Y: Training outputs [n]
            params: Model parameters (must have ell_x, lam, sigma_y attributes)
        """
        # Ensure X is 2D with shape [n, d_x]
        # If X is 1D, reshape to [n, 1] instead of [1, n]
        X = np.asarray(X)
        if X.ndim == 1:
            self.X = X.reshape(-1, 1)  # [n, 1]
        else:
            self.X = X  # Already 2D [n, d_x]
        self.Y = np.asarray(Y).ravel()
        self.n = len(Y)
        self.params = params
        
        # Set kernel parameters
        self.ell_x = params.ell_x
        self.lam = params.lam
        self.sigma_y = params.sigma_y
        
        # Compute kernel matrices
        self._compute_kernel_matrices()
    
    def _compute_kernel_matrices(self):
        """Compute kernel matrices for CKME"""
        from utils.kernels import rbf_kernel
        
        # X-space kernel (RBF) - only this is needed for weight computation
        K_x = rbf_kernel(self.X, self.X, self.ell_x)
        
        # Regularized kernel matrix for weight computation
        # A = K_x + lam * n * I (standard KRR formulation)
        A = K_x + self.lam * self.n * np.eye(self.n)
        
        # Cholesky decomposition for stable solves
        self.L = np.linalg.cholesky(A + 1e-10 * np.eye(self.n))
        
        # Y-space kernel (only needed for MMD loss, not for weight computation)
        # Store it separately if needed for MMD loss
        self.K_y = rbf_kernel(self.Y.reshape(-1, 1), self.Y.reshape(-1, 1), self.sigma_y)
    
    def compute_weights(self, X_star: np.ndarray) -> np.ndarray:
        """
        Compute simplex weights for query points (vectorized)
        
        This is the unified interface that handles both single and batch queries.
        For a single point, pass X_star with shape (1, d_x) or (d_x,).
        
        Args:
            X_star: Query points [m, d_x] or [d_x] for single point
            
        Returns:
            Weights matrix [n, m] where column j contains weights for X_star[j]
            For single point, returns [n, 1] which can be raveled to [n]
        """
        from utils.kernels import rbf_kernel
        
        X_star = np.atleast_2d(X_star)
        m = X_star.shape[0]
        
        # Compute kernel matrix for all query points (RBF)
        K_x = rbf_kernel(self.X, X_star, self.ell_x)
        
        # Solve (L L^T) A = K_x for all query points
        Y = np.linalg.solve(self.L, K_x)
        A = np.linalg.solve(self.L.T, Y)
        
        # Apply non-negativity constraint
        W = np.maximum(A, 0.0)
        
        # Handle fallback cases
        S = W.sum(axis=0)
        fallback_mask = S <= 1e-12
        
        if np.any(fallback_mask):
            W[:, fallback_mask] = K_x[:, fallback_mask]
            S[fallback_mask] = W[:, fallback_mask].sum(axis=0)
        
        # Normalize
        W = W / (S[np.newaxis, :] + 1e-12)
        
        return W
    
    # Backward compatibility aliases
    def simplex_weights(self, x_star: np.ndarray) -> np.ndarray:
        """
        Compute weights for a single query point (backward compatibility)
        
        Args:
            x_star: Single query point [d_x] or [1, d_x]
            
        Returns:
            Weights vector [n]
        """
        x_star = np.atleast_2d(x_star)
        weights = self.compute_weights(x_star)  # Returns [n, 1]
        return weights.ravel()  # Return [n]
    
    def simplex_weights_batch(self, X_star: np.ndarray) -> np.ndarray:
        """
        Compute weights for multiple query points (backward compatibility)
        
        Args:
            X_star: Query points [m, d_x]
            
        Returns:
            Weights matrix [n, m]
        """
        return self.compute_weights(X_star)

