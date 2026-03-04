#!/usr/bin/env python3
"""
DCP (Distributional Conformal Prediction) Predictor
"""

import numpy as np
from typing import Tuple
from core.models import CKMEModel


class DCPPredictor:
    """Distributional Conformal Prediction predictor"""
    
    def __init__(self, model: CKMEModel):
        self.model = model
        self.X = model.X
        self.Y = model.Y
    
    def compute_ecdf(self, x_star: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        """Compute weighted empirical CDF: F_x(y) = sum_i w_i(x) * 1{Y_i <= y}"""
        weights = self.model.compute_weights(x_star).ravel()
        
        sorted_indices = np.argsort(self.Y)
        sorted_Y = self.Y[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        y_values = np.asarray(y_values)
        mask = y_values[:, np.newaxis] >= sorted_Y[np.newaxis, :]
        ecdf_values = np.sum(sorted_weights[np.newaxis, :] * mask, axis=1)
        
        return ecdf_values
    
    def compute_calibration_scores(self, X_cal: np.ndarray, Y_cal: np.ndarray) -> np.ndarray:
        """Compute DCP calibration scores: c_i = |F_x(Y_i) - 0.5|"""
        batch_weights = self.model.compute_weights(X_cal)  # [n_train, n_cal]
        
        sorted_indices = np.argsort(self.Y)
        sorted_Y = self.Y[sorted_indices]
        sorted_weights = batch_weights[sorted_indices, :]  # [n_train, n_cal]
        
        mask = Y_cal[:, np.newaxis] >= sorted_Y[np.newaxis, :]  # [n_cal, n_train]
        ecdf_values = np.sum(sorted_weights.T * mask, axis=1)  # [n_cal]
        
        return np.abs(ecdf_values - 0.5)
    
    def predict_interval(self, x_star: np.ndarray, calibration_scores: np.ndarray, alpha: float) -> Tuple[float, float]:
        """Predict conformal prediction interval using DCP method"""
        t = np.quantile(calibration_scores, 1 - alpha)
        
        y_min, y_max = np.min(self.Y), np.max(self.Y)
        y_grid = np.linspace(y_min, y_max, 1000)
        
        ecdf_values = self.compute_ecdf(x_star, y_grid)
        valid_mask = (ecdf_values >= 0.5 - t) & (ecdf_values <= 0.5 + t)
        
        if np.any(valid_mask):
            valid_y = y_grid[valid_mask]
            return float(np.min(valid_y)), float(np.max(valid_y))
        else:
            weights = self.model.compute_weights(x_star).ravel()
            sorted_indices = np.argsort(self.Y)
            sorted_Y = self.Y[sorted_indices]
            sorted_weights = weights[sorted_indices]
            
            cumsum_weights = np.cumsum(sorted_weights)
            cumsum_weights = cumsum_weights / cumsum_weights[-1]
            
            lower_idx = np.searchsorted(cumsum_weights, alpha/2)
            upper_idx = np.searchsorted(cumsum_weights, 1-alpha/2)
            
            lower_idx = max(0, min(lower_idx, len(sorted_Y)-1))
            upper_idx = max(0, min(upper_idx, len(sorted_Y)-1))
            
            return float(sorted_Y[lower_idx]), float(sorted_Y[upper_idx])

