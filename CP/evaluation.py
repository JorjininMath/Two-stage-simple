"""
evaluation.py

Evaluation metrics for Conformal Prediction performance.

This module provides functions to evaluate CP performance, including:
- Conditional coverage: coverage rate for each unique x (averaged across replications)
- Conditional width: prediction interval width for each unique x
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import numpy as np

ArrayLike = np.ndarray


def find_unique_x(
    X: ArrayLike,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Find unique x values and return indices for each unique x.
    
    Parameters
    ----------
    X : ndarray, shape (n, d)
        Input points (may contain duplicates).
    tolerance : float, default=1e-6
        Tolerance for considering two points as the same.
        
    Returns
    -------
    unique_X : ndarray, shape (m, d)
        Unique x values.
    indices_map : list of ndarray
        For each unique x, the indices in the original X array.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    
    if n == 0:
        return np.empty((0, d)), []
    
    # Use a simple approach: round to tolerance precision and use np.unique
    # For more robust matching, could use scipy.spatial.distance
    X_rounded = np.round(X / tolerance) * tolerance
    
    # Find unique rows
    unique_X, inverse_indices = np.unique(X_rounded, axis=0, return_inverse=True)
    
    # Build indices map
    n_unique = len(unique_X)
    indices_map = []
    for i in range(n_unique):
        mask = inverse_indices == i
        indices_map.append(np.where(mask)[0])
    
    return unique_X, indices_map


def evaluate_cp(
    X_test: ArrayLike,
    Y_test: ArrayLike,
    L: ArrayLike,
    U: ArrayLike,
    alpha: float,
    tolerance: float = 1e-6,
    coverage_method: str = "score",
    model=None,
    q_hat: Optional[float] = None,
    t_grid: Optional[ArrayLike] = None,
    score_type: str = "abs_median",
) -> Dict:
    """
    Evaluate conditional CP performance given prediction intervals.
    
    This is a pure evaluation function that only computes statistics.
    It does NOT call any model or compute intervals - intervals must be
    pre-computed and provided as inputs.
    
    For each unique x, computes:
    - Conditional coverage: proportion of Y values at that x that fall within the interval
    - Conditional width: prediction interval width at that x
    
    Then averages across all unique x values.
    
    Parameters
    ----------
    X_test : ndarray, shape (n, d)
        Test input points (required). May contain duplicate x values.
    Y_test : ndarray, shape (n,)
        Corresponding true output values (required).
        Must have the same length as X_test.
    L : ndarray, shape (n,)
        Lower bounds of prediction intervals for each test point (required).
        L[i] is the lower bound for X_test[i].
    U : ndarray, shape (n,)
        Upper bounds of prediction intervals for each test point (required).
        U[i] is the upper bound for X_test[i].
    alpha : float
        Significance level (target coverage is 1 - alpha).
    tolerance : float, default=1e-6
        Tolerance for considering two x values as the same.
    coverage_method : str, default="score"
        Use "score" (R DCP-DR: cov = 1{score <= q_hat}) or "interval" (cov = 1{Y in [L,U]}).
        When "score", requires model, q_hat, t_grid.
    model : optional
        CKME model for score-based coverage. Required when coverage_method="score".
    q_hat : float, optional
        Conformal threshold. Required when coverage_method="score".
    t_grid : array-like, optional
        Y grid for CDF evaluation. Required when coverage_method="score".
    score_type : str, default="abs_median"
        Score type for score-based coverage.
        
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'conditional_coverage': dict
            - 'mean': mean conditional coverage across unique x
            - 'per_x': list of conditional coverage for each unique x
            - 'target': target coverage (1 - alpha)
        - 'conditional_width': dict
            - 'mean': mean conditional width across unique x
            - 'per_x': list of conditional width for each unique x
            - 'std': standard deviation of conditional widths
        - 'n_unique_x': number of unique x values
        - 'n_total_samples': total number of test samples
        
    Examples
    --------
    >>> # First, compute intervals using CP
    >>> L, U = cp.predict_interval(X_test, t_grid)
    >>> 
    >>> # Then, evaluate performance
    >>> results = evaluate_cp(X_test, Y_test, L, U, alpha=0.1)
    >>> print(f"Mean coverage: {results['conditional_coverage']['mean']:.4f}")
    """
    X_test = np.asarray(X_test, dtype=float)
    Y_test = np.asarray(Y_test, dtype=float).ravel()
    L = np.asarray(L, dtype=float).ravel()
    U = np.asarray(U, dtype=float).ravel()
    
    n = X_test.shape[0]
    if Y_test.shape[0] != n:
        raise ValueError(
            f"X_test and Y_test must have the same number of samples, "
            f"got {n} and {Y_test.shape[0]}"
        )
    if L.shape[0] != n:
        raise ValueError(
            f"L must have the same length as X_test, got {L.shape[0]} and {n}"
        )
    if U.shape[0] != n:
        raise ValueError(
            f"U must have the same length as X_test, got {U.shape[0]} and {n}"
        )
    
    # Step 1: Find unique x values
    unique_X, indices_map = find_unique_x(X_test, tolerance)
    n_unique = len(unique_X)
    
    if n_unique == 0:
        raise ValueError("No unique x values found in X_test")
    
    # Step 2: Evaluate for each unique x
    conditional_coverages = []
    conditional_widths = []
    
    use_score_based = (
        coverage_method == "score"
        and model is not None
        and q_hat is not None
        and t_grid is not None
    )
    if use_score_based:
        from .scores import score_from_cdf
        from CKME.coefficients import compute_ckme_coeffs
        # Vectorized: F(Y[i]|X[i]) for all i in one batch (like R; avoids n_test separate CDF calls)
        # C (n_train, n_test), G (n_train, n_test) with G[:,j]=g_{Y[j]}(Y_train), F[j]=C[:,j]^T @ G[:,j]
        C = compute_ckme_coeffs(model.L, model.kx, model.X, X_test)
        G = model.indicator.g_matrix(model.Y, Y_test)
        F_test = np.sum(C * G, axis=0).astype(float)
        np.clip(F_test, 0.0, 1.0, out=F_test)
        scores_test = score_from_cdf(F_test, score_type=score_type)
        covered = scores_test <= q_hat


    for i, (x_unique, indices) in enumerate(zip(unique_X, indices_map)):
        # Get all Y values and intervals at this x
        Y_at_x = Y_test[indices]
        L_at_x = L[indices]
        U_at_x = U[indices]
        
        L_val = float(L_at_x[0])
        U_val = float(U_at_x[0])
        
        width = U_val - L_val
        conditional_widths.append(width)
        
        if use_score_based:
            coverage_at_x = float(np.mean(covered[indices]))
        else:
            coverage_at_x = float(np.mean((Y_at_x >= L_val) & (Y_at_x <= U_val)))
        conditional_coverages.append(coverage_at_x)
    
    # Step 3: Compute statistics
    mean_conditional_coverage = float(np.mean(conditional_coverages))
    mean_conditional_width = float(np.mean(conditional_widths))
    std_conditional_width = float(np.std(conditional_widths))
    
    return {
        'conditional_coverage': {
            'mean': mean_conditional_coverage,
            'per_x': conditional_coverages,
            'target': 1.0 - alpha,
        },
        'conditional_width': {
            'mean': mean_conditional_width,
            'per_x': conditional_widths,
            'std': std_conditional_width,
        },
        'n_unique_x': n_unique,
        'n_total_samples': n,
        'unique_x': unique_X,
    }


def compute_interval_score(Y_test: ArrayLike, L: ArrayLike, U: ArrayLike, alpha: float) -> Tuple[np.ndarray, float]:
    """
    Compute interval score for prediction intervals.
    
    Interval Score = (U - L) + (2/alpha) * [L - Y]_{+} + (2/alpha) * [Y - U]_{+}
    where [x]_{+} = max(0, x)
    
    Parameters
    ----------
    Y_test : ndarray, shape (n,)
        True output values.
    L : ndarray, shape (n,)
        Lower bounds of prediction intervals.
    U : ndarray, shape (n,)
        Upper bounds of prediction intervals.
    alpha : float
        Significance level.
        
    Returns
    -------
    scores : ndarray, shape (n,)
        Interval scores for each test point.
    mean_score : float
        Mean interval score.
    """
    Y_test = np.asarray(Y_test).ravel()
    L = np.asarray(L).ravel()
    U = np.asarray(U).ravel()
    
    width = U - L
    penalty_lower = (2.0 / alpha) * np.maximum(0, L - Y_test)
    penalty_upper = (2.0 / alpha) * np.maximum(0, Y_test - U)
    
    scores = width + penalty_lower + penalty_upper
    mean_score = float(np.mean(scores))
    
    return scores, mean_score

