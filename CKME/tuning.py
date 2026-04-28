"""
tuning.py

Hyperparameter tuning for CKME using k-fold cross-validation.

This module provides:
- k-fold cross-validation for parameter selection
- Parameter grid search
- Evaluation metrics integration
- Results storage and best parameter selection

The implementation uses sklearn.model_selection.KFold for data splitting
and supports parallelization via joblib for faster computation on multi-core
systems (including M1 Mac).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import KFold

try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

from .cdf import compute_ckme_cdf
from .loss_functions import make_loss, CRPSLoss
from .parameters import ParamGrid, Params

ArrayLike = np.ndarray


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class CVResult:
    """
    Cross-validation result for a single parameter combination.

    Attributes
    ----------
    params : Params
        Parameter combination that was evaluated.

    mean_loss : float
        Mean loss across all CV folds (validation loss).

    std_loss : float
        Standard deviation of loss across CV folds.

    fold_losses : List[float]
        Loss value for each CV fold (validation loss).
    """
    params: Params
    mean_loss: float
    std_loss: float
    fold_losses: List[float] = field(default_factory=list)


@dataclass
class TuningResults:
    """
    Complete hyperparameter tuning results.

    Attributes
    ----------
    best_params : Params
        Best parameter combination (lowest mean loss).

    best_loss : float
        Best mean loss value.

    cv_results : List[CVResult]
        CV results for all parameter combinations tested.

    param_grid : ParamGrid
        Parameter grid that was searched.

    cv_folds : int
        Number of CV folds used.

    n_params_tested : int
        Total number of parameter combinations tested.
    """
    best_params: Params
    best_loss: float
    cv_results: List[CVResult] = field(default_factory=list)
    param_grid: Optional[ParamGrid] = None
    cv_folds: int = 5
    n_params_tested: int = 0


# ---------------------------------------------------------------------------
# Internal: evaluate single parameter combination on one fold
# ---------------------------------------------------------------------------

def _evaluate_single_fold(
    X_train_fold: ArrayLike,
    Y_train_fold: ArrayLike,
    X_val_fold: ArrayLike,
    Y_val_fold: ArrayLike,
    params: Params,
    t_grid: ArrayLike,
    loss_fn: CRPSLoss,
) -> float:
    """
    Evaluate a single parameter combination on a single CV fold.

    Parameters
    ----------
    X_train_fold : ndarray, shape (n_train, d)
    Y_train_fold : ndarray, shape (n_train,)
    X_val_fold : ndarray, shape (n_val, d)
    Y_val_fold : ndarray, shape (n_val,)
    params : Params
    t_grid : ndarray, shape (M,)
    loss_fn : CRPSLoss

    Returns
    -------
    val_loss : float
        Validation loss value for this fold.
    """
    F_pred_val = compute_ckme_cdf(
        X_train_fold, Y_train_fold, params,
        X_val_fold, t_grid,
        clip=True
    )
    val_loss = loss_fn.compute(F_pred_val, Y_val_fold, t_grid)
    return float(val_loss)


# ---------------------------------------------------------------------------
# Internal: evaluate single parameter combination on all folds
# ---------------------------------------------------------------------------

def _evaluate_params_cv(
    X_train: ArrayLike,
    Y_train: ArrayLike,
    params: Params,
    t_grid: ArrayLike,
    loss_fn: CRPSLoss,
    cv_folds: int,
    random_state: Optional[int],
    n_jobs: int = 1,
) -> CVResult:
    """
    Evaluate a single parameter combination using k-fold CV.

    Parameters
    ----------
    X_train : ndarray, shape (n, d)
        Full training inputs.

    Y_train : ndarray, shape (n,)
        Full training outputs.

    params : Params
        Parameter combination to evaluate.

    t_grid : ndarray, shape (M,)
        Threshold grid for CDF evaluation.

    loss_fn : CRPSLoss
        Loss function object.

    cv_folds : int
        Number of CV folds.

    random_state : int, optional
        Random seed for data shuffling.

    n_jobs : int, default=1
        Number of parallel jobs. If 1, runs sequentially.
        If > 1, uses joblib for parallelization.

    Returns
    -------
    cv_result : CVResult
        CV result for this parameter combination.
    """
    X_train = np.asarray(X_train, dtype=float)
    Y_train = np.asarray(Y_train, dtype=float).ravel()
    n = X_train.shape[0]

    # Create KFold splitter
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Prepare fold evaluations
    fold_tasks = []
    for train_idx, val_idx in kf.split(X_train):
        X_train_fold = X_train[train_idx]
        Y_train_fold = Y_train[train_idx]
        X_val_fold = X_train[val_idx]
        Y_val_fold = Y_train[val_idx]

        fold_tasks.append(
            (X_train_fold, Y_train_fold, X_val_fold, Y_val_fold)
        )

    # Evaluate all folds (parallel or sequential)
    if n_jobs > 1 and HAS_JOBLIB:
        # Parallel evaluation
        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_single_fold)(
                X_train_fold, Y_train_fold, X_val_fold, Y_val_fold,
                params, t_grid, loss_fn
            )
            for X_train_fold, Y_train_fold, X_val_fold, Y_val_fold in fold_tasks
        )
    else:
        # Sequential evaluation
        fold_results = [
            _evaluate_single_fold(
                X_train_fold, Y_train_fold, X_val_fold, Y_val_fold,
                params, t_grid, loss_fn
            )
            for X_train_fold, Y_train_fold, X_val_fold, Y_val_fold in fold_tasks
        ]

    fold_losses = np.array(fold_results, dtype=float)
    mean_loss = float(np.mean(fold_losses))
    std_loss = float(np.std(fold_losses))

    return CVResult(
        params=params,
        mean_loss=mean_loss,
        std_loss=std_loss,
        fold_losses=fold_losses.tolist(),
    )


# ---------------------------------------------------------------------------
# Main: k-fold cross-validation
# ---------------------------------------------------------------------------

def cross_validate_ckme(
    X_train: ArrayLike,
    Y_train: ArrayLike,
    param_grid: ParamGrid,
    t_grid: ArrayLike,
    loss_fn: Optional[CRPSLoss] = None,
    loss_type: str = "crps",
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> TuningResults:
    """
    Perform k-fold cross-validation for CKME parameter tuning.

    This function evaluates all parameter combinations in the grid using
    k-fold cross-validation and returns the best parameters along with
    detailed results for all combinations.

    Parameters
    ----------
    X_train : ndarray, shape (n, d)
        Training input points.

    Y_train : ndarray, shape (n,)
        Training output values.

    param_grid : ParamGrid
        Hyperparameter search grid.

    t_grid : ndarray, shape (M,)
        Threshold grid for CDF evaluation. This should cover the range
        of Y values of interest.

    loss_fn : CRPSLoss, optional
        Loss function object. If None, creates one using loss_type.

    loss_type : str, default="crps"
        Type of loss function to use if loss_fn is None.

    cv_folds : int, default=5
        Number of cross-validation folds.

    random_state : int, optional
        Random seed for data shuffling in KFold. If None, uses random seed.

    n_jobs : int, default=1
        Number of parallel jobs for parameter evaluation.
        - 1: Sequential evaluation (no parallelization)
        - >1: Parallel evaluation using joblib (uses multiple CPU cores)
        - -1: Use all available CPU cores
        Note: Parallelization works on M1 Mac and other multi-core systems.

    verbose : bool, default=False
        If True, print progress information.

    Returns
    -------
    results : TuningResults
        Tuning results containing:
        - best_params: Best parameter combination
        - best_loss: Best mean loss value
        - cv_results: Detailed CV results for all parameter combinations
        - Other metadata

    Examples
    --------
    >>> from CKME.parameters import ParamGrid
    >>> import numpy as np
    >>>
    >>> param_grid = ParamGrid(
    ...     ell_x_list=[0.3, 0.5, 0.7],
    ...     lam_list=[1e-4, 1e-3],
    ...     h_list=[0.1, 0.2]
    ... )
    >>> t_grid = np.linspace(Y_train.min(), Y_train.max(), 100)
    >>>
    >>> results = cross_validate_ckme(
    ...     X_train, Y_train, param_grid, t_grid,
    ...     cv_folds=5,
    ...     n_jobs=4  # Use 4 CPU cores
    ... )
    >>> print(f"Best params: {results.best_params}")
    >>> print(f"Best loss: {results.best_loss}")
    """
    X_train = np.atleast_2d(np.asarray(X_train, dtype=float))
    Y_train = np.asarray(Y_train, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()

    if X_train.shape[0] != Y_train.shape[0]:
        raise ValueError(
            f"X_train and Y_train must have same number of samples, "
            f"got {X_train.shape[0]} and {Y_train.shape[0]}"
        )

    if param_grid.is_empty():
        raise ValueError("param_grid is empty. Please provide at least one value for each parameter.")

    # Create loss function if not provided
    if loss_fn is None:
        loss_fn = make_loss(loss_type)

    # Handle n_jobs=-1 (use all cores)
    if n_jobs == -1:
        if HAS_JOBLIB:
            from joblib import cpu_count
            n_jobs = cpu_count()
        else:
            n_jobs = 1
            if verbose:
                print("Warning: joblib not available, using sequential evaluation")

    # Collect all parameter combinations
    param_list = list(param_grid.iter_grid())
    n_params = len(param_list)

    if verbose:
        print(f"Testing {n_params} parameter combinations with {cv_folds}-fold CV...")
        if n_jobs > 1:
            print(f"Using {n_jobs} parallel jobs")

    # Evaluate all parameter combinations
    if n_jobs > 1 and HAS_JOBLIB:
        # Parallel evaluation across parameter combinations
        cv_results = Parallel(n_jobs=n_jobs)(
            delayed(_evaluate_params_cv)(
                X_train, Y_train, params, t_grid, loss_fn,
                cv_folds, random_state, n_jobs=1  # n_jobs=1 for inner folds
            )
            for params in param_list
        )
    else:
        # Sequential evaluation
        cv_results = []
        for i, params in enumerate(param_list):
            if verbose:
                print(f"  Evaluating parameter combination {i+1}/{n_params}...")
            result = _evaluate_params_cv(
                X_train, Y_train, params, t_grid, loss_fn,
                cv_folds, random_state, n_jobs=1
            )
            cv_results.append(result)

    # Find best parameters
    best_idx = np.argmin([r.mean_loss for r in cv_results])
    best_result = cv_results[best_idx]

    if verbose:
        print(f"\nBest parameters found:")
        print(f"  ell_x={best_result.params.ell_x}, "
              f"lam={best_result.params.lam}, "
              f"h={best_result.params.h}")
        print(f"  Mean loss: {best_result.mean_loss:.6f} ± {best_result.std_loss:.6f}")

    return TuningResults(
        best_params=best_result.params,
        best_loss=best_result.mean_loss,
        cv_results=cv_results,
        param_grid=param_grid,
        cv_folds=cv_folds,
        n_params_tested=n_params,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def tune_ckme_params(
    X_train: ArrayLike,
    Y_train: ArrayLike,
    param_grid: ParamGrid,
    t_grid: ArrayLike,
    loss_type: str = "crps",
    cv_folds: int = 5,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> Tuple[Params, TuningResults]:
    """
    Convenience wrapper for parameter tuning.

    This function calls cross_validate_ckme() and returns the best parameters
    and full results as a tuple.

    Parameters
    ----------
    X_train : ndarray, shape (n, d)
        Training input points.

    Y_train : ndarray, shape (n,)
        Training output values.

    param_grid : ParamGrid
        Hyperparameter search grid.

    t_grid : ndarray, shape (M,)
        Threshold grid for CDF evaluation.

    loss_type : str, default="crps"
        Type of loss function to use.

    cv_folds : int, default=5
        Number of cross-validation folds.

    random_state : int, optional
        Random seed for data shuffling.

    n_jobs : int, default=1
        Number of parallel jobs. Use -1 for all available cores.

    verbose : bool, default=False
        If True, print progress information.

    Returns
    -------
    best_params : Params
        Best parameter combination.

    results : TuningResults
        Complete tuning results.

    Examples
    --------
    >>> best_params, results = tune_ckme_params(
    ...     X_train, Y_train, param_grid, t_grid,
    ...     cv_folds=5,
    ...     n_jobs=-1  # Use all CPU cores
    ... )
    """
    results = cross_validate_ckme(
        X_train, Y_train, param_grid, t_grid,
        loss_type=loss_type,
        cv_folds=cv_folds,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    return results.best_params, results

