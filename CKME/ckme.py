"""
ckme.py

Core CKME model for conditional CDF estimation via conditional kernel mean embedding.

This module implements the CKMEModel class, which provides:
- Model training with fixed parameters or parameter tuning via cross-validation
- CDF prediction for query inputs

Key idea
--------
Given training data (X_i, Y_i) and a query x, the CKME embedding is
represented implicitly as

    μ̂_{Y|X=x} = Σ_i c_i(x) φ_Y(Y_i),

where the coefficients c(x) are obtained by solving

    (K_X + n λ I) c(x) = k_X(X, x).

For a smooth indicator g_t(y) ≈ 1{y ≤ t}, the conditional CDF is estimated as

    F(t | x) = E[g_t(Y) | X = x] ≈ Σ_i g_t(Y_i) c_i(x).

We deliberately avoid defining a Y-kernel: all Y-side structure is
handled through the test functions (smooth indicators).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from .parameters import Params, ParamGrid
from .kernels import make_x_rbf_kernel
from .indicators import make_indicator, BaseIndicator
from .coefficients import build_cholesky_factor, compute_ckme_coeffs
from .cdf import compute_cdf_from_coeffs
from .tuning import tune_ckme_params, TuningResults

ArrayLike = np.ndarray


class CKMEModel:
    """
    Conditional Kernel Mean Embedding model for conditional CDF estimation.

    This model provides a unified interface for:
    - Training with fixed hyperparameters or parameter tuning via cross-validation
    - Predicting conditional CDFs F(t | x) for query inputs

    Usage
    -----
    # Train with fixed parameters
    model = CKMEModel(indicator_type="logistic")
    model.fit(X_train, Y_train, params=params)
    F = model.predict_cdf(X_test, t_grid)

    # Train with parameter tuning
    model = CKMEModel(indicator_type="logistic")
    model.fit(X_train, Y_train, param_grid=param_grid, t_grid=t_grid, cv_folds=5)
    F = model.predict_cdf(X_test, t_grid)

    Note
    ----
    The coefficient vector c(x) is NOT a probability vector. It can contain
    negative entries and does not sum to one, since CKME solves a regularized
    linear system rather than estimating conditional density weights. We do
    not normalize or clip c(x); if needed, we only clip the final CDF values
    F(t | x) to [0, 1].
    """

    def __init__(self, indicator_type: str = "logistic") -> None:
        """
        Initialize an empty CKME model.

        Parameters
        ----------
        indicator_type : str, default="logistic"
            Type of indicator function. Options: "logistic", "gaussian_cdf", "softplus", "step".

        Note
        ----
        The model is not trained at initialization. Call fit() to train the model.
        """
        if indicator_type not in ["logistic", "gaussian_cdf", "softplus", "step"]:
            raise ValueError(
                f"indicator_type must be one of ['logistic', 'gaussian_cdf', 'softplus', 'step'], "
                f"got {indicator_type}"
            )
        self.indicator_type = indicator_type

        # These will be set during fit()
        self.X: Optional[ArrayLike] = None
        self.Y: Optional[ArrayLike] = None
        self.params: Optional[Params] = None
        self.indicator: Optional[BaseIndicator] = None
        self.kx = None
        self.K_X: Optional[ArrayLike] = None
        self.L: Optional[ArrayLike] = None
        self.n: Optional[int] = None
        self.d: Optional[int] = None

        # Store tuning results if parameter tuning was performed
        self.tuning_results: Optional[TuningResults] = None

    def _is_fitted(self) -> bool:
        """Check if the model has been trained."""
        return self.X is not None and self.L is not None

    def fit(
        self,
        X: ArrayLike,
        Y: ArrayLike,
        param_grid: Optional[ParamGrid] = None,
        params: Optional[Params] = None,
        t_grid: Optional[ArrayLike] = None,
        loss_type: str = "crps",
        cv_folds: int = 5,
        random_state: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ) -> "CKMEModel":
        """
        Train the CKME model.

        Either params (fixed) or param_grid (CV tuning) must be provided.

        Parameters
        ----------
        X : array-like, shape (n, d)
            Training inputs.
        Y : array-like, shape (n,) or (n, 1)
            Training outputs.
        params : Params, optional
            Fixed hyperparameters. If provided, no tuning; train directly.
        param_grid : ParamGrid, optional
            Hyperparameter grid for k-fold CV tuning. Required if params not provided.
        t_grid : array-like, optional
            Threshold grid for CV. Required only when using param_grid.
        loss_type, cv_folds, random_state, n_jobs, verbose
            Used only when param_grid is provided.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Y = np.asarray(Y, dtype=float).ravel()
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples, "
                f"got {X.shape[0]} and {Y.shape[0]}"
            )

        if params is not None:
            # Fixed params: no tuning
            if verbose:
                print(f"Training with fixed params: ell_x={params.ell_x:.4f}, "
                      f"lam={params.lam:.4f}, h={params.h:.4f}")
        elif param_grid is not None and not param_grid.is_empty():
            if t_grid is None:
                raise ValueError("t_grid required for parameter tuning")
            t_grid = np.asarray(t_grid, dtype=float).ravel()
            if verbose:
                print("Starting parameter tuning with k-fold cross-validation...")
            best_params, tuning_results = tune_ckme_params(
                X_train=X,
                Y_train=Y,
                param_grid=param_grid,
                t_grid=t_grid,
                loss_type=loss_type,
                cv_folds=cv_folds,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=verbose,
            )
            self.tuning_results = tuning_results
            params = best_params
            if verbose:
                print(f"Best parameters: ell_x={params.ell_x:.4f}, "
                      f"lam={params.lam:.4f}, h={params.h:.4f}")
                print(f"Best CV loss: {tuning_results.best_loss:.6f}")
        else:
            raise ValueError("Provide either params (fixed) or param_grid (tuning)")

        # Store training data and parameters
        self.X = X
        self.Y = Y
        self.n, self.d = X.shape
        self.params = params

        # Create smooth indicator object
        self.indicator = make_indicator(self.indicator_type, params.h)

        # Build X-kernel with fixed ell_x
        self.kx = make_x_rbf_kernel(params.ell_x)

        # Precompute training Gram matrix K_X
        self.K_X = self.kx(self.X, self.X)  # shape (n, n)

        # Build Cholesky factor of (K_X + n * lam * I)
        self.L = build_cholesky_factor(self.K_X, self.n, params.lam)

        return self

    def predict_cdf(
        self,
        X_query: ArrayLike,
        t_grid: Optional[Union[ArrayLike, float]] = None,
        t: Optional[float] = None,
        clip: bool = True,
    ) -> ArrayLike:
        """
        Predict conditional CDF F(t | x) for query inputs.

        Parameters
        ----------
        X_query : array-like, shape (q, d) or (d,)
            Query input points. A single point can be passed as a 1D array.

        t_grid : array-like, shape (M,) or float, optional
            Threshold grid at which F(t_m | x) is evaluated, or a single threshold value.
            If `t` is provided, `t_grid` is ignored.

        t : float, optional
            Single threshold value. If provided, `t_grid` is ignored.

        clip : bool, default=True
            If True, clip the resulting CDF values to [0, 1]. This is sometimes
            useful numerically because the RKHS representation can produce
            values slightly outside [0, 1].

        Returns
        -------
        F : ndarray, shape (q, M) or (q,)
            Matrix of CDF values with entries F[j, m] = F(t_m | x_j).
            If a single threshold is provided (via `t` or scalar `t_grid`),
            returns shape (q,).

        Raises
        ------
        ValueError
            If the model has not been trained (fit() has not been called), or
            if neither `t_grid` nor `t` is provided.
        """
        if not self._is_fitted():
            raise ValueError("Model has not been trained. Call fit() first.")

        # Determine threshold(s) to use
        if t is not None:
            t_grid = t
        elif t_grid is None:
            raise ValueError("Either t_grid or t must be provided")

        X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
        is_single_threshold = np.isscalar(t_grid)
        t_grid = np.asarray([t_grid] if is_single_threshold else t_grid, dtype=float).ravel()

        # Compute coefficients for query points
        C = compute_ckme_coeffs(self.L, self.kx, self.X, X_query)  # shape (n, q)

        # Compute CDF using indicator method
        F = compute_cdf_from_coeffs(C, self.Y, self.indicator, t_grid, clip=clip)  # shape (q, M)

        # Return scalar result if single threshold was provided
        if is_single_threshold:
            return F[:, 0]
        return F

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk. Uses numpy's np.savez for arrays and a simple
        format that reconstructs kx and indicator on load.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=self.X,
            Y=self.Y,
            L=self.L,
            ell_x=self.params.ell_x,
            lam=self.params.lam,
            h=self.params.h,
            indicator_type=np.array([self.indicator_type], dtype=object),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CKMEModel":
        """Load model from disk."""
        data = np.load(path, allow_pickle=True)
        indicator_type = str(data["indicator_type"].item())
        model = cls(indicator_type=indicator_type)
        model.X = data["X"]
        model.Y = data["Y"]
        model.L = data["L"]
        model.n, model.d = model.X.shape
        model.params = Params(
            ell_x=float(data["ell_x"]),
            lam=float(data["lam"]),
            h=float(data["h"]),
        )
        model.indicator = make_indicator(indicator_type, model.params.h)
        model.kx = make_x_rbf_kernel(model.params.ell_x)
        model.K_X = model.kx(model.X, model.X)
        return model
