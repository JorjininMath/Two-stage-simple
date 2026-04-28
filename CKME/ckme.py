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
from .coefficients import build_cholesky_factor, build_cholesky_from_X, compute_ckme_coeffs
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
        self.r: int = 1  # replications per site (1 = old full-data mode)

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
        r: int = 1,
        verbose: bool = False,
        dtype: Optional[type] = None,
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
        dtype : numpy dtype, optional
            Working precision for the kernel matrix and Cholesky factor.
            Defaults to float64. Pass np.float32 to halve memory for large n.
            Only applies to the fixed-params path; CV tuning uses float64.
        """
        self.r = int(r)
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
        self.params = params

        # Create smooth indicator object
        self.indicator = make_indicator(self.indicator_type, params.h)

        # Build X-kernel with fixed ell_x
        self.kx = make_x_rbf_kernel(params.ell_x)

        # ------------------------------------------------------------------
        # Distinct-sites mode (r > 1): fit on n_0 unique sites instead of
        # n_0 * r_0 total points.  Mathematically equivalent to the full
        # formulation because the block structure of K_X forces all
        # coefficients within a site to be equal; averaging g_t(Y) over
        # replicates per site and solving the n_0 x n_0 system gives the
        # same CDF estimate with O(r_0^2) memory savings.
        # ------------------------------------------------------------------
        if self.r > 1:
            n_sites = X.shape[0] // self.r
            # Distinct sites: take every r-th row (sites are stored consecutively)
            self.X = X[::self.r]                      # (n_0, d)
            self.Y = Y.reshape(n_sites, self.r)        # (n_0, r)  — Y per site
            self.n = n_sites
            self.d = self.X.shape[1]
        else:
            self.X = X
            self.Y = Y
            self.n, self.d = X.shape

        # Build (K_X + n*lam*I) and its Cholesky factor in a single (n, n)
        # buffer — avoids materializing K_X separately. For large n this
        # cuts peak training memory roughly 4x. dtype=float32 halves it again.
        self.L = build_cholesky_from_X(
            self.X, params.ell_x, self.n * params.lam, dtype=dtype
        )
        if dtype is not None:
            self.X = self.X.astype(dtype, copy=False)
        # K_X is no longer cached; predict() only needs L and X.
        self.K_X = None

        return self

    def predict_cdf(
        self,
        X_query: ArrayLike,
        t_grid: Optional[Union[ArrayLike, float]] = None,
        t: Optional[float] = None,
        clip: bool = True,
        monotone: bool = False,
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

        monotone : bool, default=False
            If True, apply isotonic regression to enforce monotonicity of F(t | x)
            along t for each query point. Requires scikit-learn.
            Useful when the raw CKME estimate has minor non-monotonicities due to
            the smoothed indicator approximation or regularization.

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

        # Compute coefficients for query points: shape (n_sites, q)
        C = compute_ckme_coeffs(self.L, self.kx, self.X, X_query)

        if self.r > 1:
            # Distinct-sites mode: G_bar[j, m] = mean_k g_{t_m}(Y[j, k])
            # self.Y has shape (n_sites, r); compute per-site empirical CDF values
            Y_flat = self.Y.ravel()                               # (n_sites * r,)
            G_all  = self.indicator.g_matrix(Y_flat, t_grid)     # (n_sites * r, M)
            G_bar  = G_all.reshape(self.n, self.r, -1).mean(axis=1)  # (n_sites, M)
            F = C.T @ G_bar                                       # (q, M)
            if clip:
                F = np.clip(F, 0.0, 1.0)
        else:
            # OLD full-data mode (r=1, or single observation per site)
            F = compute_cdf_from_coeffs(C, self.Y, self.indicator, t_grid, clip=clip)  # (q, M)

        # Optionally enforce monotonicity via isotonic regression
        if monotone and not is_single_threshold and F.shape[1] > 1:
            from sklearn.isotonic import IsotonicRegression
            _ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
            t_idx = np.arange(F.shape[1], dtype=float)
            for i in range(F.shape[0]):
                F[i] = _ir.fit_transform(t_idx, F[i])

        # Return scalar result if single threshold was provided
        if is_single_threshold:
            return F[:, 0]
        return F

    def predict_quantile(
        self,
        X_query: ArrayLike,
        tau: float,
        t_grid: ArrayLike,
        monotone: bool = True,
    ) -> ArrayLike:
        """
        Predict conditional quantile q_τ(x) = inf{y : F(y | x) ≥ τ} for query inputs.

        This is CDF inversion on t_grid: for each query point, find the first
        grid value where the estimated conditional CDF reaches τ.

        Parameters
        ----------
        X_query : array-like, shape (q, d) or (d,)
            Query input points.

        tau : float
            Quantile level in (0, 1). E.g., 0.1 for 10th percentile.

        t_grid : array-like, shape (M,)
            Dense grid of Y values for inversion. Should cover the range of Y.

        monotone : bool, default=True
            If True (default), apply isotonic regression to enforce CDF monotonicity
            before inverting. Strongly recommended: a non-monotone CDF can cause
            argmax to return the wrong crossing point. Requires scikit-learn.

        Returns
        -------
        q_tau : ndarray, shape (q,)
            Estimated conditional quantile at level tau for each query point.
            If F(t | x) never reaches tau on the grid, returns t_grid[-1].
        """
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        F = self.predict_cdf(X_query, t_grid, monotone=monotone)  # shape (q, M)
        mask = F >= tau                         # shape (q, M)
        has_valid = mask.any(axis=1)            # shape (q,)
        idx = mask.argmax(axis=1)               # first True per row; 0 if no True
        n_clipped = int((~has_valid).sum())
        if n_clipped > 0:
            import warnings
            warnings.warn(
                f"predict_quantile: CDF did not reach tau={tau:.3f} for "
                f"{n_clipped}/{len(has_valid)} query points; clipping to t_grid[-1]={t_grid[-1]:.4f}. "
                "Consider widening t_grid or using percentile-based bounds.",
                stacklevel=2,
            )
        return np.where(has_valid, t_grid[idx], t_grid[-1])

    def predict_quantile_solve(
        self,
        X_query: ArrayLike,
        tau: float,
        t_lo: Optional[float] = None,
        t_hi: Optional[float] = None,
        xtol: float = 1e-6,
    ) -> ArrayLike:
        """
        Predict conditional quantile q_τ(x) by directly solving F(t|x) = τ.

        Unlike predict_quantile, this method does NOT require a t_grid.
        The algorithm branches on indicator type:

        **Smooth indicators (logistic, gaussian_cdf, softplus)**
            F(t|x) = Σᵢ cᵢ(x)·g_t(Yᵢ) is a smooth function of t.
            Uses scipy.optimize.brentq to solve F(t|x) − τ = 0.
            Avoids both discretisation error and truncation bias.
            Assumes approximate monotonicity (holds for well-regularised fits;
            CKME coefficients can be negative so strict monotonicity is not
            guaranteed).

        **Step indicator**
            F(t|x) = Σᵢ cᵢ(x)·1{Yᵢ ≤ t} is a weighted empirical CDF —
            a step function that only jumps at the training Y values.
            The exact quantile is found by sorting the training Y values,
            computing the cumulative weighted coefficient sum, and returning
            the first Y where that sum reaches τ.  This is O(n log n),
            fully vectorised over query points, and requires no root-finding.
            For r > 1: each observation (site j, rep k) gets weight cⱼ(x)/r,
            so the same sort-and-scan logic applies to the flattened arrays.

        Parameters
        ----------
        X_query : array-like, shape (q, d) or (d,)
            Query input points.

        tau : float
            Quantile level in (0, 1).

        t_lo, t_hi : float, optional
            Search bracket for brentq (smooth indicators only).
            If omitted, inferred from training Y:
              t_lo = percentile(Y, 0.1) − 5·h
              t_hi = percentile(Y, 99.9) + 5·h
            Ignored for step indicator.

        xtol : float, default=1e-6
            Absolute tolerance passed to brentq (smooth indicators only).

        Returns
        -------
        q_tau : ndarray, shape (q,)
            Estimated conditional quantile at level tau for each query point.
            Clipped to t_lo / t_hi (smooth) or Y_min / Y_max (step) when
            the cumulative sum never reaches tau inside the range.
        """
        import warnings

        if not self._is_fitted():
            raise ValueError("Model has not been trained. Call fit() first.")
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in (0, 1), got {tau}")

        X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
        q = X_query.shape[0]
        Y_flat = self.Y.ravel()  # (n_sites * r,) or (n,)

        # Compute CKME coefficients once: shape (n_sites, q)
        C = compute_ckme_coeffs(self.L, self.kx, self.X, X_query)

        # ------------------------------------------------------------------
        # Branch: step indicator — exact weighted-ECDF inversion
        # ------------------------------------------------------------------
        if self.indicator_type == "step":
            # Build flat (observation, query) arrays.
            # Each observation i has weight c_j(x) / r for query point j.
            if self.r > 1:
                # C has shape (n_sites, q); expand to (n_sites * r, q)
                C_flat = np.repeat(C, self.r, axis=0) / self.r  # (n*r, q)
            else:
                C_flat = C  # (n, q)

            # Sort observations by Y value
            sort_idx = np.argsort(Y_flat)               # (n_obs,)
            Y_sorted = Y_flat[sort_idx]                  # (n_obs,)
            C_sorted = C_flat[sort_idx, :]               # (n_obs, q)

            # Cumulative weighted sum along the Y axis: shape (n_obs, q)
            cum = np.cumsum(C_sorted, axis=0)

            # For each query point j, find first index where cum[:, j] >= tau
            mask = cum >= tau                             # (n_obs, q)
            has_valid = mask.any(axis=0)                 # (q,)
            idx = mask.argmax(axis=0)                    # (q,) — 0 if no True

            n_hi_clip = int((~has_valid).sum())
            if n_hi_clip > 0:
                warnings.warn(
                    f"predict_quantile_solve (step): cumulative weight never "
                    f"reached tau={tau:.3f} for {n_hi_clip}/{q} query points; "
                    f"clipping to Y_max={Y_sorted[-1]:.4f}.",
                    stacklevel=2,
                )
            return np.where(has_valid, Y_sorted[idx], Y_sorted[-1])

        # ------------------------------------------------------------------
        # Branch: smooth indicators — brentq root finding
        # ------------------------------------------------------------------
        from scipy.optimize import brentq

        h = self.params.h
        if t_lo is None:
            t_lo = float(np.percentile(Y_flat, 0.1)) - 5.0 * h
        if t_hi is None:
            t_hi = float(np.percentile(Y_flat, 99.9)) + 5.0 * h

        def _F_at_t(t: float, c_j: np.ndarray) -> float:
            """Evaluate F(t | x_j) for coefficient vector c_j."""
            if self.r > 1:
                g_all = self.indicator.g_vector(Y_flat, t)         # (n_sites * r,)
                g_bar = g_all.reshape(self.n, self.r).mean(axis=1) # (n_sites,)
                return float(np.dot(c_j, g_bar))
            else:
                g = self.indicator.g_vector(self.Y, t)             # (n,)
                return float(np.dot(c_j, g))

        results = np.empty(q)
        n_lo_clip = 0
        n_hi_clip = 0
        for j in range(q):
            c_j = C[:, j]
            f_lo = _F_at_t(t_lo, c_j) - tau
            f_hi = _F_at_t(t_hi, c_j) - tau

            if f_lo >= 0.0:
                results[j] = t_lo
                n_lo_clip += 1
            elif f_hi < 0.0:
                results[j] = t_hi
                n_hi_clip += 1
            else:
                results[j] = brentq(
                    lambda t: _F_at_t(t, c_j) - tau,
                    t_lo, t_hi, xtol=xtol,
                )

        if n_lo_clip > 0:
            warnings.warn(
                f"predict_quantile_solve: F(t_lo|x) >= tau for {n_lo_clip}/{q} "
                f"points; clipping to t_lo={t_lo:.4f}. Consider lowering t_lo.",
                stacklevel=2,
            )
        if n_hi_clip > 0:
            warnings.warn(
                f"predict_quantile_solve: F(t_hi|x) < tau for {n_hi_clip}/{q} "
                f"points; clipping to t_hi={t_hi:.4f}. Consider raising t_hi.",
                stacklevel=2,
            )
        return results

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
            Y=self.Y,   # shape (n_sites, r) if r>1, else (n,)
            L=self.L,
            ell_x=self.params.ell_x,
            lam=self.params.lam,
            h=self.params.h,
            indicator_type=np.array([self.indicator_type], dtype=object),
            r=np.array(self.r),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CKMEModel":
        """Load model from disk."""
        data = np.load(path, allow_pickle=True)
        indicator_type = str(data["indicator_type"].item())
        model = cls(indicator_type=indicator_type)
        model.X = data["X"]
        model.Y = data["Y"]   # (n_sites, r) if r>1, else (n,)
        model.L = data["L"]
        model.r = int(data["r"]) if "r" in data else 1  # backward compat
        model.n, model.d = model.X.shape
        model.params = Params(
            ell_x=float(data["ell_x"]),
            lam=float(data["lam"]),
            h=float(data["h"]),
        )
        model.indicator = make_indicator(indicator_type, model.params.h)
        model.kx = make_x_rbf_kernel(model.params.ell_x)
        # K_X is not cached: predict() only needs L and X.
        model.K_X = None
        return model
