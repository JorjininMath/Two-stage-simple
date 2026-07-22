"""A small CKME-style conditional CDF estimator over KME covariates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pairwise_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """NumPy-only Euclidean distances."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    squared = np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2.0 * x @ y.T
    np.maximum(squared, 0.0, out=squared)
    return np.sqrt(squared)


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    squared = np.sum(x * x, axis=1)[:, None] + np.sum(y * y, axis=1)[None, :] - 2.0 * x @ y.T
    np.maximum(squared, 0.0, out=squared)
    return squared


def condensed_pairwise_distances(x: np.ndarray) -> np.ndarray:
    distances = pairwise_distances(x, x)
    return distances[np.triu_indices(distances.shape[0], k=1)]


def rbf_kernel(x: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """RBF kernel matrix for standardized KME covariates."""

    if tau <= 0 or not np.isfinite(tau):
        raise ValueError("tau must be a positive finite value")
    return np.exp(-squared_distances(x, y) / (2.0 * tau**2))


def candidate_taus_from_distances(
    z_fit: np.ndarray,
    percentiles: tuple[int, ...] = (10, 20, 40, 60, 80),
) -> list[float]:
    """Build positive bandwidth candidates from fit-feature distances."""

    distances = condensed_pairwise_distances(z_fit)
    positive = distances[np.isfinite(distances) & (distances > 1e-12)]
    if positive.size == 0:
        return [1.0]
    raw = np.percentile(positive, percentiles)
    floor = max(float(np.percentile(positive, 1)) * 0.1, 1e-6)
    values = [max(float(v), floor) for v in raw if np.isfinite(v) and v > 0]
    unique = sorted({round(v, 12): v for v in values}.values())
    return unique if unique else [floor]


def make_cdf_grid(y_fit: np.ndarray, grid_size: int) -> np.ndarray:
    """Use empirical fit-output quantiles as CDF thresholds."""

    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    y = np.asarray(y_fit, dtype=float).ravel()
    if y.size == 0 or not np.all(np.isfinite(y)):
        raise ValueError("fit outputs must be nonempty and finite")
    levels = np.arange(1, grid_size + 1, dtype=float) / (grid_size + 1.0)
    return np.asarray(np.quantile(y, levels), dtype=float)


def cdf_response_matrix(y: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Return binary or averaged CDF responses on ``t_grid``."""

    y = np.asarray(y, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    if y.ndim == 1:
        return (y[:, None] <= t_grid[None, :]).astype(float)
    if y.ndim == 2:
        return np.mean(y[:, :, None] <= t_grid[None, None, :], axis=1)
    raise ValueError("y must be a one- or two-dimensional array")


def rearrange_cdf(raw: np.ndarray) -> np.ndarray:
    """Clip and monotonize predicted CDF curves."""

    return np.clip(np.maximum.accumulate(np.asarray(raw, dtype=float), axis=1), 0.0, 1.0)


def monotonicity_violation(raw: np.ndarray) -> np.ndarray:
    """Total downward movement in raw CDF curves before rearrangement."""

    raw = np.asarray(raw, dtype=float)
    if raw.shape[1] < 2:
        return np.zeros(raw.shape[0], dtype=float)
    return np.sum(np.maximum(0.0, raw[:, :-1] - raw[:, 1:]), axis=1)


def logistic_cdf(x: np.ndarray) -> np.ndarray:
    """Stable logistic smooth indicator."""

    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class KRRConditionalCDFEstimator:
    """Kernel-ridge estimator for ``F(t | Z)`` on a fixed threshold grid."""

    tau: float
    ridge: float
    t_grid: np.ndarray | None = None
    z_fit: np.ndarray | None = None
    alpha_: np.ndarray | None = None

    def fit(self, z_fit: np.ndarray, y_fit: np.ndarray) -> "KRRConditionalCDFEstimator":
        self.z_fit = np.asarray(z_fit, dtype=float)
        if self.z_fit.ndim != 2:
            raise ValueError("z_fit must be a two-dimensional array")
        if self.ridge <= 0 or not np.isfinite(self.ridge):
            raise ValueError("ridge must be a positive finite value")
        if self.t_grid is None:
            self.t_grid = make_cdf_grid(y_fit, grid_size=100)
        else:
            self.t_grid = np.asarray(self.t_grid, dtype=float).ravel()
        responses = cdf_response_matrix(y_fit, self.t_grid)
        if responses.shape[0] != self.z_fit.shape[0]:
            raise ValueError("z_fit and y_fit sizes differ")

        gram = rbf_kernel(self.z_fit, self.z_fit, self.tau)
        system = gram + self.ridge * np.eye(gram.shape[0])
        try:
            self.alpha_ = np.linalg.solve(system, responses)
        except np.linalg.LinAlgError:
            self.alpha_ = np.linalg.lstsq(system, responses, rcond=None)[0]
        return self

    def predict_raw(self, z_query: np.ndarray) -> np.ndarray:
        if self.z_fit is None or self.alpha_ is None:
            raise ValueError("fit must be called before prediction")
        z_query = np.asarray(z_query, dtype=float)
        return rbf_kernel(z_query, self.z_fit, self.tau) @ self.alpha_

    def predict_cdf(self, z_query: np.ndarray) -> np.ndarray:
        return rearrange_cdf(self.predict_raw(z_query))

    def validation_mse(self, z_val: np.ndarray, y_val: np.ndarray) -> float:
        if self.t_grid is None:
            raise ValueError("fit must be called before validation")
        targets = cdf_response_matrix(y_val, self.t_grid)
        pred = self.predict_cdf(z_val)
        return float(np.mean((pred - targets) ** 2))


def tune_krr_cdf(
    z_fit: np.ndarray,
    y_fit: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    t_grid: np.ndarray,
    tau_candidates: list[float] | None = None,
    ridge_candidates: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1e-1, 1.0),
) -> tuple[KRRConditionalCDFEstimator, dict[str, float | str]]:
    """Tune KRR-CDF hyperparameters by validation CDF MSE."""

    z_fit = np.asarray(z_fit, dtype=float)
    z_val = np.asarray(z_val, dtype=float)
    t_grid = np.asarray(t_grid, dtype=float)
    responses = cdf_response_matrix(y_fit, t_grid)
    targets = cdf_response_matrix(y_val, t_grid)
    tau_candidates = tau_candidates or candidate_taus_from_distances(z_fit)

    best: dict[str, float | str] | None = None
    best_alpha: np.ndarray | None = None
    for tau in tau_candidates:
        gram = rbf_kernel(z_fit, z_fit, float(tau))
        gram = 0.5 * (gram + gram.T)
        try:
            eigvals, eigvecs = np.linalg.eigh(gram)
            eigvals = np.maximum(eigvals, 0.0)
            projected = eigvecs.T @ responses
            k_val = rbf_kernel(z_val, z_fit, float(tau))
            for ridge in ridge_candidates:
                denom = eigvals[:, None] + float(ridge)
                alpha = eigvecs @ (projected / denom)
                pred = rearrange_cdf(k_val @ alpha)
                mse = float(np.mean((pred - targets) ** 2))
                if best is None or mse < float(best["val_mse"]):
                    best = {"tau": float(tau), "ridge": float(ridge), "val_mse": mse, "solver": "eigh"}
                    best_alpha = alpha
        except np.linalg.LinAlgError:
            for ridge in ridge_candidates:
                estimator = KRRConditionalCDFEstimator(float(tau), float(ridge), t_grid=t_grid).fit(z_fit, y_fit)
                mse = estimator.validation_mse(z_val, y_val)
                if best is None or mse < float(best["val_mse"]):
                    best = {"tau": float(tau), "ridge": float(ridge), "val_mse": mse, "solver": "solve"}
                    best_alpha = estimator.alpha_

    if best is None or best_alpha is None:
        raise RuntimeError("failed to tune KRR-CDF estimator")
    estimator = KRRConditionalCDFEstimator(float(best["tau"]), float(best["ridge"]), t_grid=t_grid)
    estimator.z_fit = z_fit
    estimator.alpha_ = best_alpha
    return estimator, best


@dataclass
class CKMECDFEstimator:
    """Estimate ``F(t | Z)`` by kernel-weighted smooth indicators.

    KME supplies the covariate ``Z``. This estimator is the CKME-style layer:
    it estimates the output conditional CDF using a neighborhood kernel in KME
    covariate space and a smooth indicator in output space.
    """

    adaptive: bool = True
    fixed_h_z: float | None = None
    k_neighbors: int = 50
    eta: float = 1.0
    beta: float = 1.0
    h_y: float | None = None
    z_fit: np.ndarray | None = None
    y_fit: np.ndarray | None = None
    u_fit: np.ndarray | None = None

    def fit(self, z_fit: np.ndarray, y_fit: np.ndarray, u_fit: np.ndarray | None = None) -> "CKMECDFEstimator":
        self.z_fit = np.asarray(z_fit, dtype=float)
        self.y_fit = np.asarray(y_fit, dtype=float).ravel()
        if self.z_fit.shape[0] != self.y_fit.size:
            raise ValueError("z_fit and y_fit sizes differ")
        if u_fit is None:
            u_fit = np.ones(self.y_fit.size, dtype=float)
        self.u_fit = np.asarray(u_fit, dtype=float).ravel()
        if self.u_fit.size != self.y_fit.size:
            raise ValueError("u_fit and y_fit sizes differ")

        distances = condensed_pairwise_distances(self.z_fit)
        positive = distances[distances > 1e-10]
        if self.fixed_h_z is None:
            self.fixed_h_z = float(np.median(positive)) if positive.size else 1.0
        if self.h_y is None:
            y_scale = float(np.std(self.y_fit))
            self.h_y = max(0.05 * y_scale, 1e-3)
        return self

    def _bandwidths(self, z_query: np.ndarray, u_query: np.ndarray | None) -> np.ndarray:
        if self.z_fit is None or self.u_fit is None:
            raise ValueError("fit must be called before prediction")
        z_query = np.asarray(z_query, dtype=float)
        if not self.adaptive:
            return np.full(z_query.shape[0], float(self.fixed_h_z), dtype=float)
        distances = pairwise_distances(z_query, self.z_fit)
        k_idx = min(max(int(self.k_neighbors), 1), self.z_fit.shape[0]) - 1
        d_k = np.partition(distances, k_idx, axis=1)[:, k_idx]
        positive = distances[distances > 1e-10]
        floor = max(float(np.percentile(positive, 5)) * 0.1, 1e-5) if positive.size else 1e-4
        if u_query is None:
            u_query = np.full(z_query.shape[0], float(np.median(self.u_fit)), dtype=float)
        u_scaled = np.asarray(u_query, dtype=float).ravel() / max(float(np.median(self.u_fit)), 1e-8)
        return np.maximum(self.eta * np.maximum(d_k, floor) * (1.0 + self.beta * u_scaled), floor)

    def predict_cdf(self, z_query: np.ndarray, t_grid: np.ndarray, u_query: np.ndarray | None = None) -> np.ndarray:
        """Return CDF curves with shape ``(n_query, len(t_grid))``."""

        if self.z_fit is None or self.y_fit is None or self.h_y is None:
            raise ValueError("fit must be called before prediction")
        z_query = np.asarray(z_query, dtype=float)
        t_grid = np.asarray(t_grid, dtype=float).ravel()
        distances = pairwise_distances(z_query, self.z_fit)
        h_z = self._bandwidths(z_query, u_query)
        logw = -(distances**2) / (2.0 * h_z[:, None] ** 2)
        logw -= np.max(logw, axis=1, keepdims=True)
        weights = np.exp(logw)
        weights /= np.sum(weights, axis=1, keepdims=True)
        smooth_indicators = logistic_cdf((t_grid[None, :] - self.y_fit[:, None]) / self.h_y)
        curves = weights @ smooth_indicators
        return np.maximum.accumulate(np.clip(curves, 0.0, 1.0), axis=1)

    def bootstrap_training_curves(
        self,
        z_query: np.ndarray,
        t_grid: np.ndarray,
        rng: np.random.Generator,
        n_bootstrap: int,
        u_query: np.ndarray | None = None,
    ) -> np.ndarray:
        """Bootstrap training scenarios while keeping the query representation fixed."""

        if self.z_fit is None or self.y_fit is None or self.u_fit is None:
            raise ValueError("fit must be called before bootstrap")
        curves = []
        n = self.y_fit.size
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot = CKMECDFEstimator(
                adaptive=self.adaptive,
                fixed_h_z=self.fixed_h_z,
                k_neighbors=min(self.k_neighbors, n),
                eta=self.eta,
                beta=self.beta,
                h_y=self.h_y,
            ).fit(self.z_fit[idx], self.y_fit[idx], self.u_fit[idx])
            curves.append(boot.predict_cdf(z_query, t_grid, u_query=u_query)[0])
        return np.vstack(curves)


def pointwise_band(curves: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lower, median, and upper pointwise envelope."""

    return (
        np.quantile(curves, alpha / 2.0, axis=0),
        np.quantile(curves, 0.5, axis=0),
        np.quantile(curves, 1.0 - alpha / 2.0, axis=0),
    )
