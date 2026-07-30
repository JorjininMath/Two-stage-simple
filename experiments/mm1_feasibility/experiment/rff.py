"""Random Fourier features for empirical input-distribution KMEs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def median_heuristic_sigma(
    x: np.ndarray,
    rng: np.random.Generator,
    max_points: int = 10_000,
    n_pairs: int = 20_000,
    fallback: float = 1.0,
) -> float:
    """Approximate the one-dimensional RBF median-distance bandwidth."""

    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        return fallback
    if x.size > max_points:
        x = rng.choice(x, size=max_points, replace=False)
    n_pairs = min(n_pairs, max(1, x.size * 20))
    idx_a = rng.integers(0, x.size, size=n_pairs)
    idx_b = rng.integers(0, x.size, size=n_pairs)
    distances = np.abs(x[idx_a] - x[idx_b])
    distances = distances[distances > 1e-12]
    if distances.size == 0:
        return fallback
    sigma = float(np.median(distances))
    return sigma if np.isfinite(sigma) and sigma > 1e-12 else fallback


@dataclass
class RFFTransformer1D:
    """RFF approximation of an RBF feature map on log-positive samples."""

    n_features: int = 100
    sigma: float | None = None
    omega: np.ndarray | None = None
    phase: np.ndarray | None = None

    def fit(self, observations: np.ndarray, rng: np.random.Generator) -> "RFFTransformer1D":
        if self.n_features <= 0:
            raise ValueError("n_features must be positive")
        if self.sigma is None:
            self.sigma = median_heuristic_sigma(observations, rng)
        self.omega = rng.normal(0.0, 1.0 / self.sigma, size=self.n_features)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, size=self.n_features)
        return self

    def transform(self, observations: np.ndarray) -> np.ndarray:
        if self.omega is None or self.phase is None:
            raise ValueError("fit must be called before transform")
        x = np.asarray(observations, dtype=float).reshape(-1, 1)
        return np.sqrt(2.0 / self.n_features) * np.cos(x * self.omega[None, :] + self.phase[None, :])

    def mean_embedding(self, observations: np.ndarray) -> np.ndarray:
        """Estimate the KME by averaging RFF features over finite samples."""

        return np.mean(self.transform(observations), axis=0)
