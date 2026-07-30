"""KME covariates built from finite queue input samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dgp import Scenario
from .rff import RFFTransformer1D


@dataclass
class Standardizer:
    """Fit-only standardizer for KME covariates."""

    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "Standardizer":
        self.mean_ = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("standardizer is not fitted")
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_


def rho_hat(inter_arrivals: np.ndarray, services: np.ndarray) -> float:
    """Estimate traffic intensity from finite input samples only."""

    return float(np.mean(services) / np.mean(inter_arrivals))


def input_uncertainty(scenarios: Sequence[Scenario]) -> np.ndarray:
    """Return n_A^-1/2 + n_S^-1/2 for adaptive smoothing."""

    return np.array([s.n_a**-0.5 + s.n_s**-0.5 for s in scenarios], dtype=float)


@dataclass
class KMEFeatureBuilder:
    """Build scenario covariates from empirical KMEs of input samples.

    KME estimates the input distributions. The downstream CKME-CDF estimator
    then learns the output CDF conditional on these distribution-valued
    covariates.
    """

    rff_dim: int = 100
    rff_a: RFFTransformer1D | None = None
    rff_s: RFFTransformer1D | None = None

    def fit(self, scenarios: Sequence[Scenario], rng: np.random.Generator) -> "KMEFeatureBuilder":
        log_a = np.concatenate([np.log(s.inter_arrivals) for s in scenarios])
        log_s = np.concatenate([np.log(s.services) for s in scenarios])
        self.rff_a = RFFTransformer1D(self.rff_dim).fit(log_a, rng)
        self.rff_s = RFFTransformer1D(self.rff_dim).fit(log_s, rng)
        return self

    def transform_samples(self, inter_arrivals: np.ndarray, services: np.ndarray) -> np.ndarray:
        if self.rff_a is None or self.rff_s is None:
            raise ValueError("KMEFeatureBuilder is not fitted")
        a = np.asarray(inter_arrivals, dtype=float)
        s = np.asarray(services, dtype=float)
        return np.concatenate(
            [
                self.rff_a.mean_embedding(np.log(a)),
                self.rff_s.mean_embedding(np.log(s)),
                np.array([a.size**-0.5, s.size**-0.5, rho_hat(a, s)], dtype=float),
            ]
        )

    def transform(self, scenarios: Sequence[Scenario]) -> np.ndarray:
        return np.vstack([self.transform_samples(s.inter_arrivals, s.services) for s in scenarios])


def bootstrap_query_features(
    scenario: Scenario,
    builder: KMEFeatureBuilder,
    standardizer: Standardizer,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap finite input samples and return standardized query features."""

    rows = []
    uncertainty = []
    for _ in range(n_bootstrap):
        a_star = rng.choice(scenario.inter_arrivals, size=scenario.n_a, replace=True)
        s_star = rng.choice(scenario.services, size=scenario.n_s, replace=True)
        rows.append(builder.transform_samples(a_star, s_star))
        uncertainty.append(a_star.size**-0.5 + s_star.size**-0.5)
    return standardizer.transform(np.vstack(rows)), np.asarray(uncertainty, dtype=float)
