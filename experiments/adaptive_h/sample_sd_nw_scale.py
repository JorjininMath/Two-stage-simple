"""Sample-SD plus Nadaraya-Watson scale estimator for adaptive bandwidth.

This is the active response-scale estimator for
``h(x) = c * sigma_hat(x)``.  Its name intentionally states the estimator so
that it cannot be confused with the retired IQR-based response-scale plug-in.

Design:
  1. From Stage 1 data D_0 = (X_all, Y_all) with n_0 sites x r_0 replicates,
     compute per-site sample std: sigma_site[i] = std(Y at site i, ddof=1).
  2. Smooth (X_0, sigma_site) with Nadaraya-Watson regression using a Gaussian
     kernel and Silverman's rule of thumb for the bandwidth. The bandwidth can
     be multiplied by bw_factor for diagnostic tuning.
  3. Return a callable that maps any X_query -> sigma_hat(X_query), floored at
     a small constant to avoid h(x) = 0 in low-noise regions (e.g., gibbs_s1).

Usage:
    est = SampleSdNwScaleEstimator.fit(
        stage1.X_all,
        stage1.Y_all,
        n_0=stage1.n_0,
        r_0=stage1.r_0,
    )
    h_query = c_scale * est.predict(X_query)

Notes:
  - This estimator uses replications at the same x. Without replications, use
    residual-based local scale estimation instead.
  - For Gaussian DGPs, sigma_hat(x) -> sigma(Y|x) as n_0 and r_0 grow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

_SIGMA_FLOOR = 1e-3  # Mirror adaptive_bandwidth._H_FLOOR.


def _silverman_bw(x_1d: np.ndarray) -> float:
    """Return an input-space smoothing bandwidth via Silverman's rule.

    The IQR below is used only to robustify the bandwidth of the *X-space*
    Nadaraya-Watson smoother.  It is not an IQR response-scale estimator:
    ``sigma_sites`` is always computed from per-site sample standard
    deviations.  This distinction records the project's estimator decision.
    """
    n = len(x_1d)
    if n < 2:
        return 1.0
    sd = float(np.std(x_1d, ddof=1))
    iqr = float(np.subtract(*np.percentile(x_1d, [75, 25])))
    spread = min(sd, iqr / 1.34) if iqr > 0 else sd
    if spread == 0.0:
        spread = 1.0
    return 1.06 * spread * n ** (-1 / 5)


@dataclass
class SampleSdNwScaleEstimator:
    """Nadaraya-Watson-smoothed per-site std estimate sigma_hat(x).

    Stored fields:
      X_sites : (n_0, d) site coordinates from Stage 1
      sigma_sites : (n_0,) per-site sample std estimate
      bw : (d,) NW bandwidth per dimension (Silverman's rule on X_sites)
    """
    X_sites: np.ndarray
    sigma_sites: np.ndarray
    bw: np.ndarray

    @classmethod
    def fit(
        cls,
        X_all: np.ndarray,
        Y_all: np.ndarray,
        n_0: int,
        r_0: int,
        bw: Optional[np.ndarray] = None,
        bw_factor: float = 1.0,
    ) -> "SampleSdNwScaleEstimator":
        """Build the estimator from Stage 1 raw data.

        X_all is (n_0 * r_0, d), with site i occupying rows [i*r_0, (i+1)*r_0).
        Y_all is (n_0 * r_0,).
        """
        if bw_factor <= 0:
            raise ValueError(f"bw_factor must be positive; got {bw_factor}")
        X_all = np.atleast_2d(X_all)
        Y_all = np.asarray(Y_all).ravel()
        if X_all.shape[0] != n_0 * r_0:
            raise ValueError(
                f"X_all has {X_all.shape[0]} rows but n_0 * r_0 = {n_0 * r_0}"
            )
        if Y_all.shape[0] != n_0 * r_0:
            raise ValueError(
                f"Y_all has {Y_all.shape[0]} entries but n_0 * r_0 = {n_0 * r_0}"
            )
        if r_0 < 2:
            raise ValueError(
                "SampleSdNwScaleEstimator needs r_0 >= 2 to estimate "
                f"per-site std; got r_0={r_0}"
            )

        d = X_all.shape[1]
        X_sites = X_all.reshape(n_0, r_0, d)[:, 0, :]  # one row per site
        Y_by_site = Y_all.reshape(n_0, r_0)
        sigma_sites = np.std(Y_by_site, axis=1, ddof=1)

        if bw is None:
            bw = bw_factor * np.array([_silverman_bw(X_sites[:, j]) for j in range(d)])
        else:
            bw = np.asarray(bw).ravel()
            if bw.shape[0] != d:
                raise ValueError(f"bw must have shape ({d},); got {bw.shape}")

        return cls(X_sites=X_sites, sigma_sites=sigma_sites, bw=bw)

    def predict(self, X_query: np.ndarray) -> np.ndarray:
        """Return sigma_hat(X_query), floored at _SIGMA_FLOOR."""
        Xq = np.atleast_2d(X_query)
        n_q, d = Xq.shape
        if d != self.X_sites.shape[1]:
            raise ValueError(
                f"X_query has dim {d}, sites have dim {self.X_sites.shape[1]}"
            )

        # Squared scaled distance per dimension, then sum.
        # weights[q, i] = exp(-0.5 * sum_j ((X_query[q,j] - X_sites[i,j]) / bw[j])^2)
        diff = (Xq[:, None, :] - self.X_sites[None, :, :]) / self.bw[None, None, :]
        d2 = (diff * diff).sum(axis=2)
        w = np.exp(-0.5 * d2)
        w_sum = w.sum(axis=1)
        # Guard against pathological numerical underflow at extreme query points.
        safe = w_sum > 0
        sigma_hat = np.full(n_q, float(self.sigma_sites.mean()))
        sigma_hat[safe] = (w[safe] @ self.sigma_sites) / w_sum[safe]

        return np.maximum(sigma_hat, _SIGMA_FLOOR)

    def get_h(self, X_query: np.ndarray, c_scale: float) -> np.ndarray:
        """Convenience: h(x) = c_scale * sigma_hat(x), floored at _SIGMA_FLOOR."""
        return np.maximum(c_scale * self.predict(X_query), _SIGMA_FLOOR)


# Backward-compatible symbol for notebooks or saved scripts that imported the
# old generic class name.  New code should use SampleSdNwScaleEstimator.
SampleSdNwScaleEstimator = SampleSdNwScaleEstimator
