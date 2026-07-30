"""Regression tests for adaptive-bandwidth CDF and score evaluation."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from CKME.coefficients import compute_ckme_coeffs
from CKME.ckme import CKMEModel
from CKME.indicators import make_indicator
from CKME.parameters import Params
from CP.interval import projected_quantile_interval
from experiments.adaptive_h.adaptive_bandwidth import (
    adaptive_point_scores,
    adaptive_predict_cdf,
    adaptive_predict_interval,
)


def _fit_small_model(indicator_type: str, *, r: int) -> CKMEModel:
    if r == 2:
        X_sites = np.array([[0.1], [0.5], [0.9]], dtype=float)
        X_train = np.repeat(X_sites, r, axis=0)
        Y_train = np.array([-0.4, 0.1, 0.0, 0.7, 0.5, 1.4], dtype=float)
    else:
        X_train = np.array([[0.1], [0.3], [0.6], [0.9]], dtype=float)
        Y_train = np.array([-0.4, 0.2, 0.5, 1.3], dtype=float)

    model = CKMEModel(indicator_type=indicator_type)
    return model.fit(
        X_train,
        Y_train,
        params=Params(ell_x=0.35, lam=0.03, h=0.2),
        r=r,
    )


def _scalar_adaptive_cdf(
    model: CKMEModel,
    X_query: np.ndarray,
    h_query: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Reference implementation matching the original per-query loop."""
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    h_query = np.asarray(h_query, dtype=float).ravel()
    t_grid = np.asarray(t_grid, dtype=float).ravel()
    Y_flat = model.Y.ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_query)

    F_all = np.empty((X_query.shape[0], t_grid.size), dtype=float)
    for point in range(X_query.shape[0]):
        indicator = make_indicator(model.indicator_type, float(h_query[point]))
        G = indicator.g_matrix(Y_flat, t_grid)
        if model.r > 1:
            G = G.reshape(model.n, model.r, -1).mean(axis=1)
        F_all[point] = C[:, point] @ G
    return F_all


def _scalar_raw_scores(
    model: CKMEModel,
    X_query: np.ndarray,
    Y_query: np.ndarray,
    h_query: np.ndarray,
) -> np.ndarray:
    X_query = np.atleast_2d(np.asarray(X_query, dtype=float))
    Y_query = np.asarray(Y_query, dtype=float).ravel()
    h_query = np.asarray(h_query, dtype=float).ravel()
    Y_flat = model.Y.ravel()
    C = compute_ckme_coeffs(model.L, model.kx, model.X, X_query)

    scores = np.empty(Y_query.size, dtype=float)
    for point, (response, bandwidth) in enumerate(zip(Y_query, h_query)):
        indicator = make_indicator(model.indicator_type, float(bandwidth))
        G = indicator.g_matrix(Y_flat, np.array([response]))
        if model.r > 1:
            G = G.reshape(model.n, model.r, 1).mean(axis=1)
        F_value = float(C[:, point] @ G[:, 0])
        scores[point] = abs(float(np.clip(F_value, 0.0, 1.0)) - 0.5)
    return scores


class AdaptiveBandwidthBatchingTests(unittest.TestCase):
    """Check batched evaluation against the original scalar formulas."""

    def setUp(self) -> None:
        self.X_query = np.linspace(0.12, 0.88, 7)[:, None]
        self.h_query = np.linspace(0.08, 0.45, 7)
        self.t_grid = np.linspace(-1.0, 2.0, 31)

    def test_batched_cdf_matches_scalar_for_all_indicators(self) -> None:
        indicator_types = ["logistic", "softplus", "step"]
        if importlib.util.find_spec("scipy") is not None:
            indicator_types.append("gaussian_cdf")

        for indicator_type in indicator_types:
            with self.subTest(indicator_type=indicator_type):
                model = _fit_small_model(indicator_type, r=2)
                expected = _scalar_adaptive_cdf(
                    model,
                    self.X_query,
                    self.h_query,
                    self.t_grid,
                )

                query_batched = adaptive_predict_cdf(
                    model,
                    self.X_query,
                    self.h_query,
                    self.t_grid,
                    max_batch_bytes=8_000,
                )
                threshold_batched = adaptive_predict_cdf(
                    model,
                    self.X_query,
                    self.h_query,
                    self.t_grid,
                    max_batch_bytes=200,
                )

                np.testing.assert_allclose(query_batched, expected, atol=1e-13)
                np.testing.assert_allclose(
                    threshold_batched,
                    expected,
                    atol=1e-13,
                )

    def test_unreplicated_cdf_and_clipping_match_scalar_reference(self) -> None:
        model = _fit_small_model("logistic", r=1)
        expected = _scalar_adaptive_cdf(
            model,
            self.X_query,
            self.h_query,
            self.t_grid,
        )
        actual = adaptive_predict_cdf(
            model,
            self.X_query,
            self.h_query,
            self.t_grid,
            clip=True,
            max_batch_bytes=1_000,
        )
        np.testing.assert_allclose(actual, np.clip(expected, 0.0, 1.0), atol=1e-13)

    def test_interval_uses_batched_cdf_and_canonical_projection(self) -> None:
        model = _fit_small_model("logistic", r=2)
        q_hat = 0.37
        expected_cdf = _scalar_adaptive_cdf(
            model,
            self.X_query,
            self.h_query,
            self.t_grid,
        )
        expected = projected_quantile_interval(
            expected_cdf,
            self.t_grid,
            q_hat,
        )
        actual = adaptive_predict_interval(
            model,
            self.X_query,
            self.h_query,
            self.t_grid,
            q_hat,
            max_batch_bytes=200,
        )
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_public_point_scores_match_raw_scalar_formula(self) -> None:
        for indicator_type in ["logistic", "softplus", "step"]:
            with self.subTest(indicator_type=indicator_type):
                model = _fit_small_model(indicator_type, r=2)
                Y_query = np.linspace(-0.2, 1.2, self.X_query.shape[0])
                expected = _scalar_raw_scores(
                    model,
                    self.X_query,
                    Y_query,
                    self.h_query,
                )
                actual = adaptive_point_scores(
                    model,
                    self.X_query,
                    Y_query,
                    self.h_query,
                    max_batch_bytes=200,
                )
                np.testing.assert_allclose(actual, expected, atol=1e-13)

    def test_invalid_bandwidth_inputs_raise_clear_errors(self) -> None:
        model = _fit_small_model("logistic", r=2)
        with self.assertRaisesRegex(ValueError, "one bandwidth per query"):
            adaptive_predict_cdf(
                model,
                self.X_query,
                self.h_query[:-1],
                self.t_grid,
            )
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            adaptive_predict_cdf(
                model,
                self.X_query,
                np.zeros(self.X_query.shape[0]),
                self.t_grid,
            )
        with self.assertRaisesRegex(ValueError, "max_batch_bytes"):
            adaptive_point_scores(
                model,
                self.X_query,
                np.zeros(self.X_query.shape[0]),
                self.h_query,
                max_batch_bytes=0,
            )


if __name__ == "__main__":
    unittest.main()
