"""Focused tests for the final adaptive-h benchmark simulators."""

from __future__ import annotations

import unittest

import numpy as np

from Two_stage.sim_functions import (
    get_experiment_config,
    mm1_sojourn_mean,
    mm1_sojourn_scale,
    mm1_sojourn_simulator,
    raised_floor_gauss_simulator,
    raised_floor_mean,
    raised_floor_scale,
    raised_floor_t3_simulator,
)
from experiments.adaptive_h.adaptive_bandwidth import ORACLE_SCALE


class FinalAdaptiveHSimulatorTests(unittest.TestCase):
    """Check registry metadata, distribution definitions, and oracle scales."""

    def test_registry_contains_three_one_dimensional_dgps(self) -> None:
        expected_bounds = {
            "mm1_sojourn": (0.1, 0.9),
            "raised_floor_gauss": (0.0, 2.0 * np.pi),
            "raised_floor_t3": (0.0, 2.0 * np.pi),
        }
        for name, (expected_lower, expected_upper) in expected_bounds.items():
            with self.subTest(simulator=name):
                config = get_experiment_config(name)
                lower, upper = config["bounds"]
                self.assertEqual(config["d"], 1)
                np.testing.assert_allclose(lower, [expected_lower])
                np.testing.assert_allclose(upper, [expected_upper])
                self.assertIs(config["simulator"], globals()[f"{name}_simulator"])

    def test_simulators_preserve_shape_and_are_reproducible(self) -> None:
        inputs = {
            "mm1_sojourn": np.array([[0.2], [0.5], [0.8]]),
            "raised_floor_gauss": np.array([[0.0], [np.pi], [2.0 * np.pi]]),
            "raised_floor_t3": np.array([[0.0], [np.pi], [2.0 * np.pi]]),
        }
        for name, x in inputs.items():
            with self.subTest(simulator=name):
                simulator = get_experiment_config(name)["simulator"]
                first = simulator(x, random_state=731)
                second = simulator(x, random_state=731)
                self.assertEqual(first.shape, x.shape)
                np.testing.assert_array_equal(first, second)

    def test_raised_floor_mean_scale_and_positive_floor(self) -> None:
        x = np.array([0.0, np.pi, 2.0 * np.pi])
        expected_mean = np.exp(x / 10.0) * np.sin(x)
        expected_scale = 0.10 + 0.20 * (x - np.pi) ** 2
        np.testing.assert_allclose(raised_floor_mean(x), expected_mean)
        np.testing.assert_allclose(raised_floor_scale(x), expected_scale)
        self.assertAlmostEqual(float(raised_floor_scale(np.array([np.pi]))[0]), 0.10)
        self.assertTrue(np.all(raised_floor_scale(x) >= 0.10))

    def test_gaussian_conditional_moments(self) -> None:
        n_draws = 200_000
        x_value = 1.25
        x = np.full(n_draws, x_value)
        draws = raised_floor_gauss_simulator(x, random_state=1729)
        target_mean = float(raised_floor_mean(np.array([x_value]))[0])
        target_sd = float(raised_floor_scale(np.array([x_value]))[0])
        self.assertAlmostEqual(float(np.mean(draws)), target_mean, delta=0.01 * target_sd)
        self.assertAlmostEqual(float(np.std(draws)), target_sd, delta=0.01 * target_sd)

    def test_t3_noise_is_variance_normalized(self) -> None:
        n_draws = 400_000
        x_value = 1.25
        x = np.full(n_draws, x_value)
        draws = raised_floor_t3_simulator(x, random_state=2718)
        target_mean = float(raised_floor_mean(np.array([x_value]))[0])
        target_sd = float(raised_floor_scale(np.array([x_value]))[0])
        standardized = (draws - target_mean) / target_sd
        self.assertAlmostEqual(float(np.mean(standardized)), 0.0, delta=0.015)
        self.assertAlmostEqual(float(np.std(standardized)), 1.0, delta=0.04)
        self.assertLess(float(np.std(standardized)), np.sqrt(3.0) - 0.25)

    def test_mm1_sojourn_conditional_mean_sd_and_positivity(self) -> None:
        n_draws = 250_000
        rho_value = 0.60
        rho = np.full(n_draws, rho_value)
        draws = mm1_sojourn_simulator(rho, random_state=31415)
        target = 1.0 / (1.0 - rho_value)
        self.assertTrue(np.all(draws >= 0.0))
        self.assertAlmostEqual(float(np.mean(draws)), target, delta=0.015 * target)
        self.assertAlmostEqual(float(np.std(draws)), target, delta=0.02 * target)
        np.testing.assert_allclose(mm1_sojourn_mean(rho[:3]), target)
        np.testing.assert_allclose(mm1_sojourn_scale(rho[:3]), target)

    def test_domain_validation_rejects_invalid_inputs(self) -> None:
        for rho in (np.array([0.0]), np.array([1.0]), np.array([np.nan])):
            with self.subTest(rho=rho):
                with self.assertRaises(ValueError):
                    mm1_sojourn_simulator(rho, random_state=1)

        for x in (
            np.array([-1e-6]),
            np.array([2.0 * np.pi + 1e-6]),
            np.array([np.inf]),
        ):
            with self.subTest(x=x):
                with self.assertRaises(ValueError):
                    raised_floor_gauss_simulator(x, random_state=1)
                with self.assertRaises(ValueError):
                    raised_floor_t3_simulator(x, random_state=1)

    def test_oracle_scale_matches_conditional_standard_deviation(self) -> None:
        mm1_x = np.array([[0.2], [0.5], [0.8]])
        raised_x = np.array([[0.0], [np.pi], [2.0 * np.pi]])
        np.testing.assert_allclose(
            ORACLE_SCALE["mm1_sojourn"](mm1_x),
            mm1_sojourn_scale(mm1_x),
        )
        for name in ("raised_floor_gauss", "raised_floor_t3"):
            with self.subTest(simulator=name):
                np.testing.assert_allclose(
                    ORACLE_SCALE[name](raised_x),
                    raised_floor_scale(raised_x),
                )


if __name__ == "__main__":
    unittest.main()
