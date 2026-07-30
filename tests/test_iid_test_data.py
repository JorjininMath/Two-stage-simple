"""Focused tests for protocol-matched iid test-data generation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from Two_stage.test_data import generate_test_data


def _stage2_stub(method: str, x_1: np.ndarray) -> SimpleNamespace:
    """Create the minimal Stage-2 result used by test-data generation."""
    return SimpleNamespace(
        selection_method=method,
        X_1=np.asarray(x_1, dtype=float),
        model=None,
        t_grid=np.linspace(-1.0, 1.0, 5),
        alpha=0.1,
    )


class IidTestDataTests(unittest.TestCase):
    """Verify the iid q_X branch and its boundary with legacy modes."""

    def test_iid_generation_does_not_require_candidates(self) -> None:
        stage2 = _stage2_stub("iid", np.array([[0.5]]))

        X_test, Y_test = generate_test_data(
            stage2_result=stage2,
            n_test=8,
            r_test=1,
            X_cand=None,
            simulator_func="exp1",
            random_state=41,
        )

        self.assertEqual(X_test.shape, (8, 1))
        self.assertEqual(Y_test.shape, (8,))
        self.assertTrue(np.all((0.1 <= X_test) & (X_test <= 0.9)))

    def test_iid_generation_uses_custom_qx_sampler(self) -> None:
        stage2 = _stage2_stub("iid", np.array([[0.5]]))

        def custom_qx(
            n: int,
            rng: np.random.Generator,
        ) -> np.ndarray:
            return rng.choice([0.2, 0.8], size=n).reshape(-1, 1)

        X_test, _ = generate_test_data(
            stage2_result=stage2,
            n_test=12,
            r_test=1,
            simulator_func="exp1",
            random_state=17,
            qx_sampler=custom_qx,
        )

        expected = custom_qx(12, np.random.default_rng(17))
        np.testing.assert_array_equal(X_test, expected)

    def test_iid_generation_does_not_exclude_repeated_target_locations(self) -> None:
        stage2 = _stage2_stub("iid", np.array([[0.5]]))

        def point_mass_qx(
            n: int,
            rng: np.random.Generator,
        ) -> np.ndarray:
            del rng
            return np.full((n, 1), 0.5)

        X_test, _ = generate_test_data(
            stage2_result=stage2,
            n_test=6,
            r_test=1,
            simulator_func="exp1",
            random_state=23,
            tolerance=1.0,
            qx_sampler=point_mass_qx,
        )

        np.testing.assert_array_equal(X_test, np.full((6, 1), 0.5))

    def test_iid_input_and_simulator_seeds_can_be_separated(self) -> None:
        stage2 = _stage2_stub("iid", np.array([[0.5]]))
        common = {
            "stage2_result": stage2,
            "n_test": 10,
            "r_test": 1,
            "simulator_func": "exp1",
            "random_state": 37,
        }

        X_first, Y_first = generate_test_data(
            **common,
            sim_random_state=101,
        )
        X_second, Y_second = generate_test_data(
            **common,
            sim_random_state=202,
        )
        X_repeat, Y_repeat = generate_test_data(
            **common,
            sim_random_state=101,
        )

        np.testing.assert_array_equal(X_first, X_second)
        np.testing.assert_array_equal(X_first, X_repeat)
        self.assertFalse(np.array_equal(Y_first, Y_second))
        np.testing.assert_array_equal(Y_first, Y_repeat)

    def test_iid_generation_requires_one_output_per_input(self) -> None:
        stage2 = _stage2_stub("iid", np.array([[0.5]]))

        with self.assertRaisesRegex(ValueError, "requires r_test=1"):
            generate_test_data(
                stage2_result=stage2,
                n_test=4,
                r_test=2,
                simulator_func="exp1",
                random_state=29,
            )

    def test_legacy_sampling_still_requires_candidates(self) -> None:
        stage2 = _stage2_stub("sampling", np.array([[0.5]]))

        with self.assertRaisesRegex(ValueError, "X_cand is required"):
            generate_test_data(
                stage2_result=stage2,
                n_test=4,
                r_test=1,
                X_cand=None,
                simulator_func="exp1",
                random_state=31,
            )

    def test_legacy_lhs_still_excludes_stage2_sites(self) -> None:
        stage2 = _stage2_stub("lhs", np.array([[0.5]]))

        X_test, _ = generate_test_data(
            stage2_result=stage2,
            n_test=5,
            r_test=2,
            simulator_func="exp1",
            random_state=43,
            tolerance=0.05,
        )

        X_test_sites = X_test.reshape(5, 2, 1)[:, 0, :]
        self.assertTrue(np.all(np.abs(X_test_sites[:, 0] - 0.5) > 0.05))


if __name__ == "__main__":
    unittest.main()
