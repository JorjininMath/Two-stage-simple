"""Small end-to-end smoke test for the public two-stage research API."""

from __future__ import annotations

import unittest

import numpy as np

from CKME.parameters import Params
from Two_stage import run_stage1_train, run_stage2


class TwoStagePipelineSmokeTests(unittest.TestCase):
    """Exercise Stage 1 training, iid calibration, and interval prediction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.stage1 = run_stage1_train(
            n_0=4,
            r_0=2,
            simulator_func="exp1",
            params=Params(ell_x=0.5, lam=0.01, h=0.2),
            t_grid_size=31,
            random_state=17,
        )
        cls.stage2 = run_stage2(
            stage1_result=cls.stage1,
            X_cand=None,
            n_1=10,
            r_1=1,
            simulator_func="exp1",
            method="iid",
            alpha=0.1,
            random_state=19,
        )

    def test_stage1_shapes_and_parameters(self) -> None:
        self.assertEqual(self.stage1.X_all.shape, (8, 1))
        self.assertEqual(self.stage1.Y_all.shape, (8,))
        self.assertEqual(self.stage1.X_0.shape, (4, 1))
        self.assertEqual(self.stage1.t_grid.shape, (31,))
        self.assertEqual(self.stage1.r_0, 2)

    def test_stage2_uses_iid_single_draw_calibration(self) -> None:
        self.assertEqual(self.stage2.selection_method, "iid")
        self.assertEqual(self.stage2.r_1, 1)
        self.assertEqual(self.stage2.X_stage2.shape, (10, 1))
        self.assertEqual(self.stage2.Y_stage2.shape, (10,))
        self.assertTrue(np.isfinite(self.stage2.cp.q_hat))

    def test_prediction_intervals_are_finite_and_ordered(self) -> None:
        lower, upper = self.stage2.predict_interval(
            np.array([[0.25], [0.75]], dtype=float)
        )

        self.assertEqual(lower.shape, (2,))
        self.assertEqual(upper.shape, (2,))
        self.assertTrue(np.all(np.isfinite(lower)))
        self.assertTrue(np.all(np.isfinite(upper)))
        self.assertTrue(np.all(lower <= upper))

    def test_calibration_input_and_output_seeds_are_separate(self) -> None:
        first = run_stage2(
            stage1_result=self.stage1,
            X_cand=None,
            n_1=10,
            r_1=1,
            simulator_func="exp1",
            method="iid",
            alpha=0.1,
            random_state=29,
            sim_random_state=31,
        )
        second = run_stage2(
            stage1_result=self.stage1,
            X_cand=None,
            n_1=10,
            r_1=1,
            simulator_func="exp1",
            method="iid",
            alpha=0.1,
            random_state=29,
            sim_random_state=37,
        )
        np.testing.assert_array_equal(first.X_stage2, second.X_stage2)
        self.assertFalse(np.array_equal(first.Y_stage2, second.Y_stage2))


if __name__ == "__main__":
    unittest.main()
