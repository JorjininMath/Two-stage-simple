"""Focused tests for the canonical final adaptive-h runner."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.adaptive_h.run_final_adaptive_h_benchmark import (
    ARMS,
    _equal_count_bins,
    _job_complete,
    _job_dir,
    _seed_bundle,
    run_one_job,
)


class FinalAdaptiveHRunnerTests(unittest.TestCase):
    """Exercise locked configuration invariants and one tiny real job."""

    def test_tracked_config_locks_iid_protocol_without_s0(self) -> None:
        config_path = (
            Path(__file__).resolve().parents[1]
            / "experiments"
            / "adaptive_h"
            / "final_benchmark_config.json"
        )
        config = json.loads(config_path.read_text())
        self.assertEqual(config["calibration_method"], "iid")
        self.assertEqual(config["r_cal"], 1)
        self.assertEqual(config["test_method"], "iid")
        self.assertEqual(config["r_test"], 1)
        self.assertEqual(config["stage1_design"], "grid")
        self.assertNotIn("s0", json.dumps(config).lower())

    def test_seed_bundle_names_disjoint_reproducible_streams(self) -> None:
        first = _seed_bundle(1000, 2, 1, 3)
        second = _seed_bundle(1000, 2, 1, 3)
        other = _seed_bundle(1000, 3, 1, 3)
        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), len(first))
        self.assertTrue(set(first.values()).isdisjoint(other.values()))
        self.assertEqual(first["stage1_output"], first["stage1_design"] + 1)

    def test_equal_count_bins_are_balanced(self) -> None:
        labels = _equal_count_bins(
            np.array([0.8, 0.1, 0.6, 0.4, 0.2, 0.9, 0.3, 0.7, 0.5]),
            4,
        )
        counts = np.bincount(labels)
        self.assertEqual(counts.sum(), 9)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_tiny_job_writes_complete_paired_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            job = {
                "schema_version": "test",
                "output_dir": str(output_dir),
                "overwrite": False,
                "simulator": "mm1_sojourn",
                "budget": 20,
                "macrorep": 0,
                "r0": 10,
                "stage1_design": "grid",
                "n_cal": 30,
                "n_test": 25,
                "alpha": 0.1,
                "c_scale": 1.0,
                "scale_bw_factor": 1.0,
                "group_bins": 5,
                "t_grid_size": 60,
                "t_grid_margin": 3.0,
                "params": {"ell_x": 0.3, "lam": 0.001, "h": 0.1},
                "seeds": _seed_bundle(7000, 0, 0, 0),
            }
            rows = run_one_job(job)
            path = _job_dir(output_dir, "mm1_sojourn", 20, 0)
            self.assertTrue(_job_complete(path))
            self.assertEqual({row["arm"] for row in rows}, set(ARMS))

            frames = {
                arm: pd.read_csv(path / f"per_point_{arm}.csv")
                for arm in ARMS
            }
            required = {
                "raw_score",
                "s_hat",
                "s_oracle",
                "h_over_s",
                "group_bin",
                "covered_score",
                "covered_interval",
            }
            for frame in frames.values():
                self.assertTrue(required.issubset(frame.columns))
                self.assertEqual(len(frame), 25)
            reference = frames["fixed"][["test_index", "x0", "y"]]
            for arm in ("plugin_sd_nw", "oracle"):
                pd.testing.assert_frame_equal(
                    reference,
                    frames[arm][["test_index", "x0", "y"]],
                )


if __name__ == "__main__":
    unittest.main()
