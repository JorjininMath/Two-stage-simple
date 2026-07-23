"""Regression tests for final adaptive-h postprocessing and export gates."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib as mpl
import pandas as pd

from experiments.adaptive_h.analyze_final_adaptive_h_scores import (
    analyze_file,
    paired_deltas,
)
from experiments.adaptive_h.plot_final_adaptive_h_results import _style
from experiments.adaptive_h.summarize_final_adaptive_h_benchmark import (
    advisor_table,
    main_table,
)
from tools.export_manuscript_assets import (
    Asset,
    qa_artifact_hashes_match,
    sha256,
)


class ScoreHomogeneityTests(unittest.TestCase):
    def test_analyzer_requires_every_configured_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "per_point_fixed.csv"
            pd.DataFrame(
                {
                    "macrorep": [0, 0],
                    "simulator": ["test", "test"],
                    "budget": [100, 100],
                    "arm": ["fixed", "fixed"],
                    "group_bin": [0, 1],
                    "raw_score": [0.1, 0.2],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Expected nonempty group"):
                analyze_file(path, expected_groups=10)

    def test_paired_deltas_rejects_incomplete_arms(self) -> None:
        frame = pd.DataFrame(
            {
                "macrorep": [0, 0],
                "simulator": ["test", "test"],
                "budget": [100, 100],
                "arm": ["fixed", "oracle"],
                "max_pairwise_score_ks": [0.4, 0.2],
                "mean_pairwise_score_ks": [0.3, 0.1],
                "bin_mean_score_range": [0.2, 0.1],
            }
        )
        with self.assertRaisesRegex(ValueError, "Incomplete score arms"):
            paired_deltas(frame)


class ExportGateTests(unittest.TestCase):
    def test_qa_hash_gate_rejects_changed_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "result.csv"
            source.write_text("value\n1\n", encoding="utf-8")
            asset = Asset(
                source="result.csv",
                destination="copy.csv",
                evidence_id="adaptive_h_final_sample_sd_nw",
                role="test",
            )
            report = {
                "status": "pass",
                "artifact_sha256": {"result.csv": sha256(source)},
            }
            self.assertTrue(
                qa_artifact_hashes_match(
                    report,
                    assets=(asset,),
                    repo_root=root,
                )
            )
            source.write_text("value\n2\n", encoding="utf-8")
            self.assertFalse(
                qa_artifact_hashes_match(
                    report,
                    assets=(asset,),
                    repo_root=root,
                )
            )

    def test_qa_hash_gate_requires_pass_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "result.csv"
            source.write_text("value\n1\n", encoding="utf-8")
            asset = Asset(
                source="result.csv",
                destination="copy.csv",
                evidence_id="adaptive_h_final_sample_sd_nw",
                role="test",
            )
            report = {
                "status": "fail",
                "artifact_sha256": {"result.csv": sha256(source)},
            }
            self.assertFalse(
                qa_artifact_hashes_match(
                    report,
                    assets=(asset,),
                    repo_root=root,
                )
            )


class FigureFormatTests(unittest.TestCase):
    def test_publication_font_settings(self) -> None:
        _style()
        self.assertEqual(mpl.rcParams["pdf.fonttype"], 42)
        self.assertEqual(mpl.rcParams["ps.fonttype"], 42)
        self.assertEqual(mpl.rcParams["svg.fonttype"], "none")


class AdvisorTableTests(unittest.TestCase):
    def test_table_separates_raw_and_projected_interval_coverage(self) -> None:
        summary = pd.DataFrame(
            {
                "simulator": ["mm1_sojourn"],
                "budget": [1000],
                "arm": ["fixed"],
                "mean_coverage": [0.899],
                "mcse_coverage": [0.002],
                "mean_coverage_interval": [0.855],
                "mcse_coverage_interval": [0.002],
                "mean_score_interval_disagreement": [0.049],
                "mean_width": [7.9],
                "mcse_width": [0.06],
                "mean_interval_score": [11.0],
                "mcse_interval_score": [0.1],
                "mean_worst_group_coverage_gap": [0.07],
            }
        )
        table = advisor_table(main_table(summary, budget=1000))
        self.assertEqual(table.loc[0, "Raw-score cov. (MCSE)"], "0.899 (0.002)")
        self.assertEqual(table.loc[0, "Interval cov. (MCSE)"], "0.855 (0.002)")
        self.assertEqual(table.loc[0, "Disagreement"], "0.049")


if __name__ == "__main__":
    unittest.main()
