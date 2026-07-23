"""Tests for the one-way Career OS project-status interface."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "tools" / "export_project_status.py"


def load_export_module():
    spec = importlib.util.spec_from_file_location("export_project_status", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProjectStatusExportTests(unittest.TestCase):
    def test_live_status_contains_all_stable_fields(self) -> None:
        module = load_export_module()
        payload = module.extract_status(
            (PROJECT_ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
        )

        self.assertEqual(payload["project_id"], "wk_ckme_ext")
        self.assertEqual(set(payload["fields"]), set(module.FIELD_NAMES))
        self.assertEqual(payload["last_updated"], "2026-07-23")

    def test_duplicate_or_missing_marker_is_rejected(self) -> None:
        module = load_export_module()
        with self.assertRaises(ValueError):
            module.extract_status("Last updated: 2026-07-22\n")


if __name__ == "__main__":
    unittest.main()
