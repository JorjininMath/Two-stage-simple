"""Tests for working-directory-independent project path resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project_support.paths import (
    PROJECT_ROOT_ENV_VAR,
    ProjectRootNotFoundError,
    find_project_root,
    resolve_from,
    resolve_project_path,
)


class ProjectPathResolutionTests(unittest.TestCase):
    """Verify the explicit path rules used by future experiment scripts."""

    def test_find_repo_root_from_nested_source_file(self) -> None:
        expected_root = Path(__file__).resolve().parents[1]
        nested_source = expected_root / "src" / "Two_stage" / "stage2.py"

        self.assertEqual(find_project_root(nested_source), expected_root)

    def test_find_root_from_marker_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "pyproject.toml").write_text("[project]\nname='test'\n")
            (root / "PROTOCOL.md").write_text("# Test protocol\n")
            nested = root / "one" / "two"
            nested.mkdir(parents=True)

            self.assertEqual(find_project_root(nested), root.resolve())

    def test_environment_override_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            configured_root = Path(temporary_directory).resolve()
            with patch.dict(
                os.environ,
                {PROJECT_ROOT_ENV_VAR: str(configured_root)},
                clear=False,
            ):
                self.assertEqual(find_project_root(Path("/")), configured_root)

    def test_missing_environment_override_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_root = Path(temporary_directory) / "missing"
            with patch.dict(
                os.environ,
                {PROJECT_ROOT_ENV_VAR: str(missing_root)},
                clear=False,
            ):
                with self.assertRaisesRegex(
                    ProjectRootNotFoundError,
                    PROJECT_ROOT_ENV_VAR,
                ):
                    find_project_root()

    def test_relative_project_path_uses_explicit_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            expected = (root / "analysis" / "summary.csv").resolve()

            self.assertEqual(
                resolve_project_path("analysis/summary.csv", root=root),
                expected,
            )

    def test_absolute_project_path_is_unchanged(self) -> None:
        absolute = (Path(tempfile.gettempdir()) / "external-result.csv").resolve()

        self.assertEqual(resolve_project_path(absolute), absolute)

    def test_resolve_from_uses_owning_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            experiment = Path(temporary_directory) / "adaptive_h"
            expected = (experiment / "output" / "summary.csv").resolve()

            self.assertEqual(
                resolve_from(experiment, "output/summary.csv"),
                expected,
            )

    def test_must_exist_rejects_missing_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(FileNotFoundError):
                resolve_project_path(
                    "missing.txt",
                    root=temporary_directory,
                    must_exist=True,
                )


if __name__ == "__main__":
    unittest.main()
