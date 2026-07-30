"""Regression tests for the public package names used by existing scripts."""

from __future__ import annotations

import unittest


class PublicPackageImportTests(unittest.TestCase):
    """Keep the pre-reorganization import API stable."""

    def test_ckme_public_api_imports(self) -> None:
        from CKME import CKMEModel, ParamGrid, Params, make_loss

        self.assertTrue(callable(CKMEModel))
        self.assertTrue(callable(Params))
        self.assertTrue(callable(ParamGrid))
        self.assertTrue(callable(make_loss))

    def test_cp_public_api_imports(self) -> None:
        from CP import CP

        self.assertTrue(callable(CP))

    def test_two_stage_public_api_imports(self) -> None:
        from Two_stage import (
            Stage1TrainResult,
            Stage2Result,
            run_stage1_train,
            run_stage2,
        )

        self.assertTrue(callable(Stage1TrainResult))
        self.assertTrue(callable(Stage2Result))
        self.assertTrue(callable(run_stage1_train))
        self.assertTrue(callable(run_stage2))


if __name__ == "__main__":
    unittest.main()
