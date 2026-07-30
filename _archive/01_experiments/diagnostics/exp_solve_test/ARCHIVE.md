# Archive Record: Quantile Extraction Solver Test

- **Archive ID:** `ARC-DIA-005`
- **Original path:** `_archive/exp_solve_test/`
- **Archived on:** 2026-07-22
- **Status:** Completed implementation diagnostic.
- **Purpose:** Compare grid-plus-isotonic and direct-solver quantile extraction
  for logistic and step indicators.
- **Replacement:** The tested quantile implementation in the active `src/CKME/`
  package and its current tests.
- **Key files:** `test_solve.py` and `output/solve_raw.csv`.
- **Known issues:** This is a standalone research diagnostic rather than a
  maintained automated unit test. Its path assumptions changed after archival.
- **Reopen:** Translate the relevant case into a small deterministic test in the
  active test suite instead of running this directory in place.
