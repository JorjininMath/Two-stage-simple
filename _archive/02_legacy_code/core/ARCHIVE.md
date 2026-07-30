# Archive Record: Legacy Core Package

- **Archive ID:** `ARC-COD-001`
- **Original path:** `_archive/core/`
- **Archived on:** 2026-07-22
- **Status:** Deprecated implementation.
- **Purpose:** Preserve an early general-purpose modeling package with loss,
  model, optimizer, and predictor abstractions.
- **Replacement:** The active `src/CKME/`, `src/CP/`, and `src/Two_stage/` packages.
- **Key files:** `__init__.py`, `loss/`, `models/`, `optimizers/`, and
  `predictors/`.
- **Known issues:** Interfaces and assumptions do not match the active pipeline;
  no current experiment should import this package.
- **Reopen:** Extract only a clearly identified algorithmic idea into a new
  active module with tests. Do not add the archive directory to `sys.path`.
