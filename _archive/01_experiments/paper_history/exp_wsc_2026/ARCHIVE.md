# Archive Record: WSC 2026 Reproduction Runner

- **Archive ID:** `ARC-HIS-001`
- **Original path:** `_archive/exp_wsc_2026/`
- **Archived on:** 2026-07-22
- **Status:** Historical paper reproduction.
- **Purpose:** Preserve the scripts and partial outputs used to reproduce Tables
  2--3 from the WSC 2026 study.
- **Replacement:** `experiments/adaptive_h/` for the current journal-scale workflow.
- **Key files:** `README.md`, `run_wsc_compare.py`, `run_wsc_gauss_only.py`,
  `make_tables.py`, and `pretrained_params.json`.
- **Known issues:** Commands in the historical README use the old archive path
  and should be treated as provenance, not current instructions. The experiment
  predates the locked iid calibration protocol.
- **Reopen:** Copy the folder to a temporary active location, update all paths,
  recreate its historical environment, and verify tables against the paper.
