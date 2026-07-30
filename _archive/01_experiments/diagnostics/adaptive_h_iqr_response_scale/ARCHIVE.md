# Archive Record: Retired IQR Response-Scale Plug-In

- **Archive ID:** `ARC-DIA-006`
- **Original path:** `experiments/adaptive_h/output_exp4_iqr*/` and related
  post-hoc diagnostics.
- **Archived on:** 2026-07-22
- **Status:** Retired by estimator decision; provenance only.
- **Purpose:** Preserve results and diagnostics from an earlier adaptive-h arm
  that estimated the response scale using an IQR-based rule.
- **Replacement:** `experiments/adaptive_h/sample_sd_nw_scale.py`, which uses
  per-site sample standard deviations followed by Nadaraya-Watson smoothing.
- **Key files:** `diagnose_iqr_response_scale_outputs.py`,
  `artifacts/output_exp4_iqr/`, and
  `artifacts/output_existing_diagnostics/`.
- **Known issues:** These artifacts are excluded from current result summaries,
  advisor updates, manuscript-generated assets, and active experiment commands.
  The archived diagnostic script retains its original paths and is not an
  active entrypoint.
- **Reopen:** Use only to trace historical decisions. Do not merge IQR results
  into current tables or revive the estimator without a new protocol decision.

No files were permanently deleted when this item was archived.
