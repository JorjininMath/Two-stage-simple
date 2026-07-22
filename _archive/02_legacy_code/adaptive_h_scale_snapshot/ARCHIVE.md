# Archive Record: Pre-Rename Sample-SD/NW Scale Snapshot

- **Archive ID:** `ARC-COD-003`
- **Original path:** `experiments/adaptive_h/plugin_sigma.py.bak`
- **Archived on:** 2026-07-22
- **Status:** Superseded source snapshot.
- **Purpose:** Preserve the pre-`bw_factor` implementation that was stored as a
  `.bak` file in the active experiment directory.
- **Replacement:** `experiments/adaptive_h/sample_sd_nw_scale.py`.
- **Key file:** `sample_sd_nw_scale_before_bandwidth_factor.py`.
- **Known issues:** The old generic name obscured that response scale was based
  on sample SD. This snapshot is not the retired IQR response-scale estimator.
- **Reopen:** Diff against the active implementation only when tracing code
  history; do not import it from active experiments.
