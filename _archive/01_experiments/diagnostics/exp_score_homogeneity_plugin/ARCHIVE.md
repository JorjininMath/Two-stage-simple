# Archive Record: Sample-SD plus NW Homogeneity Precursor

- **Archive ID:** `ARC-DIA-004`
- **Original path:** `_archive/exp_score_homogeneity_plugin/`
- **Archived on:** 2026-07-22
- **Status:** Superseded implementation precursor; historical diagnostic only.
- **Purpose:** Estimate sitewise scale with sample standard deviations, smooth
  it with a Nadaraya-Watson kernel, and study score homogeneity and bandwidth
  sensitivity.
- **Replacement:** The protocol-aligned implementation in `experiments/adaptive_h/`.
- **Key files:** `run_plugin.py`, `config.txt`, `plot_compare.py`,
  `plot_hsigma_scan.py`, and `output*/summary_*.csv`.
- **Known issues:** Despite the historical folder name, this is a sample-SD plus
  NW precursor, not the retired IQR plug-in. It uses an older experiment and
  calibration setup, so its outputs are excluded from current summaries,
  advisor updates, and the manuscript.
- **Reopen:** Compare individual implementation ideas with the current active
  code; do not reactivate the archived runner or copy its reported numbers.
