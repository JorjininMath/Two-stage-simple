# Archive Record: Indicator-Bandwidth Sweep

- **Archive ID:** `ARC-DIA-002`
- **Original path:** `_archive/exp_h_sweep/`
- **Archived on:** 2026-07-22
- **Status:** Completed diagnostic; not active evidence.
- **Purpose:** Sweep the smooth-indicator bandwidth and inspect CDF shape,
  density shape, quantile error, and CRPS.
- **Replacement:** `experiments/adaptive_h/` for the current bandwidth study.
- **Key files:** `run_h_sweep.py`, `plot_h_sweep.py`, and `output_*/`.
- **Known issues:** Fixed constants and old relative imports make the runner
  unsuitable as a current experiment entrypoint. It predates the locked
  adaptive-bandwidth protocol.
- **Reopen:** Copy the scripts into a newly specified diagnostic, replace path
  logic and constants with explicit configuration, then rerun from scratch.
