# Archive Record: Branin Benchmark

- **Archive ID:** `ARC-EXP-001`
- **Original path:** `_archive/exp_branin/`
- **Archived on:** 2026-07-22
- **Status:** Parked exploratory experiment.
- **Purpose:** Explore CKME-CP, DCP-DR, and hetGP on a two-dimensional Branin
  response with heteroscedastic Gaussian and Student-t noise.
- **Replacement:** None. Promote only if a two-dimensional benchmark becomes a
  defined requirement of the current paper.
- **Key files:** `spec.md`, `run_branin_compare.py`, `plot_branin.py`, and
  `output*/branin_compare_summary.csv`.
- **Known issues:** The experiment uses old relative imports and has several
  output variants whose protocol differences must be reconstructed before use.
- **Reopen:** Write a new spec, identify the exact saved run to retain, validate
  all baselines, and copy the selected workflow to an active experiment.
