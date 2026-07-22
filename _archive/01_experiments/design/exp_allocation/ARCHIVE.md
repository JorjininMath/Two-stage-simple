# Archive Record: Stage-2 Allocation Experiment

- **Archive ID:** `ARC-DES-001`
- **Original path:** `_archive/exp_allocation/`
- **Archived on:** 2026-07-22
- **Status:** Superseded experiment; historical results only.
- **Purpose:** Compare Stage-2 budget allocations and LHS, adaptive sampling,
  and mixed site selection across CKME-CP, DCP-DR, and hetGP.
- **Replacement:** `experiments/design/` and the current rules in `PROTOCOL.md`.
- **Key files:** `run_allocation_compare.py`, `plot_allocation.py`, `config.txt`,
  and `output/*/allocation_summary.csv`.
- **Known issues:** The runner assumes its old directory depth and imports active
  modules using a relative `sys.path` edit. Its calibration design predates the
  current iid target-law protocol. The output tree is about 1.2 GB and contains
  more than eleven thousand files.
- **Reopen:** Copy the whole directory to a new active experiment, replace path
  assumptions, reconcile the design with `PROTOCOL.md`, and give any new run a
  fresh identifier. Do not treat saved summaries as current paper evidence.
