# Archive Record: RLCP Reproduction

- **Archive ID:** `ARC-REP-001`
- **Original path:** `_archive/Conditional_Coverage/reproduce_rlcp/`
- **Archived on:** 2026-07-22
- **Status:** Completed external reproduction.
- **Purpose:** Reproduce selected Figure 2 results from Hore and Barber (2024)
  and retain the external RLCP repository used by the local R scripts.
- **Replacement:** `experiments/gibbs_compare/` is the active comparison workflow; this
  package remains its external provenance reference.
- **Key files:** `reproduce_notes.md`, `run_reproduction.R`, `smoke_test.R`,
  `output_min/`, `RLCP/README.md`, and `RLCP/results/gibbs_et_al_results.csv`.
- **Known issues:** `RLCP/` is a nested Git repository and must remain intact.
  Active Gibbs scripts still need their old RLCP path updated to this location.
  R dependencies, runtime, authorship, and third-party licensing remain separate
  from this project's own code.
- **Reopen:** Work from a copy, preserve the nested repository metadata, inspect
  its upstream history/license, recreate the R environment, and run the smoke
  test before the full reproduction.
