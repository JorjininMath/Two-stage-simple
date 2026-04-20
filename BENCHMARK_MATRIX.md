# Benchmark Matrix — DGPs × Experiment Modules

A single source of truth for which DGPs each experiment uses. Use this to
check coverage, spot anchor DGPs (appear in every module), and see where a
new DGP should be added.

Last updated: 2026-04-14.

## Legend

- ⭐ **Anchor** — appears in every module; backbone of the paper
- ✓ — used in this module
- ○ — planned / optional
- (blank) — not used

## Columns (experiment modules)


| Col    | Module                      | Topic             | Purpose                                                        |
| ------ | --------------------------- | ----------------- | -------------------------------------------------------------- |
| **T1** | `exp_design/`               | Allocation design | Saturation sweep, fill-factor calibration, adaptive gain curve |
| **T2** | `exp_conditional_coverage/` | CKME consistency  | CDF / quantile error vs n, fixed vs adaptive h                 |
| **T3** | `exp_nongauss/`             | Method comparison | CKME vs DCP-DR vs hetGP on non-Gaussian noise                  |
| **T4** | `exp_gibbs_compare/`        | RLCP comparison   | CKME-CP vs RLCP on Gibbs DGPs                                  |
| **T5** | `exp_stock/`                | Real data         | Stock returns (from DCP paper; d=1, strong heteroscedasticity) |


## Rows (DGPs)

All `nongauss_*` DGPs are un-normalized (scale = σ_tar(x) directly). Only
Student-t (A1) is kept active; Gamma (B2) and Gaussian-mixture (C1) families,
along with the variance-normalized variants, are archived at
`Two_stage/sim_functions/_archive/` and not used in the paper.


| DGP                   | d   | noise                | T1  | T2  | T3  | T4  | T5  | notes                                   |
| --------------------- | --- | -------------------- | --- | --- | --- | --- | --- | --------------------------------------- |
| **exp2_gauss_high** ⭐ | 1   | Gauss hetero         | ○   | ✓   | ✓   |     |     | anchor — smooth-f Gauss heteroscedastic |
| **nongauss_A1L** ⭐    | 1   | Student-t ν=3        | ○   | ✓   | ✓   |     |     | anchor — strong heavy-tail (T1 ✅ done)  |
| exp2_gauss_low        | 1   | Gauss hetero (low σ) | ○   | ○   |     |     |     | T1 control (saturated throughout)       |
| nongauss_A1S          | 1   | Student-t ν=10       | ○   | ○   | ✓   |     |     | weak heavy-tail control                 |
| exp1                  | 1   | Gauss (MG1 queue)    |     |     |     |     |     | legacy, dropped                         |
| **gibbs_s1** ⭐        | 1   | |sin(x)|             |     |     |     | ✓   |     | ✓                                       |
| gibbs_s2              | 1   | bell-shaped σ        |     | ○   |     | ✓   |     | RLCP Setting 2 (1D)                     |
| **gibbs_s1_d5**       | 5   | |sin(x)|             |     | ✓   | ○   |     |     |                                         |
| gibbs_s2_d5           | 5   | bell-shaped σ        | ○   |     |     |     |     | T1 high-d mechanism 2nd DGP (planned)   |
| stock_returns         | 1   | real (DCP paper)     |     | ○   | ○   |     | ✓   | heteroscedastic financial returns       |


## Anchor DGPs (appear across topics)

Three anchors carry the paper's cross-topic narrative:

- **exp2_gauss_high** — smooth 1D heteroscedastic Gauss. Baseline "easy" regime;
used in T1 as control, T2 as CKME convergence demo, T3 as Gauss sanity for
method comparison.
- **nongauss_A1L** — strong Student-t ν=3 heavy tail. Paper's "hard" regime;
used in T1 (with `_raw` variant), T2 (hardest consistency test), T3
(where DCP-DR / hetGP break down).
- **gibbs_s1** (1D) — RLCP Setting 1. Anchors T4 (direct RLCP comparison)
and extends to T1 via `gibbs_s1_d5` (high-d mechanism).

Any new experiment should reuse at least one anchor before adding a
module-specific DGP. This keeps cross-topic comparisons clean.

## Module-specific additions

- **T1 only**: `exp2_gauss_low` (control), `gibbs_s*_d5` (high-d only relevant
for the allocation mechanism story).
- **T3**: uses Student-t pair (A1S/L) + Gaussian anchor (`exp2_gauss_high`) for
method comparison. Gamma / mixture families are archived.
- **T4 only**: Gibbs 1D variants (RLCP comparison is 1D by construction).

## Consolidation TODO (current gaps)

1. ✅ **All synthetic DGPs registered in `Two_stage/sim_functions/`** —
  `_LOCAL_DGPS` / `examples/exp2_gauss.py` promoted (2026-04-14).
2. **Stock-returns loader** — DCP paper data already in repo; needs a shared
  loader under `data_loaders/` so T5 can be driven by a DGP-like interface.

