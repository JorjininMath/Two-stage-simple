# Experiment Index

Last reviewed: 2026-07-22

This index defines the experiment boundary for the current CKME-DCP paper. It
separates runnable code from paper-ready evidence: an experiment may run
successfully and still be a pilot, pre-protocol artifact, or historical
reproduction.

## Main Paper Line

### `experiments/adaptive_h/` -- active development

Purpose: test whether response-scale normalization improves the CKME-DCP score
and whether an implementable scale estimator approaches the oracle mechanism.

- Active estimator: per-site sample SD plus Nadaraya-Watson smoothing in
  `sample_sd_nw_scale.py`.
- Exp1--Exp3: current manuscript snapshot mechanism evidence, but pre-protocol.
- Exp4: pre-protocol sample-SD/NW provenance; not final paper evidence.
- Planned final run: specified but not yet implemented/produced in
  `planned_final_benchmark_spec.md`.
- Explicit exclusion: the IQR response-scale plug-in is archive-only.

Canonical historical-workflow commands:

```bash
python experiments/adaptive_h/run_exp4_sample_sd_nw.py --n_macro 50
python experiments/adaptive_h/summarize_exp4_sample_sd_nw.py
python experiments/adaptive_h/plot_exp4_gaussian_gap.py
python experiments/adaptive_h/plot_exp4_score_homogeneity.py --simulator all
```

Before sharing a conclusion, check `analysis/CURRENT_RESULTS.md` and
`analysis/CLAIM_EVIDENCE_MAP.md`.

## Supporting Evidence

| Directory | Role | Evidence status | Main caution |
|---|---|---|---|
| `experiments/coverage_mechanism/` | Equal-tailed score and PCP portability studies | Supporting pilots | Scope is limited to tested DGPs/budgets |
| `experiments/framing_validation/` | Scale, score-homogeneity, and epistemic diagnostics | Supporting/partial | Several gates use 5--20 macroreps; G4a missed one gate |
| `experiments/design/` | Stage-1/Stage-2 allocation ablations | Pilot | Most existing comparisons use about 5 macroreps |
| `experiments/conditional_coverage/` | Consistency and over-smoothing diagnostics | Supporting, older protocol | Oracle scale and 10 macroreps |
| `experiments/nongauss/` | DCP-DR/hetGP comparison | Refresh required | Unequal completed benchmark counts |
| `experiments/gibbs_compare/` | Gibbs/RLCP comparison | Historical supporting context | Old run sizes and archived third-party dependency |
| `experiments/onesided/` | Quantile/one-sided diagnostics | Historical diagnostic | Small older runs; no final paper claim |

Supporting evidence belongs in the main text only after its claim is promoted
in the claim-evidence map. Otherwise route it to an appendix, diagnostic note,
or archive.

## Separate or Exploratory Work

### `experiments/mm1_feasibility/`

KME/CKME feasibility for M/M/1 input uncertainty. It evaluates CDF and
quantile error and does **not** run conformal prediction. Keep it separate from
CKME-DCP coverage claims.

### `experiments/stock/` (local, ignored)

Exploratory empirical extension with local data dependencies. It is not a
public reproducibility path until data provenance and benchmark design are
reviewed.

## Compatibility Entrypoints

The old root directories `exp_adaptive_h/`, `exp_conditional_coverage/`,
`exp_design/`, `exp_gibbs_compare/`, `exp_nongauss/`, and `exp_onesided/`
contain only lightweight wrappers for a few formerly documented commands. They
do not own configs, result folders, or experiment implementations.

## Archive Boundary

Retired work is indexed in `_archive/INDEX.md` and `_archive/CATALOG.tsv`.
Important entries include:

- `ARC-DIA-006`: retired IQR response-scale plug-in outputs;
- `ARC-HIS-001`: WSC 2026 table-reproduction runner;
- `ARC-REP-001`: third-party RLCP reproduction;
- `ARC-WRT-001`: dissertation/defense derivative package.

Do not execute or edit an archived experiment in place. Read its `ARCHIVE.md`,
copy it to a new active experiment with a new run ID, then revalidate protocol,
paths, dependencies, and data provenance.
