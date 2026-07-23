# Reproduction and Validation Guide

Run commands from the repository root unless a section says otherwise. Install
the checkout first:

```bash
python -m pip install -e ".[experiments,test]"
```

## Fast Repository Checks

These checks do not run paper-scale experiments:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src experiments tests tools
python tools/export_manuscript_assets.py --check
bash -n hpc/submit_all.sh
```

The M/M/1 feasibility module uses pytest-style tests:

```bash
python -m pytest experiments/mm1_feasibility/tests -q
```

## Core API Smoke Run

`tests/test_two_stage_pipeline_smoke.py` performs a tiny Stage 1 to iid Stage 2
run. It verifies the package layout and the guarantee-bearing calibration path;
it is not a numerical experiment.

```bash
PYTHONPATH=src python -m unittest discover -s tests \
  -p 'test_two_stage_pipeline_smoke.py' -v
```

## Adaptive-Bandwidth Mechanism Runs

Existing Exp1--Exp4 scripts are runnable historical workflows. Their results
predate the locked protocol and must retain that qualification.

```bash
python experiments/adaptive_h/pretrain_params.py
python experiments/adaptive_h/run_exp1_baseline.py --n_macro 50
python experiments/adaptive_h/run_exp2_oracle.py --n_macro 50
python experiments/adaptive_h/run_exp3_csweep.py --n_macro 50
python experiments/adaptive_h/run_exp4_sample_sd_nw.py --n_macro 50
python experiments/adaptive_h/summarize_exp4_sample_sd_nw.py
```

The final sample-SD/NW paper run is **not yet implemented end-to-end**. Its
acceptance requirements are in
`experiments/adaptive_h/final_benchmark_spec.md` and
`analysis/CLAIM_EVIDENCE_MAP.md`. Do not label the historical Exp4 output as
that final run.

## Supporting Experiments

Representative entrypoints:

```bash
python experiments/conditional_coverage/run_consistency.py --help
python experiments/design/run_saturation_sweep.py --help
python experiments/nongauss/run_nongauss_compare.py --help
python experiments/gibbs_compare/run_gibbs_compare.py --help
python experiments/onesided/run_onesided_compare.py --help
```

Read `EXPERIMENT_INDEX.md` and each experiment's README/spec before launching a
large run. Several supporting outputs are pilots or older-protocol evidence.

## R Benchmarks

```bash
Rscript -e "parse(file='benchmarks/dcp/dcp_methods.R')"
Rscript -e "parse(file='benchmarks/dcp/run_one_case.R')"
Rscript benchmarks/dcp/run_one_case.R DATA_DIR OUTPUT.csv 0.1 500
```

The last command requires compatible exported case data and installed R
packages. Gibbs RLCP results depend on the archived third-party reproduction
described in `_archive/04_external_reproductions/rlcp/ARCHIVE.md`.

## Manuscript Assets and Build

The manuscript must consume `manuscript/generated/`, not experiment output
folders directly.

```bash
python tools/export_manuscript_assets.py --dry-run
python tools/export_manuscript_assets.py
python tools/export_manuscript_assets.py --check
```

Then, if a LaTeX installation is available:

```bash
cd manuscript/journal_scale_adaptive
latexmk -pdf target_aware_scale_adaptive_ckme_cp.tex
```

The generated manifest records the source, modification time, byte size, and
SHA-256 hash for each allowlisted asset.

## Archive Integrity

The archive manifest intentionally inventories all files but hashes only
control/source/compact-result files.

```bash
shasum -a 256 -c _archive/manifests/SHA256SUMS
```

Regenerate manifests only after an intentional archive change:

```bash
bash _archive/manifests/build_manifests.sh
```

## Paper-Ready Evidence Rule

A number may enter the manuscript or an advisor-ready update only when:

1. its experiment protocol and source revision are recorded;
2. the run has the required macroreplications (normally at least 50);
3. summary and per-point QA pass;
4. its claim and limitation appear in `analysis/CLAIM_EVIDENCE_MAP.md`;
5. the asset is exported with provenance rather than linked to a temporary
   output directory.
