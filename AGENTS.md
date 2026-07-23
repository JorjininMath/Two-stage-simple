# AGENTS.md

Guidance for coding and research-assistant work in this repository.

## Language and Editing

- Use Chinese for explanations and summaries.
- Use English for code, comments, docstrings, filenames, commands, and commit
  messages.
- Prefer small, reviewable diffs and preserve unrelated local changes.
- Do not delete historical research material. Move it to the indexed archive
  with an `ARCHIVE.md` record.
- This repository is public: never add secrets, credentials, private
  correspondence, or machine-specific absolute paths.

## Scientific Source of Truth

Read these before changing an experiment or manuscript claim:

1. `PROJECT_STATUS.md` -- current milestone, bottleneck, and next action.
2. `PROTOCOL.md` -- locked estimator, calibration, and evaluation rules.
3. `EXPERIMENT_INDEX.md` -- active/supporting/archive boundary.
4. `analysis/CURRENT_RESULTS.md` -- checked current interpretation.
5. `analysis/CLAIM_EVIDENCE_MAP.md` -- claim-level evidence and exclusions.

Planning notes never override the protocol or verified analysis.

## Locked Method Rules

- Active adaptive response scale: per-site sample SD followed by
  Nadaraya-Watson smoothing (`experiments/adaptive_h/sample_sd_nw_scale.py`).
- The IQR response-scale plug-in is retired. Its artifacts are archive-only and
  must not enter active summaries, advisor updates, or manuscript assets.
- Guarantee-bearing calibration uses `run_stage2(method="iid", r_1=1)`.
  Design-selected/replicated Stage-2 modes are legacy reproduction modes.
- Replicated Stage-1 CV must keep all replications of a site in one fold.
- Raw point scores define coverage; projected CDF intervals define width and
  interval score.
- Paper-facing numerical results normally require at least 50 macroreplications
  and a run manifest.

## Python Layout and API

Core packages use a `src/` layout:

- `src/CKME/` -- CKME conditional CDF model, kernels, indicators, tuning.
- `src/CP/` -- calibration, scores, projected intervals, evaluation.
- `src/Two_stage/` -- Stage 1/Stage 2 orchestration and simulators.
- `src/project_support/` -- project-root and project-relative path helpers.

Public imports remain unchanged:

```python
from CKME import CKMEModel, Params, ParamGrid
from CP import CP
from Two_stage import run_stage1_train, run_stage2
```

Install with:

```bash
python -m pip install -e ".[experiments]"
```

Direct experiment scripts add the repository root and `src/` to `sys.path` so
documented project-root commands also work before editable installation.

## Experiment Layout

- `experiments/adaptive_h/` -- main mechanism line and planned final benchmark.
- `experiments/coverage_mechanism/` -- score and portability pilots.
- `experiments/framing_validation/` -- mechanism/epistemic diagnostics.
- `experiments/design/` -- allocation and saturation ablations.
- `experiments/conditional_coverage/` -- consistency diagnostics.
- `experiments/{nongauss,gibbs_compare,onesided}/` -- supporting comparisons.
- `experiments/mm1_feasibility/` -- separate non-conformal feasibility module.
- `experiments/stock/` -- local exploratory extension (ignored).

Each active experiment owns its config, code, and ignored output directories.
Use project-relative user paths and file-relative default config/output paths.
Do not write results to old root `exp_*` compatibility directories.

Current adaptive-h commands:

```bash
python experiments/adaptive_h/run_exp4_sample_sd_nw.py --n_macro 50
python experiments/adaptive_h/summarize_exp4_sample_sd_nw.py
python experiments/adaptive_h/plot_exp4_gaussian_gap.py
python experiments/adaptive_h/plot_exp4_score_homogeneity.py --simulator all
```

These are runnable historical mechanism scripts; the protocol-aligned final
run described in `final_benchmark_spec.md` is still pending.

## Tests and Fast Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m pytest experiments/mm1_feasibility/tests -q
python -m compileall -q src experiments tests tools
python tools/export_manuscript_assets.py --check
bash -n hpc/submit_all.sh
```

Do not launch a heavy experiment merely to validate a path change. Use imports,
`--help`, smoke tests, syntax checks, and existing artifacts first.

## R and HPC

- `benchmarks/dcp/dcp_methods.R` -- DCP helper functions.
- `benchmarks/dcp/run_one_case.R` -- single exported-case benchmark runner.
- `experiments/gibbs_compare/` -- experiment-level SLURM/RLCP scripts.
- `hpc/submit_all.sh` -- root-level SLURM dispatcher.

Run SLURM commands from the repository root and review cluster-specific
account/partition/email settings before submission.

## Results, Manuscript, and Sharing

- Run facts: `experiment_logs/`.
- Checked interpretation and evidence registry: `analysis/`.
- Stable paper assets: `manuscript/generated/`.
- Active paper: `manuscript/journal_scale_adaptive/`.
- Advisor/coauthor packages: `research_updates/`.
- Shareable PDF: `paper/current/`; superseded snapshots: `paper/archive/`.

The manuscript must read exported assets rather than temporary experiment
output paths. Refresh/check the explicit allowlist with:

```bash
python tools/export_manuscript_assets.py
python tools/export_manuscript_assets.py --check
```

Advisor-returned files belong in a private dated round under
`manuscript/advisor_feedback/rounds/`. Never edit the received file in place;
record its checksum and integrate changes into the active manuscript.

## Notes and Archive

`notes/` is local working memory organized into ideas, current/archived plans,
theory, decisions, meetings, literature, and legacy drafts. Use descriptive
filenames and date decisions/meeting notes.

`_archive/` is reference-only. Start with `_archive/INDEX.md` or
`_archive/CATALOG.tsv`; every item has reopening instructions in `ARCHIVE.md`.
Large local payloads are inventoried in `_archive/manifests/FILES.tsv`.

## Career OS Boundary

`PROJECT_STATUS.md` is authoritative. Preserve its `career-os:*` marker pairs
for a future one-way importer. Career OS may display the operational summary,
but it must not become the source of scientific claims or write them back into
this repository.
