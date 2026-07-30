# Two-Stage CKME-DCP Research Project

This repository develops conditional distribution estimation and prediction
intervals using Conditional Kernel Mean Embedding (CKME) and split conformal
prediction. The active paper studies response-scale-adaptive indicator
bandwidths while keeping the guarantee-bearing calibration sample iid from the
target input law.

## Start Here

| Question | File |
|---|---|
| What is the project doing now? | [`PROJECT_STATUS.md`](PROJECT_STATUS.md) |
| Which numerical conclusions are safe to share? | [`analysis/CURRENT_RESULTS.md`](analysis/CURRENT_RESULTS.md) |
| Which claim is supported by which file? | [`analysis/CLAIM_EVIDENCE_MAP.md`](analysis/CLAIM_EVIDENCE_MAP.md) |
| What estimator/evaluation rules are locked? | [`PROTOCOL.md`](PROTOCOL.md) |
| Which experiments are active, supporting, or historical? | [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md) |
| How do I prepare an advisor update? | [`research_updates/README.md`](research_updates/README.md) |
| How do I reproduce or validate code? | [`REPRODUCE.md`](REPRODUCE.md) |

The final 50-macroreplication benchmark is complete. It supports the
**oracle scale-normalization mechanism** on the raised-floor Gaussian and
Student-\(t_3\) DGPs and shows that the implementable sample-SD plus
Nadaraya--Watson estimator increasingly tracks the local scale pattern. Its
interval-score and groupwise gains are DGP-dependent. In addition, the
projected M/M/1 fixed interval has 0.855 coverage despite 0.899 raw score-set
coverage, so projected intervals are audited separately from the
guarantee-bearing score sets. The current claim is deliberately not one of
uniform dominance. The former IQR response-scale plug-in is retired and exists
only in the searchable archive.

## Locked Method Boundary

- Stage 1 fits a CKME conditional CDF model, with site-grouped CV when training
  data contain replications.
- The active scale estimator computes a sample standard deviation at each
  replicated Stage-1 site, then smooths those values by Nadaraya-Watson
  regression: `experiments/adaptive_h/sample_sd_nw_scale.py`.
- Guarantee-bearing Stage-2 calibration uses iid inputs from the target law and
  exactly one fresh response at each input (`method="iid"`, `r_1=1`).
- Raw point-evaluated conformity scores define coverage. Monotone-projected CDF
  intervals define separately audited interval coverage, width, and interval
  score; they do not inherit the raw score-set guarantee.
- Design-selected/replicated Stage-2 modes remain available only for older
  experiment reproduction and emit a warning.

See [`PROTOCOL.md`](PROTOCOL.md) for the precise rules and edge cases.

## Installation

Python 3.10 or later is required.

```bash
git clone https://github.com/JorjininMath/Two-stage-simple.git
cd Two-stage-simple

# Core API plus experiment dependencies, installed from the src layout.
python -m pip install -e ".[experiments]"
```

The Conda environment remains available as an alternative:

```bash
conda env create -f environment.yml
conda activate ckme_env
python -m pip install -e .
```

Optional R benchmarks require `hetGP` and `quantreg`. Their implementation and
runner are isolated in [`benchmarks/dcp/`](benchmarks/dcp/).

## Minimal API Example

```python
from CKME.parameters import Params
from Two_stage import run_stage1_train, run_stage2

stage1 = run_stage1_train(
    n_0=40,
    r_0=5,
    simulator_func="exp1",
    params=Params(ell_x=0.5, lam=0.01, h=0.1),
    random_state=42,
)

stage2 = run_stage2(
    stage1_result=stage1,
    X_cand=None,          # not used by iid calibration
    n_1=200,
    r_1=1,
    simulator_func="exp1",
    method="iid",
    alpha=0.1,
    random_state=43,
)

lower, upper = stage2.predict_interval([[0.4], [0.7]])
```

Public imports remain `CKME`, `CP`, and `Two_stage` even though their source
now lives under `src/`.

## Main Adaptive-Bandwidth Workflow

The canonical workflow uses no \(S^0\) score. It trains on a fixed grid and
uses fresh iid calibration and test inputs from the same target law.

```bash
python experiments/adaptive_h/pretrain_params.py \
    --simulators mm1_sojourn,raised_floor_gauss,raised_floor_t3 \
    --out experiments/adaptive_h/pretrained_params_final.json
python experiments/adaptive_h/run_final_adaptive_h_benchmark.py \
    --n-workers 4 --executor thread
python experiments/adaptive_h/summarize_final_adaptive_h_benchmark.py
python experiments/adaptive_h/analyze_final_adaptive_h_scores.py
python experiments/adaptive_h/plot_final_adaptive_h_results.py
python experiments/adaptive_h/qa_final_adaptive_h_benchmark.py --require-final
python tools/export_manuscript_assets.py
python tools/export_manuscript_assets.py --check
```

The completed workflow is controlled by
[`experiments/adaptive_h/final_benchmark_spec.md`](experiments/adaptive_h/final_benchmark_spec.md).
The final local raw output is ignored by Git; compact QA-checked evidence is
exported under [`analysis/adaptive_h/`](analysis/adaptive_h/) and
[`manuscript/generated/`](manuscript/generated/).

Historical Exp1--Exp4 commands and old root wrappers remain provenance-only.
New scripts, configs, and outputs belong only under `experiments/`.

## Research-to-Paper Workflow

| Stage | Location | Rule |
|---|---|---|
| Capture an idea/question | `notes/inbox/` | Local working memory; not evidence |
| Develop a plan or theory | `notes/planning/`, `notes/theory/` | Must not override the protocol |
| Record a decision | `notes/decisions/` | Use a dated, descriptive filename |
| Define/run an experiment | `experiments/<topic>/` | Keep config/spec/code together |
| Record a run | `experiment_logs/` | Command, seed, settings, outputs, next action |
| Verify and interpret | `analysis/` | Separate checked results from pilots/history |
| Export paper assets | `manuscript/generated/` | Use the explicit hash-checked exporter |
| Share an update | `research_updates/` | Include evidence status and limitations |
| Write the paper | `manuscript/` | Read only stable generated assets |
| Preserve retired work | `_archive/` | Index it; do not develop in place |

Export or check the current paper-facing asset allowlist with:

```bash
python tools/export_manuscript_assets.py
python tools/export_manuscript_assets.py --check
```

## Project Structure

```text
Two-stage-simple/
├── PROJECT_STATUS.md             # one-page operational update
├── PROJECT_HISTORY.md            # dated completed work
├── PROTOCOL.md                   # locked estimator/evaluation rules
├── EXPERIMENT_INDEX.md           # active/supporting/archive boundary
├── REPRODUCE.md                  # validation and reproduction commands
├── src/
│   ├── CKME/                     # conditional CDF estimator
│   ├── CP/                       # conformal calibration and intervals
│   ├── Two_stage/                # Stage 1/Stage 2 orchestration
│   └── project_support/          # stable project-relative path helpers
├── experiments/
│   ├── adaptive_h/               # main method, final benchmark, and historical diagnostics
│   ├── coverage_mechanism/       # score/mechanism pilots
│   ├── framing_validation/       # epistemic/aleatoric diagnostics
│   ├── design/                   # design and allocation ablations
│   ├── conditional_coverage/     # consistency diagnostics
│   ├── nongauss/                 # R benchmark comparison
│   ├── gibbs_compare/            # Gibbs/RLCP supporting comparison
│   ├── onesided/                 # quantile and one-sided diagnostics
│   ├── mm1_feasibility/          # separate input-uncertainty feasibility
│   └── stock/                    # local exploratory extension (ignored)
├── analysis/                     # result registry and claim-evidence map
├── experiment_logs/              # local run records and templates
├── notes/                        # local ideas/plans/theory/decisions
├── manuscript/
│   ├── journal_scale_adaptive/   # active LaTeX manuscript
│   ├── reports/                  # paper-facing experiment reports
│   ├── generated/                # exported tables/figures with provenance
│   └── advisor_feedback/         # private returned-version intake workflow
├── research_updates/             # advisor/coauthor update templates/packages
├── paper/
│   ├── current/                  # at most one approved shareable PDF
│   └── archive/YYYY-MM/          # superseded shared snapshots
├── benchmarks/                   # external benchmark implementations
├── hpc/                          # top-level SLURM dispatcher
├── tools/                        # asset/provenance utilities
├── tests/                        # core import/path/pipeline smoke tests
└── _archive/                     # indexed retired work and manifests
```

The old top-level `exp_*` directories contain compatibility wrappers only and
are not sources of experiment code or results.

## Advisor-Returned Versions

Put each returned manuscript in a private dated round:

```text
manuscript/advisor_feedback/rounds/YYYY-MM-DD-R01-short-label/
```

Keep the returned file unchanged under `received/`, record its checksum and the
exact sent snapshot in `manifest.yaml`, and integrate accepted edits into the
active manuscript using the checklist/log. See
[`manuscript/advisor_feedback/README.md`](manuscript/advisor_feedback/README.md).

## Archive and Career OS

Use [`_archive/INDEX.md`](_archive/INDEX.md) for a human-readable inventory and
[`_archive/CATALOG.tsv`](_archive/CATALOG.tsv) for filtering. Every archived
item has an `ARCHIVE.md`; large payloads remain local but are listed in the file
manifest. Nothing was permanently deleted in the 2026-07-22 reorganization.

`PROJECT_STATUS.md` is authoritative for this project. Its stable
`career-os:*` markers are designed for a future one-way importer into Career
OS. Until that importer exists, update Career OS manually; Career OS should not
write scientific claims back into this repository.

The machine-readable handoff is already available without changing Career OS:

```bash
python tools/export_project_status.py --check
python tools/export_project_status.py
```

## Citation

```bibtex
@misc{ckme_two_stage,
  author = {Jin Zhao},
  title  = {Two-Stage Conditional Kernel Mean Embedding and Conformal Prediction},
  year   = {2026},
  url    = {https://github.com/JorjininMath/Two-stage-simple}
}
```

The citation will be replaced by the journal record when available. This
project is licensed under Apache 2.0; see [`LICENSE`](LICENSE).
