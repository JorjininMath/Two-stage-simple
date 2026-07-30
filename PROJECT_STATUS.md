# Project Status

Last updated: 2026-07-23

This is the one-page operational update for the CKME two-stage project. It
records the current scientific boundary, the next actions, and recent completed
work. Detailed plans belong in `notes/`; run-level facts belong in
`experiment_logs/`; evidence interpretations belong in `analysis/`.

Keep the `career-os` comments unchanged. A future importer may update only the
text between each matching `start` and `end` marker.

## Quick Update

<!-- career-os:latestResult:start -->
- **Latest result:** The 50-macroreplication final adaptive-h benchmark passed
  QA. The sample-SD plus Nadaraya--Watson plug-in increasingly tracks the scale
  and approaches oracle score behavior on the two raised-floor DGPs, but its
  interval-score and groupwise gains are DGP-dependent. Projected-interval
  coverage must be reported separately from raw score-set coverage.
<!-- career-os:latestResult:end -->

<!-- career-os:bottleneck:start -->
- **Bottleneck:** The primary evidence gap is closed. The remaining paper-level
  choices are global-multiplier/scale-smoother refinement, projection-aware
  interval construction, a null reference for the KS diagnostic, and the
  minimal refreshed external baseline set.
<!-- career-os:bottleneck:end -->

<!-- career-os:nextAction:start -->
- **Next action:** Review the advisor-ready adaptive-h draft and decide whether
  to refine the plug-in before adding external baselines, or present the mixed
  final result as the method's current empirical boundary.
<!-- career-os:nextAction:end -->

<!-- career-os:advisorAsk:start -->
- **Advisor ask:** After the protocol-aligned adaptive-bandwidth evidence is
  packaged, confirm the minimal external baseline set and which supporting
  experiments belong in the main paper versus the appendix.
<!-- career-os:advisorAsk:end -->

<!-- career-os:nextDecision:start -->
- **Next decision:** Choose the smallest refreshed external comparison that is
  sufficient for the journal paper after the primary adaptive-bandwidth result
  passes its evidence checks.
<!-- career-os:nextDecision:end -->

## Current Milestone

Use the completed, traceable adaptive-bandwidth evidence package to settle the
paper's empirical claim and the smallest next comparison/refinement scope.

## Locked Scientific Boundary

- Adaptive bandwidth is `h(x) = c * s_hat(x)`, where `s_hat(x)` is the per-site
  sample standard deviation smoothed by Nadaraya--Watson regression.
- Calibration inputs are iid from the declared target law, with `r_1 = 1`.
- Raw point-evaluated scores define the coverage guarantee; monotone-projected
  intervals define width and interval-score reporting.
- Paper-facing numbers require at least 50 macroreplications unless explicitly
  labeled as a pilot.
- The IQR-based plug-in is historical only and is excluded from active code,
  current evidence summaries, advisor updates, and the manuscript.

The controlling method document is [`PROTOCOL.md`](PROTOCOL.md). If this status
page conflicts with the protocol or an active experiment `spec.md`, the protocol
and specification take precedence.

## Current Evidence

| Evidence item | Status | Current use |
| --- | --- | --- |
| Fixed versus oracle scale normalization | Diagnostic | Mechanism evidence only |
| Oracle multiplier sensitivity | Diagnostic | Default-value check only |
| Sample-SD plus NW plug-in under the locked protocol | Complete; mixed performance | Main implementable method |
| Raw-score homogeneity | Complete | Qualified mechanism evidence |
| iid target-law calibration check | Complete | Validity audit |
| Current shareable paper snapshot | Ready: `paper/current/CKME_Adaptive_H.pdf` | Advisor review |

## Active Tasks

- [x] ~~**P1:** Run the final sample-SD plus Nadaraya--Watson
  adaptive-bandwidth experiment under the locked protocol.~~ (2026-07-23)
- [x] ~~**P1:** Save and audit raw conformity scores for the homogeneity
  analysis.~~ (2026-07-23)
- [x] ~~**P1:** Verify iid target-law calibration with `r_1 = 1` in the final
  workflow.~~ (2026-07-23)
- [x] ~~**P2:** Create a result manifest linking commands, settings, summaries,
  tables, figures, and QA.~~ (2026-07-23)
- [x] ~~**P2:** Produce the first dated advisor update from checked evidence.~~
  (2026-07-23)
- [ ] **P3:** Refresh only the external baselines selected for the paper.

## Waiting / Blocked

- [ ] External-baseline scope -- decide after the main adaptive-bandwidth
  evidence is checked.
- [ ] Main-text versus appendix routing for supporting experiments -- confirm
  during the next paper-structure review.

## Recently Completed

- [x] ~~Implemented and tested the no-\(S^0\), iid target-law final
  adaptive-bandwidth workflow, including the iid test-data bug fix and three
  registered final DGPs.~~ (2026-07-23)
- [x] ~~Completed 50 paired macroreplications over three DGPs and four budgets,
  with raw-score, scale, interval, seed, and manifest diagnostics.~~
  (2026-07-23)
- [x] ~~Completed score analysis, publication figures, final QA, and
  hash-checked manuscript asset export.~~ (2026-07-23)
- [x] ~~Updated the journal draft and final experiment report with the
  qualified mixed plug-in result.~~ (2026-07-23)
- [x] ~~Built and visually verified the unversioned advisor-ready paper
  snapshot under `paper/current/`.~~ (2026-07-23)
- [x] ~~Reorganized core code, experiments, notes, results, manuscript assets,
  paper snapshots, and archive into role-based directories with stable paths.~~
  (2026-07-22)
- [x] ~~Added a checked result registry, claim-to-evidence map, and
  hash-verified manuscript asset exporter.~~ (2026-07-22)
- [x] ~~Moved the IQR response-scale branch and approximately 877 MB of related
  outputs/diagnostics into an archive-only provenance record.~~ (2026-07-22)
- [x] ~~Added private workflows for advisor-returned versions and dated
  advisor/coauthor updates.~~ (2026-07-22)
- [x] ~~Removed the IQR-based plug-in from the active workflow and current
  paper-facing documentation.~~ (2026-07-22)
- [x] ~~Synchronized the active method description with the sample-SD plus
  Nadaraya--Watson plug-in.~~ (2026-07-21)
- [x] ~~Separated historical plug-in outputs from the current evidence set.~~
  (2026-07-21)
- [x] ~~Locked iid target-law calibration, grouped CV, conformal-quantile edge
  handling, and the two-layer evaluation protocol.~~ (2026-07-17)
- [x] ~~Unified interval extraction through monotone projection and generalized
  inversion.~~ (2026-07-16)

When this list exceeds roughly 15--20 items, retain only the newest entries here
and move the older entries to [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md).

## Decisions Needed

- [ ] Which external baselines are essential for the main paper?
- [ ] Which design and conditional-coverage experiments belong in the appendix?
- [ ] What evidence threshold should trigger creation of the next shareable PDF?

## Career OS Connection

This file is the authoritative CKME operational summary. Career OS currently
tracks this work as `wk_ckme_ext`, but synchronization remains manual. A future
one-way importer may read only the five `career-os:*` marker blocks near the top
of this file. Career OS must not overwrite protocol, evidence, or manuscript
content in this repository. `python tools/export_project_status.py` exposes the
marker blocks as JSON when the Career OS importer is ready.

## Key Links

- Scientific protocol: [`PROTOCOL.md`](PROTOCOL.md)
- Active experiment boundary: [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md)
- Main adaptive-bandwidth specification:
  [`experiments/adaptive_h/final_benchmark_spec.md`](experiments/adaptive_h/final_benchmark_spec.md)
- Current results: [`analysis/CURRENT_RESULTS.md`](analysis/CURRENT_RESULTS.md)
- Claim-to-evidence map:
  [`analysis/CLAIM_EVIDENCE_MAP.md`](analysis/CLAIM_EVIDENCE_MAP.md)
- Current paper plan:
  [`notes/planning/current/paper/`](notes/planning/current/paper/)
- Active manuscript:
  [`manuscript/journal_scale_adaptive/`](manuscript/journal_scale_adaptive/)
- Paper snapshot status: [`paper/README.md`](paper/README.md)
- Advisor-update workspace: [`research_updates/`](research_updates/)
