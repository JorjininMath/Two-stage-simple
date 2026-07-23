# Project Status

Last updated: 2026-07-22

This is the one-page operational update for the CKME two-stage project. It
records the current scientific boundary, the next actions, and recent completed
work. Detailed plans belong in `notes/`; run-level facts belong in
`experiment_logs/`; evidence interpretations belong in `analysis/`.

Keep the `career-os` comments unchanged. A future importer may update only the
text between each matching `start` and `end` marker.

## Quick Update

<!-- career-os:latestResult:start -->
- **Latest result:** Existing fixed-versus-oracle adaptive-bandwidth evidence is
  diagnostic only. It qualitatively supports scale normalization, but the
  protocol-aligned sample-SD plus Nadaraya--Watson plug-in still needs its final
  run.
<!-- career-os:latestResult:end -->

<!-- career-os:bottleneck:start -->
- **Bottleneck:** The current implementable plug-in, raw-score homogeneity, and
  iid target-law calibration have not yet been verified together in a final
  paper-scale run.
<!-- career-os:bottleneck:end -->

<!-- career-os:nextAction:start -->
- **Next action:** Run the locked sample-SD plus Nadaraya--Watson workflow with
  iid target-law calibration, save the required raw scores, and complete its QA
  before promoting any numerical claim.
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

Complete the protocol-aligned adaptive-bandwidth experiment and produce one
advisor-ready evidence package whose claims, figures, summaries, run settings,
and source files are traceable.

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
| Sample-SD plus NW plug-in under the locked protocol | Pending final run | Intended main implementable method |
| Raw-score homogeneity | Pending final run | Intended mechanism evidence |
| iid target-law calibration check | Pending final run | Required validity audit |
| Current shareable paper snapshot | Not available | Rebuild after method and evidence synchronization |

## Active Tasks

- [ ] **P1:** Run the final sample-SD plus Nadaraya--Watson adaptive-bandwidth
  experiment under the locked protocol.
- [ ] **P1:** Save and audit raw conformity scores for the homogeneity analysis.
- [ ] **P1:** Verify iid target-law calibration with `r_1 = 1` in the final
  workflow.
- [ ] **P2:** Create a result manifest linking commands, settings, summaries,
  tables, figures, and QA.
- [ ] **P2:** Produce the first dated advisor update from checked evidence.
- [ ] **P3:** Refresh only the external baselines selected for the paper.

## Waiting / Blocked

- [ ] External-baseline scope -- decide after the main adaptive-bandwidth
  evidence is checked.
- [ ] Main-text versus appendix routing for supporting experiments -- confirm
  during the next paper-structure review.

## Recently Completed

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
