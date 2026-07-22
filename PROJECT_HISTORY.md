# Project History

This file is the durable, date-ordered record of completed project-level work.
The short operational view stays in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).
When completed items age out of that page, move them here without removing the
strikethrough or completion date.

Do not use this file as a scientific source of truth. Method changes are
controlled by [`PROTOCOL.md`](PROTOCOL.md) and correctness changes are explained
in [`CHANGELOG.md`](CHANGELOG.md).

## 2026 Q3

### Workspace, provenance, and communication

- [x] ~~Reorganized core code and experiments around `src/` and
  `experiments/`, preserving selected old command wrappers.~~ (2026-07-22)
- [x] ~~Created project status/history, results registry, claim-evidence map,
  research-update workflow, and advisor-feedback intake workflow.~~
  (2026-07-22)
- [x] ~~Created a searchable archive catalog plus file and checksum manifests;
  no research payload was permanently deleted.~~ (2026-07-22)
- [x] ~~Separated current versus archived paper snapshots and added stable,
  hash-verified manuscript asset exports.~~ (2026-07-22)

### Paper method and evidence boundary

- [x] ~~Removed the IQR-based plug-in from the active workflow and current
  paper-facing documentation.~~ (2026-07-22)
- [x] ~~Synchronized the active method description with the sample-SD plus
  Nadaraya--Watson plug-in.~~ (2026-07-21)
- [x] ~~Separated historical plug-in outputs from the current evidence set.~~
  (2026-07-21)

### Correctness and protocol lock

- [x] ~~Locked iid target-law calibration with one fresh response per
  calibration input.~~ (2026-07-17)
- [x] ~~Locked site-grouped cross-validation for replicated training data.~~
  (2026-07-17)
- [x] ~~Corrected the conformal-quantile edge case to return the whole space
  when the requested finite-sample level is unattainable.~~ (2026-07-17)
- [x] ~~Established the raw-score guarantee layer and projected-interval
  reporting layer.~~ (2026-07-17)
- [x] ~~Unified interval extraction through monotone projection and generalized
  inversion.~~ (2026-07-16)
