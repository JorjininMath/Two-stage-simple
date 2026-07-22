# Results and Analysis Hub

This directory is the project-level index for verified numerical evidence. It
does not replace experiment outputs. Raw and summarized results remain beside
their runners under `experiments/`; this hub records what those files can and
cannot currently support.

## Start Here

| File | Purpose |
| --- | --- |
| `CURRENT_RESULTS.md` | Short, advisor-ready snapshot of the strongest current findings and the main evidence gap |
| `CLAIM_EVIDENCE_MAP.md` | Paper claims linked to exact source files, with safe wording and prohibited overclaims |
| `results_registry.yaml` | Machine-readable source of truth for evidence status, paths, run sizes, and refresh requirements |
| `adaptive_h/README.md` | Detailed adaptive-bandwidth analysis notes |

## Evidence Classes

- **Current paper snapshot:** an artifact is referenced by the active journal
  draft or its paper-facing report. This label does not imply that it satisfies
  the locked protocol. The May 2026 adaptive-h Exp1--Exp3 outputs are in this
  class and are mechanism evidence only.
- **Supporting or pilot:** useful evidence from a diagnostic, ablation, or
  feasibility study. It must be labeled with its actual macroreplication count
  and cannot silently become a main-paper result.
- **Legacy or provenance:** retained to explain earlier decisions or reproduce
  an older report. It is not a current claim source.
- **Missing refresh:** an intended claim has no protocol-aligned artifact set
  yet. The required output paths and acceptance conditions are recorded before
  the run is launched.

## Promotion Rule

An experiment can be promoted to final paper evidence only after all of the
following are true:

1. its estimator and calibration design match `PROTOCOL.md`, or every
   deviation is explicit in the experiment specification;
2. paper-facing numbers use at least 50 macroreplications;
3. a run manifest records settings, seeds, and source revision;
4. the summary can be traced to raw or per-point outputs;
5. the claim is updated in `results_registry.yaml` and
   `CLAIM_EVIDENCE_MAP.md`.

The retired IQR-based **response-scale** plug-in is excluded from current
evidence. Its archived artifacts may be consulted for provenance only. An IQR
used inside Silverman's rule for the *input-space* Nadaraya--Watson bandwidth
is a different calculation and does not change the response-scale estimator,
which remains per-site sample SD.

## Manuscript Assets

Only explicitly allowlisted files are copied into `manuscript/generated/`.
Run:

```bash
python tools/export_manuscript_assets.py
python tools/export_manuscript_assets.py --check
```

The export records source paths and SHA-256 hashes in
`manuscript/generated/adaptive_h_assets_manifest.json`. Exp4 and every retired
IQR response-scale artifact are intentionally absent from that allowlist.
