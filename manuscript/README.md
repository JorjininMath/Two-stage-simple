# Manuscript And Formal Reports

This folder is for polished writing artifacts, not daily experiment logs.

Current layout:

- `journal_scale_adaptive/`: active journal-draft material.
- `reports/`: formal experiment reports and paper-facing summaries.
- `generated/`: allowlisted tables/figures copied from checked result sources;
  see `adaptive_h_assets_manifest.json` for hashes and provenance.
- `advisor_feedback/`: private intake/checklist workflow for returned versions.

Daily experiment records should go to `experiment_logs/`. Post-hoc analysis
maps and diagnostic summaries should go to `analysis/`. Generated experiment
outputs should remain beside their experiment scripts in `output_*` folders.
The active manuscript should read stable copies from `generated/`, refreshed by
`python tools/export_manuscript_assets.py`, rather than reading output folders
directly.
