# Research Updates

This folder turns checked project evidence into compact updates for an advisor,
coauthor, collaborator, or research talk. It is a communication layer, not the
scientific source of truth.

## Layout

```text
research_updates/
  templates/                  Tracked, reusable update templates
  current/                    Local/private working update
  archive/YYYY-MM-DD-topic/   Local/private frozen update packages
```

The `current/` and `archive/` contents should remain gitignored by default.
When an update is ready to share, create a self-contained dated package with
copied figures and tables rather than links to temporary output directories.

## Evidence Rule

Every result in an update must identify:

- its evidence status (`diagnostic`, `partial`, `checked`, `advisor-ready`, or
  `manuscript-ready`);
- the run or output location;
- the analysis note or summary supporting the interpretation;
- its limitation or remaining check.

Archived, superseded, or IQR-based plug-in results must not appear in a current
advisor update.

## Workflow

1. Copy `templates/advisor_update.md` into a dated folder under `current/`.
2. Fill the bottom line, changes, checked evidence, unsupported claims, and
   decisions needed.
3. Copy only the selected figures/tables into the package.
4. Add the relevant Git commit, protocol date, run IDs, and source paths.
5. After sharing, move the complete folder to `archive/YYYY-MM-DD-topic/` and
   do not edit it in place. Start a new update for subsequent work.
