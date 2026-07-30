# Project Archive

This directory preserves retired CKME research material that remains useful for
provenance, comparison, or future reuse. Archived material is not part of the
active paper workflow and must not be cited as current evidence without a new
review.

Nothing in this reorganization was permanently deleted. Whole directory trees
were moved intact so that historical outputs stay beside the code and notes
that produced them.

## Directory map

| Directory | Purpose |
|---|---|
| `01_experiments/` | Retired experiments, grouped by research role |
| `02_legacy_code/` | Superseded core implementations and early experiment code |
| `03_writing_derivatives/` | Dissertation, defense, and other derived writing packages |
| `04_external_reproductions/` | Third-party code and local reproductions |
| `05_project_history/` | Superseded project-level plans and snapshots |
| `90_quarantine/` | Recoverable temporary, duplicate, or not-yet-classified material |
| `manifests/` | Search and integrity manifests for the archive |

Start with [`INDEX.md`](INDEX.md) for a human-readable inventory or
[`CATALOG.tsv`](CATALOG.tsv) for filtering in a spreadsheet or script. Every
archived item has an `ARCHIVE.md` record with its original path, status,
replacement, known issues, key files, and reopening instructions.

## Finding material

From the repository root:

```bash
rg -i "allocation|score homogeneity|RLCP" \
  _archive/INDEX.md _archive/CATALOG.tsv _archive/**/ARCHIVE.md
```

For a complete file-level search, use `_archive/manifests/FILES.tsv` or `rg`
directly under `_archive/`.

## Reopening an archived item

1. Read the item's `ARCHIVE.md` and identify its active replacement.
2. Check whether relative paths or imports were broken by archival relocation.
3. Copy the item to a new active location; do not develop inside `_archive/`.
4. Revalidate the scientific protocol, dependencies, and data provenance.
5. Give the reopened work a new experiment/run identifier and document which
   archived item it came from.

Archive contents are reference-only. `90_quarantine/` is also non-destructive:
files placed there remain recoverable until an explicit, separately reviewed
deletion decision is made.
