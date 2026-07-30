# Advisor-Returned Manuscript Versions

This local/private workspace preserves manuscript files returned by an advisor
and records how each comment was integrated. The original returned file is
evidence and must never be edited in place.

## Layout

```text
advisor_feedback/
  templates/round/
    manifest.yaml
    received/
    action-checklist.md
    integration-log.md
  rounds/
    YYYY-MM-DD-R01-short-label/
```

Actual folders under `rounds/` should remain gitignored because returned drafts
and comments may contain private correspondence or unpublished material.

## Start A Feedback Round

1. Copy `templates/round/` to
   `rounds/YYYY-MM-DD-R01-short-label/`.
2. Put the untouched returned `.docx`, `.pdf`, `.tex`, or archive in
   `received/`.
3. Record the received file, the sent manuscript snapshot, and checksums in
   `manifest.yaml`.
4. Translate comments into `action-checklist.md`; do not silently omit rejected
   or deferred suggestions.
5. Apply accepted edits to the active manuscript, not to the returned file.
6. Record the integrating commit and manuscript location in
   `integration-log.md`.

If the advisor returns TeX source, diff it against the exact sent snapshot
before integration. Never overwrite the active manuscript with a returned file.
