# Archive Manifests

`FILES.tsv` is a complete, repository-relative file inventory generated after
the 2026-07-22 reorganization. It records path, byte size, and modification
timestamp for each archived file.

`SHA256SUMS` covers archive control records and human-authored source,
configuration, note, and compact tabular-result files. Very large generated
output trees are represented in `FILES.tsv` rather than fully hashed; in
particular, `exp_allocation/output/` contains more than eleven thousand files.
This keeps routine archive validation fast while preserving a searchable record
of every file.

If a large archived experiment is reopened, create a dedicated full checksum
manifest for that item before moving or modifying it.
