# External Benchmarks

Benchmark implementations are separated from the CKME estimator and paper
experiments so their dependencies and provenance are visible.

| Directory | Methods | Called by |
|---|---|---|
| `dcp/` | DCP-DR, DCP-QR, and hetGP | `experiments/nongauss/` |

The RLCP third-party reproduction is retained under
`_archive/04_external_reproductions/rlcp/`; the Gibbs comparison reads its
saved results but does not treat the archived repository as active project
code.
