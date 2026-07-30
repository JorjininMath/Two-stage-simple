# DCP and hetGP Benchmark

- `dcp_methods.R` contains the DCP helper functions.
- `run_one_case.R` reads one exported Python experiment case and writes
  per-point benchmark intervals.

From the repository root:

```bash
Rscript benchmarks/dcp/run_one_case.R DATA_DIR OUTPUT.csv 0.1 500
```

R packages: `hetGP` and `quantreg` (plus their dependencies). The runner is
path-stable because it sources `dcp_methods.R` relative to its own file.
