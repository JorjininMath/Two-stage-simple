# exp3_test: Compare CKME, DCP-DR, hetGP (exp_test or exp3)

**exp_test**: Branin–Hoo (2D), fixed Gaussian noise. **exp3**: same mean, Student-t noise. See `Two_stage/sim_functions/`.

## What it does

- **Stage1** and **Stage2** sizes come from **config.txt**: `n_0`, `r_0`, `n_1`, `r_1` (and `simulator_func`, e.g. exp_test).
- One Stage2 case per run (design method from `--method`: lhs / sampling / mixed).
- For each macrorep: **CKME** (Python), **DCP-DR** and **hetGP** (R); mean **coverage**, **width**, **interval score** over test points.
- Averages over `--n_macro` macroreps and writes a summary table.

## Requirements

- Python: CKME, Two_stage, numpy, pandas.
- R: `quantreg`, `hetGP`; script uses `exp_stage2_impact_arc/run_benchmarks_one_case.R` and project root `dcp_r.R`.

## Run

From project root:

```bash
# Default: config.txt (exp_test, n_0=100, r_0=10, n_1=200, r_1=5), 5 macroreps, method=lhs
python exp3_test/run_exp3_compare.py

# Fewer macroreps, different design method
python exp3_test/run_exp3_compare.py --n_macro 2 --method sampling

# Custom config and output
python exp3_test/run_exp3_compare.py --config exp3_test/config.txt --output_dir exp3_test/output
```

## Output

- **exp3_test/output/exp3_compare_summary.csv**: columns `method`, `mean_coverage`, `sd_coverage`, `mean_width`, `sd_width`, `mean_interval_score`, `sd_interval_score`, `n_macroreps`.
- Per-macrorep data under `output/macrorep_<k>/case_<name>/` (CSVs and benchmarks.csv for R).
