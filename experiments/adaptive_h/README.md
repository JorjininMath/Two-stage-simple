# Adaptive-Bandwidth Experiment

This directory contains two distinct layers:

1. **Runnable historical mechanism experiments**: Exp1--Exp4 scripts and their
   existing ignored outputs. Exp1--Exp3 support the oracle mechanism only;
   Exp4 is a pre-protocol sample-SD/NW result and is provenance, not final
   paper evidence.
2. **Planned final benchmark**:
   `final_benchmark_spec.md`. Its final DGP names, budgets, manifest,
   and diagnostic fields are not all implemented yet.

The only active implementable response-scale estimator is
`sample_sd_nw_scale.py`: per-site sample standard deviation followed by
Nadaraya-Watson smoothing. The IQR response-scale branch and its outputs are
retired under
`_archive/01_experiments/diagnostics/adaptive_h_iqr_response_scale/`.

Current runnable Exp4 commands:

```bash
python experiments/adaptive_h/run_exp4_sample_sd_nw.py --n_macro 50
python experiments/adaptive_h/summarize_exp4_sample_sd_nw.py
python experiments/adaptive_h/plot_exp4_gaussian_gap.py
python experiments/adaptive_h/plot_exp4_score_homogeneity.py --simulator all
```

Use `analysis/CURRENT_RESULTS.md` before sharing numerical conclusions. The
next paper-facing run is the protocol-aligned final benchmark described in the
planned specification; it has not been completed yet.
