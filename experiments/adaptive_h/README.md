# Adaptive-Bandwidth Experiment

This directory contains two distinct layers:

1. **Canonical final benchmark**: the locked iid-calibrated comparison in
   `final_benchmark_spec.md`, implemented by
   `run_final_adaptive_h_benchmark.py`.
2. **Runnable historical mechanism experiments**: Exp1--Exp4 scripts and their
   existing ignored outputs. Exp1--Exp3 support the oracle mechanism only;
   Exp4 is a pre-protocol sample-SD/NW result and is provenance, not final
   paper evidence.

The only active implementable response-scale estimator is
`sample_sd_nw_scale.py`: per-site sample standard deviation followed by
Nadaraya-Watson smoothing. The IQR response-scale branch and its outputs are
retired under
`_archive/01_experiments/diagnostics/adaptive_h_iqr_response_scale/`.

Canonical final commands:

```bash
python experiments/adaptive_h/run_final_adaptive_h_benchmark.py \
    --n-workers 4 --executor thread
python experiments/adaptive_h/summarize_final_adaptive_h_benchmark.py
python experiments/adaptive_h/analyze_final_adaptive_h_scores.py
python experiments/adaptive_h/plot_final_adaptive_h_results.py
python experiments/adaptive_h/qa_final_adaptive_h_benchmark.py --require-final
```

The runner uses no \(S^0\) score. Stage~1 uses a fixed grid; calibration and
test inputs are iid from the same target law with one output per input. The
local ignored output contains 600 completed jobs and 1,800 per-arm point files.
Stable compact evidence is exported to `analysis/adaptive_h/` and
`manuscript/generated/`.

Use `analysis/CURRENT_RESULTS.md` before sharing numerical conclusions. The
final result is mixed: the plug-in increasingly tracks the scale and moves the
score diagnostic toward oracle on the raised-floor DGPs, but it does not
uniformly improve interval score or groupwise coverage. Raw score-set and
projected-interval coverage must be reported separately; M/M/1 fixed is the
material disagreement case.
