# Legacy adaptive-bandwidth entry points

This directory is compatibility-only. Active code, configuration, and outputs
belong under `experiments/adaptive_h/`; do not add new experiment logic here.

| Legacy command | Canonical module |
|---|---|
| `python exp_adaptive_h/pretrain_params.py` | `python -m experiments.adaptive_h.pretrain_params` |
| `python exp_adaptive_h/run_exp4_plugin.py` | `python -m experiments.adaptive_h.run_exp4_sample_sd_nw` |
| `python exp_adaptive_h/summarize_exp4.py` | `python -m experiments.adaptive_h.summarize_exp4_sample_sd_nw` |
| `python exp_adaptive_h/plot_exp4a.py` | `python -m experiments.adaptive_h.plot_exp4_gaussian_gap` |
| `python exp_adaptive_h/plot_exp4b.py` | `python -m experiments.adaptive_h.plot_exp4_score_homogeneity` |

The old “plugin” filename now forwards to the sample-standard-deviation plus
Nadaraya–Watson implementation. The retired IQR-based response-scale plug-in is
not part of the active workflow.
