# Experiment Index

This file records the active project boundary for the current journal-scale
CKME-DCP work. It is intentionally narrower than the full repository history.

## Main paper line

### `exp_adaptive_h/`

Primary workflow for the current target-aware, scale-adaptive CKME-DCP paper.
It uses the existing scalar-h core pipeline and evaluates adaptive `h(x)` at the
experiment layer.

Key commands:

```bash
python exp_adaptive_h/pretrain_params.py
python exp_adaptive_h/run_exp4_plugin.py --n_macro 50
python exp_adaptive_h/summarize_exp4.py
python exp_adaptive_h/plot_exp4a.py
python exp_adaptive_h/plot_exp4b.py --simulator all
```

The WSC-style Gaussian DGP is kept here as `wsc_gauss`; the old WSC reproduction
runner is no longer an active experiment.

## Supporting evidence

### `exp_design/`

Design-comparison and adaptive-allocation evidence. Use this to discuss the role
of the Stage 2 site-selection score, not as the main adaptive-bandwidth result.

### `exp_nongauss/`

Non-Gaussian benchmark evidence against R baselines. This folder still reflects
the broader historical six-DGP plan, while the current simulator registry keeps
the active Student-t A1 variants. Refresh this folder before treating it as a
fully reproducible public workflow.

### `exp_gibbs_compare/`

Gibbs/RLCP comparison workflow. Useful as supporting context for heteroscedastic
coverage behavior.

### `exp_conditional_coverage/`

Consistency and diagnostics. The nested `_archive_old/` folder is legacy
material and should not be part of the current paper narrative.

### `exp_onesided/`

Quantile-estimation diagnostics and one-sided score experiments. Treat as
diagnostic support rather than the current main paper path.

## Exploratory or local-only

### `exp_stock/` (local-only, ignored)

Exploratory real-data extension using archived returns data. It depends on
local archived inputs and should not be presented as a public reproducibility
path without a separate data/benchmark cleanup.

### `ckme_dcp_mm1/`

Feasibility-only KME/CKME input-uncertainty module for M/M/1 queueing outputs.
It does not run conformal prediction and should not be used as evidence for the
main CKME-DCP coverage claims.

## Archived or legacy

### `_archive/exp_wsc_2026/`

Historical WSC 2026 reproduction scripts, config, table formatter, and partial
local outputs. The `wsc_gauss` DGP remains registered in `Two_stage/` and is
used by `exp_adaptive_h/`, but the WSC reproduction runner is no longer active.

### `_archive/` and nested old experiment folders

Superseded implementations, old reports, local outputs, and diagnostic trials.
Do not use these folders as public-facing entrypoints unless they are refreshed
and explicitly re-promoted.
