# Adaptive-h Analysis Map

This folder indexes the existing adaptive-bandwidth experiment results and
diagnostics without moving generated outputs.

## GPT Diagnostic Spec Coverage

The existing-output diagnostic pass is here:

- script: `exp_adaptive_h/diagnose_existing_adaptive_h.py`
- output root: `exp_adaptive_h/output_existing_diagnostics/`
- report: `exp_adaptive_h/output_existing_diagnostics/existing_diagnostic_report.md`

Coverage of the requested diagnostic spec:

| diagnostic item | status | current artifact |
| --- | --- | --- |
| marginal coverage near 0.9 | available | `existing_diag_summary.csv` |
| effective ratio `h(x)/s(x)` | available | `existing_diag_summary.csv`, `diag_effective_ratio_exp4_iqr_budget_max.png` |
| group/bin coverage deviation | available | `existing_diag_bin.csv`, `existing_diag_summary.csv` |
| interval width and interval score | available | `existing_diag_summary.csv`, `diag_width_to_scale_exp4_iqr_budget_max.png` |
| c-sweep behavior | available | `diag_csweep_exp3.png` |
| plug-in versus oracle | available | `existing_diag_scale_estimation.csv`, `existing_diag_red_flags.csv` |
| grid-boundary red flags | available | `existing_diag_red_flags.csv` |
| raw-score homogeneity KS/q90 | not available from current outputs | `existing_diag_score_homogeneity.csv` explains that raw scores were not saved |
| exact `R_s=1,2,4,8` trend | not tested by existing DGPs | would require a separate controlled simulator |

## Rasi Lab-style Mapping

| Rasi Lab category | Current path | Role | Action |
| --- | --- | --- | --- |
| Experiments | `exp_adaptive_h/` | runners, summarizers, plotters, generated `output_*` folders | keep in place |
| Daily experiment logs | `experiment_logs/exp_adaptive_h/` | local day-to-day run notes | use daily template |
| Analysis | `analysis/exp_adaptive_h/` | stable map of diagnostics and interpretation-ready artifacts | keep tracked |
| Manuscript/report | `manuscript/reports/`, `manuscript/journal_scale_adaptive/` | formal paper-facing writeups | update only after diagnostics are stable |

## Current Interpretation

The existing examples support the adaptive-h mechanism qualitatively. Marginal
coverage remains close to the split-conformal target, oracle adaptive bandwidth
flattens `h(x)/s(x)`, and oracle improves worst-bin deviation and interval score
relative to fixed bandwidth in the current Exp4-IQR summary.

The two limitations are:

- exact score-homogeneity metrics need raw conformity scores saved per test
  point;
- exact `R_s=1,2,4,8` trends require a controlled DGP and should not be claimed
  from the current four examples.

## Recommended Next Steps

- Add raw conformity score saving to future adaptive-h runners if score
  homogeneity becomes a central empirical claim.
- Keep generated diagnostic CSV/PNG files in `exp_adaptive_h/output_*` folders.
- Use `experiment_logs/templates/daily_experiment_report.md` for each daily run.
- Promote only stable, paper-facing conclusions into `manuscript/reports/` or
  the journal draft.
