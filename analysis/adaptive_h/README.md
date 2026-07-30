# Adaptive-h Analysis Map

This folder indexes adaptive-bandwidth evidence without moving generated
outputs. The current implementable method is fixed by `PROTOCOL.md` and
`experiments/adaptive_h/final_benchmark_spec.md`: estimate the response
scale by the per-site sample
standard deviation, smooth the sitewise estimates by Nadaraya-Watson regression,
and use `h(x) = c * s_hat(x)`.

## Current Evidence Status

| evidence item | status | paper role |
| --- | --- | --- |
| fixed versus oracle scale normalization | diagnostic | mechanism evidence |
| oracle multiplier sensitivity | diagnostic | default-value check |
| sample-SD plus NW plug-in under the locked protocol | complete, mixed performance | main implementable method |
| raw-score homogeneity | complete | mechanism diagnostic |
| iid target-law calibration check | complete | validity audit |

Older plug-in output analyses are retained only for local provenance. They are
not part of the current evidence set and should not be used in the manuscript,
formal reports, or advisor updates.

## Rasi Lab-style Mapping

| Rasi Lab category | Current path | Role | Action |
| --- | --- | --- | --- |
| Experiments | `experiments/adaptive_h/` | runners, summarizers, plotters, generated `output_*` folders | keep in place |
| Daily experiment logs | `experiment_logs/adaptive_h/` | local day-to-day run notes | use daily template |
| Analysis | `analysis/adaptive_h/` | stable map of diagnostics and interpretation-ready artifacts | keep tracked |
| Manuscript/report | `manuscript/reports/`, `manuscript/journal_scale_adaptive/` | formal paper-facing writeups | update only after diagnostics are stable |

## Current Interpretation

The final 50-macroreplication run supports oracle scale normalization on the
raised-floor Gaussian and Student-\(t_3\) DGPs. The sample-SD plus NW estimator
increasingly tracks the scale as budget grows and moves the maximum pairwise
raw-score KS diagnostic toward the oracle on those two DGPs. Interval score and
groupwise coverage are mixed, and M/M/1 does not show the same oracle score
advantage. M/M/1 fixed also has a material raw-score/projected-interval
coverage discrepancy. Do not claim uniform plug-in dominance, uniform scale
consistency, or guaranteed coverage for the projected interval.

## Recommended Next Steps

- Treat `final_run_manifest.json`, `final_summary.csv`,
  `final_scale_diagnostics_summary.csv`,
  `final_score_homogeneity_summary.csv`, and `final_qa_report.md` as the
  checked compact evidence set.
- Use `plot_data/` to reconstruct or audit the five exported final figures.
- Keep full per-point outputs in ignored `experiments/adaptive_h/output_*`
  folders.
- Use `experiment_logs/templates/daily_experiment_report.md` for each daily run.
- Keep the mixed-outcome limitation in every advisor update and manuscript
  claim.
