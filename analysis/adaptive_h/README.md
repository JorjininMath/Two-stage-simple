# Adaptive-h Analysis Map

This folder indexes adaptive-bandwidth evidence without moving generated
outputs. The current implementable method is fixed by `PROTOCOL.md` and
`PROTOCOL.md` and
`experiments/adaptive_h/final_benchmark_spec.md`: estimate the response
scale by the per-site sample
standard deviation, smooth the sitewise estimates by Nadaraya-Watson regression,
and use `h(x) = c * s_hat(x)`.

## Current Evidence Status

| evidence item | status | paper role |
| --- | --- | --- |
| fixed versus oracle scale normalization | diagnostic | mechanism evidence |
| oracle multiplier sensitivity | diagnostic | default-value check |
| sample-SD plus NW plug-in under the locked protocol | pending final run | main implementable method |
| raw-score homogeneity | pending final run | mechanism evidence |
| iid target-law calibration check | pending final run | validity audit |

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

The existing examples support the oracle scale-normalization mechanism
qualitatively: the adaptive rule flattens `h(x)/s(x)` and improves bin-wise
stability and interval score relative to fixed bandwidth in the tested settings.
The implementable sample-SD plus NW method still needs a protocol-aligned final
run before it can support a paper-facing numerical claim.

## Recommended Next Steps

- Add raw conformity score saving to future adaptive-h runners if score
  homogeneity becomes a central empirical claim.
- Run only the sample-SD plus NW plug-in in the final workflow.
- Keep generated diagnostic CSV/PNG files in
  `experiments/adaptive_h/output_*` folders.
- Use `experiment_logs/templates/daily_experiment_report.md` for each daily run.
- Promote only stable, paper-facing conclusions into `manuscript/reports/` or
  the journal draft.
