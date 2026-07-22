# Current Results Snapshot

Last verified against local result files: **2026-07-22**.

## Bottom Line

The strongest result currently visible in the journal draft is the
**oracle scale-normalization mechanism**, not the implementable plug-in. In the
May 1, 2026 Exp2 run (50 paired macroreplications per DGP), oracle adaptive
bandwidth reduced the median paired worst-bin coverage deviation on all four
DGPs and improved paired interval score in 98--100% of macroreplications. The
four median oracle-minus-fixed worst-bin changes were `-0.033`, `-0.016`,
`-0.061`, and `-0.021`; marginal coverage remained near 0.90.

The Exp3 multiplier sweep (50 macroreplications on `nongauss_A1L`) supports
`c=1` as a practical default, not as a uniquely optimal value. Worst-bin
deviation was `0.105` at `c=1` and `0.104` at `c=2`; mean interval score was
`5.276` and `5.251`, respectively.

These Exp1--Exp3 artifacts predate the locked protocol. They use the older
paper workflow and therefore support a **mechanism statement only**. They do
not establish the final finite-sample coverage result or the performance of
the implementable sample-SD plus Nadaraya--Watson plug-in.

## Current Paper Snapshot Evidence

| Result | Evidence | What is safe to share |
| --- | --- | --- |
| Fixed-h reference | `experiments/adaptive_h/output_exp1/exp1_summary.csv`, `exp1_table.tex` | Across four DGPs and 50 macroreps, mean marginal coverage ranged from 0.8967 to 0.9027; local coverage variation remained visible. |
| Oracle versus fixed | `experiments/adaptive_h/output_exp2/exp2_paired_deltas.csv`, `exp2_table.tex` | Oracle scale normalization improved paired worst-bin deviation on all four tested DGPs and paired interval score in 98--100% of runs. On `exp1`, it increased mean width by about 0.300, so the result is not “uniformly narrower.” |
| Multiplier sensitivity | `experiments/adaptive_h/output_exp3/exp3_summary.csv`, `exp3_table.tex` | Performance changed smoothly over `c in {0.3, 0.5, 1, 2}`; `c=1` is a defensible default with little worst-bin improvement left at `c=2`. |

## Supporting and Pilot Evidence

- **Score-choice pilot (2026-07-16; 50 macroreps per budget).** The
  width-optimized score had median width gains near 3%, but rare tail-chasing
  failures made its mean width 14.0% worse than equal-tailed at budget 600 and
  7.6% worse at budget 2,000. Source:
  `experiments/coverage_mechanism/pilot_score_design/report_pilot_score_design.tex`.
- **PCP portability sub-study (2026-07-16; 50 macroreps per DGP).** The report
  records a 4.4-fold compression of the ten-bin coverage range after
  replication-based scale normalization. This is portability evidence, not a
  replacement for the paper's equal-tailed CKME-DCP score.
- **Framing validation (2026-07-07/08).** G1 and G2 support scale-linked
  coverage and score heterogeneity; G3 shows that `u_tail` localized an
  epistemic hotspot much more sharply than interval width; G4b supports a
  diagnose-then-retune workflow. G2 uses 10 macroreps, G3 uses 5, and G4 uses
  20, so these remain supporting diagnostics. G4a did not meet its proposed
  CDF-error gate and must stay labeled partial.
- **Conditional-coverage consistency (10 macroreps).** Fixed-h errors decline
  with sample size on two Gaussian DGPs. The oracle `h(x)=2 sigma(x)` run shows
  an L1/L3 trade-off: pre-CP tail-quantile error remains structurally biased
  while post-CP coverage improves. This is a diagnostic phenomenon, not the
  final plug-in experiment.
- **Design ablations (mostly 5 macroreps).** Existing one-dimensional
  allocation comparisons are nearly flat on interval score; a d=5 pilot shows
  a modest adaptive gain when Stage 1 is under-fit and a tie after Stage 1 is
  enlarged. Treat the regime interpretation as a pilot until refreshed.

## Legacy or Provenance Only

- `experiments/adaptive_h/output_exp4/` contains a 50-macrorep sample-SD/NW
  run from the earlier workflow. It has no run manifest and its per-point
  files omit required final diagnostics such as `raw_score`, `s_hat`,
  `s_oracle`, `h_over_s`, and `group_bin`. It is not final paper evidence.
- The retired IQR response-scale plug-in and related diagnostics are archived
  under `_archive/01_experiments/diagnostics/adaptive_h_iqr_response_scale/`.
  They must not be cited as current evidence.
- The non-Gaussian R-baseline outputs have unequal completed macrorep counts
  (CKME 50; DCP-DR/hetGP 32--48 depending on DGP), while Gibbs and one-sided
  outputs use older small-run workflows. They require a public reproducibility
  refresh before promotion.
- The M/M/1 input-uncertainty module is KME/CKME feasibility work only; it does
  not provide conformal coverage evidence.

## Critical Missing Refresh

The next claim-closing artifact is a locked-protocol run of the implementable
`plugin_sd_nw` arm against fixed and oracle arms. It should create:

```text
experiments/adaptive_h/output_final_adaptive_h/manifest.json
experiments/adaptive_h/output_final_adaptive_h/per_arm.csv
experiments/adaptive_h/output_final_adaptive_h/paired_deltas.csv
experiments/adaptive_h/output_final_adaptive_h/summary.csv
experiments/adaptive_h/output_final_adaptive_h/per_point/...
```

The run must use iid target-law calibration with one response per calibration
input, at least 50 macroreplications, the final DGP set and budgets in
`experiments/adaptive_h/planned_final_benchmark_spec.md`, and the required
raw-score/scale diagnostics.
Until those files exist and pass audit, advisor updates should say:

> The oracle mechanism is supported; the final protocol-aligned sample-SD/NW
> plug-in comparison is the remaining main empirical gap.
