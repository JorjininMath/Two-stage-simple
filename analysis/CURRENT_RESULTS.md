# Current Results Snapshot

Last verified against local result files: **2026-07-23**.

## Bottom Line

The protocol-aligned final adaptive-h benchmark is complete: 50 paired
macroreplications for three DGPs and four Stage-1 budgets, with 1,000 iid
target-law calibration pairs and 1,000 fresh iid test pairs per job. All
raw-score marginal coverages at the main budget `B=1000` are between `0.898`
and `0.902` (MCSE about `0.002`).

The guarantee belongs to raw score-set membership, not automatically to the
monotone-projected reporting interval. At `B=1000`, M/M/1 fixed has raw
coverage `0.8987`, projected-interval coverage `0.8546`, and
score-set/interval disagreement `0.04874`. The other eight combinations have
projected coverage between `0.8997` and `0.9026` and disagreement between
`0.0033` and `0.0086`.

The strongest mechanism result is qualified. On the raised-floor Gaussian and
Student-`t3` DGPs, oracle scale normalization reduces the maximum pairwise
raw-score KS diagnostic from `0.531` to `0.296` and from `0.718` to `0.437`.
The sample-SD plus Nadaraya--Watson plug-in moves this diagnostic toward oracle
(`0.468` and `0.539`) and its mean absolute relative scale error falls
substantially over `B=100` to `B=1000`. On M/M/1, however, fixed and oracle KS
are similar (`0.224` and `0.231`) and the plug-in value is `0.286`.

Practical performance is also mixed. At `B=1000`, the plug-in is narrower than
fixed bandwidth on all three DGPs, but interval score improves only for the
Student-`t3` model (`3.486` versus `3.576`). It is slightly worse for Gaussian
noise (`3.372` versus `3.334`) and substantially worse for M/M/1 (`12.568`
versus `10.983`). The safe conclusion is therefore that the plug-in
increasingly, but only partially, tracks the scale pattern and moves the score
diagnostic toward oracle on the raised-floor DGPs; it does not uniformly
dominate fixed bandwidth in interval score or groupwise coverage.

## Final Paper Evidence

| Result | Evidence | What is safe to share |
| --- | --- | --- |
| Locked final benchmark | `analysis/adaptive_h/final_run_manifest.json`, `final_summary.csv`, `final_paired_deltas.csv` | The iid target-law, one-output-per-input benchmark is complete with 50 macroreplications and nominal raw score-set marginal coverage; projected-interval coverage is a separate reporting metric. |
| Plug-in scale tracking | `analysis/adaptive_h/final_scale_diagnostics_summary.csv` | Raw oracle-relative scale error decreases with Stage-1 budget for all three DGPs, but includes the untuned global-multiplier difference and is not a uniform-consistency result. |
| Raw-score mechanism | `analysis/adaptive_h/final_score_homogeneity_summary.csv`, `plot_data/final_raw_score_homogeneity.csv` | Oracle and plug-in reduce the maximum pairwise KS diagnostic on the two raised-floor DGPs; this does not generalize to M/M/1, and the statistic has no null calibration. |
| Numerical and asset QA | `analysis/adaptive_h/final_qa_report.md`, `manuscript/generated/adaptive_h_assets_manifest.json` | QA passes with one rare small-budget reporting-grid warning; all exported final assets are hash-bound to the QA pass. |

## Historical Mechanism Evidence

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

## Final Refresh Complete

The claim-closing run created:

```text
experiments/adaptive_h/output_final_adaptive_h/manifest.json
experiments/adaptive_h/output_final_adaptive_h/per_arm.csv
experiments/adaptive_h/output_final_adaptive_h/paired_deltas.csv
experiments/adaptive_h/output_final_adaptive_h/summary.csv
experiments/adaptive_h/output_final_adaptive_h/jobs/.../per_point_*.csv
```

It uses iid target-law calibration with one response per input, 50
macroreplications, the final DGPs and budgets, and complete raw-score/scale
diagnostics. The full output stays local and ignored; compact checked evidence
is exported under `analysis/adaptive_h/`.

Advisor updates should say:

> Oracle scale normalization reduces score heterogeneity on the two
> raised-floor DGPs. The sample-SD/NW plug-in increasingly tracks the scale and
> moves the score diagnostic toward oracle there, but its interval and
> groupwise benefits are DGP-dependent. Raw score-set and projected-interval
> coverage are reported separately.
