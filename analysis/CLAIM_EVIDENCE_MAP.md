# Claim-to-Evidence Map

This map separates statements that can be made now from claims that still need
a protocol-aligned refresh. Exact numerical sources are listed so an advisor
update, talk, or manuscript revision can be checked without searching output
folders.

| ID | Proposed claim | Status | Exact evidence | Safe wording now | Do not claim |
| --- | --- | --- | --- | --- | --- |
| `AH-MECH-01` | Local response-scale normalization improves groupwise behavior of the CKME-DCP score. | Current paper snapshot; mechanism only | `experiments/adaptive_h/output_exp2/exp2_paired_deltas.csv`; `experiments/adaptive_h/output_exp2/exp2_table.tex`; `experiments/framing_validation/output_gate1/gate1_correlations.csv`; `experiments/framing_validation/output_gate2/gate2_ks_summary.csv` | “Oracle scale normalization improved paired worst-bin deviation on all four tested DGPs and reduced score heterogeneity most clearly on `exp1`.” | “The implementable plug-in has been validated under the final protocol.” |
| `AH-EFF-01` | Adaptive bandwidth improves interval efficiency. | Current paper snapshot; qualified | `experiments/adaptive_h/output_exp2/exp2_summary.csv`; `experiments/adaptive_h/output_exp2/exp2_paired_deltas.csv` | “Oracle adaptive bandwidth improved paired interval score in 98--100% of macroreps.” | “Oracle adaptive bandwidth is always narrower.” (`exp1` mean width increased by about 0.300.) |
| `AH-C-01` | `c=1` is a stable default multiplier. | Current paper snapshot | `experiments/adaptive_h/output_exp3/exp3_summary.csv`; `experiments/adaptive_h/output_exp3/exp3_table.tex` | “The sweep is not knife-edge; `c=1` is a practical default and `c=2` gives little further worst-bin improvement.” | “`c=1` is statistically optimal.” |
| `AH-PLUGIN-01` | Sample-SD plus Nadaraya--Watson scale estimation retains the oracle benefit. | **Missing refresh** | Expected: `experiments/adaptive_h/output_final_adaptive_h/manifest.json`, `paired_deltas.csv`, `summary.csv`, and per-point diagnostics | “The method is implemented; the protocol-aligned final comparison is pending.” | Any final numerical plug-in claim based on the old `output_exp4/` tree. |
| `VALID-TARGET-01` | Split conformal attains finite-sample marginal coverage under matched target-law calibration. | Theory/protocol fixed; final numerical audit missing | `PROTOCOL.md`; expected final adaptive-h manifest and raw-score coverage output | “The guarantee is tied to iid calibration and test pairs from the same target law.” | Using older LHS/mixed replicated-calibration experiments as direct evidence for the locked guarantee. |
| `SCORE-ROBUST-01` | Equal-tailed scoring is more robust than estimated width optimization in the tested skewed DGP. | Supporting, 50-macrorep pilot | `experiments/coverage_mechanism/pilot_score_design/pilot_n50_b600.csv`; `pilot_n50_b2000.csv`; `report_pilot_score_design.tex` | “Typical optimized runs were about 3% narrower, but rare failures reversed the mean at both budgets.” | “Width optimization can never help,” or generalization beyond the tested DGP and budgets. |
| `PCP-SCALE-01` | The response-scale normalization principle transfers to a sample-based PCP score. | Supporting, 50-macrorep sub-study | `experiments/coverage_mechanism/pilot_score_design/pilot_pcp_n50.csv`; `report_pilot_score_design.tex` | “In the tested heteroscedastic DGP, scale normalization compressed the reported ten-bin coverage range 4.4-fold.” | “Scaled PCP dominates CKME-DCP.” The report records a set-size premium. |
| `EPI-DIAG-01` | Epistemic diagnostics can localize model error more sharply than CP width. | Supporting diagnostic | `experiments/framing_validation/output_gate3/gate3_summary.csv`; `experiments/framing_validation/README.md` | “In the five-macrorep bump diagnostic, separation indices were 4.9 for width, 32 for oracle CDF L2, and 112 for `u_tail`.” | A general finite-sample theorem or a paper-level quantitative conclusion from five macroreps. |
| `EPI-ACT-01` | Diagnostics can distinguish data scarcity from model capacity limits. | Supporting, mixed outcome | `experiments/framing_validation/output_gate4a/gate4a_metrics.csv`; `output_gate4b/gate4b_metrics.csv`; `experiments/framing_validation/README.md` | “G4b supported retuning after targeted sampling hit a bias floor; G4a only supported hotspot detection/allocation and missed its CDF-error gate.” | “Targeted `u_tail` allocation reliably improves CDF error.” |
| `DESIGN-REGIME-01` | Adaptive Stage-2 allocation is most useful when Stage 1 is under-fit. | Pilot | `experiments/design/report.md`; `experiments/design/output/expB_var_epi/gibbs_s1_d5/summary.csv`; `experiments/design/output/expB_var_epi/gibbs_s1_d5_n500r10/summary.csv` | “The d=5 pilot showed about a 2.1% interval-score gain at the smaller Stage-1 design and a tie at the larger design.” | A calibrated universal saturation law; most design runs use five macroreps. |
| `CONSIST-01` | Fixed-h CKME-DCP errors decline with sample size on the tested Gaussian DGPs. | Supporting, 10 macroreps | `experiments/conditional_coverage/output_consistency_fixed/summary_exp1.csv`; `summary_exp2.csv`; `experiments/conditional_coverage/report.md` | “All reported fixed-h L1/L2/L3 error summaries decrease over the tested sample sizes.” | A proven convergence rate or a conclusion for the final plug-in estimator. |

## Explicit Exclusion: Retired IQR Response-Scale Plug-in

`_archive/01_experiments/diagnostics/adaptive_h_iqr_response_scale/` is
provenance only. No table, figure, summary, advisor update, or manuscript claim
should use it as evidence for the current method.

This exclusion does not prohibit robust IQR calculations used solely to set an
**input-space** smoothing bandwidth, such as the Silverman rule in
`experiments/adaptive_h/sample_sd_nw_scale.py`. The response-scale observations
used by the current plug-in are per-site sample standard deviations.
