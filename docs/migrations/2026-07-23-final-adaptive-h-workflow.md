# Final Adaptive-h Workflow Migration

Date: 2026-07-23

Status: complete; final documentation commit recorded after validation

## Purpose

This migration closes the final adaptive-bandwidth workflow gap. It separates
scale-adaptive output smoothing from adaptive input design, removes `S^0` from
the final experiment, and produces a protocol-aligned evidence package that can
be audited, shared with an advisor, and consumed by the manuscript.

The retired IQR response-scale plug-in remains archive-only and was not restored
or used.

## Renamed Files

| old path | new path | reason |
| --- | --- | --- |
| `experiments/adaptive_h/planned_final_benchmark_spec.md` | `experiments/adaptive_h/final_benchmark_spec.md` | The specification now controls an implemented and completed workflow. |
| `manuscript/journal_scale_adaptive/target_aware_scale_adaptive_ckme_cp.tex` | `manuscript/journal_scale_adaptive/scale_adaptive_ckme_cp.tex` | The filename now matches the paper's scale-adaptive CKME-DCP main contribution. |
| `manuscript/reports/adaptive_h_report.tex` | `manuscript/reports/adaptive_h_final_report.tex` | The report now describes the locked final benchmark rather than historical oracle diagnostics. |

## P0 Correctness Changes

1. `src/Two_stage/test_data.py`
   - Added an explicit iid test-data branch.
   - iid inputs are sampled directly from the registered target law `q_X`.
   - `X_cand`, `S^0`, Stage-2 selected sites, and site-exclusion logic are not
     used for iid test data.
   - iid test data use one output per input.

2. `src/Two_stage/stage2.py`
   - Added a separate simulator-output seed so input and output streams are
     explicit and independently reproducible.

3. `src/Two_stage/sim_functions/`
   - Added `mm1_sojourn`, with
     `Y | rho ~ Exponential(rate=1-rho)` and `s(rho)=1/(1-rho)`.
   - Added `raised_floor_gauss`.
   - Added `raised_floor_t3`, using variance-normalized `t_3 / sqrt(3)`.
   - Registered all final DGPs and their oracle response scales.

4. Historical conflict removal
   - The final configuration uses iid calibration and test inputs with
     `r_cal=r_test=1`.
   - Stage-1 budgets are `100, 250, 500, 1000`.
   - Historical LHS/replicated Exp4 remains provenance-only.

## P1 Workflow Changes

The canonical entrypoints are:

```bash
python experiments/adaptive_h/run_final_adaptive_h_benchmark.py \
    --n-workers 4 --executor thread
python experiments/adaptive_h/summarize_final_adaptive_h_benchmark.py
python experiments/adaptive_h/analyze_final_adaptive_h_scores.py
python experiments/adaptive_h/plot_final_adaptive_h_results.py
python experiments/adaptive_h/qa_final_adaptive_h_benchmark.py --require-final
python tools/export_manuscript_assets.py
python tools/export_manuscript_assets.py --check
```

The runner:

- uses no `S^0` score;
- shares Stage-1, calibration, test data, and named seed streams across fixed,
  oracle, and sample-SD plus NW arms;
- checkpoints every DGP--budget--macroreplication job;
- records the scientific configuration hash, source revision, code state,
  pretrained parameters, and seed scheme;
- saves raw-score and projected-interval layers separately;
- writes per-point `raw_score`, `s_hat`, `s_oracle`, `h_over_s`, `group_bin`,
  interval-boundary, and pairing diagnostics.

The adaptive CDF implementation now uses memory-bounded batching and exposes a
public raw point-score function. Regression tests verify scalar equivalence for
all supported smooth-step families.

## Final Run

Configuration:

| setting | value |
| --- | --- |
| DGPs | `mm1_sojourn`, `raised_floor_gauss`, `raised_floor_t3` |
| Stage-1 budgets | `100, 250, 500, 1000` |
| Stage-1 replications | `r0=10` |
| calibration | 1,000 iid target-law pairs, one output per input |
| test | 1,000 fresh iid target-law pairs, one output per input |
| arms | `fixed`, `plugin_sd_nw`, `oracle` |
| macroreplications | 50 |
| completed jobs | 600 |
| per-point arm files | 1,800 |
| base seed | `2026072351` |
| source commit recorded by run | `22f13b0f649a872edabe27aec1c09da893ff8769` |

Pretrained Stage-1 parameters:

| DGP | `ell_x` | `lam` | fixed `h` |
| --- | ---: | ---: | ---: |
| M/M/1 | 0.3 | 0.001 | 0.1 |
| raised-floor Gaussian | 0.5 | 0.001 | 0.1 |
| raised-floor Student-`t3` | 1.0 | 0.001 | 0.05 |

## Final Evidence Boundary

At `B=1000`, every raw-score marginal coverage is between 0.898 and 0.902.
Oracle scale normalization substantially reduces raw-score heterogeneity on the
two raised-floor DGPs. The sample-SD plus NW estimator increasingly tracks the
scale and moves the score diagnostic toward oracle on those two DGPs, but
interval score and groupwise coverage are mixed. M/M/1 does not reproduce the
same oracle score advantage. Its fixed arm also has raw coverage 0.899 versus
projected-interval coverage 0.855, with 0.049 membership disagreement.

Safe conclusion:

> The plug-in increasingly tracks the response scale and moves the score
> diagnostic toward oracle on the raised-floor DGPs, but it does not uniformly
> dominate fixed bandwidth in interval score or groupwise coverage. Raw
> score-set and projected-interval coverage are separate quantities.

## QA and Publication Assets

Final QA passes with zero errors and one warning. Two of 1,800 arm files exceed
a 2 percent projected-interval grid-boundary rate, both fixed-bandwidth
`B=100` runs. The mean boundary rate is 0.000214, no `B=1000` arm is affected,
and raw-score coverage is unaffected.

The figure pipeline produces PDF, SVG, PNG, and TIFF versions of five figures.
PDFs use TrueType fonts, SVG labels remain text, and compact source-data CSVs
are exported. The QA report records SHA-256 hashes for every final asset; the
exporter refuses to publish a changed or unverified source.

Checked compact evidence is under:

- `analysis/adaptive_h/`
- `analysis/adaptive_h/plot_data/`
- `manuscript/generated/`

Full raw output remains ignored under:

- `experiments/adaptive_h/output_final_adaptive_h/`

## Validation

- 41 pre-postprocessing unit tests passed after the P0/P1 implementation.
- Five focused postprocessing/export regression tests were added.
- Smoke, 20-macroreplication pilot, and 50-macroreplication final QA passed.
- Python compilation and `git diff --check` passed.
- The final LaTeX report and journal manuscript are rebuilt after asset export.

## Git Record

- Code implementation commit:
  `22f13b0f649a872edabe27aec1c09da893ff8769`
- Final documentation/evidence commit: recorded in the repository history after
  the validation and build steps in this migration.
