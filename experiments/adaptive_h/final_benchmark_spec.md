# Final Adaptive-Bandwidth Benchmark Specification

This document specifies the **canonical final** adaptive-bandwidth benchmark.
The implementation entrypoint is `run_final_adaptive_h_benchmark.py`, with
defaults in `final_benchmark_config.json`. Historical Exp1--Exp4 runners remain
available only for provenance and mechanism diagnostics.

## Goal

Evaluate whether scale-adaptive indicator bandwidths

```text
h(x) = c * s_hat(x)
```

reduce the fixed-bandwidth distortion caused by heterogeneous response scales.
The experiment targets score construction, interval efficiency, and local/group
coverage stability. Split conformal calibration remains the source of
finite-sample marginal coverage.

The experiment does not use the diagnostic score \(S^0\). Stage-1 training uses
a fixed space-filling grid. Calibration and test inputs are fresh iid draws
from the same target law \(q_X\), so output-side bandwidth adaptation is not
confounded with adaptive design and the split-conformal guarantee applies.

## Main Data-Generating Processes

All three main DGPs are one-dimensional. The first and third share the same
mean and scale so that the non-Gaussian case isolates tail behavior.

| Simulator | Domain | Model | Role |
| --- | --- | --- | --- |
| `raised_floor_gauss` | `[0, 2*pi]` | `Y = exp(x/10) sin(x) + s(x) epsilon`, `epsilon ~ N(0,1)` | Clean location-scale baseline |
| `mm1_sojourn` | `[0.1, 0.9]` | `Y | rho ~ Exponential(rate=1-rho)` with `mu=1`, `lambda=rho` | Exact stationary M/M/1 sojourn-time benchmark |
| `raised_floor_t3` | `[0, 2*pi]` | `Y = exp(x/10) sin(x) + s(x) epsilon`, `epsilon ~ t_3 / sqrt(3)` | Heavy-tail stress at the same conditional SD |

For the two raised-floor DGPs,

```text
s(x) = 0.10 + 0.20 * (x - pi)^2.
```

This replaces the older WSC scale `0.01 + 0.20 * (x - pi)^2` in the final
adaptive-h figures. The raised floor avoids near-zero scale regions that create
large plotting artifacts and simplifies the theory assumption
`0 < s_min <= s(x) <= s_max`.

For `mm1_sojourn`, the stationary M/M/1 sojourn-time distribution is

```text
Y | rho ~ Exponential(rate = 1 - rho),  rho in [0.1, 0.9],
s(rho) = SD(Y | rho) = 1 / (1 - rho).
```

For `raised_floor_t3`, dividing by `sqrt(3)` standardizes a Student-\(t_3\)
variable to unit variance. Consequently, `s(x)` is the conditional standard
deviation in both raised-floor DGPs; the two cases differ only in tail shape.

Historical simulators such as `wsc_gauss`, `gibbs_s1`, and `nongauss_A1L`
should remain registered for backward compatibility, but they are not the
default final adaptive-h benchmark set.

## Experimental Arms

Each macrorep trains Stage 1 once for a given DGP and Stage 1 budget, collects
one Stage 2 calibration set, and evaluates all arms on the same test set.

| Arm | Bandwidth rule | Purpose |
| --- | --- | --- |
| `fixed` | Scalar `h_fixed` selected by Stage 1 CV | Baseline CKME-DCP |
| `oracle` | `h(x) = c * s(x)` | Mechanism benchmark |
| `plugin_sd_nw` | `h(x) = c * s_hat_NW(x)` | Implementable adaptive-h method |

Use the same CKME `ell_x` and `lam` across the three arms. The fixed arm uses
the CV-selected scalar `h_fixed`; adaptive arms only replace the output-side
indicator bandwidth at calibration and prediction time.

## Plug-in Scale Estimator

The final plug-in estimator uses replicated Stage 1 observations. For Stage 1
sites `x_i` with `r_0` replications,

```text
s_site(x_i) = sqrt( sum_j (Y_ij - mean_j Y_ij)^2 / (r_0 - 1) ).
```

Smooth the sitewise estimates with Gaussian Nadaraya-Watson regression:

```text
s_hat_NW(x) =
    sum_i K_b(x, x_i) s_site(x_i) / sum_i K_b(x, x_i).
```

The final benchmark freezes Silverman's rule with `bw_factor=1.0` before
calibration. It is recorded in the run manifest and is never chosen with oracle
scale information or separately inside a macrorep. Any alternative
`bw_factor` values are sensitivity analyses, not the primary method.

## Default Settings

Unless a script argument overrides them, use:

| Quantity | Value |
| --- | ---: |
| `alpha` | `0.10` |
| `c` | `1.0` |
| `n_macro` | `50` |
| `t_grid_size` | `1000` |
| Stage 1 design | equal-spaced grid |
| Calibration method | iid from `q_X` |
| Calibration pairs `n_cal` | `1000` |
| Calibration reps per input | `1` |
| Test method | fresh iid from the same `q_X` |
| Test pairs `n_test` | `1000` |
| Test reps per input | `1` |

For the Stage 1 budget sweep, fix `r_0 = 10` and use

```text
B = n_0 * r_0 in {100, 250, 500, 1000}
n_0 in {10, 25, 50, 100}.
```

The `B=1000` row is the main table setting. The full budget sweep is used for
plug-in-vs-oracle convergence diagnostics. A smoke run can use `n_macro=5`; a
pilot run can use `n_macro=20`. Increase to `n_macro=100` only if final error
bars are too unstable.

## Parameter Training

Run Stage 1 CV before the main experiment:

```bash
python experiments/adaptive_h/pretrain_params.py \
    --simulators mm1_sojourn,raised_floor_gauss,raised_floor_t3 \
    --out experiments/adaptive_h/pretrained_params_final.json
```

The CV grid should tune `ell_x`, `lam`, and scalar `h_fixed` using CRPS. Store
the selected values in
`experiments/adaptive_h/pretrained_params_final.json`. The same `ell_x` and
`lam` are used for fixed, oracle, and plug-in arms.

## Required Outputs

The main runner writes only to ignored output directories:

```text
experiments/adaptive_h/output_final_adaptive_h/
```

Required tracked code should not depend on files inside that output directory.
Each run writes:

| File | Content |
| --- | --- |
| `manifest.json` | simulator list, seeds, budgets, data sizes, CV params, `bw_factor`, git commit if available |
| `per_arm.csv` | one row per `(macrorep, simulator, budget, arm)` |
| `paired_deltas.csv` | paired plug-in minus oracle and adaptive minus fixed metrics |
| `summary.csv` | mean, SD, and Monte Carlo SE by `(simulator, budget, arm)` |
| `per_point.csv` files | test-point diagnostics for reproducible plotting |

The per-point files should include:

```text
x0, y, L, U, covered_interval, covered_score, raw_score,
width, interval_score, h_query, s_oracle, s_hat, h_over_s,
group_bin, y_in_grid, L_at_grid_lo, U_at_grid_hi
```

Saving `raw_score` is important for score-homogeneity diagnostics. Previous
outputs without raw scores can support coverage and width diagnostics, but not
the strongest score-distribution claim.

## Primary Metrics

Report these metrics by simulator, budget, and arm:

| Metric | Interpretation |
| --- | --- |
| Marginal coverage | CP sanity check against `1 - alpha` |
| Mean width | Efficiency |
| Interval score | Combined calibration and sharpness |
| Mean group coverage gap | Average absolute binwise gap from `1 - alpha` |
| Worst group coverage gap | Stability stress test |
| Plug-in minus oracle gap | Estimation penalty |
| Adaptive minus fixed gap | Benefit of adaptive bandwidth |
| `h(x) / s(x)` diagnostics | Direct fixed-h distortion and adaptive-h correction |

Use equal-count bins along `x0` for one-dimensional group coverage, with
`K=10` as the default.

## Figures and Tables

Minimum final artifacts:

1. Scale-function figure for the three DGPs.
2. Main table at `B=1000`: coverage, width, interval score, and group gap.
3. Budget-sweep plot: plug-in-vs-oracle gap against `B`.
4. Binwise coverage plot for `fixed`, `plugin_sd_nw`, and `oracle`.
5. Effective-ratio diagnostic: `h(x) / s(x)` by arm.

## Optional Appendix Experiment

A high-dimensional robustness appendix can be added after the main one-dimensional
results are stable:

```text
X_j iid Unif(-1.5, 1.5), j = 1,...,d
Y = 0.3 X_1 + sqrt(1 + 0.3 |X_1|) epsilon
d in {2, 5, 20}
```

This appendix tests irrelevant covariates and dimension sensitivity. It should
not replace the three main adaptive-h DGPs.

## GitHub-Ready Code Reorganization Checklist

1. Add final simulator names instead of mutating historical ones:
   - `raised_floor_gauss`
   - `raised_floor_t3`
   - matching oracle scale functions in `adaptive_bandwidth.py`
2. Use the final adaptive-h defaults:
   - default simulators are the three main DGPs above
   - default budgets are `{100, 250, 500, 1000}`
   - calibration and test inputs are iid from `q_X`, with one output per input
3. Use sample-SD plus NW smoothing as the only final plug-in estimator.
4. Save additional diagnostics:
   - `raw_score`
   - `s_oracle`
   - `s_hat`
   - `h_over_s`
   - `group_bin`
5. Add a run manifest:
   - command-line settings
   - data sizes
   - seeds
   - pretrained parameters
   - selected NW `bw_factor`
6. Keep the canonical entrypoint separate from historical scripts:
   - keep old `run_exp1_baseline.py`, `run_exp2_oracle.py`, `run_exp3_csweep.py`,
     and `run_exp4_sample_sd_nw.py` if they are still needed for old outputs
   - use `run_final_adaptive_h_benchmark.py` for the final spec
7. Keep generated artifacts out of Git:
   - output directories, logs, cache files, PNGs, and LaTeX build products should
     remain ignored
   - tracked public files should be code, configs, specs, selected small tables,
     and manuscript-ready source files
8. Refresh public documentation:
   - point `README.md` and `EXPERIMENT_INDEX.md` to this spec
   - keep the historical plan under `notes/planning/archived/`
