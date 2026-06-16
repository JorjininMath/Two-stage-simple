# KME/CKME Feasibility for M/M/1 Input Uncertainty

This experiment checks only the KME/CKME component for queueing-output
prediction under input uncertainty. It does not run conformal prediction, does
not claim conformal coverage, and does not implement a Lam-style
input-uncertainty confidence band.

The supervised unit is one queueing scenario. For scenario `j`, the learner
observes finite input samples

```text
D_{j,A} = inter-arrival samples
D_{j,S} = service-time samples
```

and one or more output replications `Y_{j,r}`, where each output is the average
sojourn time of the first 10 customers. Individual raw input observations
`A_{j,r}` and `S_{j,r}` are not paired with `Y_j` as supervised rows. They are
embedded into one scenario-level KME feature

```text
Z_j = (KME_A(D_{j,A}), KME_S(D_{j,S}), n_A^-1/2, n_S^-1/2, rho_hat).
```

Hidden true rates `(lambda_j, mu_j, rho_j)` are used only to generate data and
oracle Monte Carlo CDFs for evaluation. They are not used as features.

## Main Feasibility Run

From the parent repository root:

```bash
python3 -m ckme_dcp_mm1.experiment.run_kme_feasibility \
    --dgp exp_mm1 \
    --n-fit 1500 \
    --n-val 500 \
    --n-test 1000 \
    --r-train 1 \
    --r-oracle 5000 \
    --rff-dim 100 \
    --grid-size 100 \
    --n-seeds 20 \
    --out-prefix results/kme_feas_exp_mm1
```

Equivalently, from inside `ckme_dcp_mm1/`, use:

```bash
python3 -m experiment.run_kme_feasibility ...
```

Smoke test:

```bash
python3 -m ckme_dcp_mm1.experiment.run_kme_feasibility \
    --dgp exp_mm1 \
    --n-fit 300 \
    --n-val 100 \
    --n-test 200 \
    --r-train 1 \
    --r-oracle 500 \
    --rff-dim 50 \
    --grid-size 50 \
    --n-seeds 2 \
    --out-prefix results/smoke_kme_feas
```

The runner writes:

- `ckme_dcp_mm1/results/<prefix>_metrics.csv`
- `ckme_dcp_mm1/results/<prefix>_test_predictions.csv`
- `ckme_dcp_mm1/results/<prefix>_hyperparams.csv`
- `ckme_dcp_mm1/results/<prefix>_config.json`

## Model

RFF approximates separate RBF kernel mean embeddings for log inter-arrival and
log service samples. RFF bandwidths are chosen by a median-distance heuristic on
fit-split log observations only.

The CKME layer is a kernel ridge regression estimator for CDF values on a grid
of fit-output quantiles. For each threshold `t_l`, it regresses
`I(Y <= t_l)` on standardized KME features. Hyperparameters are selected on the
validation split by CDF MSE over the threshold grid. Test scenarios are used
only for oracle-CDF evaluation.

## Metrics

For each test scenario, the hidden queueing rates generate an oracle Monte
Carlo CDF on the same threshold grid. The main diagnostics are:

- integrated squared CDF error;
- integrated absolute CDF error;
- absolute quantile error at `0.1`, `0.5`, and `0.9`;
- raw monotonicity violation before rearranging CDF predictions;
- the same errors by input sample size `n_A in {20, 50, 200, 1000}`;
- diagnostic traffic groups using true `rho <= 0.75` versus `rho > 0.75`.

These are feasibility diagnostics, not coverage guarantees.

## Summaries And Plots

Summarize seed-level metrics:

```bash
python3 -m ckme_dcp_mm1.experiment.summarize_kme_feasibility \
    --input "ckme_dcp_mm1/results/*_metrics.csv" \
    --out ckme_dcp_mm1/results/kme_feas_summary.csv
```

Plot metrics and example CDF curves:

```bash
python3 -m ckme_dcp_mm1.experiment.plot_kme_feasibility \
    --metrics ckme_dcp_mm1/results/smoke_kme_feas_metrics.csv \
    --predictions ckme_dcp_mm1/results/smoke_kme_feas_test_predictions.csv \
    --out-dir ckme_dcp_mm1/figures \
    --log-dir experiment_logs/ckme_dcp_mm1
```

The plot command also writes a timestamped Markdown experiment log with the
run settings, aggregate results, generated figure paths, and the meaning of
each figure. The figures use run indices rather than displaying random seeds.
When prediction rows are available, it also writes `sup_error_vs_rho.png`,
which plots scenario-level sup CDF error against true traffic intensity and
colors points by finite input sample size.

## Existing Visual Diagnostic

The older CDF-band script remains available:

```bash
python3 -m ckme_dcp_mm1.experiment.plot_cdf_bands \
    --seed 20260604 \
    --n-train 800 \
    --rff-dim 80 \
    --n-bootstrap 200 \
    --n-oracle 20000 \
    --target-group highU_heavy \
    --out figures/kme_ckme_cdf_bands.png
```

That script creates a visual bootstrap envelope for one representative
scenario. It is useful for inspecting how finite input samples affect the KME
representation, but it is not the main feasibility experiment and is not a
confidence band. The figure uses a three-row diagnostic layout: CDF curves with
pointwise bootstrap diagnostic bands, CDF error relative to the oracle CDF, and
no-IU/IU band half-widths.
