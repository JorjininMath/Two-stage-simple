# exp_adaptive_h — Experiment Plan

Concrete plan for Exp1-4 of `Experiment_adaptive_h.md`. Exp5 (interaction with adaptive design) lives in `exp_design/`.

## DGPs (all 1D)

| Name           | Domain        | Noise type      | sigma(x) shape                 | Role                         |
|----------------|---------------|-----------------|--------------------------------|------------------------------|
| `wsc_gauss`    | [0, 2π]       | Gaussian        | smooth U: 0.01+0.20(x-π)^2     | clean baseline (s = sigma)   |
| `gibbs_s1`     | [-3, 3]       | Gaussian        | interior zero: |sin(x)|        | extreme low-sigma stress     |
| `exp1`         | [0.1, 0.9]    | Gaussian (MG1)  | boundary explosion             | extreme high-sigma stress    |
| `nongauss_A1L` | [0, 2π]       | Student-t ν=3   | smooth U: 0.01+0.20(x-π)^2     | heavy-tail (s ≠ sigma)       |

For all 4, oracle h(x) = c * s(x) where s(x) is the **scale** function (NOT response std σ(Y|x)). For Gaussian DGPs s = sigma; for nongauss_A1L sigma(Y|x) = s(x)·√3.

## Adaptive-h infrastructure (prerequisite)

Current pipeline only supports scalar h. Adaptive h(x) requires extending:
- `CKME/parameters.py`: `Params.h` accepts callable
- `CKME/indicators.py`: per-sample h
- `CKME/ckme.py`, `CKME/cdf.py`: thread h(x_i) through CDF evaluation
- `CP/scores.py`, `CP/calibration.py`: thread h(x_i) through CP score

This is **Step 3** of the implementation order; Exp1 can run on the existing scalar-h pipeline.

## Exp1 — Baseline (fixed h, CV-tuned)

- All 4 DGPs with config.txt defaults
- Stage 1 LHS, Stage 2 LHS + equal reps, fixed h from CV
- **Output**: per-DGP conditional coverage curve as the "fixed-h" reference. Reused as a baseline by Exp2-4.

## Exp2 — Oracle adaptive h vs fixed h

- All 4 DGPs, two lines per DGP: fixed h (Exp1 result) vs oracle h(x) = c·s(x), c=1.0
- **Primary metric**: conditional coverage deviation `|cov(x) − (1−α)|` vs x. Worst-bin and mean-bin reported.
- **Decision rule**: if oracle does not clearly beat fixed h on at least 3 of 4 DGPs → Score Homogeneity story is in trouble; pause and reconsider.

## Exp3 — c sensitivity (oracle regime)

- Fix one DGP: `nongauss_A1L` (heavy-tail + heteroscedastic; primary stress test)
- Sweep c ∈ {0.3, 0.5, 1.0, 2.0}, still using true s(x)
- **Output**: conditional coverage deviation vs c. Goal: show plateau, not knife-edge optimum.
- **Note**: c is settled here under oracle; Exp4 plug-in inherits the chosen c (default 1.0 unless Exp3 reveals a clear winner).

## Exp4 — Plug-in adaptive h: descend to reality

Two diagnostic targets, decoupled by DGP type.

### Exp4a — Gaussian DGPs (wsc_gauss, gibbs_s1, exp1)

- Goal: empirically validate Gap Theorem `|cov_plug − cov_oracle| = O(N_1^{−α})`
- Sweep Stage 1 budget n_0·r_0 ∈ {50, 100, 250, 500}
- 3 lines per budget: fixed h / plug-in h(x)=c·σ̂(x) / oracle h(x)=c·s(x)
- **Plot**: |cov_plug − cov_oracle| vs Stage 1 budget; expect monotone decay

### Exp4b — Student-t DGP (nongauss_A1L)

- Plug-in σ̂(x) → σ(Y|x) = s(x)·√3, NOT the oracle target s(x)
- **Hypothesis**: CP calibration absorbs the constant √3 scaling → conditional coverage of plug-in still ≈ oracle
- **Two diagnostic figures**:
  1. Conditional coverage deviation vs x: 3 lines (fixed / plug-in / oracle); expect plug-in ≈ oracle, both clearly better than fixed
  2. Calibration quantile ratio q̂_plug / q̂_oracle vs x: expect ratio ≈ 1/√3 ≈ 0.577 (mechanism evidence — CP is doing the absorption)

If both figures confirm: this becomes the paper's "Robustness to scale misspecification" result, stronger than the Gaussian convergence in 4a because it shows robustness to **estimator–oracle mismatch**, not just to estimation noise.

## Caveat (for Discussion section)

The √3 absorption argument relies on the misspecification factor being x-independent (here ν is constant). Under x-varying tail heaviness ν(x), the plug-in–oracle gap becomes x-dependent and CP can no longer absorb it; this regime would need a heavy-tail-aware scale estimator and is left as future work.

## Implementation order

1. ✅ Skeleton: config.txt, config_utils.py, Exp_plan.md, pretrain_params.py
2. Run pretrain_params.py → fixed-h baselines for 4 DGPs (Exp1)
3. **Adaptive-h infrastructure** (the listed code changes above)
4. Exp2 runner + plotter
5. Exp3 c-sweep
6. Exp4a + Exp4b runners + dual diagnostic plots
