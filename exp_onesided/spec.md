# One-Sided Prediction Experiment: Spec

## Research question

Does CKME-based conditional CDF estimation yield better plug-in one-sided quantile
bounds than linear quantile regression, especially under non-Gaussian conditional laws?

## Method comparison (estimation level only, no calibration)

| Method | Primary object | How q̂_τ(x) is obtained |
|--------|---------------|------------------------|
| QR     | q̂_τ(x) directly | linear quantile regression at level τ |
| CKME   | F̂(y\|x) first   | CDF inversion: q̂_τ(x) = inf{y : F̂(y\|x) ≥ τ} |

CKME is CDF-first (like DCP-DR); QR is quantile-first (like CQR).
No conformal calibration — this is a pure plug-in quantile comparison.

## Simulators (3 cases)

| Name | Type | Noise |
|------|------|-------|
| `exp1` | MG1 queue, 1D | Heteroscedastic Gaussian |
| `exp2_test` | sinusoidal, 1D | Skewed bimodal mixture |
| `nongauss_B2L` | sinusoidal, 1D | Centered Gamma, k=2 (strong skew) |

Rationale:
- `exp1`: sanity check — Gaussian noise, QR should work fine
- `exp2_test`: existing skewed/bimodal case already in codebase
- `nongauss_B2L`: strong asymmetric skew (Gamma), hardest case for linear QR

## One-sided bounds (plug-in, no calibration)

**Lower bound** at level τ:
```
L(x) = q̂_τ(x)      target: P(Y ≥ L(X)) ≈ 1−τ
```

**Upper bound** at level τ:
```
U(x) = q̂_τ(x)      target: P(Y ≤ U(X)) ≈ τ
```

Use τ = 0.05 (lower, 95% lower coverage) and τ = 0.95 (upper, 95% upper coverage).

## Evaluation metrics

Report lower and upper bounds **separately**.

- **Empirical coverage**: actual P(Y ≥ L(X)) and P(Y ≤ U(X)) vs nominal
- **Pinball loss** at level τ: ρ_τ(y, q̂) = (τ−1)(y−q̂)·1{y<q̂} + τ(y−q̂)·1{y≥q̂}
- **Conditional coverage by x-bin**: reveals local adaptation failure

## Paper positioning

> We compare CKME and QR at the estimation level: QR directly fits conditional
> quantiles, while CKME estimates the full conditional CDF and inverts for quantiles.
> No calibration is applied; the question is whether CKME's nonparametric
> distribution estimate yields more accurate plug-in quantile bounds, particularly
> under skewed or non-Gaussian conditional laws where linear QR is misspecified.

## Code to implement

1. `CKME/ckme.py`: add `predict_quantile(X, tau)` — invert `predict_cdf` on t_grid
2. New experiment script: `exp_onesided/run_onesided_compare.py`
   - fits CKME and linear QR on same training data
   - evaluates plug-in lower/upper bounds on test set
   - reports coverage and pinball loss per simulator
