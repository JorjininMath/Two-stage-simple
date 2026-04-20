# Diagnostic Spec: Why does L1 fail to converge under adaptive h?

## Motivation

The main `exp_conditional_coverage` experiment found that the adaptive bandwidth
`h(x) = 2 σ(x)` produces:

- **L1** (pre-CP quantile error `|F̂⁻¹(τ|x) − q_τ(x)|`): **flat** in `n`,
  ≈ 1.4–2.0 for `τ ∈ {0.05, 0.95}` on exp1.
- **L3** (conditional coverage error): converges with slope ≈ −0.39, near the
  reference `−2/5` rate.

The current `L2` metric in the report is **post-CP** — it uses the CP score
quantile `q̂` to invert `F̂` at probability levels `0.5 ± q̂`, which is **not**
the same as inverting `F̂` at `α/2` and `1−α/2`. So the L1 → L2 → L3 chain
claimed in `spec.md` is broken: L2 inherits CP's bias-absorption, L1 does not.

This diagnostic asks: **is the L1 failure caused by `h(x) = c·σ(x)` being too
large (over-smoothing the CKME CDF), and is the resulting bias structural
(invariant in `n`)?**

## Diagnostics planned (this round: 1, 2, 4 — pure post-processing)

| # | Name | Input | Signature we look for |
|---|------|-------|-----------------------|
| 1 | `h(x)` vs spread of `Y|x` | `h_at_x`, oracle `q_lo/q_hi` | adaptive `h(x)` comparable to or larger than IQ-gap → over-smoothing |
| 2 | `q̂_τ(x)` vs `q_τ(x)` at `n=64` and `n=8192` | `q_lo_hat`, `q_hi_hat`, `q_lo_oracle`, `q_hi_oracle` | adaptive curves overlap at both `n` (no shrinkage) and lie outside oracle (outward bias) |
| 4 | bias / variance decomposition of `q̂_τ` across macroreps | same | adaptive bias² flat in `n`, variance shrinks → bias-dominated structural error |

Diagnostics 3 (c-scan) and 5 (CDF overlay) are deferred — only run if 1/2/4
are inconclusive.

## Predictions (recorded before running)

- D1: `h(x) = 2σ(x)` is comparable to `(q_hi − q_lo)/2` by construction
  (since `q_hi − q_lo ≈ 3.29 σ`); the question is whether it overwhelms the
  smooth-indicator scale used by `F̂`. Expect ratio `h(x) / IQ-gap ≈ 0.6`,
  enough to noticeably smooth tails.
- D2: `q̂_lo_adaptive(x)` lies systematically **below** `q_lo_oracle(x)`,
  `q̂_hi_adaptive(x)` lies **above** `q_hi_oracle(x)`, in both `n=64` and
  `n=8192`, with the curves nearly identical → structural outward bias.
  Fixed h is closer to oracle and visibly improves between `n=64` and
  `n=8192`.
- D4: Adaptive bias² ≫ variance, both at `n=64` and `n=8192`; bias² flat,
  variance shrinks ~ `1/n`. Fixed h: bias² and variance both shrink.

## Output

```
diagnostics/output/
  diag1_h_vs_spread.png
  diag1_h_vs_spread.csv
  diag2_quantile_curves.png
  diag4_bias_var.png
  diag4_bias_var.csv
```

A diagnostic report (`diag_report.md` + PDF) summarises all three results and
states whether the over-smoothing hypothesis is confirmed.
