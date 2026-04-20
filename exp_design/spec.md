# exp_design: Stage Design Experiment

## Goal

Study the budget design of CKME-DCP's two-stage procedure:
- Stage 1 (training) vs Stage 2 (calibration) budget split
- Stage 2 internal: n_1 vs r_1 allocation
- S^0-guided adaptive site selection vs uniform LHS
- Interaction with adaptive h

## Budget Framework

Total budget B = B_train + B_cal = n_0 * r_0 + n_1 * r_1 (fixed).

Since we do NOT refit the model in Stage 2:
- Stage 1 = Training (model quality)
- Stage 2 = Calibration (q_hat precision)

See `notes/budget_design_framework.md` for full theory.

## DGPs

Start with exp2_gauss (from `examples/exp2_gauss.py`):
- True function: f(x) = exp(x/10) * sin(x), x in [0, 2*pi]
- Noise: Gaussian, heteroscedastic, sigma(x) = sigma_base + sigma_slope * (x-pi)^2
- Low noise: sigma_base=0.1, sigma_slope=0.05 (rho ~ 5.9)
- High noise: sigma_base=0.1, sigma_slope=0.20 (rho ~ 20.7)

Future DGPs (from `examples/`): exp1, exp2_student, multi-dim, stock returns.

## Experiment Design

### Exp A: Stage 2 internal allocation (n_1 vs r_1)

Fixed: n_0=250, r_0=20, B_2=5000, method=mixed
Sweep: (n_1, r_1) in [(100, 50), (250, 20), (500, 10), (1000, 5)]

### Exp B: S^0 vs LHS (Ablation 1)

Fixed: n_0=250, r_0=20, n_1=500, r_1=10
Sweep: method in [mixed, lhs]

### Exp C: B_1/B_2 ratio (Ablation 5)

Fixed: B_total=10000
Sweep: (n_0, r_0, n_1, r_1) in [
  (400, 20, 200, 10),   # more Stage 1
  (250, 20, 500, 10),   # balanced (current default)
  (150, 20, 700, 10),   # more Stage 2
]

## Metrics

- Coverage (marginal): P(L <= Y <= U), target 0.90
- Width: E[U - L]
- Interval Score (IS): Winkler score
- Max bin deviation: max_k |coverage(G_k) - 0.90| over K=10 equal-width bins
- Per-point CSV for post-hoc group coverage analysis

## Parameters

- n_macro: 20+ (for publication)
- alpha: 0.1
- n_test: 1000, r_test: 1
- n_cand: 1000
- t_grid_size: 500
- Hyperparameters: CV-tuned per DGP (pretrained_params.json)
