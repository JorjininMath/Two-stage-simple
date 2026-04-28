# exp_wsc — WSC 2026 Paper Experiments (Tables 2–3)

This folder reproduces **Tables 2 and 3** from:

> *Two-Stage Adaptive Design for Conditional Distribution Estimation*, WSC 2026.

---

## Experimental Setup

| Item | Value |
|------|-------|
| **DGP** | `wsc_gauss` (Exp 1, Gaussian) and `nongauss_A1L` (Exp 2, Student-t ν=3) |
| **True function** | f(x) = exp(x/10)·sin(x),  x ∈ [0, 2π] |
| **Noise std** | σ(x) = 0.01 + 0.2·(x − π)² |
| **Stage 1 budget** | n₀ = 500 sites × r₀ = 10 reps = 5 000 samples (fixed) |
| **Stage 2 budgets** | (n₁, r₁) ∈ {(100, 50), (200, 25), (500, 10)} |
| **Site selection** | LHS / Adaptive (∝ S⁰) / Mixture (ρ = 0.7) |
| **Benchmarks** | DCP-DR, hetGP (R; skipped automatically if R is unavailable) |
| **Coverage target** | 1 − α = 90%  (α = 0.1) |
| **Macroreps** | 50 (default) |

---

## Steps to Reproduce

### Step 0 — Install dependencies

From the project root:

```bash
conda env create -f environment.yml
conda activate ckme_env
```

R benchmarks (DCP-DR and hetGP) additionally require:
```r
install.packages(c("quantreg", "hetGP", "mvtnorm", "MASS"))
```
If R is not available, CKME results are still saved; benchmark columns will be `NaN`.

---

### Step 1 — Pre-tune hyperparameters (run once)

Runs k-fold CV on a pilot dataset to find the best `(ell_x, lam, h)` for each DGP.
Results are saved to `exp_wsc/pretrained_params.json` and loaded automatically in Step 2.

```bash
python exp_wsc/pretrain_params.py
```

Optional arguments:
```
--n_pilot 100      # pilot design sites  (default: 100)
--r_pilot 10       # replications per site (default: 10)
--cv_folds 5       # CV folds (default: 5)
--seed 0
```

---

### Step 2 — Run the main experiment

Sequential (single core, ~8–12 h for 50 macroreps):
```bash
python exp_wsc/run_wsc_compare.py --n_macro 50
```

Parallel (recommended on a multi-core machine):
```bash
python exp_wsc/run_wsc_compare.py --n_macro 50 --n_workers 8
```

Quick smoke test (1 macrorep, ~5 min):
```bash
python exp_wsc/run_wsc_compare.py --n_macro 1
```

Output is written to `exp_wsc/output/`:
```
output/
  wsc_per_macrorep.csv   — one row per (macrorep × DGP × budget × method)
  wsc_summary.csv        — aggregated mean ± sd across macroreps
  macrorep_<k>/          — per-macrorep raw data (X0/Y0/X1/Y1/X_test/Y_test)
```

---

### Step 3 — Generate Tables 2–3

```bash
python exp_wsc/make_tables.py
```

Prints Tables 2 and 3 to stdout in a readable text format.  Each cell shows
`mean (sd)` across macroreps for Coverage, Width, and Interval Score.

---

## Output Files

| File | Description |
|------|-------------|
| `output/wsc_summary.csv` | Main results table (mean ± sd per DGP × budget × method) |
| `output/wsc_per_macrorep.csv` | Raw macrorep results (useful for plotting or extra analysis) |
| `pretrained_params.json` | CV-tuned hyperparameters (regenerate with `pretrain_params.py`) |

---

## Metrics

| Metric | Definition |
|--------|-----------|
| **Coverage** | P(L ≤ Y ≤ U); target = 0.90 |
| **Width** | E[U − L] |
| **Interval Score (IS)** | (U−L) + (2/α)(L−Y)₊ + (2/α)(Y−U)₊ |

---

## Notes

- `wsc_gauss` is registered in `Two_stage/sim_functions/__init__.py` as  
  `make_exp2_gauss_simulator(sigma_base=0.01, sigma_slope=0.20)`.
- `nongauss_A1L` is `make_nongauss_A1_simulator(nu=3.0)` from `sim_nongauss_A1.py`.
- Both DGPs share the same σ(x); only the noise distribution differs.
