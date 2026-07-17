# exp_framing_validation

Mechanism-validation gates for the aleatoric/epistemic framing of the
journal paper (see `manuscript/journal_scale_adaptive/paper_storyline.md`,
Section 6). Each gate tests one falsifiable claim the intro relies on.
G1-G3 were run 2026-07-07 and passed (G3 in graded form). G4 was run
2026-07-08: G4b passed strongly, while G4a passed only as a
diagnostic/action demonstration.

| Gate | Script | Claim | Data source |
|------|--------|-------|-------------|
| G1 | `gate1_coverage_vs_scale.py` | Fixed-h bin coverage varies systematically with noise scale s(x); oracle h(x) flattens it | post-hoc on `exp_adaptive_h/output_exp2` (50 macroreps, no new simulation) |
| G2 | `gate2_score_homogeneity.py` | Fixed-h conformity-score distribution is heterogeneous across x-bins (KS vs permutation null); oracle h(x) homogenizes it | pilot re-run of the exp2 pipeline with IDENTICAL seeds (BASE_SEED=20260501), saving raw scores |
| G3 | `gate3_epistemic_diagnostics.py` | Under flat noise + a localized misfit, epistemic diagnostics (bootstrap tail-quantile variance, oracle CDF L2) localize the hotspot far better than CP interval width | standalone mini-DGP (no simulator registration) |
| G4 | `gate4_fix_epistemic.py` (spec: `spec_gate4.md`) | Diagnose -> act -> verify: u_tail-targeted Stage-2 sampling tests a data-scarcity hotspot (G4a); when the hotspot is capacity-limited, sampling alone fails and a lack-of-fit statistic routes to a model fix (G4b) | standalone mini-DGPs extending the G3 harness (20 macroreps, run 2026-07-08; G4a N1=15/20 follow-ups added 2026-07-08) |

## Usage (from project root)

```bash
python exp_framing_validation/gate1_coverage_vs_scale.py
python exp_framing_validation/gate2_score_homogeneity.py --n_macro 10 --n_workers 4
python exp_framing_validation/gate2_score_homogeneity.py --analyze_only   # re-plot only
python exp_framing_validation/gate3_epistemic_diagnostics.py
python exp_framing_validation/gate4_fix_epistemic.py --part all --n_macro 20
python exp_framing_validation/gate4_fix_epistemic.py --analyze_only   # re-plot + summary only
```

## Outputs (gitignored)

- `output_gate1/` — gate1_coverage_vs_scale.png, gate1_bin_coverage.csv, gate1_correlations.csv
- `output_gate2/` — macrorep_*/case_*/scores.csv (+meta.json), gate2_ks_profile.png, gate2_bin_q90.png, gate2_ks_summary.csv, gate2_bin_stats.csv
- `output_gate3/` — gate3_epistemic_diagnostics.png, gate3_curves.csv, gate3_summary.csv
- `output_gate4a/`, `output_gate4b/` — gate4{a,b}_metrics.csv, _curves.csv, _sites.csv, _lof.csv, figures (+ gate4b_ellx_cv.csv, gate4b_ellx_selected.csv)

## Headline results (2026-07-07 run)

- **G1 PASS**: fixed h gives Spearman rho(coverage−0.9, s(x)) of −0.93/−0.75/−0.95/−0.87
  (wsc_gauss/gibbs_s1/exp1/nongauss_A1L); oracle flattens exp1+gibbs fully,
  wsc/A1L partially (residual over-coverage at x=pi where s(x)~0.01 —
  epistemic floor).
- **G2 PASS**: fixed-h per-bin KS 3-5x above permutation null in 3/4 DGPs;
  local score-q90 varies up to 3x around the global q_hat. Oracle fully
  homogenizes exp1 (0/10 bins above null), partially wsc/A1L (same
  epistemic floor at x=pi); gibbs_s1 near-null under both arms
  (h << s(x) a.e. -> uniformly saturated scores).
- **G3 PASS (graded)**: marginal coverage 0.90 in both arms. Separation
  index (bump-window max / background median): width 4.9, oracle CDF L2
  32, u_tail 112. CP width does widen at the hotspot but 6-23x less
  sharply, and the global q_hat inflates (0.337 -> 0.361), taxing all x.
  The clean claim is "width cannot localize/attribute", not "width stays
  flat".

## Gate 4 results (2026-07-08 run, 20 macroreps, implemented from spec_gate4.md)

- **G4b PASS (core result)**: capacity-limited hotspot (bump sd=0.03 <<
  ell_x=0.1). Adding 600 targeted Stage-2 points leaves hot CDF L2
  unchanged (pre 0.122 -> 0.119; utail_fixed/lhs_fixed ratio 0.96 —
  bias floor confirmed). CV-retuning ell_x (15/20 macroreps pick 0.02)
  cuts hot CDF L2 ~9x (ratio 0.107, Wilcoxon p=9.5e-07). LOF separates
  the two states by ~26x (fixed median 289 vs retune median 11;
  P(fixed>10)=1.00). Absolute back-below-10 rate is only 0.30 — partly
  because 5/20 CV runs select ell_x=0.05 and stay genuinely misfit
  (LOF 60-104, hot CDF L2 0.046 — LOF correctly flags the CV failure),
  partly because threshold 10 is tight for a max-statistic. Use a
  RELATIVE rule (LOF drop >10x, or below background 99% quantile) in
  the paper. Trade-off found: retune raises global u_tail and q_hat
  (0.364 -> 0.383) — local bias is traded for global variance, which
  CP absorbs as slightly wider intervals.
- **G4a PARTIAL**: mechanism validated — u_tail flags the thinned
  window, targeted allocation concentrates 44/60 sites there (LHS 18),
  and post SI(u_tail) drops 375 -> 21 (lhs 255, oracle 411: only the
  u_tail policy removes the u_tail hotspot). But at B=60 the CDF-error
  advantage over LHS is marginal (ratio 0.97, p=0.08) because even the
  ORACLE allocation only reaches 0.86 — LHS already drops enough points
  in the window; remaining hot error is ell_x smoothing bias, not
  variance. q_hat refund negligible (0.3407 -> 0.3377 in ALL arms).
  Follow-up starved-budget runs on the same paired design did **not**
  rescue the CDF-error gate: N_1=20 gives utail/lhs hot CDF L2 ratio
  1.03 (oracle/lhs 0.88), and N_1=15 gives 1.01 (oracle/lhs 0.90).
  Thus the limiting issue is not only saturation; raw u_tail allocation
  removes the diagnostic hotspot but is not aligned enough with the
  CDF-error hotspot in this G4a DGP. A positive C1 demo needs a revised
  DGP or a different allocation rule (for example, flag by u_tail and
  then allocate more evenly within the flagged region). Coverage 0.90 in
  all arms/parts.
