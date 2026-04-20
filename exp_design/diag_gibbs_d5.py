"""Diagnostic: why does CKME's CDF saturate below 1 on gibbs_s1_d5?

Trains Stage 1 on gibbs_s1_d5 (n_0=100, r_0=10, CV), then on a handful of
candidate x's reports:
  - CV-selected (ell_x, lam, h)
  - t_grid range and Y_all range
  - F(t_lo|x), F(t_hi|x), max CDF over t_grid
  - c(x) diagnostics: sum, frac negative
If F(t_hi|x) is well below 1 for many x, tail-based S^0 is degenerate.
"""
from __future__ import annotations
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np

from exp_design.run_design_compare import load_dgp, _train_stage1, PARAM_GRID
from Two_stage.design import generate_space_filling_design


def main():
    dgp_name = "gibbs_s1_d5"
    cfg = load_dgp(dgp_name)
    d = cfg["d"]
    bounds = cfg["bounds"]

    n_0, r_0, t_grid_size = 100, 10, 500
    seed = 2026

    res = _train_stage1(dgp_name, cfg, n_0, r_0, t_grid_size, seed + 2, params=None)
    print(f"CV-selected params: ell_x={res.params.ell_x}, lam={res.params.lam}, h={res.params.h}")
    print(f"Y_all:  min={res.Y_all.min():.3f}  max={res.Y_all.max():.3f}  "
          f"mean={res.Y_all.mean():.3f}  std={res.Y_all.std():.3f}")
    t_grid = res.t_grid
    print(f"t_grid: [{t_grid[0]:.3f}, {t_grid[-1]:.3f}]  (M={len(t_grid)})")
    print()

    rng = np.random.default_rng(seed + 100)
    X_cand = generate_space_filling_design(n=20, d=d, method="lhs",
                                           bounds=bounds, random_state=seed + 101)
    F = res.model.predict_cdf(X_cand, t_grid)  # (20, M)
    print(f"F̂(t|x) on 20 candidates:")
    print(f"  F(t_lo|x):  mean={F[:, 0].mean():.4f}  max={F[:, 0].max():.4f}")
    print(f"  F(t_hi|x):  mean={F[:, -1].mean():.4f}  min={F[:, -1].min():.4f}  max={F[:, -1].max():.4f}")
    print(f"  max over t_grid per x:  mean={F.max(axis=1).mean():.4f}  "
          f"min={F.max(axis=1).min():.4f}")
    print(f"  min over t_grid per x:  mean={F.min(axis=1).mean():.4f}  "
          f"max={F.min(axis=1).max():.4f}")
    print()

    from CKME.coefficients import compute_ckme_coeffs
    C = compute_ckme_coeffs(res.model.L, res.model.kx, res.model.X, X_cand)
    sums = C.sum(axis=0)
    neg_frac = (C < 0).mean(axis=0)
    print(f"c(x) sums:       mean={sums.mean():.4f}  min={sums.min():.4f}  max={sums.max():.4f}")
    print(f"c(x) neg frac:   mean={neg_frac.mean():.4f}  max={neg_frac.max():.4f}")
    print(f"c(x) range:      [{C.min():.4f}, {C.max():.4f}]")


if __name__ == "__main__":
    main()
