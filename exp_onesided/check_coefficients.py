"""
check_coefficients.py

Diagnostic: inspect CKME coefficient vectors c(x) for exp2.

For each query point x, c(x) is the solution to:
    (K_X + n*lam*I) c(x) = k_X(X_train, x)

Theory says c(x) is NOT a probability vector:
  - entries can be negative
  - sum is not guaranteed to be 1

This script empirically checks:
  1. sum(c(x)) for each query point  (are they ≈ 1?)
  2. fraction of negative entries
  3. fraction of entries outside [0, 1]

Usage:
    python exp_onesided/check_coefficients.py
    python exp_onesided/check_coefficients.py --n_train 2000 --n_query 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from CKME.coefficients import build_cholesky_factor, compute_ckme_coeffs
from CKME.kernels import make_x_rbf_kernel
from CKME.parameters import Params
from Two_stage.sim_functions import get_experiment_config

R_TRAIN    = 10
_PARAM_PATH = Path(__file__).resolve().parent / "pretrained_params.json"


def _load_params(sim_name: str) -> Params:
    with open(_PARAM_PATH) as f:
        raw = json.load(f)
    if sim_name not in raw:
        raise KeyError(f"'{sim_name}' not found in pretrained_params.json. "
                       f"Available: {list(raw.keys())}")
    p = raw[sim_name]
    return Params(ell_x=p["ell_x"], lam=p["lam"], h=p["h"])


def check_coefficients(sim_name: str, n_train: int, n_query: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    params = _load_params(sim_name)
    print(f"Simulator: {sim_name}")
    print(f"Params: ell_x={params.ell_x:.4f}  lam={params.lam:.2e}  h={params.h:.4f}")
    print(f"n_train={n_train}  n_query={n_query}  seed={seed}\n")

    sim_cfg   = get_experiment_config(sim_name)
    simulator = sim_cfg["simulator"]
    x_lo = float(sim_cfg["bounds"][0].item())
    x_hi = float(sim_cfg["bounds"][1].item())

    # Training data
    X_sites = rng.uniform(x_lo, x_hi, size=(n_train, 1))

    # Query points (uniform grid)
    X_query = np.linspace(x_lo, x_hi, n_query).reshape(-1, 1)

    # Build kernel and Cholesky factor
    kx      = make_x_rbf_kernel(params.ell_x)
    K_sites = kx(X_sites, X_sites)
    L       = build_cholesky_factor(K_sites, n_train, params.lam)

    # Compute coefficient matrix: shape (n_train, n_query)
    C = compute_ckme_coeffs(L, kx, X_sites, X_query)  # (n_train, n_query)

    # ── Per-query stats ───────────────────────────────────────────────────────
    col_sums     = C.sum(axis=0)          # (n_query,)
    col_neg_frac = (C < 0).mean(axis=0)  # fraction of negative entries per query
    col_out_frac = ((C < 0) | (C > 1)).mean(axis=0)  # outside [0,1]

    print("=" * 60)
    print(f"{'Metric':<35} {'min':>8} {'mean':>8} {'max':>8}")
    print("-" * 60)
    print(f"{'sum(c(x))':<35} {col_sums.min():>8.4f} {col_sums.mean():>8.4f} {col_sums.max():>8.4f}")
    print(f"{'|sum(c(x)) - 1|':<35} {np.abs(col_sums-1).min():>8.4f} {np.abs(col_sums-1).mean():>8.4f} {np.abs(col_sums-1).max():>8.4f}")
    print(f"{'frac negative entries':<35} {col_neg_frac.min():>8.4f} {col_neg_frac.mean():>8.4f} {col_neg_frac.max():>8.4f}")
    print(f"{'frac outside [0,1]':<35} {col_out_frac.min():>8.4f} {col_out_frac.mean():>8.4f} {col_out_frac.max():>8.4f}")
    print("=" * 60)

    # ── Global stats ──────────────────────────────────────────────────────────
    print(f"\nGlobal coefficient range : [{C.min():.6f}, {C.max():.6f}]")
    print(f"Global frac negative     : {(C < 0).mean():.4f}")
    print(f"Global frac outside [0,1]: {((C < 0) | (C > 1)).mean():.4f}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    sum_ok  = np.allclose(col_sums, 1.0, atol=1e-6)
    nonneg  = (C >= 0).all()
    in_unit = ((C >= 0) & (C <= 1)).all()

    print("\n--- Verdict ---")
    print(f"  sum(c(x)) ≈ 1 for all query points : {'YES' if sum_ok  else 'NO'}")
    print(f"  all c_i(x) >= 0                    : {'YES' if nonneg  else 'NO'}")
    print(f"  all c_i(x) in [0, 1]               : {'YES' if in_unit else 'NO'}")

    if not sum_ok:
        print("\n  NOTE: c(x) does NOT sum to 1 — consistent with theory (regularized")
        print("        linear system, not a normalized kernel smoother).")
    if not nonneg:
        print("\n  NOTE: c(x) has negative entries — also consistent with theory.")


def main():
    parser = argparse.ArgumentParser(description="Check CKME coefficient properties")
    parser.add_argument("--sim",     type=str, default="exp2",
                        help="Simulator name (must be in pretrained_params.json)")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_query", type=int, default=100)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()
    check_coefficients(args.sim, args.n_train, args.n_query, args.seed)


if __name__ == "__main__":
    main()
