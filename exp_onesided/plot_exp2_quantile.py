"""
plot_exp2_quantile.py

Visualize CKME conditional quantile estimation on exp2 (Gaussian noise).

For chosen x values, compare:
  - Estimated conditional CDF  F̂(t | x)   vs  true  F(t | x) = Φ((t - f(x)) / σ(x))
  - Estimated quantile function q̂_τ(x)     vs  true  q_τ(x) = f(x) + σ(x) Φ⁻¹(τ)

True quantities are available analytically because exp2 uses Gaussian noise:
    Y | X=x  ~  N(f(x), σ²(x))
    f(x)  = exp(x/10) sin(x)
    σ(x)  = 0.01 + 0.2(x − π)²

Params are loaded from exp_onesided/pretrained_params.json (key "exp2").

Layout:
  Row 0: one panel per x — conditional CDF (estimated vs true)
  Row 1: one panel per x — quantile function q(τ) (estimated vs true)

Usage:
    python exp_onesided/plot_exp2_quantile.py
    python exp_onesided/plot_exp2_quantile.py --x 1.0 3.14159 5.5
    python exp_onesided/plot_exp2_quantile.py --n_train 500 --r_train 15 --seed 7
    python exp_onesided/plot_exp2_quantile.py --save exp_onesided/output/exp2_quantile.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from CKME.ckme import CKMEModel
from CKME.parameters import Params
from Two_stage.sim_functions import exp2_simulator
from Two_stage.sim_functions.exp2 import (
    exp2_true_function,
    exp2_noise_variance_function,
    EXP2_X_BOUNDS,
)

# ── style ─────────────────────────────────────────────────────────────────────
C_CKME  = "#2166ac"
C_TRUE  = "#d6604d"
LW      = 2.0
FILL_A  = 0.12
TAU_GRID_SIZE = 300   # resolution for quantile function plot


# ── analytic helpers ──────────────────────────────────────────────────────────

def true_cdf(t: np.ndarray, x: float) -> np.ndarray:
    """F(t | x) = Φ((t − f(x)) / σ(x))"""
    mu  = exp2_true_function(np.array([x]))[0]
    sig = np.sqrt(exp2_noise_variance_function(np.array([x]))[0])
    return norm.cdf(t, loc=mu, scale=sig)


def true_quantile(tau: np.ndarray, x: float) -> np.ndarray:
    """q_τ(x) = f(x) + σ(x) Φ⁻¹(τ)"""
    mu  = exp2_true_function(np.array([x]))[0]
    sig = np.sqrt(exp2_noise_variance_function(np.array([x]))[0])
    return mu + sig * norm.ppf(tau)


# ── training data ──────────────────────────────────────────────────────────────

def generate_train_data(
    n_sites: int,
    r_reps: int,
    rng: np.random.Generator,
    x_query: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, Y, t_grid) for exp2.

    If x_query is provided, t_grid is extended to cover the true 0.1%–99.9%
    quantile range at each query point, ensuring predict_quantile never clips.
    """
    x_lo, x_hi = float(EXP2_X_BOUNDS[0].item()), float(EXP2_X_BOUNDS[1].item())
    X_sites = rng.uniform(x_lo, x_hi, size=(n_sites, 1))
    X = np.repeat(X_sites, r_reps, axis=0)   # (n_sites*r_reps, 1), site order matches CKMEModel
    Y = exp2_simulator(X.ravel(), random_state=int(rng.integers(0, 2**31)))

    # percentile-based t_grid (see CLAUDE.md)
    Y_lo = np.percentile(Y, 0.5)
    Y_hi = np.percentile(Y, 99.5)

    # Also cover the true quantile range at query points (avoid clipping)
    if x_query is not None:
        for xq in x_query:
            xq_arr = np.array([xq])
            mu  = float(exp2_true_function(xq_arr).item())
            sig = float(np.sqrt(exp2_noise_variance_function(xq_arr).item()))
            Y_lo = min(Y_lo, mu + sig * norm.ppf(0.001))
            Y_hi = max(Y_hi, mu + sig * norm.ppf(0.999))

    margin = 0.10 * (Y_hi - Y_lo)
    t_grid = np.linspace(Y_lo - margin, Y_hi + margin, 500)
    return X, Y, t_grid


# ── main plot ─────────────────────────────────────────────────────────────────

def make_figure(
    x_values: list[float],
    n_train: int,
    r_train: int,
    seed: int,
    monotone: bool,
) -> plt.Figure:
    rng = np.random.default_rng(seed)

    # Load params
    param_path = _root / "exp_onesided" / "pretrained_params.json"
    if not param_path.exists():
        raise FileNotFoundError(
            f"pretrained_params.json not found at {param_path}.\n"
            "Run: python exp_onesided/pretrain_params.py --sims exp2"
        )
    with open(param_path) as f:
        all_params = json.load(f)
    if "exp2" not in all_params:
        raise KeyError(
            "'exp2' key not found in pretrained_params.json.\n"
            "Run: python exp_onesided/pretrain_params.py --sims exp2"
        )
    p = all_params["exp2"]
    params = Params(ell_x=p["ell_x"], lam=p["lam"], h=p["h"])
    print(f"Loaded exp2 params: ell_x={params.ell_x:.4f}, lam={params.lam:.4g}, h={params.h:.4f}")

    # Train CKME (t_grid extended to cover true quantile range at query points)
    X, Y, t_grid = generate_train_data(n_train, r_train, rng, x_query=x_values)
    model = CKMEModel(indicator_type="logistic")
    model.fit(X, Y, params=params, r=r_train)
    print(f"Trained on {n_train} sites × {r_train} reps  (t_grid: {t_grid[0]:.2f} → {t_grid[-1]:.2f})")

    # Query CKME at chosen x values
    X_query = np.array([[x] for x in x_values])  # shape (K, 1)
    F_hat = model.predict_cdf(X_query, t_grid, monotone=monotone)  # (K, M)

    # Tau grid for quantile panels
    tau_grid = np.linspace(0.02, 0.98, TAU_GRID_SIZE)
    # predict_quantile takes a single tau; vectorise over full tau_grid.
    # monotone=True (isotonic) is used here so the CDF is forced to reach 1.0
    # and quantile inversion is well-defined across [0.02, 0.98].
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Q_hat = np.column_stack([
            model.predict_quantile(X_query, tau=float(tau), t_grid=t_grid, monotone=True)
            for tau in tau_grid
        ])  # (K, T)

    K = len(x_values)
    fig, axes = plt.subplots(
        2, K,
        figsize=(4.5 * K, 8),
        squeeze=False,
    )
    fig.suptitle(
        f"exp2 — CKME quantile estimation  "
        f"(n={n_train}×{r_train}, seed={seed})\n"
        f"ell_x={params.ell_x:.3f}, lam={params.lam:.2g}, h={params.h:.3f}",
        fontsize=11, fontweight="bold",
    )

    for k, x in enumerate(x_values):
        mu  = float(exp2_true_function(np.array([x])).item())
        sig = float(np.sqrt(exp2_noise_variance_function(np.array([x])).item()))

        # ── Row 0: Conditional CDF ─────────────────────────────────────────
        ax0 = axes[0, k]
        ax0.plot(t_grid, F_hat[k], color=C_CKME, lw=LW, label="CKME  F̂(t|x)")
        ax0.plot(t_grid, true_cdf(t_grid, x), color=C_TRUE, lw=LW,
                 ls="--", label="True  F(t|x)")
        ax0.set_xlabel("t", fontsize=10)
        ax0.set_ylabel("CDF", fontsize=10)
        ax0.set_title(
            f"x = {x:.3f}\nμ = {mu:.2f},  σ = {sig:.3f}",
            fontsize=10,
        )
        ax0.legend(fontsize=9, framealpha=0.8)
        ax0.grid(True, ls=":", alpha=0.4)
        ax0.set_ylim(-0.05, 1.05)

        # ── Row 1: Quantile function q(τ) ──────────────────────────────────
        ax1 = axes[1, k]
        q_true = true_quantile(tau_grid, x)
        ax1.plot(tau_grid, Q_hat[k], color=C_CKME, lw=LW, label="CKME  q̂_τ(x)")
        ax1.plot(tau_grid, q_true,   color=C_TRUE,  lw=LW, ls="--", label="True  q_τ(x)")
        ax1.set_xlabel("τ", fontsize=10)
        ax1.set_ylabel("quantile", fontsize=10)
        ax1.set_title(f"Quantile function  (x = {x:.3f})", fontsize=10)
        ax1.legend(fontsize=9, framealpha=0.8)
        ax1.grid(True, ls=":", alpha=0.4)

    fig.tight_layout()
    return fig


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot CKME conditional quantile estimation on exp2"
    )
    parser.add_argument(
        "--x", type=float, nargs="+",
        default=[np.pi / 2, np.pi, 3 * np.pi / 2],
        help="Query x values (default: π/2, π, 3π/2)",
    )
    parser.add_argument(
        "--n_train", type=int, default=300,
        help="Number of training sites (default: 300)",
    )
    parser.add_argument(
        "--r_train", type=int, default=10,
        help="Replications per training site (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--monotone", action="store_true",
        help="Apply isotonic regression to enforce CDF monotonicity",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Output path for PNG. If omitted, shows interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig = make_figure(
        x_values=args.x,
        n_train=args.n_train,
        r_train=args.r_train,
        seed=args.seed,
        monotone=args.monotone,
    )
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
