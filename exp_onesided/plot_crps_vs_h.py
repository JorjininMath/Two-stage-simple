"""
plot_crps_vs_h.py

Sweep h (indicator bandwidth) while fixing ell_x and lam, and plot
mean CRPS vs h for the exp2 simulator.

CRPS uses the discrete 1/M estimator:
    CRPS_i = (1/M) * sum_m [F_hat(t_m | x_i) - 1{Y_i <= t_m}]^2

Usage:
    python exp_onesided/plot_crps_vs_h.py
    python exp_onesided/plot_crps_vs_h.py --ell_x 0.5 --lam 0.001 --n_macro 5
    python exp_onesided/plot_crps_vs_h.py --h_values 0.01,0.05,0.1,0.2,0.5,1.0
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CKME.ckme import CKMEModel
from CKME.parameters import Params
from Two_stage.sim_functions import get_experiment_config

SIMULATOR = "exp2"


# ---------------------------------------------------------------------------
# CRPS (1/M form)
# ---------------------------------------------------------------------------

def _compute_crps_perpoint(
    F_pred: np.ndarray,
    Y_true: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    CRPS_i = (1/M) * sum_m [F_hat(t_m|x_i) - 1{Y_i <= t_m}]^2

    F_pred : (n, M)
    Y_true : (n,)
    t_grid : (M,)

    Returns crps : (n,)
    """
    empirical = (Y_true[:, None] <= t_grid[None, :]).astype(float)  # (n, M)
    squared_diff = (F_pred - empirical) ** 2                         # (n, M)
    return np.mean(squared_diff, axis=1)                             # (n,)


# ---------------------------------------------------------------------------
# Single run: fix ell_x, lam; vary h
# ---------------------------------------------------------------------------

def _run_one_macrorep(
    seed: int,
    h_values: list[float],
    ell_x: float,
    lam: float,
    n_train: int,
    r_train: int,
    n_test: int,
    t_grid_size: int,
) -> np.ndarray:
    """
    For one macrorep, fit CKME for each h and return mean CRPS per h.

    Returns crps_by_h : (len(h_values),)
    """
    rng = np.random.default_rng(seed)

    sim_cfg = get_experiment_config(SIMULATOR)
    simulator = sim_cfg["simulator"]
    x_lo = float(sim_cfg["bounds"][0].item())
    x_hi = float(sim_cfg["bounds"][1].item())

    # Training data — shared across all h values
    X_sites = rng.uniform(x_lo, x_hi, size=(n_train, 1))
    Y_reps = [
        simulator(X_sites.ravel(), random_state=int(rng.integers(0, 2**31)))
        for _ in range(r_train)
    ]
    X_train = np.tile(X_sites, (r_train, 1))
    Y_train = np.concatenate(Y_reps)

    # Test data — uniform grid
    X_test = np.linspace(x_lo, x_hi, n_test).reshape(-1, 1)
    Y_test = simulator(X_test.ravel(), random_state=int(rng.integers(0, 2**31)))

    # t_grid — shared across all h values
    Y_lo = np.percentile(Y_train, 0.5)
    Y_hi = np.percentile(Y_train, 99.5)
    y_margin = 0.10 * (Y_hi - Y_lo)
    t_grid = np.linspace(Y_lo - y_margin, Y_hi + y_margin, t_grid_size)

    crps_by_h = np.empty(len(h_values))

    for i, h in enumerate(h_values):
        params = Params(ell_x=ell_x, lam=lam, h=h)
        ckme = CKMEModel(indicator_type="logistic")
        ckme.fit(X_train, Y_train, params=params)

        F_pred = ckme.predict_cdf(X_test, t_grid=t_grid)  # (n_test, M)
        crps_per_point = _compute_crps_perpoint(F_pred, Y_test, t_grid)
        crps_by_h[i] = float(np.mean(crps_per_point))

        print(f"  h={h:.4f}  mean CRPS={crps_by_h[i]:.5f}")

    return crps_by_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot mean CRPS vs h for exp2 (fixed ell_x, lam)"
    )
    parser.add_argument("--ell_x", type=float, default=0.23853323044733007,
                        help="Fixed RBF length scale (tuned from crps_vs_ellx_lam sweep)")
    parser.add_argument("--lam", type=float, default=0.0002782559402207126,
                        help="Fixed regularization (tuned from crps_vs_ellx_lam sweep)")
    parser.add_argument(
        "--h_values",
        type=str,
        default=None,
        help=(
            "Comma-separated h values to sweep, e.g. 0.01,0.05,0.1,0.2,0.5. "
            "If omitted, uses 20 log-spaced values from 0.01 to 1.0."
        ),
    )
    parser.add_argument("--n_h", type=int, default=10,
                        help="Number of log-spaced h values (used if --h_values not set)")
    parser.add_argument("--h_min", type=float, default=0.001)
    parser.add_argument("--h_max", type=float, default=0.5)
    parser.add_argument("--n_macro", type=int, default=1,
                        help="Number of macroreps for stability (default: 1)")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--r_train", type=int, default=5)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--t_grid_size", type=int, default=500)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output"),
    )
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save figure. Defaults to output_dir/crps_vs_h.png")
    args = parser.parse_args()

    # Build h grid
    if args.h_values:
        h_values = [float(v.strip()) for v in args.h_values.split(",")]
    else:
        h_values = list(np.logspace(np.log10(args.h_min), np.log10(args.h_max), args.n_h))

    print(f"Simulator : {SIMULATOR}")
    print(f"Fixed     : ell_x={args.ell_x}, lam={args.lam}")
    print(f"h sweep   : {[round(h, 5) for h in h_values]}")
    print(f"n_macro   : {args.n_macro}")

    # Run macroreps
    all_crps = []  # list of (n_h,) arrays

    for macro_k in range(args.n_macro):
        seed = args.base_seed + macro_k
        print(f"\n=== Macrorep {macro_k} (seed={seed}) ===")
        crps_by_h = _run_one_macrorep(
            seed=seed,
            h_values=h_values,
            ell_x=args.ell_x,
            lam=args.lam,
            n_train=args.n_train,
            r_train=args.r_train,
            n_test=args.n_test,
            t_grid_size=args.t_grid_size,
        )
        all_crps.append(crps_by_h)

    crps_mat = np.stack(all_crps, axis=0)  # (n_macro, n_h)
    crps_mean = crps_mat.mean(axis=0)
    crps_std = crps_mat.std(axis=0)

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    df_rows = []
    for macro_k, row in enumerate(all_crps):
        for h, c in zip(h_values, row):
            df_rows.append({"macrorep": macro_k, "h": h, "crps": c})
    df = pd.DataFrame(df_rows)
    csv_path = os.path.join(args.output_dir, "crps_vs_h.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV -> {csv_path}")

    # Plot
    h_arr = np.array(h_values)
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(h_arr, crps_mean, color="steelblue", lw=2.0, marker="o",
            markersize=4, label="mean CRPS")
    if args.n_macro > 1:
        ax.fill_between(
            h_arr,
            crps_mean - crps_std,
            crps_mean + crps_std,
            color="steelblue", alpha=0.25, label="±1 std"
        )

    # Mark minimum
    best_idx = int(np.argmin(crps_mean))
    ax.axvline(h_arr[best_idx], color="tomato", lw=1.2, ls="--",
               label=f"best h={h_arr[best_idx]:.4f}")
    ax.scatter([h_arr[best_idx]], [crps_mean[best_idx]],
               color="tomato", zorder=5, s=60)

    ax.set_xscale("log")
    ax.set_xlabel("h  (indicator bandwidth, log scale)")
    ax.set_ylabel("Mean CRPS  (1/M form)")
    ax.set_title(
        f"CRPS vs h  [{SIMULATOR}, ell_x={args.ell_x}, lam={args.lam}, "
        f"n_macro={args.n_macro}]",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    save_path = args.save or os.path.join(args.output_dir, "crps_vs_h.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
