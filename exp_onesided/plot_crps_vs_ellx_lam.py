"""
plot_crps_vs_ellx_lam.py

Sweep (ell_x, lam) on a 2D grid while fixing h, and plot a heatmap
of mean CRPS for the exp2 simulator.

CRPS uses the discrete 1/M estimator:
    CRPS_i = (1/M) * sum_m [F_hat(t_m | x_i) - 1{Y_i <= t_m}]^2

Usage:
    python exp_onesided/plot_crps_vs_ellx_lam.py
    python exp_onesided/plot_crps_vs_ellx_lam.py --h 0.1 --n_macro 3
    python exp_onesided/plot_crps_vs_ellx_lam.py --n_ellx 8 --n_lam 8
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
    empirical = (Y_true[:, None] <= t_grid[None, :]).astype(float)
    squared_diff = (F_pred - empirical) ** 2
    return np.mean(squared_diff, axis=1)


# ---------------------------------------------------------------------------
# Single macrorep: sweep 2D grid
# ---------------------------------------------------------------------------

def _run_one_macrorep(
    seed: int,
    ellx_values: list[float],
    lam_values: list[float],
    h: float,
    n_train: int,
    r_train: int,
    n_test: int,
    t_grid_size: int,
) -> np.ndarray:
    """
    Fit CKME for each (ell_x, lam) pair and return mean CRPS matrix.

    Returns crps_grid : (n_ellx, n_lam)
    """
    rng = np.random.default_rng(seed)

    sim_cfg = get_experiment_config(SIMULATOR)
    simulator = sim_cfg["simulator"]
    x_lo = float(sim_cfg["bounds"][0].item())
    x_hi = float(sim_cfg["bounds"][1].item())

    # Training data — shared across all (ell_x, lam) combinations
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

    # t_grid — shared
    Y_lo = np.percentile(Y_train, 0.5)
    Y_hi = np.percentile(Y_train, 99.5)
    y_margin = 0.10 * (Y_hi - Y_lo)
    t_grid = np.linspace(Y_lo - y_margin, Y_hi + y_margin, t_grid_size)

    n_ellx = len(ellx_values)
    n_lam = len(lam_values)
    crps_grid = np.empty((n_ellx, n_lam))

    for i, ell_x in enumerate(ellx_values):
        for j, lam in enumerate(lam_values):
            params = Params(ell_x=ell_x, lam=lam, h=h)
            ckme = CKMEModel(indicator_type="logistic")
            ckme.fit(X_train, Y_train, params=params)

            F_pred = ckme.predict_cdf(X_test, t_grid=t_grid)  # (n_test, M)
            crps_per_point = _compute_crps_perpoint(F_pred, Y_test, t_grid)
            crps_grid[i, j] = float(np.mean(crps_per_point))

            print(
                f"  ell_x={ell_x:.4f}  lam={lam:.5f}  "
                f"mean CRPS={crps_grid[i, j]:.5f}"
            )

    return crps_grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2D heatmap of CRPS over (ell_x, lam) for exp2"
    )
    parser.add_argument("--h", type=float, default=0.1,
                        help="Fixed indicator bandwidth (default: 0.1)")
    parser.add_argument("--ellx_min", type=float, default=0.1)
    parser.add_argument("--ellx_max", type=float, default=5.0)
    parser.add_argument("--n_ellx", type=int, default=10,
                        help="Number of log-spaced ell_x values (default: 10)")
    parser.add_argument("--lam_min", type=float, default=1e-4)
    parser.add_argument("--lam_max", type=float, default=1.0)
    parser.add_argument("--n_lam", type=int, default=10,
                        help="Number of log-spaced lam values (default: 10)")
    parser.add_argument("--n_macro", type=int, default=1,
                        help="Number of macroreps (default: 1)")
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
                        help="Path to save figure. Defaults to output_dir/crps_vs_ellx_lam.png")
    args = parser.parse_args()

    ellx_values = list(np.logspace(np.log10(args.ellx_min), np.log10(args.ellx_max), args.n_ellx))
    lam_values = list(np.logspace(np.log10(args.lam_min), np.log10(args.lam_max), args.n_lam))

    print(f"Simulator  : {SIMULATOR}")
    print(f"Fixed h    : {args.h}")
    print(f"ell_x grid : {[round(v, 4) for v in ellx_values]}")
    print(f"lam grid   : {[round(v, 6) for v in lam_values]}")
    print(f"n_macro    : {args.n_macro}")

    # Accumulate over macroreps
    all_grids = []
    for macro_k in range(args.n_macro):
        seed = args.base_seed + macro_k
        print(f"\n=== Macrorep {macro_k} (seed={seed}) ===")
        grid = _run_one_macrorep(
            seed=seed,
            ellx_values=ellx_values,
            lam_values=lam_values,
            h=args.h,
            n_train=args.n_train,
            r_train=args.r_train,
            n_test=args.n_test,
            t_grid_size=args.t_grid_size,
        )
        all_grids.append(grid)

    crps_mean = np.mean(all_grids, axis=0)  # (n_ellx, n_lam)

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for macro_k, grid in enumerate(all_grids):
        for i, ell_x in enumerate(ellx_values):
            for j, lam in enumerate(lam_values):
                rows.append({
                    "macrorep": macro_k,
                    "ell_x": ell_x,
                    "lam": lam,
                    "crps": grid[i, j],
                })
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "crps_vs_ellx_lam.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV -> {csv_path}")

    # Find best (ell_x, lam)
    best_i, best_j = np.unravel_index(np.argmin(crps_mean), crps_mean.shape)
    print(
        f"Best: ell_x={ellx_values[best_i]:.4f}  "
        f"lam={lam_values[best_j]:.6f}  "
        f"CRPS={crps_mean[best_i, best_j]:.5f}"
    )

    # ---------------------------------------------------------------------------
    # Plot: heatmap
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        crps_mean,
        origin="lower",
        aspect="auto",
        cmap="viridis_r",  # reversed: darker = lower CRPS = better
        extent=[
            np.log10(lam_values[0]), np.log10(lam_values[-1]),
            np.log10(ellx_values[0]), np.log10(ellx_values[-1]),
        ],
    )
    plt.colorbar(im, ax=ax, label="Mean CRPS (1/M form)")

    # Mark best point
    ax.scatter(
        np.log10(lam_values[best_j]),
        np.log10(ellx_values[best_i]),
        color="tomato", s=120, marker="*", zorder=5,
        label=f"best: ell_x={ellx_values[best_i]:.3f}, lam={lam_values[best_j]:.5f}",
    )

    # Axis ticks: show original values
    lam_ticks = np.log10(lam_values)
    ellx_ticks = np.log10(ellx_values)
    ax.set_xticks(lam_ticks[::max(1, len(lam_values) // 6)])
    ax.set_xticklabels(
        [f"$10^{{{v:.1f}}}$" for v in lam_ticks[::max(1, len(lam_values) // 6)]],
        fontsize=8,
    )
    ax.set_yticks(ellx_ticks[::max(1, len(ellx_values) // 6)])
    ax.set_yticklabels(
        [f"$10^{{{v:.1f}}}$" for v in ellx_ticks[::max(1, len(ellx_values) // 6)]],
        fontsize=8,
    )

    ax.set_xlabel("lam  (log scale)")
    ax.set_ylabel("ell_x  (log scale)")
    ax.set_title(
        f"CRPS heatmap  [{SIMULATOR}, h={args.h}, n_macro={args.n_macro}]",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    save_path = args.save or os.path.join(args.output_dir, "crps_vs_ellx_lam.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {save_path}")
    plt.close()


if __name__ == "__main__":
    main()
