"""
plot_intervals.py

Plot prediction intervals on test points for one macrorep.

2x3 figure:
  Row 1: exp2 (Gaussian)        — CKME fixed / CKME adaptive / DCP-DR
  Row 2: nongauss_A1L (Student-t) — CKME fixed / CKME adaptive / DCP-DR

Usage:
  python dissertation_use/plot_intervals.py
  python dissertation_use/plot_intervals.py --macrorep 2 --save dissertation_use/output/intervals.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

SIMULATORS = [
    ("exp2",          "Exp1: Gaussian noise"),
    ("nongauss_A1L",  r"Exp2: Student-$t$ ($\nu$=3) noise"),
]

METHODS = [
    {
        "label": "CKME (fixed h)",
        "L_col": "L", "U_col": "U", "cov_col": "covered_score",
        "color": "#2166ac",
    },
    {
        "label": "CKME (adaptive h)",
        "L_col": "L_adaptive", "U_col": "U_adaptive", "cov_col": "covered_adaptive",
        "color": "#b2182b",
    },
    {
        "label": "DCP-DR",
        "L_col": "L_dr", "U_col": "U_dr", "cov_col": "covered_score_dr",
        "color": "#d6604d",
    },
    {
        "label": "DCP-QR",
        "L_col": "L_qr", "U_col": "U_qr", "cov_col": "covered_score_qr",
        "color": "#4dac26",
    },
]

_PI = np.pi


def _true_mean(x):
    return np.exp(x / 10) * np.sin(x)


def main():
    parser = argparse.ArgumentParser(description="Plot prediction intervals on test points")
    parser.add_argument("--output_dir", type=str, default="dissertation_use/output")
    parser.add_argument("--macrorep", type=int, default=0)
    parser.add_argument("--site_method", type=str, default="lhs")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    target = 1 - args.alpha

    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
    fig.suptitle(f"Prediction Intervals on Test Points (macrorep {args.macrorep})",
                 fontsize=13, y=0.98)

    for row, (sim, sim_label) in enumerate(SIMULATORS):
        csv_path = out_dir / f"macrorep_{args.macrorep}" / f"case_{sim}_{args.site_method}" / "per_point.csv"
        if not csv_path.exists():
            for col in range(len(METHODS)):
                axes[row, col].set_title(f"{sim_label} — no data", fontsize=9)
            continue

        df = pd.read_csv(csv_path)
        x = df["x0"].values
        y = df["y"].values
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        # True mean curve
        x_fine = np.linspace(0, 2 * _PI, 200)
        mu_fine = _true_mean(x_fine)

        for col, m in enumerate(METHODS):
            ax = axes[row, col]

            L_col, U_col, cov_col = m["L_col"], m["U_col"], m["cov_col"]
            if L_col not in df.columns or U_col not in df.columns:
                ax.set_title(f"{m['label']} — missing", fontsize=9)
                continue

            L = df[L_col].values[order]
            U = df[U_col].values[order]

            # Interval band
            ax.fill_between(x_sorted, L, U, alpha=0.25, color=m["color"],
                            label="interval")

            # True mean
            ax.plot(x_fine, mu_fine, "k-", linewidth=0.8, alpha=0.5, label="true mean")

            # Test points: covered vs not
            if cov_col in df.columns:
                cov = df[cov_col].values[order]
                covered_mask = cov == 1
                ax.scatter(x_sorted[covered_mask], y_sorted[covered_mask],
                           s=4, c="gray", alpha=0.4, zorder=2, label="covered")
                ax.scatter(x_sorted[~covered_mask], y_sorted[~covered_mask],
                           s=12, c="red", marker="x", zorder=3, label="missed")
                cov_rate = cov.mean()
            else:
                ax.scatter(x_sorted, y_sorted, s=4, c="gray", alpha=0.4, zorder=2)
                cov_rate = np.nan

            # Width stats
            width = U - L
            mean_w = np.nanmean(width)

            ax.set_title(f"{m['label']}\ncov={cov_rate:.3f}, width={mean_w:.2f}",
                         fontsize=9)

            if col == 0:
                ax.set_ylabel(sim_label, fontsize=9)
            if row == 1:
                ax.set_xlabel("x")
                ax.set_xticks([0, _PI, 2 * _PI])
                ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])

            if row == 0 and col == 0:
                ax.legend(fontsize=6, loc="upper left")

    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
