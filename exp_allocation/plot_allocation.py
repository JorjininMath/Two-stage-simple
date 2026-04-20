"""
plot_allocation.py

Plot exp_allocation results for exp1: coverage / width / interval score
vs Stage-2 resource allocation (n_1, r_1 pairs, fixed budget n_1*r_1=5000).

Three methods compared: CKME, DCP-DR, hetGP.
Three site-selection strategies: lhs, mixed, sampling.

Layout: 3 rows (coverage / width / IS) × 3 cols (lhs / mixed / sampling)
x-axis: allocation label "n1=100\nr0=50", ...

Usage:
    python exp_allocation/plot_allocation.py
    python exp_allocation/plot_allocation.py --sim exp1 --save exp_allocation/output/exp1/allocation_plot.png
    python exp_allocation/plot_allocation.py --input exp_allocation/output/exp1/allocation_summary.csv
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── style ────────────────────────────────────────────────────────────────────
METHOD_COLORS = {"CKME": "#2166ac", "DCP-DR": "#d6604d", "hetGP": "#4dac26"}
METHOD_MARKERS = {"CKME": "o", "DCP-DR": "s", "hetGP": "^"}
SITE_METHODS = ["lhs", "mixed", "sampling"]
BENCH_METHODS = ["CKME", "DCP-DR", "hetGP"]
TARGET_COV = 0.90   # alpha=0.1 → 90% target
LW = 1.8
MS = 7


def load_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            "Run run_allocation_compare.py first."
        )
    df = pd.read_csv(path)
    return df


def alloc_label(n1: int, r1: int) -> str:
    return f"n₁={n1}\nr₁={r1}"


def make_figure(df: pd.DataFrame, sim: str) -> plt.Figure:
    sub = df[df["n_1"].notna()].copy()

    # sort by n_1 ascending (100 → 250 → 1000)
    allocs = sorted(sub[["n_1", "r_1"]].drop_duplicates().itertuples(index=False),
                    key=lambda t: t.n_1)
    x_labels = [alloc_label(int(a.n_1), int(a.r_1)) for a in allocs]
    x_pos = list(range(len(allocs)))

    metrics = [
        ("coverage", "Coverage",       "Coverage (target = 90%)"),
        ("width",    "Width",           "Interval width"),
        ("IS",       "Interval score",  "Interval score (Winkler)"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=False)
    fig.suptitle(
        f"exp_allocation — {sim}\nCoverage / Width / IS  vs  Stage-2 budget allocation",
        fontsize=12, fontweight="bold",
    )

    for row_idx, (metric_key, metric_label, row_title) in enumerate(metrics):
        for col_idx, site_method in enumerate(SITE_METHODS):
            ax = axes[row_idx][col_idx]
            sm_sub = sub[sub["method"] == site_method]

            for bm in BENCH_METHODS:
                mean_col = f"{bm}_{metric_key}_mean"
                std_col  = f"{bm}_{metric_key}_std"
                if mean_col not in sm_sub.columns:
                    continue

                means, stds = [], []
                for a in allocs:
                    row = sm_sub[(sm_sub["n_1"] == a.n_1) & (sm_sub["r_1"] == a.r_1)]
                    if row.empty:
                        means.append(np.nan)
                        stds.append(0.0)
                    else:
                        means.append(float(row[mean_col].iloc[0]))
                        stds.append(float(row[std_col].iloc[0]))

                means, stds = np.array(means), np.array(stds)
                c = METHOD_COLORS[bm]
                ax.plot(x_pos, means, color=c, lw=LW,
                        marker=METHOD_MARKERS[bm], markersize=MS, label=bm, zorder=3)
                ax.fill_between(x_pos, means - stds, means + stds,
                                color=c, alpha=0.12)

            # target coverage reference line
            if metric_key == "coverage":
                ax.axhline(TARGET_COV, color="gray", ls="--", lw=1.0, alpha=0.7,
                           label=f"target {int(TARGET_COV*100)}%")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.set_ylabel(metric_label, fontsize=9)
            ax.grid(True, ls=":", alpha=0.4)

            if row_idx == 0:
                ax.set_title(f"site selection: {site_method}", fontsize=10, fontweight="bold")
            if row_idx == 2:
                ax.set_xlabel("allocation (n₁, r₁)", fontsize=9)

            ax.legend(fontsize=7, framealpha=0.8)

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(script_dir, "output", "exp1", "allocation_summary.csv")
    parser = argparse.ArgumentParser(description="Plot exp_allocation results")
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--sim", type=str, default="exp1",
                        help="Simulator name (used for title)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save path for PNG. If omitted, show interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_summary(args.input)
    fig = make_figure(df, args.sim)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
