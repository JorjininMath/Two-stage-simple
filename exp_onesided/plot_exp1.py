"""
plot_exp1.py

Dedicated comparison plot for exp1 (MG1 queue, Gaussian noise):
CKME vs QR quantile estimation error across training sizes.

Layout: 2x2 grid
  [row 0] mean |err|  at tau=0.05  |  mean |err|  at tau=0.95
  [row 1] sup  |err|  at tau=0.05  |  sup  |err|  at tau=0.95

Usage:
    python exp_onesided/plot_exp1.py
    python exp_onesided/plot_exp1.py --save exp_onesided/output_scaling/exp1_comparison.png
    python exp_onesided/plot_exp1.py --input exp_onesided/output_scaling/scaling_summary.csv
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── style constants ──────────────────────────────────────────────────────────
COLORS = {"CKME": "#2166ac", "QR": "#d6604d"}
MARKERS = {"CKME": "o", "QR": "s"}
FILL_ALPHA = 0.15
LINE_LW = 2.0
MARKER_SIZE = 6


def load_summary(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Summary file not found: {path}\n"
            "Run run_ckme_scaling.py first."
        )
    df = pd.read_csv(path)
    return df[df["simulator"] == "exp1"].copy()


def _draw_method(ax, df: pd.DataFrame, metric_mean: str, metric_std: str,
                 tau: float, method: str, n_vals: list[int]) -> None:
    sub = df[(df["tau"] == tau) & (df["method"] == method)]
    means, stds = [], []
    for n in n_vals:
        row = sub[sub["n_train"] == n]
        if row.empty:
            means.append(np.nan)
            stds.append(0.0)
        else:
            means.append(float(row[metric_mean].iloc[0]))
            stds.append(float(row[metric_std].iloc[0]))
    means, stds = np.array(means), np.array(stds)
    c = COLORS[method]
    ax.plot(n_vals, means,
            color=c, lw=LINE_LW, marker=MARKERS[method],
            markersize=MARKER_SIZE, label=method, zorder=3)
    ax.fill_between(n_vals, means - stds, means + stds,
                    color=c, alpha=FILL_ALPHA)


def make_figure(df: pd.DataFrame) -> plt.Figure:
    n_vals = sorted(df["n_train"].unique())
    methods = ["CKME", "QR"]
    taus = [0.05, 0.95]

    # rows: metric (mean / sup), cols: tau (0.05 / 0.95)
    panels = [
        # (metric_mean_col,    metric_std_col,      y-label,             title)
        ("mean_abs_err_mean", "mean_abs_err_std",  "Mean |q̂ − q_true|", "τ = 0.05"),
        ("mean_abs_err_mean", "mean_abs_err_std",  "Mean |q̂ − q_true|", "τ = 0.95"),
        ("sup_abs_err_mean",  "sup_abs_err_std",   "Sup |q̂ − q_true|",  "τ = 0.05"),
        ("sup_abs_err_mean",  "sup_abs_err_std",   "Sup |q̂ − q_true|",  "τ = 0.95"),
    ]
    tau_per_panel = [0.05, 0.95, 0.05, 0.95]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle(
        "exp1 — MG1 queue (Gaussian noise)\nCKME vs QR quantile estimation error",
        fontsize=12, fontweight="bold",
    )

    for ax, (mm, ms, ylabel, tau_label), tau in zip(
        axes.ravel(), panels, tau_per_panel
    ):
        for method in methods:
            _draw_method(ax, df, mm, ms, tau, method, n_vals)

        ax.set_xscale("log")
        ax.set_xticks(n_vals)
        ax.set_xticklabels([str(n) for n in n_vals], fontsize=9)
        ax.set_xlabel("n_train", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"{ylabel.split()[0]} absolute error  ({tau_label})", fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=9, framealpha=0.8)

    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    default_input = os.path.join(
        os.path.dirname(__file__), "output_scaling", "scaling_summary.csv"
    )
    parser = argparse.ArgumentParser(description="Plot exp1 CKME vs QR comparison")
    parser.add_argument(
        "--input", type=str, default=default_input,
        help="Path to scaling_summary.csv",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Output path for PNG (e.g. exp_onesided/output_scaling/exp1_comparison.png). "
             "If omitted, shows interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_summary(args.input)
    fig = make_figure(df)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved → {args.save}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
