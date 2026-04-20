"""
plot_exp2.py

Dedicated comparison plot for exp2 (sin + Gaussian noise):
CKME vs QR quantile estimation error across training sizes.

Layout: 2 x 2 grid
  [row 0] mean |err|  at tau=0.05  |  mean |err|  at tau=0.95
  [row 1] sup  |err|  at tau=0.05  |  sup  |err|  at tau=0.95

Plot types:
  --plot_type line     Mean ± std band (default)
  --plot_type boxplot  One box per n_train, distribution over macroreps

Usage:
    python exp_onesided/plot_exp2.py
    python exp_onesided/plot_exp2.py --plot_type boxplot
    python exp_onesided/plot_exp2.py --method CKME --plot_type boxplot
    python exp_onesided/plot_exp2.py --save exp_onesided/output_scaling/exp2_comparison.png
    python exp_onesided/plot_exp2.py --plot_type boxplot --save exp_onesided/output_scaling/exp2_box.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── style constants ───────────────────────────────────────────────────────────
COLORS  = {"CKME": "#2166ac", "QR": "#d6604d"}
MARKERS = {"CKME": "o", "QR": "s"}
LS      = {"CKME": "-", "QR": "--"}
FILL_ALPHA = 0.15
LINE_LW    = 2.0
MARKER_SIZE = 6

PANEL_SPECS = [
    # (col,            ylabel,                    title,                    tau)
    ("mean_abs_err", "Mean |q̂ − q_true|", "Mean absolute error  (τ=0.05)", 0.05),
    ("mean_abs_err", "Mean |q̂ − q_true|", "Mean absolute error  (τ=0.95)", 0.95),
    ("sup_abs_err",  "Sup |q̂ − q_true|",  "Sup absolute error   (τ=0.05)", 0.05),
    ("sup_abs_err",  "Sup |q̂ − q_true|",  "Sup absolute error   (τ=0.95)", 0.95),
]


# ── data loading ──────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw results not found: {path}\n"
            "Run run_ckme_scaling.py first."
        )
    df = pd.read_csv(path)
    df = df[df["simulator"] == "exp2"].copy()
    if df.empty:
        raise ValueError(
            "No rows for simulator='exp2' in the raw CSV.\n"
            "Make sure SIMULATORS includes 'exp2' in run_ckme_scaling.py."
        )
    if "method" not in df.columns:
        df["method"] = "CKME"
    return df


# ── line plot helpers ─────────────────────────────────────────────────────────

def _draw_line(ax, df: pd.DataFrame, col: str, tau: float, method: str,
               n_vals: list[int]) -> None:
    sub = df[(df["tau"] == tau) & (df["method"] == method)]
    means, stds = [], []
    for n in n_vals:
        vals = sub[sub["n_train"] == n][col].values
        means.append(np.mean(vals) if len(vals) else np.nan)
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    means, stds = np.array(means), np.array(stds)
    c = COLORS.get(method, "gray")
    ax.plot(n_vals, means,
            color=c, lw=LINE_LW, ls=LS.get(method, "-"),
            marker=MARKERS.get(method, "o"), markersize=MARKER_SIZE,
            label=method, zorder=3)
    ax.fill_between(n_vals, means - stds, means + stds,
                    color=c, alpha=FILL_ALPHA)


# ── boxplot helpers ───────────────────────────────────────────────────────────

def _draw_boxplot(ax, df: pd.DataFrame, col: str, tau: float, method: str,
                  n_vals: list[int], positions: list[float]) -> None:
    sub  = df[(df["tau"] == tau) & (df["method"] == method)]
    data = [sub[sub["n_train"] == n][col].values for n in n_vals]
    c = COLORS.get(method, "gray")
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.30,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", lw=1.5),
        boxprops=dict(facecolor=c, alpha=0.35),
        whiskerprops=dict(color=c, lw=1.2),
        capprops=dict(color=c, lw=1.2),
        flierprops=dict(marker="x", color=c, markersize=4, alpha=0.6),
    )
    # invisible line for legend entry
    ax.plot([], [], color=c, lw=2.5, label=method)


# ── figure builder ────────────────────────────────────────────────────────────

def make_figure(df: pd.DataFrame,
                methods: list[str],
                plot_type: str = "line") -> plt.Figure:
    n_vals = sorted(df["n_train"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(
        "exp2 — sin + Gaussian noise\nCKME vs QR quantile estimation error",
        fontsize=12, fontweight="bold",
    )

    # For boxplot: side-by-side offsets for multiple methods
    n_methods = len(methods)
    step = 0.35  # gap between method groups in boxplot mode
    offsets = {m: (i - (n_methods - 1) / 2) * step
               for i, m in enumerate(methods)}

    for ax, (col, ylabel, title, tau) in zip(axes.ravel(), PANEL_SPECS):
        for method in methods:
            sub = df[(df["tau"] == tau) & (df["method"] == method)]
            if sub.empty:
                continue
            if plot_type == "boxplot":
                pos = [i + offsets[method] for i in range(len(n_vals))]
                _draw_boxplot(ax, df, col, tau, method, n_vals, pos)
            else:
                _draw_line(ax, df, col, tau, method, n_vals)

        if plot_type == "boxplot":
            ax.set_xticks(range(len(n_vals)))
            ax.set_xticklabels([str(n) for n in n_vals], fontsize=9)
            ax.set_xlabel("n_train", fontsize=10)
        else:
            ax.set_xscale("log")
            ax.set_xticks(n_vals)
            ax.set_xticklabels([str(n) for n in n_vals], fontsize=9)
            ax.set_xlabel("n_train", fontsize=10)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=9, framealpha=0.8)

    type_tag = " [boxplot]" if plot_type == "boxplot" else " [line]"
    n_macro = df["macrorep"].nunique() if "macrorep" in df.columns else "?"
    fig.text(0.5, 0.01, f"n_macroreps = {n_macro}{type_tag}",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    default_input = os.path.join(
        os.path.dirname(__file__), "output_scaling", "scaling_raw.csv"
    )
    parser = argparse.ArgumentParser(description="Plot exp2 CKME vs QR comparison")
    parser.add_argument(
        "--input", type=str, default=default_input,
        help="Path to scaling_raw.csv (default: output_scaling/scaling_raw.csv)",
    )
    parser.add_argument(
        "--plot_type", type=str, default="line",
        choices=["line", "boxplot"],
        help="'line' = mean±std band (default); 'boxplot' = box per n_train over macroreps",
    )
    parser.add_argument(
        "--method", type=str, default=None,
        choices=["CKME", "QR"],
        help="Plot only this method. If omitted, both are shown.",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Output path for PNG. If omitted, shows interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_raw(args.input)

    available_methods = sorted(df["method"].unique())
    if args.method is not None:
        methods = [args.method]
    else:
        methods = [m for m in ["CKME", "QR"] if m in available_methods]

    fig = make_figure(df, methods=methods, plot_type=args.plot_type)

    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved -> {args.save}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
