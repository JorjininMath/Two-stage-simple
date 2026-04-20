"""
plot_ckme_scaling.py

Plot CKME quantile estimation error vs training data size.

Reads output from run_ckme_scaling.py:
  output_scaling/scaling_raw.csv     — one row per (macrorep, n_train, sim, tau)
  output_scaling/scaling_summary.csv — mean ± std aggregated

Figure layout: one figure per simulator (3 total), each with 4 subplots (2x2):
  top-left     — mean |err| for τ=0.05
  top-right    — mean |err| for τ=0.95
  bottom-left  — sup  |err| for τ=0.05
  bottom-right — sup  |err| for τ=0.95

Usage:
    python exp_onesided/plot_ckme_scaling.py
    python exp_onesided/plot_ckme_scaling.py --save_dir exp_onesided/output_scaling/
    python exp_onesided/plot_ckme_scaling.py --method CKME
    python exp_onesided/plot_ckme_scaling.py --method QR --save_dir exp_onesided/output_scaling/
    python exp_onesided/plot_ckme_scaling.py --plot_type boxplot --save_dir exp_onesided/output_scaling/
    python exp_onesided/plot_ckme_scaling.py --plot_type boxplot --method CKME --save_dir exp_onesided/output_scaling/
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIMULATORS = ["exp1", "exp2", "nongauss_A1L"]
SIM_LABELS = {
    "exp1":         "exp1  (MG1 queue, Gaussian noise)",
    "exp2":         "exp2  (sin, Gaussian noise)",
    "nongauss_A1L": "A1L   (Student-t  ν=3, heavy tail)",
}
TAUS = [0.05, 0.95]
TAU_COLORS  = {0.05: "steelblue", 0.95: "tomato"}
METHOD_LS   = {"CKME": "-", "QR": "--"}
METHOD_MK   = {"CKME": "o", "QR": "s"}


def load_data(output_dir: str) -> pd.DataFrame:
    raw_path = os.path.join(output_dir, "scaling_raw.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Not found: {raw_path}\nRun run_ckme_scaling.py first.")
    return pd.read_csv(raw_path)


def _draw_line(ax, sub: pd.DataFrame, col: str, tau: float, method: str, n_vals: list):
    means, stds = [], []
    for n in n_vals:
        vals = sub[sub["n_train"] == n][col].values
        means.append(np.mean(vals))
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    means, stds = np.array(means), np.array(stds)
    color = TAU_COLORS[tau]
    ls = METHOD_LS.get(method, "-")
    mk = METHOD_MK.get(method, "o")
    ax.plot(n_vals, means, color=color, ls=ls, lw=1.8, marker=mk, markersize=4,
            label=method)
    ax.fill_between(n_vals, means - stds, means + stds, alpha=0.12, color=color)


def _draw_boxplot(ax, sub: pd.DataFrame, col: str, tau: float, method: str, n_vals: list,
                  offset: float = 0.0):
    """Draw one boxplot per n_train, distribution over macroreps.

    offset: small horizontal jitter so multiple methods don't overlap on log-scale.
    """
    color = TAU_COLORS[tau]
    positions = list(range(len(n_vals)))
    data = [sub[sub["n_train"] == n][col].values for n in n_vals]

    bp = ax.boxplot(
        data,
        positions=[p + offset for p in positions],
        widths=0.35,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", lw=1.5),
        boxprops=dict(facecolor=color, alpha=0.35),
        whiskerprops=dict(color=color, lw=1.2),
        capprops=dict(color=color, lw=1.2),
        flierprops=dict(marker="x", color=color, markersize=4, alpha=0.6),
    )
    # invisible line for legend
    ax.plot([], [], color=color, lw=2, label=method)


def plot_one_simulator(
    raw: pd.DataFrame,
    sim: str,
    save_path: str | None = None,
    method_filter: str | None = None,
    plot_type: str = "line",
):
    """One figure, 4 subplots (metric x tau), methods compared within each subplot.

    Parameters
    ----------
    method_filter : str or None
        If given (e.g. "CKME" or "QR"), only that method is plotted.
        If None, all methods are plotted together (original behaviour).
    plot_type : "line" or "boxplot"
        "line"    — mean ± std band (original behaviour)
        "boxplot" — one box per n_train, distribution over macroreps
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    label = SIM_LABELS[sim]
    suffix = f"  [{method_filter}]" if method_filter else ""
    type_tag = "  [boxplot]" if plot_type == "boxplot" else ""
    fig.suptitle(label + suffix + type_tag, fontsize=12, fontweight="bold")

    sub_sim = raw[raw["simulator"] == sim].copy()
    n_vals  = sorted(sub_sim["n_train"].unique())
    if "method" in sub_sim.columns:
        sub_sim["method"] = sub_sim["method"].fillna("CKME")
    methods = sorted(sub_sim["method"].unique()) if "method" in sub_sim.columns else ["CKME"]

    if method_filter is not None:
        methods = [m for m in methods if m == method_filter]

    panel_specs = [
        ("mean_abs_err", "mean |q̂ − q_true|", "Mean absolute error (τ=0.05)", 0.05),
        ("mean_abs_err", "mean |q̂ − q_true|", "Mean absolute error (τ=0.95)", 0.95),
        ("sup_abs_err",  "sup |q̂ − q_true|",  "Sup absolute error (τ=0.05)",  0.05),
        ("sup_abs_err",  "sup |q̂ − q_true|",  "Sup absolute error (τ=0.95)",  0.95),
    ]

    # small horizontal offsets so two methods don't overlap in boxplot mode
    method_offsets = {m: (i - (len(methods) - 1) / 2) * 0.2
                      for i, m in enumerate(methods)}

    for ax, (col, ylabel, title, tau) in zip(axes.ravel(), panel_specs):
        for method in methods:
            sub = sub_sim[(sub_sim["tau"] == tau) & (sub_sim["method"] == method)]
            if sub.empty:
                continue
            if plot_type == "boxplot":
                _draw_boxplot(ax, sub, col, tau, method, n_vals,
                              offset=method_offsets[method])
            else:
                _draw_line(ax, sub, col, tau, method, n_vals)

        if plot_type == "boxplot":
            positions = list(range(len(n_vals)))
            ax.set_xticks(positions)
            ax.set_xticklabels([str(n) for n in n_vals], fontsize=8)
            ax.set_xlabel("n_train", fontsize=9)
        else:
            ax.set_xscale("log")
            ax.set_xticks(n_vals)
            ax.set_xticklabels([str(n) for n in n_vals], fontsize=8)
            ax.set_xlabel("n_train", fontsize=9)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
        plt.close(fig)
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CKME scaling results")
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(os.path.dirname(__file__), "output_scaling"),
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory to save figures. If omitted, show interactively. "
             "Saves scaling_<sim>.png (or scaling_<sim>_<method>.png) for each simulator.",
    )
    parser.add_argument(
        "--method", type=str, default=None,
        choices=["CKME", "QR"],
        help="Plot only this method. If omitted, all methods are shown together.",
    )
    parser.add_argument(
        "--plot_type", type=str, default="line",
        choices=["line", "boxplot"],
        help="'line' = mean±std band (default); 'boxplot' = box per n_train over macroreps.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw = load_data(args.output_dir)

    for sim in SIMULATORS:
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            type_tag = "_box" if args.plot_type == "boxplot" else ""
            method_tag = f"_{args.method}" if args.method else ""
            fname = f"scaling_{sim}{method_tag}{type_tag}.png"
            save_path = os.path.join(args.save_dir, fname)
        else:
            save_path = None
        plot_one_simulator(raw, sim, save_path=save_path,
                           method_filter=args.method, plot_type=args.plot_type)


if __name__ == "__main__":
    main()
