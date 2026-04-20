"""
Plot exp_design results: Exp A (n1 vs r1) and Exp B (S^0 vs LHS).

Usage:
  python exp_design/plot_design.py --exp A --output_root exp_design/output/expA_s1small
  python exp_design/plot_design.py --exp B --output_root exp_design/output/expB_s1small
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DGPS = ["exp1", "exp2_gauss_low", "exp2_gauss_high"]
METRICS = ["coverage", "width", "interval_score"]
METRIC_LABELS = {"coverage": "Coverage", "width": "Width", "interval_score": "Interval Score"}
ALPHA = 0.1


def load_summary(root: Path, dgp: str) -> pd.DataFrame:
    p = root / dgp / "summary.csv"
    if not p.exists():
        print(f"  missing: {p}")
        return None
    return pd.read_csv(p)


def plot_exp_A(root: Path, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, metric in zip(axes, METRICS):
        for dgp in DGPS:
            df = load_summary(root, dgp)
            if df is None:
                continue
            # Parse n_1 from label "n1_{n1}_r1_{r1}"
            df = df.copy()
            df["n_1"] = df["label"].str.extract(r"n1_(\d+)_").astype(int)
            agg = df.groupby("n_1").agg(
                mean=(metric, "mean"), se=(metric, "sem")
            ).sort_index()
            ax.errorbar(agg.index, agg["mean"], yerr=agg["se"],
                        marker="o", label=dgp, capsize=4)
        ax.set_xscale("log")
        ax.set_xlabel(r"$n_1$ (sites)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"Exp A: {METRIC_LABELS[metric]} vs $n_1$")
        if metric == "coverage":
            ax.axhline(1 - ALPHA, color="gray", ls="--", lw=1, label="target")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


def plot_exp_B(root: Path, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    # (label, display_name)
    methods = [
        ("method_lhs", "LHS"),
        ("method_sampling_tail", "sampling (tail)"),
        ("method_sampling_variance", "sampling (var)"),
        ("method_sampling_epistemic", "sampling (epistemic)"),
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ax, metric in zip(axes, METRICS):
        all_means = {lbl: [] for lbl, _ in methods}
        all_ses = {lbl: [] for lbl, _ in methods}
        for dgp in DGPS:
            df = load_summary(root, dgp)
            if df is None:
                for lbl, _ in methods:
                    all_means[lbl].append(np.nan)
                    all_ses[lbl].append(np.nan)
                continue
            for lbl, _ in methods:
                sub = df[df["label"] == lbl]
                all_means[lbl].append(sub[metric].mean())
                all_ses[lbl].append(sub[metric].sem())

        x = np.arange(len(DGPS))
        n_m = len(methods)
        w = 0.8 / n_m
        for i, ((lbl, disp), col) in enumerate(zip(methods, colors)):
            ax.bar(x + (i - (n_m - 1) / 2) * w, all_means[lbl], w,
                   yerr=all_ses[lbl], label=disp, color=col, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(DGPS, fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"Exp B: {METRIC_LABELS[metric]}")
        if metric == "coverage":
            ax.axhline(1 - ALPHA, color="gray", ls="--", lw=1, label="target")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=["A", "B"], required=True)
    parser.add_argument("--output_root", type=str, required=True,
                        help="Directory containing {dgp}/summary.csv")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.output_root)
    save_path = Path(args.save_path) if args.save_path else root / f"exp{args.exp}_plot.png"

    if args.exp == "A":
        plot_exp_A(root, save_path)
    else:
        plot_exp_B(root, save_path)


if __name__ == "__main__":
    main()
