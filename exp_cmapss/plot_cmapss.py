"""
plot_cmapss.py

Slide-ready figures for the C-MAPSS CKME+CP experiment.

Figures
-------
  --mode degradation : Fig 1 — engine degradation + late-stage risk region
  --mode coverage    : Fig 2 — coverage vs width bar chart (all vs late-stage)
  --mode trajectory  : Fig 3 — single engine trajectory with prediction intervals
  --mode binplot     : Fig 4 — interval score / coverage by RUL bin

Usage
-----
python exp_cmapss/plot_cmapss.py --mode coverage \
    --results exp_cmapss/output/tables/cmapss_results.csv \
    --pred    exp_cmapss/output/tables/cmapss_predictions.csv \
    --save    exp_cmapss/output/figures/coverage.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "Ridge (point only)": "#aaaaaa",
    "Ridge + split CP":   "#4daf4a",
    "CKME Stage 1 CP":    "#377eb8",
    "CKME Stage 2 CP":    "#e41a1c",
    "Stage 1":            "#377eb8",
    "Stage 2":            "#e41a1c",
}

def _set_backend(save: str | None):
    """Switch to Agg only when saving; otherwise keep interactive backend."""
    if save is not None:
        matplotlib.use("Agg")


def _set_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })


# ---------------------------------------------------------------------------
# Fig 1: Degradation illustration (uses raw RUL trajectory)
# ---------------------------------------------------------------------------

def plot_degradation(
    pred_path: str,
    late_rul: int = 30,
    save: str | None = None,
):
    """
    Show true RUL declining over cycles for several test engines,
    highlight the late-stage high-risk zone.
    Uses cmapss_predictions.csv (already has rul_true in test order).
    """
    _set_backend(save)
    _set_style()
    pred_df = pd.read_csv(pred_path)
    Y_true = pred_df["rul_true"].values

    fig, ax = plt.subplots(figsize=(8, 4))

    # Approximate "cycles" index within the test set (not per-engine; illustrative)
    x = np.arange(len(Y_true))
    ax.plot(x, Y_true, color="steelblue", lw=1, alpha=0.7, label="True RUL")
    ax.axhline(late_rul, color="#e41a1c", lw=1.5, ls="--", label=f"High-risk threshold (RUL={late_rul})")
    ax.fill_between(x, 0, late_rul, alpha=0.10, color="#e41a1c", label="High-risk zone")

    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Remaining Useful Life (RUL)")
    ax.set_title("C-MAPSS FD001 — RUL Distribution (test set)")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2: Coverage vs Width bar chart
# ---------------------------------------------------------------------------

def plot_coverage_width(
    results_path: str,
    alpha: float = 0.1,
    late_rul: int = 30,
    save: str | None = None,
):
    """
    Bar chart: coverage (left) and width (right) for each method,
    grouped by subset (all vs late-stage).
    """
    _set_backend(save)
    _set_style()
    df = pd.read_csv(results_path)

    subsets = df["subset"].unique()
    methods = df["method"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bar_width = 0.18
    x = np.arange(len(subsets))

    for ax, metric, ylabel, title in [
        (axes[0], "coverage",  "Empirical Coverage", "Coverage (target = {:.0f}%)".format((1-alpha)*100)),
        (axes[1], "width",     "Average Interval Width", "Average Interval Width"),
    ]:
        for j, method in enumerate(methods):
            vals = []
            for subset in subsets:
                row = df[(df["method"] == method) & (df["subset"] == subset)]
                vals.append(float(row[metric].values[0]) if len(row) else float("nan"))
            offset = (j - len(methods) / 2 + 0.5) * bar_width
            color = METHOD_COLORS.get(method, f"C{j}")
            ax.bar(x + offset, vals, bar_width, label=method, color=color, alpha=0.85, edgecolor="white")

        if metric == "coverage":
            ax.axhline(1 - alpha, color="black", lw=1.2, ls="--", label=f"Target {(1-alpha)*100:.0f}%")

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("<=", "≤") for s in subsets], fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_ylim(bottom=0)

    fig.suptitle("C-MAPSS FD001 — CKME+CP Two-Stage Results", fontsize=13, y=1.02)
    fig.tight_layout()

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3: Single engine trajectory with intervals
# ---------------------------------------------------------------------------

def plot_trajectory(
    pred_path: str,
    results_path: str | None = None,
    late_rul: int = 30,
    engine_idx: int = 0,      # which test engine to show (by position in sorted order)
    save: str | None = None,
):
    """
    For one representative test engine: x-axis = cycle index,
    y-axis = true RUL + Stage 1 and Stage 2 intervals.
    """
    _set_backend(save)
    _set_style()
    pred_df = pd.read_csv(pred_path)
    Y_true = pred_df["rul_true"].values

    # Identify engine boundaries: consecutive groups where RUL is decreasing
    # (approximate; in test data engines are sequential)
    # We detect engine breaks by RUL jumps
    breaks = [0]
    for i in range(1, len(Y_true)):
        if Y_true[i] > Y_true[i - 1] + 5:   # large jump up = new engine
            breaks.append(i)
    breaks.append(len(Y_true))

    if engine_idx >= len(breaks) - 1:
        engine_idx = 0

    lo, hi = breaks[engine_idx], breaks[engine_idx + 1]
    cycles = np.arange(hi - lo)
    rul_true = Y_true[lo:hi]
    L1 = pred_df["L_stage1"].values[lo:hi]
    U1 = pred_df["U_stage1"].values[lo:hi]
    L2 = pred_df["L_stage2"].values[lo:hi]
    U2 = pred_df["U_stage2"].values[lo:hi]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.fill_between(cycles, L1, U1, alpha=0.20, color=METHOD_COLORS["CKME Stage 1 CP"], label="Stage 1 90% CI")
    ax.fill_between(cycles, L2, U2, alpha=0.25, color=METHOD_COLORS["CKME Stage 2 CP"], label="Stage 2 90% CI")
    ax.plot(cycles, rul_true, color="black", lw=2, label="True RUL")
    ax.plot(cycles, pred_df["rul_pred_ckme"].values[lo:hi], color="gray", lw=1.2, ls="--", label="CKME median")
    ax.axhline(late_rul, color="#e41a1c", lw=1.2, ls=":", alpha=0.8, label=f"High-risk (RUL={late_rul})")
    ax.fill_between(cycles, 0, late_rul, alpha=0.06, color="#e41a1c")

    ax.set_xlabel("Cycle (within engine)")
    ax.set_ylabel("RUL")
    ax.set_title(f"Engine trajectory — test engine {engine_idx + 1} (Stage 1 vs Stage 2 intervals)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4: Metrics by RUL bin
# ---------------------------------------------------------------------------

def plot_binplot(
    pred_path: str,
    alpha: float = 0.1,
    late_rul: int = 30,
    save: str | None = None,
):
    """
    Coverage and interval score broken down by RUL bins.
    Shows where Stage 2 improves over Stage 1.
    """
    _set_backend(save)
    _set_style()

    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate import rul_bin_analysis
    pred_df = pd.read_csv(pred_path)
    bin_df = rul_bin_analysis(pred_df, alpha=alpha)

    bins = bin_df["rul_bin"].unique()
    x = np.arange(len(bins))
    bar_width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric, ylabel, title in [
        (axes[0], "coverage",       "Empirical Coverage", "Coverage by RUL Bin"),
        (axes[1], "interval_score", "Interval Score",     "Interval Score by RUL Bin"),
    ]:
        for j, method in enumerate(["Stage 1", "Stage 2"]):
            vals = []
            for b in bins:
                row = bin_df[(bin_df["method"] == method) & (bin_df["rul_bin"] == b)]
                vals.append(float(row[metric].values[0]) if len(row) else float("nan"))
            offset = (j - 0.5) * bar_width
            color = METHOD_COLORS.get(f"CKME {method} CP", f"C{j}")
            ax.bar(x + offset, vals, bar_width, label=method, color=color, alpha=0.85, edgecolor="white")

        if metric == "coverage":
            ax.axhline(1 - alpha, color="black", lw=1.2, ls="--", label=f"Target {(1-alpha)*100:.0f}%")

        # highlight high-risk bins
        for i, b in enumerate(bins):
            try:
                hi_val = int(b.split(",")[1].replace("]", "").strip())
                if hi_val <= late_rul:
                    ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="#e41a1c")
            except Exception:
                pass

        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)

    # legend: red shading = high-risk
    red_patch = mpatches.Patch(color="#e41a1c", alpha=0.15, label=f"High-risk zone (RUL≤{late_rul})")
    axes[0].add_patch(red_patch)
    axes[0].legend(fontsize=8)

    fig.suptitle("C-MAPSS FD001 — Stage 1 vs Stage 2 by RUL Bin", fontsize=13, y=1.02)
    fig.tight_layout()

    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["degradation", "coverage", "trajectory", "binplot"],
                        default="coverage")
    parser.add_argument("--results", default="exp_cmapss/output/tables/cmapss_results.csv")
    parser.add_argument("--pred",    default="exp_cmapss/output/tables/cmapss_predictions.csv")
    parser.add_argument("--alpha",   type=float, default=0.1)
    parser.add_argument("--late_rul", type=int,  default=30)
    parser.add_argument("--engine_idx", type=int, default=0)
    parser.add_argument("--save",    default=None)
    args = parser.parse_args()

    if args.mode == "degradation":
        plot_degradation(args.pred, late_rul=args.late_rul, save=args.save)
    elif args.mode == "coverage":
        plot_coverage_width(args.results, alpha=args.alpha, late_rul=args.late_rul, save=args.save)
    elif args.mode == "trajectory":
        plot_trajectory(args.pred, late_rul=args.late_rul, engine_idx=args.engine_idx, save=args.save)
    elif args.mode == "binplot":
        plot_binplot(args.pred, alpha=args.alpha, late_rul=args.late_rul, save=args.save)


if __name__ == "__main__":
    main()
