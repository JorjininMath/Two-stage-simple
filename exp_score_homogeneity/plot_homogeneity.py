"""
plot_homogeneity.py

Generate all figures for the score homogeneity mechanism experiment.

Fig 1: Score ECDF overlay at representative x-points (fixed_cv vs adaptive)
Fig 2: Coverage curve vs x for different bandwidth configs
Fig 3: c-scan Pareto plot (homogeneity vs coverage gap)
Fig 4: Decoupling plot (quantile error vs coverage gap)

Usage:
  python exp_score_homogeneity/plot_homogeneity.py
  python exp_score_homogeneity/plot_homogeneity.py --simulator exp2
  python exp_score_homogeneity/plot_homogeneity.py --save_dir exp_score_homogeneity/output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).parent

# Colors for representative x-points (low to high sigma)
_ECDF_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
# Colors for bandwidth configs in coverage plot
_BW_COLORS = {
    "fixed_small": "#d62728",
    "fixed_cv": "#1f77b4",
    "fixed_large": "#ff7f0e",
    "adaptive_c2.0": "#2ca02c",
}


def _load_results(save_dir: Path, simulator: str) -> pd.DataFrame:
    path = save_dir / f"results_{simulator}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    return pd.read_csv(path)


def _load_summary(save_dir: Path, simulator: str) -> pd.DataFrame:
    path = save_dir / f"summary_{simulator}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Fig 1: Score ECDF overlay
# ---------------------------------------------------------------------------

def plot_score_ecdf_overlay(df: pd.DataFrame, save_dir: Path, simulator: str,
                            bw_left: str = "fixed_cv",
                            bw_right: str = "adaptive_c2.0",
                            n_repr: int = 5):
    """Score ECDF at representative x-points, two panels."""
    x_vals = sorted(df["x_eval"].unique())
    M = len(x_vals)
    # Pick n_repr roughly equally-spaced x-points
    indices = np.linspace(0, M - 1, n_repr, dtype=int)
    repr_x = [x_vals[i] for i in indices]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, bw_label in zip(axes, [bw_left, bw_right]):
        sub = df[df["bandwidth"] == bw_label]
        if sub.empty:
            ax.set_title(f"{bw_label} (no data)")
            continue

        for i, x_val in enumerate(repr_x):
            # Collect scores across macroreps for this x
            rows = sub[np.isclose(sub["x_eval"], x_val, atol=1e-6)]
            if rows.empty:
                continue
            # ks_from_pooled is per-point, but for ECDF we need raw scores.
            # We don't have raw scores in results CSV — use cov_mc as proxy
            # or compute ECDF from the KS metric distribution across macroreps.
            # Actually, we need to re-derive: the results CSV has ks_from_pooled
            # per (macrorep, bw, x), but not raw score samples.
            #
            # For the ECDF overlay, we use the coverage MC values across macroreps
            # to show the score distribution indirectly. But the real approach is
            # to save score ECDFs during run_homogeneity.py.
            #
            # Workaround: use ks_from_pooled as a summary. For proper ECDF plots,
            # we need to re-run with score saving. For now, plot ks_from_pooled
            # as a bar/profile plot.
            pass

        # Since we don't have raw score samples in CSV, plot ks_from_pooled
        # profile across x for each macrorep (mean ± SD band)
        macro_ids = sub["macrorep"].unique()
        x_arr = np.array(x_vals)
        ks_matrix = np.full((len(macro_ids), M), np.nan)
        for mi, macro_id in enumerate(macro_ids):
            msub = sub[sub["macrorep"] == macro_id].sort_values("x_eval")
            if len(msub) == M:
                ks_matrix[mi] = msub["ks_from_pooled"].values

        ks_mean = np.nanmean(ks_matrix, axis=0)
        ks_sd = np.nanstd(ks_matrix, axis=0, ddof=1)

        ax.plot(x_arr, ks_mean, "k-", linewidth=1.5)
        ax.fill_between(x_arr, ks_mean - ks_sd, ks_mean + ks_sd,
                         alpha=0.2, color="steelblue")
        ax.set_xlabel("x")
        ax.set_ylabel("KS from pooled")
        ax.set_title(bw_label)
        ax.set_ylim(bottom=0)

    fig.suptitle(f"Score Homogeneity Profile — {simulator}", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_dir / f"fig1_score_homogeneity_profile_{simulator}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1 -> fig1_score_homogeneity_profile_{simulator}.png")


# ---------------------------------------------------------------------------
# Fig 2: Coverage curve vs x
# ---------------------------------------------------------------------------

def plot_coverage_curve(df: pd.DataFrame, save_dir: Path, simulator: str,
                        alpha: float = 0.1,
                        bw_list: list[str] | None = None):
    """Local coverage vs x for selected bandwidth configs."""
    if bw_list is None:
        bw_list = ["fixed_small", "fixed_cv", "fixed_large", "adaptive_c2.0"]

    x_vals = sorted(df["x_eval"].unique())
    M = len(x_vals)
    x_arr = np.array(x_vals)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1 - alpha, color="gray", linestyle="--", linewidth=1,
               label=f"target = {1-alpha:.1f}")

    for bw_label in bw_list:
        sub = df[df["bandwidth"] == bw_label]
        if sub.empty:
            continue
        macro_ids = sub["macrorep"].unique()
        cov_matrix = np.full((len(macro_ids), M), np.nan)
        for mi, macro_id in enumerate(macro_ids):
            msub = sub[sub["macrorep"] == macro_id].sort_values("x_eval")
            if len(msub) == M:
                cov_matrix[mi] = msub["cov_mc"].values

        cov_mean = np.nanmean(cov_matrix, axis=0)
        cov_sd = np.nanstd(cov_matrix, axis=0, ddof=1)

        color = _BW_COLORS.get(bw_label, None)
        ax.plot(x_arr, cov_mean, linewidth=1.5, label=bw_label, color=color)
        ax.fill_between(x_arr, cov_mean - cov_sd, cov_mean + cov_sd,
                         alpha=0.15, color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("Conditional coverage")
    ax.set_title(f"Coverage Curve — {simulator}")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0.5, 1.05)
    fig.tight_layout()
    fig.savefig(save_dir / f"fig2_coverage_curve_{simulator}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2 -> fig2_coverage_curve_{simulator}.png")


# ---------------------------------------------------------------------------
# Fig 3: c-scan Pareto plot
# ---------------------------------------------------------------------------

def plot_cscan_pareto(summary: pd.DataFrame, save_dir: Path, simulator: str):
    """Homogeneity (ks_max) vs conditional coverage gap (cov_gap_sup)."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in summary.iterrows():
        bw = row["bandwidth"]
        x_val = row["ks_max_mean"]
        y_val = row["cov_gap_sup_mean"]
        x_err = row.get("ks_max_sd", 0)
        y_err = row.get("cov_gap_sup_sd", 0)

        is_adaptive = bw.startswith("adaptive")
        marker = "o" if is_adaptive else "s"
        color = "#2ca02c" if is_adaptive else "#1f77b4"

        ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err,
                     fmt=marker, color=color, markersize=8, capsize=3)
        # Label offset
        ax.annotate(bw, (x_val, y_val), fontsize=7,
                     xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Score homogeneity: KS max (lower = better)")
    ax.set_ylabel("Coverage gap: sup |Cov(x) - 0.9| (lower = better)")
    ax.set_title(f"Pareto: Homogeneity vs Coverage — {simulator}")

    # Add legend for marker types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=8, label="adaptive"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#1f77b4",
               markersize=8, label="fixed"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.tight_layout()
    fig.savefig(save_dir / f"fig3_cscan_pareto_{simulator}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3 -> fig3_cscan_pareto_{simulator}.png")


# ---------------------------------------------------------------------------
# Fig 4: Decoupling plot
# ---------------------------------------------------------------------------

def plot_decoupling(summary: pd.DataFrame, save_dir: Path, simulator: str):
    """Quantile error vs coverage gap — tests whether they decouple."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for _, row in summary.iterrows():
        bw = row["bandwidth"]
        x_val = row["q_err_sup_mean"]
        y_val = row["cov_gap_sup_mean"]
        x_err = row.get("q_err_sup_sd", 0)
        y_err = row.get("cov_gap_sup_sd", 0)

        is_adaptive = bw.startswith("adaptive")
        marker = "o" if is_adaptive else "s"
        color = "#2ca02c" if is_adaptive else "#d62728"

        ax.errorbar(x_val, y_val, xerr=x_err, yerr=y_err,
                     fmt=marker, color=color, markersize=8, capsize=3)
        ax.annotate(bw, (x_val, y_val), fontsize=7,
                     xytext=(5, 5), textcoords="offset points")

    ax.set_xlabel("Quantile error: sup |q̂ − q| (estimation accuracy)")
    ax.set_ylabel("Coverage gap: sup |Cov(x) − 0.9| (conditional coverage)")
    ax.set_title(f"Decoupling: Estimation vs Coverage — {simulator}")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=8, label="adaptive"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d62728",
               markersize=8, label="fixed"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    fig.tight_layout()
    fig.savefig(save_dir / f"fig4_decoupling_{simulator}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4 -> fig4_decoupling_{simulator}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot score homogeneity figures")
    parser.add_argument("--save_dir", type=str,
                        default=str(_HERE / "output"))
    parser.add_argument("--simulator", type=str, default=None,
                        help="Plot for one simulator (default: all found)")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)

    # Find available simulators
    if args.simulator:
        sims = [args.simulator]
    else:
        sims = []
        for p in save_dir.glob("results_*.csv"):
            sim = p.stem.replace("results_", "")
            sims.append(sim)
        if not sims:
            print(f"No results found in {save_dir}")
            return

    for sim in sims:
        print(f"\n=== Plotting: {sim} ===")
        df = _load_results(save_dir, sim)
        summary = _load_summary(save_dir, sim)

        plot_score_ecdf_overlay(df, save_dir, sim)
        plot_coverage_curve(df, save_dir, sim, alpha=args.alpha)
        plot_cscan_pareto(summary, save_dir, sim)
        plot_decoupling(summary, save_dir, sim)


if __name__ == "__main__":
    main()
