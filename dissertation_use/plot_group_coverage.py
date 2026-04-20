"""
plot_group_coverage.py

Group conditional coverage + width from dissertation experiment.

2x2 figure: rows = DGP, cols = Coverage / Width.

Usage:
  python dissertation_use/plot_group_coverage.py
  python dissertation_use/plot_group_coverage.py --save dissertation_use/output/group_coverage.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

SIMULATORS = [
    ("exp2",          "Exp 1: Gaussian noise"),
    ("nongauss_A1L",  r"Exp 2: Student-$t$ ($\nu=3$) noise"),
]

METHODS = [
    ("CKME (fixed $h$)",    "covered_score",    "width",          "#2166ac", "-",  "o"),
    ("CKME (adaptive $h$)", "covered_adaptive",  "width_adaptive", "#b2182b", "-",  "s"),
    ("DCP-DR",              "covered_score_dr",  "width_dr",       "#7570b3", "--", "^"),
    ("DCP-QR",              "covered_score_qr",  "width_qr",       "#1b9e77", "-.", "D"),
]

X_LO, X_HI = 0.0, 2 * np.pi


def load_per_point(output_dir: Path, sim: str, site_method: str) -> list[pd.DataFrame]:
    paths = sorted(output_dir.glob(f"macrorep_*/case_{sim}_{site_method}/per_point.csv"))
    if not paths:
        print(f"  Warning: no per_point.csv for {sim}/{site_method}", file=sys.stderr)
        return []
    return [pd.read_csv(p) for p in paths]


def compute_bin_metric(
    dfs: list[pd.DataFrame], col: str, n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    edges = np.linspace(X_LO, X_HI, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    per_macro = []
    for df in dfs:
        if col not in df.columns:
            continue
        x = df["x0"].values
        y = df[col].values
        bins = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        bin_val = np.array([y[bins == b].mean() if (bins == b).sum() > 0 else np.nan
                            for b in range(n_bins)])
        per_macro.append(bin_val)
    if not per_macro:
        return centers, np.full(n_bins, np.nan), np.full(n_bins, np.nan), 0
    stacked = np.vstack(per_macro)
    return centers, np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0), len(per_macro)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="dissertation_use/output")
    parser.add_argument("--n_bins", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--site_method", type=str, default="lhs")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    target = 1 - args.alpha

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    summary_rows = []

    for row, (sim, label) in enumerate(SIMULATORS):
        ax_cov = axes[row, 0]
        ax_wid = axes[row, 1]
        dfs = load_per_point(out_dir, sim, args.site_method)

        if not dfs:
            ax_cov.text(0.5, 0.5, "no data", transform=ax_cov.transAxes,
                        ha="center", va="center", fontsize=12, color="gray")
            ax_wid.text(0.5, 0.5, "no data", transform=ax_wid.transAxes,
                        ha="center", va="center", fontsize=12, color="gray")
            continue

        n_m = 0
        for method_label, cov_col, wid_col, color, ls, marker in METHODS:
            # Coverage
            centers, cov_mean, cov_std, n_m_this = compute_bin_metric(
                dfs, cov_col, args.n_bins)
            n_m = max(n_m, n_m_this)
            if n_m_this == 0 or np.isnan(cov_mean).all():
                continue

            ax_cov.plot(centers, cov_mean, color=color, linestyle=ls,
                        linewidth=1.8, marker=marker, markersize=4,
                        markerfacecolor="white", markeredgewidth=1.2,
                        label=method_label)
            ax_cov.fill_between(centers, cov_mean - cov_std, cov_mean + cov_std,
                                color=color, alpha=0.08, linewidth=0)

            # Width
            _, wid_mean, wid_std, n_w = compute_bin_metric(
                dfs, wid_col, args.n_bins)
            if n_w > 0 and not np.isnan(wid_mean).all():
                ax_wid.plot(centers, wid_mean, color=color, linestyle=ls,
                            linewidth=1.8, marker=marker, markersize=4,
                            markerfacecolor="white", markeredgewidth=1.2,
                            label=method_label)
                ax_wid.fill_between(centers, wid_mean - wid_std, wid_mean + wid_std,
                                    color=color, alpha=0.08, linewidth=0)

            summary_rows.append({
                "simulator": sim, "method": method_label,
                "overall_cov": np.nanmean(cov_mean),
                "max_bin_cov_dev": np.nanmax(np.abs(cov_mean - target)),
                "overall_width": np.nanmean(wid_mean) if n_w > 0 else np.nan,
                "n_macroreps": n_m_this,
            })

        # Coverage panel
        ax_cov.axhline(target, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_cov.set_xlim(X_LO, X_HI)
        ax_cov.set_ylim(0.55, 1.05)
        ax_cov.set_ylabel("Coverage")
        ax_cov.set_title(label, fontweight="bold")

        # Width panel
        ax_wid.set_xlim(X_LO, X_HI)
        ax_wid.set_ylabel("Interval width")
        ax_wid.set_title(label, fontweight="bold")

        # Only show n_macroreps annotation on first row
        if row == 0:
            ax_cov.annotate(f"$n_{{\\mathrm{{macro}}}}={n_m}$",
                            xy=(0.98, 0.02), xycoords="axes fraction",
                            ha="right", va="bottom", fontsize=9, color="gray")

    # Shared x-axis formatting
    for col in range(2):
        ax = axes[-1, col]
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels(["$0$", r"$\pi$", r"$2\pi$"])
        ax.set_xlabel("$x$")
        axes[0, col].set_xticklabels([])

    # Single shared legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   frameon=True, fancybox=False, edgecolor="gray",
                   bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    # Save summary
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        csv_path = out_dir / "group_coverage_summary.csv"
        summary.to_csv(csv_path, index=False)
        print(f"Summary saved to {csv_path}")
        print("\nSummary:")
        print(summary.to_string(index=False))

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
