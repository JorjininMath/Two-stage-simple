"""
plot_nongauss.py

Plot per-x coverage and interval width for the non-Gaussian experiment.

Layout (4 rows x 3 cols):
  Row 0 — Small: Coverage  (A1S | B2S | C1S)
  Row 1 — Small: Width     (A1S | B2S | C1S)
  Row 2 — Large: Coverage  (A1L | B2L | C1L)
  Row 3 — Large: Width     (A1L | B2L | C1L)
Each subplot shows 3 methods: CKME, DCP-DR, hetGP.

Usage (from project root):
    python exp_nongauss/plot_nongauss.py
    python exp_nongauss/plot_nongauss.py --output_dir exp_nongauss/output --n_bins 25 --site_method lhs
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

# (sim_name, column_index, display_label)
SMALL_SIMS = [
    ("nongauss_A1S", 0, r"A1-S: Student-t ($\nu$=10)"),
    ("nongauss_B2S", 1, r"B2-S: Gamma ($k$=9)"),
    ("nongauss_C1S", 2, r"C1-S: Mixture ($\pi$=0.02)"),
]
LARGE_SIMS = [
    ("nongauss_A1L", 0, r"A1-L: Student-t ($\nu$=3)"),
    ("nongauss_B2L", 1, r"B2-L: Gamma ($k$=2)"),
    ("nongauss_C1L", 2, r"C1-L: Mixture ($\pi$=0.10)"),
]

# Columns in per_point.csv for each method
METHODS = {
    "CKME":   {"cov": "covered_interval",      "width": "width"},
    "DCP-DR": {"cov": "covered_interval_dr",   "width": "width_dr"},
    "hetGP":  {"cov": "covered_interval_hetgp","width": "width_hetgp"},
}
METHOD_COLORS = {"CKME": "#2166ac", "DCP-DR": "#d6604d", "hetGP": "#4dac26"}
METHOD_LS     = {"CKME": "-",       "DCP-DR": "--",      "hetGP": "-."}


def _bin_series(x: np.ndarray, y: np.ndarray, n_bins: int):
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers, means, stds = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (x >= lo) & (x < hi)
        if mask.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        means.append(y[mask].mean())
        stds.append(y[mask].std())
    return np.array(centers), np.array(means), np.array(stds)


def load_per_point(output_dir: Path, sim: str, site_method: str) -> list[pd.DataFrame]:
    case_glob = sorted(output_dir.glob(f"macrorep_*/case_{sim}_{site_method}/per_point.csv"))
    if not case_glob:
        print(f"  Warning: no per_point.csv found for {sim}/{site_method}", file=sys.stderr)
        return []
    return [pd.read_csv(p) for p in case_glob]


def plot_panel(ax, dfs: list[pd.DataFrame], metric: str, col_key: str,
               n_bins: int, alpha_target: float | None = None):
    for mname, cols in METHODS.items():
        col = cols[col_key]
        all_means = []
        x_ref = None
        for df in dfs:
            if col not in df.columns:
                continue
            x = df["x0"].values
            y = df[col].values
            centers, means, _ = _bin_series(x, y, n_bins)
            all_means.append(means)
            if x_ref is None:
                x_ref = centers

        if not all_means or x_ref is None:
            continue

        stacked = np.vstack(all_means)
        grand_mean = stacked.mean(axis=0)
        grand_std  = stacked.std(axis=0)

        ax.plot(x_ref, grand_mean,
                color=METHOD_COLORS[mname], linestyle=METHOD_LS[mname],
                linewidth=1.8, label=mname)
        if stacked.shape[0] > 1:
            ax.fill_between(x_ref,
                            grand_mean - grand_std,
                            grand_mean + grand_std,
                            color=METHOD_COLORS[mname], alpha=0.15)

    if alpha_target is not None and metric == "coverage":
        ax.axhline(1 - alpha_target, color="black", linestyle=":", linewidth=1.2,
                   label=f"target ({1-alpha_target:.0%})")

    ax.set_xlabel("x")
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    if metric == "coverage":
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Coverage")
    else:
        ax.set_ylabel("Width")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  type=str, default=None)
    parser.add_argument("--n_bins",      type=int, default=25)
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--site_method", type=str, default="lhs")
    parser.add_argument("--save",        type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_nongauss" / "output"

    fig, axes = plt.subplots(4, 3, figsize=(13, 12))
    fig.suptitle("Non-Gaussian experiment: coverage and width vs x", fontsize=13)

    # Row group labels
    for row, label in [(0, "Small — Coverage"), (1, "Small — Width"),
                       (2, "Large — Coverage"), (3, "Large — Width")]:
        axes[row, 0].set_ylabel(f"{label}\n")

    for sim, col_idx, label in SMALL_SIMS:
        dfs = load_per_point(out_dir, sim, args.site_method)
        n_macro = len(dfs)
        title_suffix = f"(CKME: {n_macro} macroreps)" if n_macro > 0 else "(no data)"

        ax_cov = axes[0, col_idx]
        ax_cov.set_title(f"{label}\n{title_suffix}", fontsize=9)
        if dfs:
            plot_panel(ax_cov, dfs, "coverage", "cov", args.n_bins, args.alpha)
        if col_idx == 0:
            ax_cov.legend(fontsize=8)

        ax_wid = axes[1, col_idx]
        if dfs:
            plot_panel(ax_wid, dfs, "width", "width", args.n_bins)

    for sim, col_idx, label in LARGE_SIMS:
        dfs = load_per_point(out_dir, sim, args.site_method)
        n_macro = len(dfs)
        title_suffix = f"(CKME: {n_macro} macroreps)" if n_macro > 0 else "(no data)"

        ax_cov = axes[2, col_idx]
        ax_cov.set_title(f"{label}\n{title_suffix}", fontsize=9)
        if dfs:
            plot_panel(ax_cov, dfs, "coverage", "cov", args.n_bins, args.alpha)
        if col_idx == 0:
            ax_cov.legend(fontsize=8)

        ax_wid = axes[3, col_idx]
        if dfs:
            plot_panel(ax_wid, dfs, "width", "width", args.n_bins)

    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
