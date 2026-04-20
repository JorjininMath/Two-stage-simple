"""
analyze_group_coverage.py

Group conditional coverage analysis for the non-Gaussian experiment.
Bins test points into K equal-width x-bins, computes coverage per bin
per method per macrorep, and produces a summary CSV + 2x3 figure.

Paper Section 3.3 (Group Conditional Coverage).

Usage (from project root):
    python exp_nongauss/analyze_group_coverage.py
    python exp_nongauss/analyze_group_coverage.py --n_bins 10 --save exp_nongauss/output/group_coverage.png
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

# Simulator definitions (same order as plot_nongauss.py)
SMALL_SIMS = [
    ("nongauss_A1S", r"A1-S: Student-$t$ ($\nu$=10)"),
    ("nongauss_B2S", r"B2-S: Gamma ($k$=9)"),
    ("nongauss_C1S", r"C1-S: Mixture ($\pi$=0.02)"),
]
LARGE_SIMS = [
    ("nongauss_A1L", r"A1-L: Student-$t$ ($\nu$=3)"),
    ("nongauss_B2L", r"B2-L: Gamma ($k$=2)"),
    ("nongauss_C1L", r"C1-L: Mixture ($\pi$=0.10)"),
]

# Coverage columns per method
COVERAGE_COLS = {
    "CKME":   "covered_score",
    "DCP-DR": "covered_score_dr",
    "hetGP":  "covered_interval_hetgp",
}

METHOD_COLORS = {"CKME": "#2166ac", "DCP-DR": "#d6604d", "hetGP": "#4dac26"}
METHOD_LS     = {"CKME": "-",       "DCP-DR": "--",      "hetGP": "-."}

X_LO, X_HI = 0.0, 2 * np.pi


def load_per_point(output_dir: Path, sim: str, site_method: str) -> list[pd.DataFrame]:
    paths = sorted(output_dir.glob(f"macrorep_*/case_{sim}_{site_method}/per_point.csv"))
    if not paths:
        print(f"  Warning: no per_point.csv for {sim}/{site_method}", file=sys.stderr)
        return []
    return [pd.read_csv(p) for p in paths]


def compute_bin_coverage(
    dfs: list[pd.DataFrame],
    cov_col: str,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (bin_centers, mean_coverage, std_coverage, n_macroreps)."""
    edges = np.linspace(X_LO, X_HI, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    per_macro = []
    for df in dfs:
        if cov_col not in df.columns:
            continue
        x = df["x0"].values
        y = df[cov_col].values
        bins = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
        bin_cov = np.array([y[bins == b].mean() if (bins == b).sum() > 0 else np.nan
                            for b in range(n_bins)])
        per_macro.append(bin_cov)

    if not per_macro:
        return centers, np.full(n_bins, np.nan), np.full(n_bins, np.nan), 0

    stacked = np.vstack(per_macro)
    return centers, np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0), len(per_macro)


def build_summary(output_dir: Path, site_method: str, n_bins: int) -> pd.DataFrame:
    rows = []
    for sim, _ in SMALL_SIMS + LARGE_SIMS:
        dfs = load_per_point(output_dir, sim, site_method)
        if not dfs:
            continue
        edges = np.linspace(X_LO, X_HI, n_bins + 1)
        for method, cov_col in COVERAGE_COLS.items():
            centers, means, stds, n_m = compute_bin_coverage(dfs, cov_col, n_bins)
            for i in range(n_bins):
                rows.append({
                    "simulator": sim,
                    "method": method,
                    "bin_lo": edges[i],
                    "bin_hi": edges[i + 1],
                    "bin_center": centers[i],
                    "cov_mean": means[i],
                    "cov_std": stds[i],
                    "n_macroreps": n_m,
                })
    return pd.DataFrame(rows)


def plot_group_coverage(summary: pd.DataFrame, alpha: float, save_path: Path | None):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    fig.suptitle("Group Conditional Coverage by x-bin", fontsize=13, y=1.02)

    sim_groups = [SMALL_SIMS, LARGE_SIMS]
    row_labels = ["Small departure", "Large departure"]

    for row, (sims, rlabel) in enumerate(zip(sim_groups, row_labels)):
        for col, (sim, label) in enumerate(sims):
            ax = axes[row, col]
            sub = summary[summary["simulator"] == sim]

            for method in COVERAGE_COLS:
                msub = sub[sub["method"] == method]
                if msub.empty or msub["cov_mean"].isna().all():
                    continue
                x = msub["bin_center"].values
                y = msub["cov_mean"].values
                s = msub["cov_std"].values
                ax.plot(x, y, color=METHOD_COLORS[method],
                        linestyle=METHOD_LS[method], linewidth=1.8, label=method)
                ax.fill_between(x, y - s, y + s,
                                color=METHOD_COLORS[method], alpha=0.15)

            ax.axhline(1 - alpha, color="black", linestyle=":", linewidth=1.2,
                       label=f"target ({1-alpha:.0%})")
            ax.set_xlim(X_LO, X_HI)
            ax.set_xticks([0, np.pi, 2 * np.pi])
            ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
            ax.set_ylim(0.6, 1.05)
            ax.set_xlabel("x")

            n_m = sub["n_macroreps"].iloc[0] if not sub.empty else 0
            ax.set_title(f"{label}\n({n_m} macroreps)", fontsize=9)

            if col == 0:
                ax.set_ylabel(f"{rlabel}\nCoverage")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Group conditional coverage analysis")
    parser.add_argument("--output_dir",  type=str, default=None)
    parser.add_argument("--n_bins",      type=int, default=10)
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--site_method", type=str, default="lhs")
    parser.add_argument("--save",        type=str, default=None,
                        help="Save figure path (default: show interactively)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_nongauss" / "output"

    summary = build_summary(out_dir, args.site_method, args.n_bins)
    csv_path = out_dir / "group_coverage_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Summary saved to {csv_path}")

    # Print marginal summary per simulator x method
    agg = summary.groupby(["simulator", "method"]).agg(
        overall_cov=("cov_mean", "mean"),
        max_deviation=("cov_mean", lambda s: (s - (1 - args.alpha)).abs().max()),
    ).reset_index()
    print("\nOverall coverage and max bin deviation from target:")
    print(agg.to_string(index=False))

    save_path = Path(args.save) if args.save else None
    plot_group_coverage(summary, args.alpha, save_path)


if __name__ == "__main__":
    main()
