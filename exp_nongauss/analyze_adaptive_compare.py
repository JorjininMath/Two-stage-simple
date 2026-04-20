"""
analyze_adaptive_compare.py

Compare fixed-h vs adaptive-h CKME group coverage.
Reads from the adaptive output dir (which contains both fixed and adaptive columns).

Produces two figures:
  1. Group coverage by x-bin (2x3): CKME-fixed, CKME-adaptive, DCP-DR, target
  2. Max bin deviation vs K (2x3): CKME-fixed, CKME-adaptive, DCP-DR, T-G2 floor

Usage (from project root):
    python exp_nongauss/analyze_adaptive_compare.py
    python exp_nongauss/analyze_adaptive_compare.py --output_dir exp_nongauss/output_adaptive_c2.00
    python exp_nongauss/analyze_adaptive_compare.py --save_dir exp_nongauss/output_adaptive_c2.00
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

SMALL_SIMS = [
    ("nongauss_A1S", r"A1-S ($\nu$=10)"),
    ("nongauss_B2S", r"B2-S ($k$=9)"),
    ("nongauss_C1S", r"C1-S ($\pi$=0.02)"),
]
LARGE_SIMS = [
    ("nongauss_A1L", r"A1-L ($\nu$=3)"),
    ("nongauss_B2L", r"B2-L ($k$=2)"),
    ("nongauss_C1L", r"C1-L ($\pi$=0.10)"),
]

METHODS = {
    "CKME (fixed $h$)":    "covered_score",
    "CKME (adaptive $h$)": "covered_score_adaptive",
    "DCP-DR":              "covered_score_dr",
}
COLORS  = {"CKME (fixed $h$)": "#2166ac", "CKME (adaptive $h$)": "#e66101", "DCP-DR": "#d6604d"}
LS      = {"CKME (fixed $h$)": "--",      "CKME (adaptive $h$)": "-",       "DCP-DR": "-."}
MARKERS = {"CKME (fixed $h$)": "s",       "CKME (adaptive $h$)": "o",       "DCP-DR": "^"}

X_LO, X_HI = 0.0, 2 * np.pi
K_LIST = [5, 10, 20, 40]


def load_per_point(output_dir: Path, sim: str, site_method: str) -> list[pd.DataFrame]:
    paths = sorted(output_dir.glob(f"macrorep_*/case_{sim}_{site_method}/per_point.csv"))
    return [pd.read_csv(p) for p in paths] if paths else []


# ---------------------------------------------------------------------------
# Figure 1: Group coverage by x-bin
# ---------------------------------------------------------------------------

def compute_bin_coverage(
    dfs: list[pd.DataFrame], cov_col: str, n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return centers, np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    stacked = np.vstack(per_macro)
    return centers, np.nanmean(stacked, axis=0), np.nanstd(stacked, axis=0)


def plot_group_coverage(output_dir: Path, site_method: str, n_bins: int,
                        alpha: float, save_path: Path | None):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    fig.suptitle(
        r"Group Coverage: Fixed $h$ vs Adaptive $h(x)=c\,\sigma(x)$",
        fontsize=13, y=1.02,
    )
    sim_groups = [SMALL_SIMS, LARGE_SIMS]
    row_labels = ["Small departure", "Large departure"]

    for row, (sims, rlabel) in enumerate(zip(sim_groups, row_labels)):
        for col, (sim, label) in enumerate(sims):
            ax = axes[row, col]
            dfs = load_per_point(output_dir, sim, site_method)
            if not dfs:
                ax.set_title(f"{label}\n(no data)", fontsize=9)
                continue

            for method, cov_col in METHODS.items():
                centers, means, stds = compute_bin_coverage(dfs, cov_col, n_bins)
                if np.isnan(means).all():
                    continue
                ax.plot(centers, means, color=COLORS[method],
                        linestyle=LS[method], linewidth=1.8, label=method)
                ax.fill_between(centers, means - stds, means + stds,
                                color=COLORS[method], alpha=0.12)

            ax.axhline(1 - alpha, color="black", linestyle=":", linewidth=1.2,
                       label=f"target ({1-alpha:.0%})")
            ax.set_xlim(X_LO, X_HI)
            ax.set_xticks([0, np.pi, 2 * np.pi])
            ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
            ax.set_ylim(0.6, 1.05)
            ax.set_xlabel("x")
            ax.set_title(label, fontsize=10)

            if col == 0:
                ax.set_ylabel(f"{rlabel}\nCoverage")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="lower left")
            ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Figure 2: Max bin deviation vs K
# ---------------------------------------------------------------------------

def max_bin_deviation(
    dfs: list[pd.DataFrame], cov_col: str, K: int, alpha: float,
) -> tuple[float, float]:
    edges = np.linspace(X_LO, X_HI, K + 1)
    target = 1 - alpha
    per_macro = []
    for df in dfs:
        if cov_col not in df.columns:
            continue
        x = df["x0"].values
        y = df[cov_col].values
        bins = np.clip(np.digitize(x, edges) - 1, 0, K - 1)
        bin_cov = np.array([y[bins == b].mean() if (bins == b).sum() > 0 else np.nan
                            for b in range(K)])
        max_dev = np.nanmax(np.abs(bin_cov - target))
        per_macro.append(max_dev)
    if not per_macro:
        return np.nan, np.nan
    arr = np.array(per_macro)
    return float(arr.mean()), float(arr.std() / np.sqrt(len(arr)))


def binomial_floor(K: int, n_test: int, eta: float = 0.05) -> float:
    n_min = n_test / K
    return float(np.sqrt(np.log(2 * K / eta) / (2 * n_min)))


def plot_K_sweep(output_dir: Path, site_method: str, alpha: float,
                 n_test: int, save_path: Path | None):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    fig.suptitle(
        r"Max Bin Deviation vs $K$: Fixed $h$ vs Adaptive $h(x)$",
        fontsize=13, y=1.02,
    )
    sim_groups = [SMALL_SIMS, LARGE_SIMS]
    row_labels = ["Small departure", "Large departure"]

    for row, (sims, rlabel) in enumerate(zip(sim_groups, row_labels)):
        for col, (sim, label) in enumerate(sims):
            ax = axes[row, col]
            dfs = load_per_point(output_dir, sim, site_method)
            if not dfs:
                ax.set_title(f"{label}\n(no data)", fontsize=9)
                continue

            for method, cov_col in METHODS.items():
                means, ses = [], []
                for K in K_LIST:
                    m, s = max_bin_deviation(dfs, cov_col, K, alpha)
                    means.append(m)
                    ses.append(s)
                if all(np.isnan(means)):
                    continue
                ax.errorbar(K_LIST, means, yerr=ses,
                            marker=MARKERS[method], color=COLORS[method],
                            label=method, linewidth=1.8, capsize=3, markersize=6)

            # T-G2 binomial floor
            K_dense = np.linspace(K_LIST[0], K_LIST[-1], 100)
            floor_dense = [binomial_floor(int(k), n_test) for k in K_dense]
            ax.plot(K_dense, floor_dense, "k:", linewidth=1.5, label="T-G2 floor")

            ax.set_xlim(K_LIST[0] - 1, K_LIST[-1] + 2)
            ax.set_ylim(0, 0.55)
            ax.set_xlabel("$K$ (number of bins)")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)

            if col == 0:
                ax.set_ylabel(f"{rlabel}\nMax bin deviation")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(output_dir: Path, site_method: str, alpha: float):
    print("\nMax bin deviation summary (K=10):")
    print(f"{'Simulator':>16s}  {'CKME-fixed':>11s}  {'CKME-adapt':>11s}  {'DCP-DR':>11s}")
    for sim, _ in SMALL_SIMS + LARGE_SIMS:
        dfs = load_per_point(output_dir, sim, site_method)
        if not dfs:
            continue
        vals = {}
        for method, cov_col in METHODS.items():
            m, s = max_bin_deviation(dfs, cov_col, 10, alpha)
            vals[method] = f"{m:.3f}±{s:.3f}" if not np.isnan(m) else "    N/A    "
        print(f"{sim:>16s}  {vals['CKME (fixed $h$)']:>11s}  "
              f"{vals['CKME (adaptive $h$)']:>11s}  {vals['DCP-DR']:>11s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  type=str, default=None)
    parser.add_argument("--n_bins",      type=int, default=10)
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--site_method", type=str, default="lhs")
    parser.add_argument("--n_test",      type=int, default=1000)
    parser.add_argument("--save_dir",    type=str, default=None,
                        help="Directory to save figures (default: show)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (
        _root / "exp_nongauss" / "output_adaptive_c2.00")

    print_summary(out_dir, args.site_method, args.alpha)

    save_dir = Path(args.save_dir) if args.save_dir else None

    plot_group_coverage(
        out_dir, args.site_method, args.n_bins, args.alpha,
        save_path=(save_dir / "group_coverage_compare.png") if save_dir else None,
    )
    plot_K_sweep(
        out_dir, args.site_method, args.alpha, args.n_test,
        save_path=(save_dir / "K_sweep_compare.png") if save_dir else None,
    )


if __name__ == "__main__":
    main()
