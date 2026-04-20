"""
analyze_group_coverage_K.py

Ablation 6: Group coverage vs bin count K.
Post-hoc re-binning of existing per_point.csv data.

Produces a figure with:
  - x-axis: K (number of bins)
  - y-axis: max bin deviation |Cov_k - (1-alpha)|
  - Lines: CKME, DCP-DR, hetGP, theoretical binomial floor
  - Layout: 2x3 (Small/Large x A1/B2/C1)

Usage (from project root):
    python exp_nongauss/analyze_group_coverage_K.py
    python exp_nongauss/analyze_group_coverage_K.py --save exp_nongauss/output/ablation6_K.png
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

COVERAGE_COLS = {
    "CKME":   "covered_score",
    "DCP-DR": "covered_score_dr",
    "hetGP":  "covered_interval_hetgp",
}

METHOD_COLORS = {"CKME": "#2166ac", "DCP-DR": "#d6604d", "hetGP": "#4dac26"}
METHOD_MARKERS = {"CKME": "o", "DCP-DR": "s", "hetGP": "^"}

X_LO, X_HI = 0.0, 2 * np.pi
K_LIST = [5, 10, 20, 40]


def load_per_point(output_dir: Path, sim: str, site_method: str) -> list[pd.DataFrame]:
    paths = sorted(output_dir.glob(f"macrorep_*/case_{sim}_{site_method}/per_point.csv"))
    if not paths:
        return []
    return [pd.read_csv(p) for p in paths]


def max_bin_deviation(
    dfs: list[pd.DataFrame], cov_col: str, K: int, alpha: float,
) -> tuple[float, float]:
    """Return (mean, se) of max-bin-deviation across macroreps."""
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
    """T-G2 binomial fluctuation term: sqrt(log(2K/eta) / (2 * n_min))."""
    n_min = n_test / K  # approximate for equal-width bins + LHS
    return float(np.sqrt(np.log(2 * K / eta) / (2 * n_min)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--site_method", type=str, default="lhs")
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_nongauss" / "output"

    # Compute max bin deviation for each (sim, method, K)
    rows = []
    for sim, _ in SMALL_SIMS + LARGE_SIMS:
        dfs = load_per_point(out_dir, sim, args.site_method)
        if not dfs:
            print(f"  No data for {sim}, skipping")
            continue
        for method, cov_col in COVERAGE_COLS.items():
            for K in K_LIST:
                mean_dev, se_dev = max_bin_deviation(dfs, cov_col, K, args.alpha)
                rows.append({
                    "simulator": sim, "method": method, "K": K,
                    "max_bin_dev_mean": mean_dev, "max_bin_dev_se": se_dev,
                })

    df_result = pd.DataFrame(rows)
    csv_path = out_dir / "ablation6_K_sweep.csv"
    df_result.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print summary
    print("\nMax bin deviation by K:")
    pivot = df_result.pivot_table(
        index=["simulator", "method"], columns="K", values="max_bin_dev_mean",
    )
    print(pivot.to_string(float_format="%.3f"))

    # Theoretical floor
    floors = {K: binomial_floor(K, args.n_test) for K in K_LIST}
    print("\nBinomial floor (T-G2):")
    for K, f in floors.items():
        print(f"  K={K:2d}: {f:.3f}")

    # --- Figure: 2x3 grid ---
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    fig.suptitle(
        r"Ablation 6: Max bin deviation vs $K$  (T-G2 validation)",
        fontsize=13, y=1.02,
    )

    sim_groups = [SMALL_SIMS, LARGE_SIMS]
    row_labels = ["Small departure", "Large departure"]

    for row, (sims, rlabel) in enumerate(zip(sim_groups, row_labels)):
        for col, (sim, label) in enumerate(sims):
            ax = axes[row, col]
            sub = df_result[df_result["simulator"] == sim]

            for method in COVERAGE_COLS:
                msub = sub[sub["method"] == method]
                if msub.empty or msub["max_bin_dev_mean"].isna().all():
                    continue
                ax.errorbar(
                    msub["K"].values, msub["max_bin_dev_mean"].values,
                    yerr=msub["max_bin_dev_se"].values,
                    marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                    label=method, linewidth=1.8, capsize=3, markersize=6,
                )

            # Binomial floor curve
            K_dense = np.linspace(K_LIST[0], K_LIST[-1], 100)
            floor_dense = [binomial_floor(int(k), args.n_test) for k in K_dense]
            ax.plot(K_dense, floor_dense, "k:", linewidth=1.5, label="T-G2 floor")

            ax.set_xlim(K_LIST[0] - 1, K_LIST[-1] + 2)
            ax.set_ylim(0, 0.55)
            ax.set_xlabel("K (number of bins)")
            ax.set_title(label, fontsize=10)
            ax.grid(alpha=0.3)

            if col == 0:
                ax.set_ylabel(f"{rlabel}\nMax bin deviation")
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure: {save_path}")

    if args.show:
        plt.show()

    if not args.save and not args.show:
        default_path = out_dir / "ablation6_K.png"
        plt.savefig(default_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure: {default_path}")


if __name__ == "__main__":
    main()
