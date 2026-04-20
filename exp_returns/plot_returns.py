"""
plot_returns.py

Visualise CKME + R benchmark results on stock returns data.

Figures:
  1. Conditional coverage vs lrealvol bin — all methods overlaid
  2. Conditional width    vs lrealvol bin — all methods overlaid
  3. Per-round summary bar (coverage + width + IS) for CKME

Usage (from project root):
    python exp_returns/plot_returns.py --save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Method display config: (coverage_col, width_col, color, marker, label)
METHODS = [
    ("covered_ckme",  "width_ckme",  "black",      "o", "CKME"),
    ("covered_dr",    "width_dr",    "royalblue",   "^", "DCP-DR"),
    ("covered_qr",    "width_qr",    "steelblue",   "s", "DCP-QR"),
    ("covered_cqr",   "width_cqr",   "darkorange",  "D", "CQR"),
    ("covered_cqrm",  "width_cqrm",  "orange",      "v", "CQR-m"),
    ("covered_cqrr",  "width_cqrr",  "gold",        "P", "CQR-r"),
    ("covered_reg",   "width_reg",   "seagreen",    "x", "CP-OLS"),
    ("covered_loc",   "width_loc",   "mediumvioletred", "+", "CP-loc"),
]


def load_all_rounds(output_dir: Path, n_rounds: int = 5):
    pp_frames, bin_frames = [], []
    for r in range(n_rounds):
        pp_path  = output_dir / f"round_{r}" / "per_point.csv"
        bin_path = output_dir / f"round_{r}" / "binned.csv"
        if not pp_path.exists():
            continue
        df_pp  = pd.read_csv(pp_path);  df_pp["round"]  = r + 1
        df_bin = pd.read_csv(bin_path); df_bin["round"] = r + 1
        pp_frames.append(df_pp)
        bin_frames.append(df_bin)
    df_pp  = pd.concat(pp_frames,  ignore_index=True) if pp_frames  else pd.DataFrame()
    df_bin = pd.concat(bin_frames, ignore_index=True) if bin_frames else pd.DataFrame()
    return df_pp, df_bin


def aggregate_bins(df_bin: pd.DataFrame) -> pd.DataFrame:
    cov_cols   = [c for c in df_bin.columns if c.startswith("covered")]
    width_cols = [c for c in df_bin.columns if c.startswith("width")]
    agg_dict   = {c: "mean" for c in cov_cols + width_cols}
    agg_dict["x_mid"] = "mean"
    agg_dict["n"]     = "sum"
    return df_bin.groupby("bin").agg(agg_dict).reset_index()


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    parser = argparse.ArgumentParser(description="Plot returns experiment results")
    parser.add_argument("--output_dir", type=str, default="exp_returns/output")
    parser.add_argument("--n_rounds",   type=int, default=5)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--save",       action="store_true")
    args = parser.parse_args()

    out_dir = _root / args.output_dir
    df_pp, df_bin = load_all_rounds(out_dir, n_rounds=args.n_rounds)

    if df_pp.empty:
        print(f"No results found in {out_dir}. Run run_returns.py first.")
        return

    agg        = aggregate_bins(df_bin)
    target_cov = 1.0 - args.alpha

    # Filter to methods that actually exist in data
    avail = [(cov, wid, col, mkr, lbl)
             for cov, wid, col, mkr, lbl in METHODS
             if cov in agg.columns and wid in agg.columns]

    # ------------------------------------------------------------------ #
    # Figure 1 : Conditional coverage vs bin                              #
    # ------------------------------------------------------------------ #
    fig1, ax1 = plt.subplots(figsize=(11, 5))
    for cov_col, _, color, marker, label in avail:
        lw = 2.5 if label == "CKME" else 1.5
        ax1.plot(agg["bin"], agg[cov_col],
                 marker=marker, lw=lw, color=color, label=label, markersize=5)
    ax1.axhline(target_cov, ls="--", color="gray", lw=1, label=f"Target {target_cov:.0%}")
    ax1.set_xlabel("Volatility bin (low → high lrealvol)", fontsize=12)
    ax1.set_ylabel("Conditional coverage", fontsize=12)
    ax1.set_title(f"Conditional coverage — {target_cov:.0%} prediction intervals", fontsize=13)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.set_ylim(0.5, 1.05)
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(axis="y", alpha=0.3)
    fig1.tight_layout()
    if args.save:
        p = out_dir / "plot_conditional_coverage.png"
        fig1.savefig(p, dpi=150); print(f"Saved {p}")
    else:
        plt.show()

    # ------------------------------------------------------------------ #
    # Figure 2 : Conditional width vs bin                                 #
    # ------------------------------------------------------------------ #
    fig2, ax2 = plt.subplots(figsize=(11, 5))
    for _, wid_col, color, marker, label in avail:
        lw = 2.5 if label == "CKME" else 1.5
        ax2.plot(agg["bin"], agg[wid_col],
                 marker=marker, lw=lw, color=color, label=label, markersize=5)
    ax2.set_xlabel("Volatility bin (low → high lrealvol)", fontsize=12)
    ax2.set_ylabel("Mean interval width (%)", fontsize=12)
    ax2.set_title("Conditional interval width", fontsize=13)
    ax2.legend(fontsize=9, ncol=2)
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    if args.save:
        p = out_dir / "plot_conditional_width.png"
        fig2.savefig(p, dpi=150); print(f"Saved {p}")
    else:
        plt.show()

    # ------------------------------------------------------------------ #
    # Figure 3 : Per-round CKME summary bar                               #
    # ------------------------------------------------------------------ #
    summary_path = out_dir / "returns_summary.csv"
    if summary_path.exists():
        df_sum = pd.read_csv(summary_path)
        df_rounds = df_sum[df_sum["round"] != "Overall"].copy()
        df_rounds["round"] = df_rounds["round"].astype(int)

        fig3, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, col, ylabel, color in [
            (axes[0], "CKME_coverage", "Coverage",        "steelblue"),
            (axes[1], "CKME_width",    "Mean width (%)",  "darkorange"),
            (axes[2], "CKME_IS",       "Interval score",  "seagreen"),
        ]:
            if col not in df_rounds.columns:
                continue
            ax.bar(df_rounds["round"], df_rounds[col], color=color, alpha=0.7)
            if col == "CKME_coverage":
                ax.axhline(target_cov, ls="--", color="gray", lw=1.5,
                           label=f"Target {target_cov:.0%}")
                ax.legend(fontsize=9)
                ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
                ax.set_ylim(0.7, 1.05)
            ax.set_xlabel("Round"); ax.set_ylabel(ylabel); ax.set_title(f"CKME {ylabel}")
            ax.set_xticks(df_rounds["round"]); ax.grid(axis="y", alpha=0.3)
        fig3.suptitle("CKME + split CP — 5-round rolling evaluation", fontsize=13, y=1.01)
        fig3.tight_layout()
        if args.save:
            p = out_dir / "plot_round_summary.png"
            fig3.savefig(p, dpi=150, bbox_inches="tight"); print(f"Saved {p}")
        else:
            plt.show()

    # ------------------------------------------------------------------ #
    # Print compact summary table                                         #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    cov_cols = [c for c in df_pp.columns if c.startswith("covered")]
    wid_cols = [c for c in df_pp.columns if c.startswith("width")]
    rows = []
    for cov_col, wid_col in zip(cov_cols, wid_cols):
        method = cov_col.replace("covered_", "").upper()
        rows.append({
            "method"  : method,
            "coverage": df_pp[cov_col].mean(),
            "width"   : df_pp[wid_col].mean(),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
