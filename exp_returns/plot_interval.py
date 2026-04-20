"""
plot_interval.py

Plot prediction interval vs actual returns, one figure per method.

Methods with L/U columns: ckme, qr (DCP-QR), dr (DCP-DR)

Usage (from project root):
    python exp_returns/plot_interval.py --save
    python exp_returns/plot_interval.py --round 2 --save
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

METHODS = [
    ("ckme", "CKME",    "steelblue"),
    ("qr",   "DCP-QR",  "darkorange"),
    ("dr",   "DCP-DR",  "seagreen"),
]


def plot_one(ax, idx, Y, L, U, cov, color, label):
    ax.fill_between(idx, L, U, alpha=0.25, color=color, label=f"{label} 90% 预测区间")
    ax.plot(idx, U, color=color, lw=0.8, alpha=0.7)
    ax.plot(idx, L, color=color, lw=0.8, alpha=0.7)
    ax.scatter(idx[cov],  Y[cov],  color="seagreen", s=8,  alpha=0.7, label="区间内",   zorder=3)
    ax.scatter(idx[~cov], Y[~cov], color="crimson",  s=14, alpha=0.9,
               marker="x", label="超出区间", zorder=4)
    ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.set_ylabel("日收益率 Y (%)", fontsize=12)
    ax.set_xlabel("测试集交易日（时间顺序）", fontsize=11)
    ax.set_title(f"{label} 预测区间 vs 实际日收益率", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.2)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "PingFang HK"
    plt.rcParams["axes.unicode_minus"] = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--round",      type=int,  default=0)
    parser.add_argument("--output_dir", type=str,  default="exp_returns/output")
    parser.add_argument("--save",       action="store_true")
    args = parser.parse_args()

    out_dir = _root / args.output_dir
    pp_path = out_dir / f"round_{args.round}" / "per_point.csv"
    if not pp_path.exists():
        print(f"File not found: {pp_path}\nRun run_returns.py first.")
        return

    df  = pd.read_csv(pp_path)
    n   = len(df)
    idx = np.arange(n)
    Y   = df["Y"].values

    for key, label, color in METHODS:
        if f"L_{key}" not in df.columns:
            continue
        L   = df[f"L_{key}"].values
        U   = df[f"U_{key}"].values
        cov = df[f"covered_{key}"].values.astype(bool)

        fig, ax = plt.subplots(figsize=(13, 5))
        plot_one(ax, idx, Y, L, U, cov, color, label)
        fig.tight_layout()

        if args.save:
            p = out_dir / f"plot_interval_{key}_round{args.round}.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            print(f"Saved {p}")
        else:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
