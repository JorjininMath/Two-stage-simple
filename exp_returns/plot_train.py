"""
plot_train.py

Time-series plot of training data (D0 + D1) for one rolling round.
D0 = model fitting data, D1 = CP calibration data.

Usage (from project root):
    python exp_returns/plot_train.py --save
    python exp_returns/plot_train.py --round 2 --save
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


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "PingFang HK"
    plt.rcParams["axes.unicode_minus"] = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--round",      type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="exp_returns/output")
    parser.add_argument("--save",       action="store_true")
    args = parser.parse_args()

    data_dir = _root / args.output_dir / f"round_{args.round}" / "data"
    if not data_dir.exists():
        print(f"Data not found: {data_dir}")
        return

    Y0 = pd.read_csv(data_dir / "Y0.csv", header=None).values.flatten()
    Y1 = pd.read_csv(data_dir / "Y1.csv", header=None).values.flatten()

    n0, n1  = len(Y0), len(Y1)
    idx0    = np.arange(n0)
    idx1    = np.arange(n0, n0 + n1)
    Y_all   = np.concatenate([Y0, Y1])
    idx_all = np.arange(n0 + n1)

    fig, ax = plt.subplots(figsize=(13, 5))

    # D0: model fitting
    ax.scatter(idx0, Y0, color="steelblue", s=4, alpha=0.5, label="训练数据 D0（模型拟合）", zorder=2)
    # D1: CP calibration
    ax.scatter(idx1, Y1, color="darkorange", s=4, alpha=0.5, label="校准数据 D1（区间校准）", zorder=2)

    # Dividing line between D0 and D1
    ax.axvline(n0, color="gray", lw=1.2, ls="--", alpha=0.8)
    ax.text(n0 + n1 * 0.01, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 8,
            "D1 校准段", fontsize=9, color="gray", va="top")

    ax.axhline(0, color="gray", lw=0.7, ls="--", alpha=0.4)
    ax.set_ylabel("日收益率 Y (%)", fontsize=12)
    ax.set_xlabel("训练集交易日（时间顺序）", fontsize=11)
    ax.set_title("历史日收益率训练数据", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    if args.save:
        p = _root / args.output_dir / f"plot_train_round{args.round}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
