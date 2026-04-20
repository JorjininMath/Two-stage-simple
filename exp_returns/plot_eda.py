"""
plot_eda.py

EDA scatter: daily return Y vs lagged realized volatility X, with quantile bands.

Usage (from project root):
    python exp_returns/plot_eda.py --save
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

from exp_returns.preprocess import load_returns_data


def _setup_cjk_font(plt) -> str:
    """
    Select an available CJK font to avoid garbled Chinese text.
    Returns the chosen font name, or "default" if none is found.
    """
    from matplotlib import font_manager

    candidates = [
        "PingFang HK",
        "PingFang SC",
        "Heiti SC",
        "Songti SC",
        "STHeiti",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)

    if chosen is not None:
        plt.rcParams["font.family"] = chosen
    else:
        # Keep a wide fallback list for different OS/font packs.
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = candidates + ["DejaVu Sans"]
        chosen = "default"

    plt.rcParams["axes.unicode_minus"] = False
    return chosen


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen_font = _setup_cjk_font(plt)

    parser = argparse.ArgumentParser()
    parser.add_argument("--save",       action="store_true")
    parser.add_argument("--output_dir", type=str, default="exp_returns/output")
    args = parser.parse_args()

    out_dir = _root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    Y, X = load_returns_data()
    x = X[:, 0]

    q33, q67 = np.quantile(x, [1/3, 2/3])
    regime = np.where(x <= q33, "低波动", np.where(x <= q67, "中波动", "高波动"))
    regime_colors = {"低波动": "steelblue", "中波动": "darkorange", "高波动": "crimson"}

    fig, ax = plt.subplots(figsize=(11, 6))

    # Scatter (every 4th point for readability)
    idx = np.arange(0, len(Y), 4)
    for reg, col in regime_colors.items():
        mask = regime[idx] == reg
        ax.scatter(x[idx][mask], Y[idx][mask],
                   color=col, alpha=0.2, s=5, label=reg, rasterized=True)

    # Quantile bands via bin statistics
    bins    = np.linspace(x.min(), x.max(), 31)
    bin_idx = np.clip(np.digitize(x, bins) - 1, 0, len(bins) - 2)
    x_mid, q05, q25, q50, q75, q95 = [], [], [], [], [], []
    for b in range(len(bins) - 1):
        y_b = Y[bin_idx == b]
        if len(y_b) < 10:
            continue
        x_mid.append(0.5 * (bins[b] + bins[b + 1]))
        q05.append(np.quantile(y_b, 0.05))
        q25.append(np.quantile(y_b, 0.25))
        q50.append(np.quantile(y_b, 0.50))
        q75.append(np.quantile(y_b, 0.75))
        q95.append(np.quantile(y_b, 0.95))

    x_mid = np.array(x_mid)
    ax.fill_between(x_mid, q05, q95, alpha=0.15, color="black", label="5%–95% 分位数带")
    ax.fill_between(x_mid, q25, q75, alpha=0.28, color="black", label="25%–75% 分位数带")
    ax.plot(x_mid, q50, color="black", lw=2.0, label="中位数")

    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("滞后已实现波动率 X (%)", fontsize=12)
    ax.set_ylabel("日收益率 Y (%)", fontsize=12)
    ax.set_title("收益率分布随波动率变化", fontsize=13)
    ax.legend(fontsize=10, ncol=2, loc="upper right")
    ax.grid(alpha=0.2)

    fig.tight_layout()

    if args.save:
        p = out_dir / "plot_eda.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Font used: {chosen_font}")
        print(f"Saved {p}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
