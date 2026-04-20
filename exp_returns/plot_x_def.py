"""
plot_x_def.py

Explanatory figure for the covariate X (lagged realized volatility).

Usage (from project root):
    python exp_returns/plot_x_def.py --save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    plt.rcParams["font.family"] = "PingFang HK"
    plt.rcParams["axes.unicode_minus"] = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--save",       action="store_true")
    parser.add_argument("--output_dir", type=str, default="exp_returns/output")
    args = parser.parse_args()

    out_dir = _root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # ------------------------------------------------------------------ #
    # Title
    # ------------------------------------------------------------------ #
    ax.text(5, 9.4, "协变量 X：滞后已实现波动率",
            ha="center", va="center", fontsize=16, fontweight="bold")

    # ------------------------------------------------------------------ #
    # Equation (rendered via mathtext)
    # ------------------------------------------------------------------ #
    ax.text(5, 7.8,
            r"$X_t = \sqrt{\sum_{s=t-22}^{t-1} r_s^2}$",
            ha="center", va="center", fontsize=22)

    ax.text(5, 6.3,
            r"$r_s$：第 $s$ 天市场日收益率（%）",
            ha="center", va="center", fontsize=12, color="#444444")

    # ------------------------------------------------------------------ #
    # Timeline diagram
    # ------------------------------------------------------------------ #
    # Draw timeline bar
    tl_y   = 4.8
    tl_x0  = 1.2
    tl_x1  = 8.8
    n_days = 23   # t-22 ... t (24 ticks, show selected)

    ax.annotate("", xy=(tl_x1 + 0.2, tl_y), xytext=(tl_x0 - 0.1, tl_y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5))

    # Tick marks and labels
    xs = np.linspace(tl_x0, tl_x1, n_days)
    window_color = "steelblue"
    target_color = "crimson"

    for i, xp in enumerate(xs):
        if i == n_days - 1:
            col = target_color
            h   = 0.35
        else:
            col = window_color
            h   = 0.22
        ax.plot([xp, xp], [tl_y - h, tl_y + h], color=col, lw=1.5)

    # Shade the 22-day window
    ax.fill_betweenx([tl_y - 0.22, tl_y + 0.22],
                     xs[0], xs[-2],
                     color=window_color, alpha=0.15)

    # Brace / label for window
    ax.annotate("", xy=(xs[-2], tl_y - 0.6), xytext=(xs[0], tl_y - 0.6),
                arrowprops=dict(arrowstyle="<->", color=window_color, lw=1.5))
    ax.text((xs[0] + xs[-2]) / 2, tl_y - 1.0,
            "过去 22 个交易日\n用于计算 $X_t$",
            ha="center", va="center", fontsize=10.5, color=window_color)

    # Label t-22, t-1, t
    for xi, lbl in [(xs[0], "$t-22$"), (xs[-2], "$t-1$"), (xs[-1], "$t$")]:
        col = target_color if lbl == "$t$" else window_color
        ax.text(xi, tl_y + 0.55, lbl,
                ha="center", va="bottom", fontsize=10, color=col)

    # Label for t (today)
    ax.text(xs[-1], tl_y - 0.75,
            "当天\n预测目标 $Y_t$",
            ha="center", va="top", fontsize=10, color=target_color)

    # ------------------------------------------------------------------ #
    # Bottom note
    # ------------------------------------------------------------------ #
    ax.text(5, 1.1,
            "X 越大 → 市场波动越剧烈 → 收益率分布越宽",
            ha="center", va="center", fontsize=11.5,
            color="black",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff", edgecolor="#aaaacc", lw=1))

    fig.tight_layout()

    if args.save:
        p = out_dir / "plot_x_def.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved {p}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
