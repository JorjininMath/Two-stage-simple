"""
Plot Adaptive Gain Curve (Δ IS vs B_total) and width-vs-B_total from
saturation-sweep outputs.

Usage:
  python exp_design/plot_adaptive_gain_curve.py \\
      --dgps nongauss_A1L_raw gibbs_s1_d5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent / "output" / "saturation"
COLORS = {
    "lhs": "#444444",
    "sampling_tail": "#d62728",
    "sampling_epistemic": "#1f77b4",
}

# Provisional n_0^*(d). To be replaced by calibrate_fill_factor.py output.
N0_STAR_PROVISIONAL = {1: 100, 5: 300}

# DGP → d mapping for regime shading
DGP_DIM = {
    "nongauss_A1L_raw": 1,
    "nongauss_A1S_raw": 1,
    "nongauss_B2L_raw": 1,
    "exp2_gauss_low": 1,
    "exp2_gauss_high": 1,
    "gibbs_s1_d5": 5,
    "gibbs_s2_d5": 5,
}

REGIME_COLORS = {
    "starved": "#ffe5e5",
    "intermediate": "#fff7d6",
    "saturated": "#e5f4e5",
}


def shade_regimes(ax, dgp: str, r_fixed: int = 10) -> None:
    """Shade ax background by (starved / intermediate / saturated) regime in B_total."""
    d = DGP_DIM.get(dgp)
    if d is None or d not in N0_STAR_PROVISIONAL:
        return
    n0_star = N0_STAR_PROVISIONAL[d]
    # B_total = 2 * n_0 * r_fixed (balanced 5:5)
    b_starved = 2 * (0.3 * n0_star) * r_fixed
    b_sat = 2 * n0_star * r_fixed
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], b_starved, color=REGIME_COLORS["starved"], alpha=0.5, zorder=0)
    ax.axvspan(b_starved, b_sat, color=REGIME_COLORS["intermediate"], alpha=0.5, zorder=0)
    ax.axvspan(b_sat, xlim[1], color=REGIME_COLORS["saturated"], alpha=0.5, zorder=0)
    ax.set_xlim(xlim)


def load_agg(dgp: str) -> pd.DataFrame:
    p = ROOT / dgp / "summary_agg.csv"
    return pd.read_csv(p)


def plot_for_dgps(dgps: list[str], out_path: Path) -> None:
    n = len(dgps)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 8), squeeze=False)

    for j, dgp in enumerate(dgps):
        agg = load_agg(dgp)
        # Pivots
        is_p = agg.pivot(index="B_total", columns="variant", values="IS")
        is_se = agg.pivot(index="B_total", columns="variant", values="IS_se")
        wd_p = agg.pivot(index="B_total", columns="variant", values="width")
        wd_se = agg.pivot(index="B_total", columns="variant", values="width_se")

        # Δ IS panel
        ax = axes[0, j]
        b = is_p.index.to_numpy()
        for var in ["sampling_tail", "sampling_epistemic"]:
            if var not in is_p.columns:
                continue
            delta = (is_p["lhs"] - is_p[var]).to_numpy()
            se = np.sqrt(is_se["lhs"].to_numpy() ** 2 + is_se[var].to_numpy() ** 2)
            ax.errorbar(b, delta, yerr=se, marker="o", capsize=3,
                        color=COLORS[var], label=var.replace("sampling_", ""))
        ax.axhline(0, color="k", lw=0.6, ls="--")
        ax.set_xscale("log")
        ax.set_xlim(b.min() * 0.9, b.max() * 1.1)
        shade_regimes(ax, dgp)
        ax.set_xlabel("B_total (log)")
        ax.set_ylabel("Δ IS = IS(LHS) − IS(adaptive)")
        d = DGP_DIM.get(dgp, "?")
        n0s = N0_STAR_PROVISIONAL.get(d, "?")
        ax.set_title(f"{dgp}  (d={d}, provisional n₀*={n0s})\nΔ IS ± 1 SE")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)

        # Width panel
        ax = axes[1, j]
        for var in ["lhs", "sampling_tail", "sampling_epistemic"]:
            if var not in wd_p.columns:
                continue
            ax.errorbar(b, wd_p[var].to_numpy(), yerr=wd_se[var].to_numpy(),
                        marker="o", capsize=3, color=COLORS[var],
                        label=var.replace("sampling_", ""))
        ax.set_xscale("log")
        ax.set_xlim(b.min() * 0.9, b.max() * 1.1)
        shade_regimes(ax, dgp)
        ax.set_xlabel("B_total (log)")
        ax.set_ylabel("Width")
        ax.set_title("Interval width ± 1 SE")
        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)

    # Figure-level regime legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=REGIME_COLORS["starved"], alpha=0.7, label="starved (ρ<0.3)"),
        Patch(color=REGIME_COLORS["intermediate"], alpha=0.7, label="intermediate (0.3≤ρ<1)"),
        Patch(color=REGIME_COLORS["saturated"], alpha=0.7, label="saturated (ρ≥1)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(
        "Regime shading from provisional n₀*(d) "
        "(to be finalized by calibrate_fill_factor.py)",
        y=-0.01, fontsize=9, style="italic",
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgps", nargs="+",
                    default=["nongauss_A1L_raw", "gibbs_s1_d5"])
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "_figs" / "adaptive_gain_curve.png"))
    args = ap.parse_args()
    plot_for_dgps(args.dgps, Path(args.out))


if __name__ == "__main__":
    main()
