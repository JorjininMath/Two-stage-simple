"""
plot_exp3.py

Two figures for Exp3 (c-sweep on nongauss_A1L):

1. exp3_coverage_curves.png — single panel, conditional coverage curves overlaid
   for fixed h plus each c. Lets us see whether the curve shape is robust to c.

2. exp3_metric_vs_c.png — 2x2 panels of summary metrics vs c, with horizontal
   reference line at the fixed-h value:
     (a) worst-bin |cov - (1-alpha)|
     (b) mean-bin  |cov - (1-alpha)|
     (c) mean width
     (d) mean interval score
   Goal: show plateau (flat region around c=1.0), not knife-edge optimum.

Reads:
    exp_adaptive_h/output_exp3/macrorep_{k}/case_nongauss_A1L_{arm}/per_point.csv

Writes:
    exp_adaptive_h/output_exp3/exp3_coverage_curves.png
    exp_adaptive_h/output_exp3/exp3_metric_vs_c.png

Usage (from project root):
    python exp_adaptive_h/plot_exp3.py
    python exp_adaptive_h/plot_exp3.py --output_dir exp_adaptive_h/output_exp3 --n_bins 25
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Two_stage.sim_functions import get_experiment_config

SIMULATOR = "nongauss_A1L"


def _bin_coverage(df: pd.DataFrame, bin_edges: np.ndarray) -> np.ndarray:
    x = df["x0"].to_numpy()
    cov = df["covered_score"].to_numpy()
    n_bins = len(bin_edges) - 1
    out = np.full(n_bins, np.nan)
    idx = np.clip(np.searchsorted(bin_edges, x, side="right") - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out[b] = cov[mask].mean()
    return out


def _macrorep_metrics(df: pd.DataFrame, bin_edges: np.ndarray, target: float) -> dict:
    cov = df["covered_score"].to_numpy()
    width = df["width"].to_numpy()
    is_ = df["interval_score"].to_numpy()
    cov_bins = _bin_coverage(df, bin_edges)
    valid = ~np.isnan(cov_bins)
    devs = np.abs(cov_bins[valid] - target)
    return {
        "marginal":   cov.mean(),
        "worst_bin":  devs.max() if devs.size else np.nan,
        "mean_bin":   devs.mean() if devs.size else np.nan,
        "mean_width": width.mean(),
        "mean_is":    is_.mean(),
    }


def _discover_arms(out_dir: Path) -> list[str]:
    arms = set()
    pat = re.compile(rf"^case_{re.escape(SIMULATOR)}_(.+)$")
    for case_dir in out_dir.glob("macrorep_*/case_*"):
        m = pat.match(case_dir.name)
        if m:
            arms.add(m.group(1))
    return sorted(arms, key=lambda s: (s != "fixed", float(s[1:]) if s.startswith("c") else 0.0))


def _load_arm_paths(out_dir: Path, arm: str) -> list[Path]:
    return sorted(out_dir.glob(f"macrorep_*/case_{SIMULATOR}_{arm}/per_point.csv"))


def _arm_color_label(arm: str, cmap):
    if arm == "fixed":
        return "black", "fixed h (CV)"
    c_val = float(arm[1:])
    return cmap(c_val), f"oracle c={c_val:g}"


def main():
    parser = argparse.ArgumentParser(description="Plot Exp3 c-sweep")
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_exp3")
    parser.add_argument("--n_bins",     type=int, default=20)
    parser.add_argument("--alpha",      type=float, default=0.1)
    args = parser.parse_args()

    out_dir = (_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not out_dir.exists():
        print(f"ERROR: {out_dir} not found.", file=sys.stderr)
        sys.exit(1)

    target = 1.0 - args.alpha
    arms = _discover_arms(out_dir)
    if not arms:
        print(f"ERROR: no case_{SIMULATOR}_* directories under {out_dir}", file=sys.stderr)
        sys.exit(1)

    cfg = get_experiment_config(SIMULATOR)
    x_lo = float(cfg["bounds"][0][0])
    x_hi = float(cfg["bounds"][1][0])
    bin_edges = np.linspace(x_lo, x_hi, args.n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    c_vals = [float(a[1:]) for a in arms if a.startswith("c")]
    cmin = min(c_vals) if c_vals else 0.3
    cmax = max(c_vals) if c_vals else 2.0
    norm = plt.Normalize(vmin=cmin * 0.9, vmax=cmax * 1.1)
    cmap = lambda c: plt.cm.viridis(norm(c))  # noqa: E731

    # ---------- Figure 1: conditional coverage curves overlay ----------
    fig1, ax1 = plt.subplots(1, 1, figsize=(9, 6))
    arm_metrics: dict[str, dict] = {}

    for arm in arms:
        paths = _load_arm_paths(out_dir, arm)
        if not paths:
            continue
        cov_per = np.empty((len(paths), args.n_bins))
        marg = np.empty(len(paths))
        per_rep_metrics = []
        for i, p in enumerate(paths):
            df = pd.read_csv(p)
            cov_per[i] = _bin_coverage(df, bin_edges)
            marg[i] = df["covered_score"].mean()
            per_rep_metrics.append(_macrorep_metrics(df, bin_edges, target))

        med = np.nanmedian(cov_per, axis=0)
        color, label = _arm_color_label(arm, cmap)
        lw = 2.5 if arm == "fixed" else 1.6
        ls = "--" if arm == "fixed" else "-"
        ax1.plot(bin_centers, med, color=color, lw=lw, ls=ls,
                 marker="o", ms=3, label=f"{label} (marg={float(np.median(marg)):.3f})")

        m_df = pd.DataFrame(per_rep_metrics)
        arm_metrics[arm] = {
            "c":           float(arm[1:]) if arm.startswith("c") else float("nan"),
            "marginal":    float(np.median(marg)),
            "worst_bin":   float(m_df["worst_bin"].median()),
            "worst_bin_q1": float(m_df["worst_bin"].quantile(0.25)),
            "worst_bin_q3": float(m_df["worst_bin"].quantile(0.75)),
            "mean_bin":    float(m_df["mean_bin"].median()),
            "mean_bin_q1": float(m_df["mean_bin"].quantile(0.25)),
            "mean_bin_q3": float(m_df["mean_bin"].quantile(0.75)),
            "mean_width":  float(m_df["mean_width"].mean()),
            "mean_width_sd": float(m_df["mean_width"].std(ddof=1)),
            "mean_is":     float(m_df["mean_is"].mean()),
            "mean_is_sd":  float(m_df["mean_is"].std(ddof=1)),
        }

    ax1.axhline(target, ls=":", color="gray", alpha=0.7,
                label=f"target $1-\\alpha={target:.2f}$")
    ax1.set_xlabel("x")
    ax1.set_ylabel("conditional coverage (median over macroreps)")
    ax1.set_xlim(x_lo, x_hi)
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_title(
        f"Exp3 (nongauss_A1L): conditional coverage vs x, c-sweep "
        f"of oracle h(x)=c·s(x)"
    )
    fig1.tight_layout()
    out1 = out_dir / "exp3_coverage_curves.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved: {out1}")

    # ---------- Figure 2: metric vs c (4 panels) ----------
    if not c_vals:
        print("No oracle c arms found; skipping metric_vs_c figure.")
        return

    sweep_arms = [a for a in arms if a.startswith("c")]
    sweep_c = np.array([arm_metrics[a]["c"] for a in sweep_arms])
    fixed_metrics = arm_metrics.get("fixed")

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("worst_bin", "worst-bin |cov - (1-α)|", "viridis", True),
        ("mean_bin",  "mean-bin |cov - (1-α)|",  "viridis", True),
        ("mean_width", "mean width E[U-L]",      "viridis", False),
        ("mean_is",   "mean interval score",     "viridis", False),
    ]
    for ax, (key, ylabel, _, with_iqr) in zip(axes2.ravel(), panels):
        ys = np.array([arm_metrics[a][key] for a in sweep_arms])
        if with_iqr and (key + "_q1") in arm_metrics[sweep_arms[0]]:
            lo = np.array([arm_metrics[a][key + "_q1"] for a in sweep_arms])
            hi = np.array([arm_metrics[a][key + "_q3"] for a in sweep_arms])
            ax.fill_between(sweep_c, lo, hi, alpha=0.2, color="tab:blue",
                            label="25-75% across macroreps")
        elif (key + "_sd") in arm_metrics[sweep_arms[0]]:
            sds = np.array([arm_metrics[a][key + "_sd"] for a in sweep_arms])
            ax.fill_between(sweep_c, ys - sds, ys + sds, alpha=0.2,
                            color="tab:blue", label="±1 sd")
        ax.plot(sweep_c, ys, "o-", color="tab:blue", lw=2, ms=7,
                label="oracle h(x)=c·s(x)")
        if fixed_metrics is not None:
            ax.axhline(fixed_metrics[key], ls="--", color="black",
                       label=f"fixed h: {fixed_metrics[key]:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("c")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.set_xticks(sweep_c)
        ax.set_xticklabels([f"{c:g}" for c in sweep_c])

    fig2.suptitle(
        "Exp3 (nongauss_A1L): summary metrics vs c. "
        "Goal — plateau around c=1, not knife-edge.",
        fontsize=13, y=1.00,
    )
    fig2.tight_layout()
    out2 = out_dir / "exp3_metric_vs_c.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
