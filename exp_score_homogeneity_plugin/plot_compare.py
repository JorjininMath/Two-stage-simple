"""
plot_compare.py

Side-by-side comparison of oracle sigma vs plug-in sigma_hat for the
score homogeneity experiment.

Produces:
  figA_cscan_compare.png   — ks_max and cov_gap_sup vs c (3 sims x 2 metrics)
  figB_coverage_curve_compare_{sim}.png — coverage vs x, oracle/plugin/fixed
  figC_sigma_hat_vs_true_{sim}.png — sigma_hat(x) vs sigma_true(x)

Usage:
  python exp_score_homogeneity_plugin/plot_compare.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).parent
_ORACLE_DIR = _HERE.parent / "exp_score_homogeneity" / "output"
_PLUGIN_DIR = _HERE / "output"

_SIMS = ["mild", "exp2", "nongauss_B2L"]


def _parse_c(bw: str) -> float | None:
    m = re.match(r"adaptive_c([\d.]+)", bw)
    return float(m.group(1)) if m else None


def _cscan_rows(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary.copy()
    rows["c_val"] = rows["bandwidth"].apply(_parse_c)
    return rows[rows["c_val"].notna()].sort_values("c_val")


def _fixed_rows(summary: pd.DataFrame) -> dict:
    out = {}
    for bw in ["fixed_small", "fixed_cv", "fixed_large"]:
        row = summary[summary["bandwidth"] == bw]
        if len(row):
            out[bw] = row.iloc[0]
    return out


# ---------------------------------------------------------------------------
# Fig A: c-scan curves (3 sims × 2 metrics)
# ---------------------------------------------------------------------------

def plot_cscan_compare(save_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    metric_cols = [("ks_max_mean", "ks_max_sd", "KS max (homogeneity)"),
                    ("cov_gap_sup_mean", "cov_gap_sup_sd", "sup coverage gap")]

    for j, sim in enumerate(_SIMS):
        try:
            oracle_sum = pd.read_csv(_ORACLE_DIR / f"summary_{sim}.csv")
            plugin_sum = pd.read_csv(_PLUGIN_DIR / f"summary_{sim}.csv")
        except FileNotFoundError as e:
            print(f"  SKIP {sim}: {e}")
            continue

        o_scan = _cscan_rows(oracle_sum)
        p_scan = _cscan_rows(plugin_sum)
        o_fixed = _fixed_rows(oracle_sum)

        for i, (mean_col, sd_col, ylabel) in enumerate(metric_cols):
            ax = axes[i, j]

            ax.errorbar(o_scan["c_val"], o_scan[mean_col],
                         yerr=o_scan.get(sd_col, 0),
                         fmt="o-", color="#1f77b4", label="oracle $\\sigma$",
                         capsize=3, linewidth=1.5)
            ax.errorbar(p_scan["c_val"], p_scan[mean_col],
                         yerr=p_scan.get(sd_col, 0),
                         fmt="s--", color="#d62728",
                         label="plug-in $\\hat\\sigma$",
                         capsize=3, linewidth=1.5)

            # Horizontal reference: fixed_cv and fixed_small
            for bw_key, col, style in [("fixed_cv", "#2ca02c", ":"),
                                         ("fixed_small", "#888888", "-.")]:
                if bw_key in o_fixed:
                    ax.axhline(o_fixed[bw_key][mean_col], color=col,
                               linestyle=style, linewidth=1, alpha=0.8,
                               label=bw_key if i == 0 and j == 0 else None)

            if i == 0:
                ax.set_title(sim, fontsize=12)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 1:
                ax.set_xlabel("c (adaptive bandwidth scale)")
            ax.grid(True, alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=4, fontsize=10)

    fig.suptitle("Oracle vs Plug-in $\\hat\\sigma$: c-scan comparison",
                 fontsize=13, y=1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Fig B: coverage curve vs x (oracle / plug-in / fixed), one c
# ---------------------------------------------------------------------------

def plot_coverage_curve_compare(sim: str, c_show: float, alpha: float,
                                  save_path: Path):
    try:
        o_df = pd.read_csv(_ORACLE_DIR / f"results_{sim}.csv")
        p_df = pd.read_csv(_PLUGIN_DIR / f"results_{sim}.csv")
    except FileNotFoundError as e:
        print(f"  SKIP {sim}: {e}")
        return

    bw_label = f"adaptive_c{c_show:.1f}"

    def _cov_vs_x(df, bw):
        sub = df[df["bandwidth"] == bw]
        if sub.empty:
            return None, None, None
        x_vals = np.sort(sub["x_eval"].unique())
        macro_ids = sub["macrorep"].unique()
        M = len(x_vals)
        mat = np.full((len(macro_ids), M), np.nan)
        for mi, mid in enumerate(macro_ids):
            msub = sub[sub["macrorep"] == mid].sort_values("x_eval")
            if len(msub) == M:
                mat[mi] = msub["cov_mc"].values
        return x_vals, np.nanmean(mat, axis=0), np.nanstd(mat, axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(1 - alpha, color="gray", linestyle="--", linewidth=1,
               label=f"target = {1 - alpha:.2f}")

    for df, tag, color in [(o_df, f"oracle {bw_label}", "#1f77b4"),
                            (p_df, f"plug-in {bw_label}", "#d62728")]:
        x_arr, cmean, csd = _cov_vs_x(df, bw_label)
        if x_arr is None:
            continue
        ax.plot(x_arr, cmean, "-", color=color, linewidth=1.6, label=tag)
        ax.fill_between(x_arr, cmean - csd, cmean + csd, alpha=0.15, color=color)

    # Fixed baselines from oracle file (same data)
    for bw, color, style in [("fixed_small", "#888888", "-.")]:
        x_arr, cmean, _ = _cov_vs_x(o_df, bw)
        if x_arr is not None:
            ax.plot(x_arr, cmean, style, color=color, linewidth=1.2, label=bw)

    ax.set_xlabel("x")
    ax.set_ylabel("Conditional coverage")
    ax.set_title(f"Coverage curve — {sim}, c={c_show}")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Fig C: sigma_hat(x) vs sigma_true(x) from plug-in per-point results
# ---------------------------------------------------------------------------

def plot_sigma_diagnostic(sim: str, save_path: Path):
    try:
        p_df = pd.read_csv(_PLUGIN_DIR / f"results_{sim}.csv")
    except FileNotFoundError as e:
        print(f"  SKIP {sim}: {e}")
        return

    # Average sigma_hat across macroreps at each x
    sub = p_df[p_df["bandwidth"] == "adaptive_c1.0"]  # any adaptive row
    if sub.empty:
        return

    grp = sub.groupby("x_eval").agg(
        sigma_hat_mean=("sigma_hat", "mean"),
        sigma_hat_sd=("sigma_hat", "std"),
        sigma_true=("sigma_true", "first"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grp["x_eval"], grp["sigma_true"], "k-",
             linewidth=1.8, label=r"$\sigma_{\rm true}(x)$")
    ax.plot(grp["x_eval"], grp["sigma_hat_mean"], "r--",
             linewidth=1.5, label=r"$\hat\sigma(x)$ (kernel smoothed)")
    ax.fill_between(grp["x_eval"],
                     grp["sigma_hat_mean"] - grp["sigma_hat_sd"],
                     grp["sigma_hat_mean"] + grp["sigma_hat_sd"],
                     alpha=0.2, color="red")

    rmse = float(np.sqrt(np.mean((grp["sigma_hat_mean"] - grp["sigma_true"]) ** 2)))
    h_sig = float(sub["h_sigma"].iloc[0]) if "h_sigma" in sub else float("nan")
    ax.set_title(f"{sim}  (h$_\\sigma$={h_sig:.3f}, RMSE={rmse:.3f})")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\sigma$")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str,
                         default=str(_HERE / "output"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--c_show", type=float, default=2.0)
    args = parser.parse_args()

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cscan_compare(out_dir / "figA_cscan_compare.png")

    for sim in _SIMS:
        plot_coverage_curve_compare(
            sim, args.c_show, args.alpha,
            out_dir / f"figB_coverage_curve_compare_{sim}.png")
        plot_sigma_diagnostic(
            sim, out_dir / f"figC_sigma_hat_vs_true_{sim}.png")


if __name__ == "__main__":
    main()
