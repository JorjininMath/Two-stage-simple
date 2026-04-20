"""
plot_consistency.py

Generate all figures for the asymptotic consistency experiment:

  1. fig_loglog_{sim}.png  — L1/L2/L3 metrics vs n (log-log), one per simulator
  2. fig_coverage_curves_{sim}.png — Cov_n(x) vs x for each n, one per simulator
  3. fig_compare_fixed_vs_adaptive_{sim}.png — mae_cov + sup_err side-by-side, one per sim

Usage:
  python exp_conditional_coverage/plot_consistency.py
  python exp_conditional_coverage/plot_consistency.py --sims exp1
  python exp_conditional_coverage/plot_consistency.py --output_dir exp_conditional_coverage/output_consistency_fixed
  python exp_conditional_coverage/plot_consistency.py --no_adaptive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REF_SLOPE = -0.4          # theoretical nonparametric rate (d=1): -2/5
ALPHA_SHADE = 0.20        # fill_between transparency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fit_slope(ns: np.ndarray, means: np.ndarray) -> tuple[float, float]:
    """OLS slope on log-log scale. Returns (slope, intercept)."""
    valid = (ns > 0) & (means > 0) & np.isfinite(means)
    if valid.sum() < 2:
        return float("nan"), float("nan")
    log_n = np.log(ns[valid])
    log_m = np.log(means[valid])
    b = np.polyfit(log_n, log_m, 1)
    return float(b[0]), float(b[1])


def _ref_line(ns: np.ndarray, means: np.ndarray, slope: float = REF_SLOPE) -> np.ndarray:
    """Reference slope line anchored at largest n."""
    anchor_n = ns[-1]
    anchor_m = means[np.isfinite(means)][-1]
    if anchor_m <= 0:
        return np.full_like(ns, float("nan"), dtype=float)
    log_ref = np.log(anchor_m) + slope * (np.log(ns) - np.log(anchor_n))
    return np.exp(log_ref)


def _label_slope(slope: float) -> str:
    if np.isnan(slope):
        return "(n/a)"
    return f"(slope {slope:+.2f})"


# ---------------------------------------------------------------------------
# Figure 1: log-log convergence (L1, L2, L3) — one figure per simulator
# ---------------------------------------------------------------------------
def plot_loglog(df: pd.DataFrame, sim: str, save_path: Path) -> None:
    """
    Three-panel log-log plot:
      Row 0: L3 — mae_cov, sup_err_cov
      Row 1: L2 — mae_ep_L, mae_ep_U
      Row 2: L1 — mae_q_lo, mae_q_hi
    """
    ns = df["n_0"].values.astype(float)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    fig.suptitle(f"Asymptotic consistency — {sim}", fontsize=12, y=1.01)

    panels = [
        # (ax, title, metric pairs [(col_mean, col_sd, label, color)])
        (axes[0], "L3: Conditional coverage", [
            ("mae_cov_mean",     "mae_cov_sd",      "MAE-Cov",  "tab:blue"),
            ("sup_err_cov_mean", "sup_err_cov_sd",  "Sup-Err",  "tab:orange"),
        ]),
        (axes[1], "L2: Interval endpoints", [
            ("mae_ep_L_mean", "mae_ep_L_sd", r"MAE $\hat{L}-q_{\alpha/2}$",   "tab:green"),
            ("mae_ep_U_mean", "mae_ep_U_sd", r"MAE $\hat{U}-q_{1-\alpha/2}$", "tab:red"),
        ]),
        (axes[2], "L1: Quantile (pre-CP)", [
            ("mae_q_lo_mean", "mae_q_lo_sd", r"MAE $\hat{q}_{\alpha/2}$",   "tab:purple"),
            ("mae_q_hi_mean", "mae_q_hi_sd", r"MAE $\hat{q}_{1-\alpha/2}$", "tab:brown"),
        ]),
    ]

    for ax, title, metrics in panels:
        ref_anchored = False
        for col_mean, col_sd, label, color in metrics:
            if col_mean not in df.columns:
                continue
            means = df[col_mean].values.astype(float)
            sds   = df[col_sd].values.astype(float) if col_sd in df.columns else np.zeros_like(means)

            slope, intercept = _fit_slope(ns, means)
            lbl = f"{label} {_label_slope(slope)}"

            ax.loglog(ns, means, "o-", color=color, linewidth=1.8, markersize=5, label=lbl)
            # SD band (log-space fill; clip at small positive)
            lo = np.maximum(means - sds, 1e-9)
            hi = means + sds
            ax.fill_between(ns, lo, hi, alpha=ALPHA_SHADE, color=color, linewidth=0)

            # Reference slope line (anchored once per panel, grey dashed)
            if not ref_anchored and np.any(means > 0):
                ref = _ref_line(ns, means, REF_SLOPE)
                ax.loglog(ns, ref, "--", color="grey", linewidth=1.2,
                          label=f"Ref slope {REF_SLOPE:+.1f}")
                ref_anchored = True

        ax.set_xlabel("n", fontsize=10)
        ax.set_ylabel("Error", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(int(n)) for n in ns], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Figure 2: Coverage curves Cov_n(x) vs x — one figure per simulator
# ---------------------------------------------------------------------------
def plot_coverage_curves(df_raw: pd.DataFrame, sim: str, alpha: float,
                         save_path: Path) -> None:
    """
    For each n_0 value: plot mean Cov_n(x) ± 1SD across macroreps vs x.
    """
    ns = sorted(df_raw["n_0"].unique())
    cmap = plt.get_cmap("viridis", len(ns))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(1 - alpha, color="black", linestyle="--", linewidth=1.2,
               label=f"Target {1-alpha:.0%}", zorder=5)

    for i, n in enumerate(ns):
        sub = df_raw[df_raw["n_0"] == n]
        # mean and sd of cov_mc across macroreps at each x_eval
        grp = sub.groupby("x_eval")["cov_mc"].agg(["mean", "std"]).reset_index()
        x   = grp["x_eval"].values
        mu  = grp["mean"].values
        sd  = grp["std"].values
        color = cmap(i)
        ax.plot(x, mu, "-", color=color, linewidth=1.6, label=f"n={int(n)}")
        ax.fill_between(x, mu - sd, mu + sd, alpha=ALPHA_SHADE, color=color, linewidth=0)

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel(r"$\widehat{\mathrm{Cov}}_n(x)$", fontsize=10)
    ax.set_title(f"Conditional coverage curves — {sim}", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Figure 2b: Width curves W_n(x) vs x — one figure per simulator
# ---------------------------------------------------------------------------
def plot_width_curves(df_raw: pd.DataFrame, sim: str, save_path: Path) -> None:
    """
    For each n_0 value: plot mean width W_n(x) = U-L ± 1SD across macroreps vs x.
    Also overlay oracle width = q_hi_oracle - q_lo_oracle (identical across macroreps).
    """
    ns = sorted(df_raw["n_0"].unique())
    cmap = plt.get_cmap("viridis", len(ns))

    # Compute width columns
    df_raw = df_raw.copy()
    df_raw["width"] = df_raw["U"] - df_raw["L"]
    df_raw["oracle_width"] = df_raw["q_hi_oracle"] - df_raw["q_lo_oracle"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Oracle width (same for all n, use first available)
    sub0 = df_raw[df_raw["n_0"] == ns[0]]
    grp0 = sub0.groupby("x_eval")["oracle_width"].mean().reset_index()
    ax.plot(grp0["x_eval"].values, grp0["oracle_width"].values,
            "k--", linewidth=2.0, label="Oracle width", zorder=5)

    for i, n in enumerate(ns):
        sub = df_raw[df_raw["n_0"] == n]
        grp = sub.groupby("x_eval")["width"].agg(["mean", "std"]).reset_index()
        x   = grp["x_eval"].values
        mu  = grp["mean"].values
        sd  = grp["std"].values
        color = cmap(i)
        ax.plot(x, mu, "-", color=color, linewidth=1.6, label=f"n={int(n)}")
        ax.fill_between(x, mu - sd, mu + sd, alpha=ALPHA_SHADE, color=color, linewidth=0)

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("Width (U − L)", fontsize=10)
    ax.set_title(f"Prediction interval width — {sim}", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Figure 3: Compare fixed vs adaptive (mae_cov + sup_err)
# ---------------------------------------------------------------------------
def plot_compare_fixed_adaptive(
    df_fixed: pd.DataFrame,
    df_adapt: pd.DataFrame,
    sim: str,
    save_path: Path,
) -> None:
    """Two panels: mae_cov and sup_err_cov, fixed vs adaptive overlaid."""
    ns = df_fixed["n_0"].values.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(f"Fixed vs Adaptive h — {sim}", fontsize=11)

    metrics = [
        ("mae_cov_mean",     "mae_cov_sd",     "MAE-Cov",  axes[0]),
        ("sup_err_cov_mean", "sup_err_cov_sd", "Sup-Err",  axes[1]),
    ]

    styles = [
        (df_fixed, "Fixed h (CV)",    "tab:blue",   "o-"),
        (df_adapt, "Adaptive h (c=2)", "tab:orange", "s--"),
    ]

    for col_mean, col_sd, title, ax in metrics:
        for df, lbl, color, marker in styles:
            if col_mean not in df.columns:
                continue
            means = df[col_mean].values.astype(float)
            sds   = df[col_sd].values.astype(float) if col_sd in df.columns else np.zeros_like(means)
            slope, _ = _fit_slope(ns, means)
            ax.loglog(ns, means, marker, color=color, linewidth=1.8, markersize=5,
                      label=f"{lbl} {_label_slope(slope)}")
            lo = np.maximum(means - sds, 1e-9)
            hi = means + sds
            ax.fill_between(ns, lo, hi, alpha=ALPHA_SHADE, color=color, linewidth=0)

        ref = _ref_line(ns, df_fixed[col_mean].values.astype(float), REF_SLOPE)
        ax.loglog(ns, ref, "--", color="grey", linewidth=1.1,
                  label=f"Ref slope {REF_SLOPE:+.1f}")

        ax.set_xlabel("n", fontsize=10)
        ax.set_ylabel("Error", fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(int(n)) for n in ns], fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot consistency experiment figures")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Primary output dir (default: auto-detect output_consistency_fixed/)")
    parser.add_argument("--adaptive_dir", type=str, default=None,
                        help="Adaptive-h output dir (default: auto-detect output_consistency_adaptive_c2.00/)")
    parser.add_argument("--sims", type=str, default="exp1,exp2",
                        help="Comma-separated simulators")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--no_adaptive", action="store_true",
                        help="Skip fixed-vs-adaptive comparison figure")
    args = parser.parse_args()

    sims = [s.strip() for s in args.sims.split(",")]

    fixed_dir = Path(args.output_dir) if args.output_dir else (_HERE / "output_consistency_fixed")
    adapt_dir = (
        Path(args.adaptive_dir) if args.adaptive_dir
        else (_HERE / "output_consistency_adaptive_c2.00")
    )

    if not fixed_dir.exists():
        print(f"ERROR: output dir not found: {fixed_dir}")
        sys.exit(1)

    for sim in sims:
        print(f"\n=== {sim} ===")

        # --- Load summary ---
        sum_path = fixed_dir / f"summary_{sim}.csv"
        if not sum_path.exists():
            print(f"  summary not found: {sum_path}, skipping")
            continue
        df_sum = pd.read_csv(sum_path)

        # --- Load raw results ---
        raw_path = fixed_dir / f"results_{sim}.csv"
        df_raw = pd.read_csv(raw_path) if raw_path.exists() else None

        # Figure 1: log-log
        plot_loglog(df_sum, sim, fixed_dir / f"fig_loglog_{sim}.png")

        # Figure 2: coverage curves + width curves
        if df_raw is not None:
            plot_coverage_curves(df_raw, sim, args.alpha,
                                 fixed_dir / f"fig_coverage_curves_{sim}.png")
            plot_width_curves(df_raw, sim,
                              fixed_dir / f"fig_width_curves_{sim}.png")
        else:
            print("  raw results not found, skipping coverage/width curves")

        # Width curves for adaptive dir too
        if not args.no_adaptive and adapt_dir.exists():
            raw_adapt = adapt_dir / f"results_{sim}.csv"
            if raw_adapt.exists():
                df_raw_adapt = pd.read_csv(raw_adapt)
                plot_coverage_curves(df_raw_adapt, sim, args.alpha,
                                     adapt_dir / f"fig_coverage_curves_{sim}.png")
                plot_width_curves(df_raw_adapt, sim,
                                  adapt_dir / f"fig_width_curves_{sim}.png")

        # Figure 3: fixed vs adaptive
        if not args.no_adaptive and adapt_dir.exists():
            sum_adapt = adapt_dir / f"summary_{sim}.csv"
            if sum_adapt.exists():
                df_adapt = pd.read_csv(sum_adapt)
                plot_compare_fixed_adaptive(
                    df_sum, df_adapt, sim,
                    fixed_dir / f"fig_compare_fixed_vs_adaptive_{sim}.png",
                )
            else:
                print(f"  adaptive summary not found for {sim}, skipping comparison")
        elif not args.no_adaptive:
            print(f"  adaptive dir not found ({adapt_dir}), skipping comparison")

    print("\nDone.")


if __name__ == "__main__":
    main()
