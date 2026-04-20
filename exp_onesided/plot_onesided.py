"""
plot_onesided.py

Visualize one-sided quantile comparison: CKME vs QR.

Figure modes:
  --mode bounds   (default): 3×2 grid — one row per simulator, columns = lower/upper bound
                              Shows q_true, q_ckme, q_qr, and scatter of Y vs x
  --mode coverage:           Bar chart of empirical coverage per (simulator, tau, method)
  --mode sup_error:          Boxplot of sup_x |q_hat - q_true| across macroreps
  --mode mean_error:         Boxplot of mean_x |q_hat - q_true| across macroreps
  --mode tailprob:           Line plot of delta_tau(x) on one macrorep
  --mode tailprob_t:         Plot P(Y>t | X=x) for fixed t values (true vs CKME)
  --mode tailprob_t_agg:     Aggregate over macroreps: median + IQR over x

Usage:
    python exp_onesided/plot_onesided.py
    python exp_onesided/plot_onesided.py --mode coverage
    python exp_onesided/plot_onesided.py --mode sup_error
    python exp_onesided/plot_onesided.py --mode tailprob --macrorep 0
    python exp_onesided/plot_onesided.py --macrorep 0 --save exp_onesided/output/bounds.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SIMULATORS = ["exp1", "exp2", "nongauss_B2L"]
SIM_LABELS = {
    "exp1": "exp1 (Gaussian)",
    "exp2": "exp2 (heteroscedastic Gaussian)",
    "nongauss_B2L": "nongauss_B2L (Gamma strong skew)",
}
TAUS = [0.05, 0.95]


def tau_tag(tau: float) -> str:
    return f"{tau:.2f}"


def load_perpoint(output_dir: str, macrorep: int) -> dict:
    """Load all per-point CSVs for a macrorep. Returns {(sim, tau): DataFrame}."""
    macro_dir = os.path.join(output_dir, f"macrorep_{macrorep}")
    data = {}
    for sim in SIMULATORS:
        for tau in TAUS:
            path = os.path.join(macro_dir, f"{sim}_tau{tau_tag(tau)}_perpoint.csv")
            if os.path.exists(path):
                data[(sim, tau)] = pd.read_csv(path)
    return data


def plot_bounds(data: dict, save_path: str | None = None):
    """3×2 grid: rows = simulators, cols = lower (tau=0.05) / upper (tau=0.95)."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="row")
    fig.suptitle("One-sided quantile bounds: true vs CKME vs QR (plug-in)", fontsize=13)

    tau_cols = {0.05: 0, 0.95: 1}

    for row, sim in enumerate(SIMULATORS):
        for tau, col in tau_cols.items():
            ax = axes[row, col]
            key = (sim, tau)
            if key not in data:
                ax.set_visible(False)
                continue

            df = data[key]
            x = df["x"].values

            # CKME quantile line
            ax.plot(x, df["q_ckme"].values, color="steelblue", lw=1.8,
                    label="CKME q^")

            # QR quantile line (if available)
            if "q_qr" in df.columns and df["q_qr"].notna().any():
                ax.plot(x, df["q_qr"].values, color="tomato", lw=1.8,
                        ls="--", label="QR q^")

            # True quantile line (if available)
            if "q_true" in df.columns and df["q_true"].notna().any():
                ax.plot(x, df["q_true"].values, color="black", lw=1.6,
                        ls="-", label="True q")

            ax.set_title(f"{SIM_LABELS[sim]}  τ={tau}", fontsize=8.5)
            ax.set_xlabel("x")
            ax.set_ylabel("y")

            if row == 0 and col == 0:
                ax.legend(fontsize=7, markerscale=3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_coverage(summary_path: str, save_path: str | None = None):
    """Bar chart of empirical coverage per (simulator, tau, method)."""
    if not os.path.exists(summary_path):
        print(f"Summary not found: {summary_path}")
        return

    df = pd.read_csv(summary_path)
    agg = df.groupby(["simulator", "tau", "method"])["coverage"].mean().reset_index()

    sims = SIMULATORS
    taus = TAUS
    methods = sorted(agg["method"].unique())
    colors = {"CKME": "steelblue", "QR": "tomato"}

    n_groups = len(sims) * len(taus)
    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.2), 5))

    bar_width = 0.35
    x_pos = np.arange(n_groups)
    labels = []

    for m_idx, method in enumerate(methods):
        heights = []
        for sim in sims:
            for tau in taus:
                row = agg[(agg["simulator"] == sim) & (agg["tau"] == tau) &
                          (agg["method"] == method)]
                heights.append(row["coverage"].values[0] if len(row) else np.nan)
                if m_idx == 0:
                    labels.append(f"{SIM_LABELS[sim]}\nτ={tau}")
        offset = (m_idx - (len(methods) - 1) / 2) * bar_width
        ax.bar(x_pos + offset, heights, bar_width,
               label=method, color=colors.get(method, "gray"), alpha=0.8)

    # Nominal one-sided coverage line (tau=0.05 or 0.95 -> target 0.95)
    ax.axhline(0.95, color="black", lw=1.2, ls="--", label="Nominal target=0.95")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("One-sided empirical coverage: CKME vs QR")
    ax.set_ylim(0.7, 1.02)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    else:
        plt.show()
    plt.close()


def plot_sup_error(sup_error_path: str, save_path: str | None = None):
    """Boxplot of sup_x |q_hat - q_true| by simulator/tau/method."""
    if not os.path.exists(sup_error_path):
        print(f"Sup-error summary not found: {sup_error_path}")
        return

    df = pd.read_csv(sup_error_path)
    if df.empty:
        print("Sup-error summary is empty.")
        return

    fig, axes = plt.subplots(1, len(TAUS), figsize=(12, 4.5), sharey=True)
    if len(TAUS) == 1:
        axes = [axes]

    colors = {"CKME": "steelblue", "QR": "tomato"}
    methods = ["CKME", "QR"]

    for ax, tau in zip(axes, TAUS):
        data = df[df["tau"] == tau]
        labels = []
        positions = []
        values = []
        pos = 1
        for sim in SIMULATORS:
            for method in methods:
                v = data[
                    (data["simulator"] == sim) & (data["method"] == method)
                ]["sup_abs_quantile_error"].dropna().values
                if len(v) == 0:
                    continue
                values.append(v)
                positions.append(pos)
                labels.append(f"{sim}\n{method}")
                pos += 1
            pos += 1  # blank gap between simulators

        bp = ax.boxplot(values, positions=positions, widths=0.7, patch_artist=True)
        for patch, label in zip(bp["boxes"], labels):
            color = colors["CKME"] if "CKME" in label else colors["QR"]
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"sup_x |q_hat - q_true|, tau={tau:.2f}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel("Sup absolute quantile error")

    fig.suptitle("One-sided quantile sup error across macroreps", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    plt.close()


def plot_mean_error(mean_error_path: str, save_path: str | None = None):
    """Boxplot of mean_x |q_hat - q_true| by simulator/tau/method."""
    if not os.path.exists(mean_error_path):
        print(f"Mean-error summary not found: {mean_error_path}")
        return

    df = pd.read_csv(mean_error_path)
    if df.empty:
        print("Mean-error summary is empty.")
        return

    fig, axes = plt.subplots(1, len(TAUS), figsize=(12, 4.5), sharey=True)
    if len(TAUS) == 1:
        axes = [axes]

    colors = {"CKME": "steelblue", "QR": "tomato"}
    methods = ["CKME", "QR"]

    for ax, tau in zip(axes, TAUS):
        data = df[df["tau"] == tau]
        labels = []
        positions = []
        values = []
        pos = 1
        for sim in SIMULATORS:
            for method in methods:
                v = data[
                    (data["simulator"] == sim) & (data["method"] == method)
                ]["mean_abs_quantile_error"].dropna().values
                if len(v) == 0:
                    continue
                values.append(v)
                positions.append(pos)
                labels.append(f"{sim}\n{method}")
                pos += 1
            pos += 1  # blank gap between simulators

        bp = ax.boxplot(values, positions=positions, widths=0.7, patch_artist=True)
        for patch, label in zip(bp["boxes"], labels):
            color = colors["CKME"] if "CKME" in label else colors["QR"]
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"mean_x |q_hat - q_true|, tau={tau:.2f}")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel("Mean absolute quantile error")

    fig.suptitle("One-sided quantile mean error across macroreps", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    plt.close()


def plot_tailprob(data: dict, save_path: str | None = None):
    """Plot true vs CKME tail probability at true quantile on one macrorep."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex="row")
    fig.suptitle("Tail probability at true quantile: CKME vs true", fontsize=13)

    tau_cols = {0.05: 0, 0.95: 1}
    for row, sim in enumerate(SIMULATORS):
        for tau, col in tau_cols.items():
            ax = axes[row, col]
            key = (sim, tau)
            if key not in data:
                ax.set_visible(False)
                continue
            df = data[key]
            if "tail_prob_ckme_at_qtrue" not in df.columns:
                ax.text(0.5, 0.5, "tail_prob columns missing", ha="center", va="center")
                continue
            x = df["x"].values
            ax.plot(x, df["tail_prob_true_at_qtrue"].values, color="black", lw=1.6,
                    label="True 1−F(q_true|x)")
            ax.plot(x, df["tail_prob_ckme_at_qtrue"].values, color="steelblue", lw=1.8,
                    ls="--", label="CKME 1−F̂(q_true|x)")
            ax.axhline(1.0 - tau, color="gray", lw=1.0, ls=":", label=f"Nominal {1-tau:.2f}")
            ax.set_title(f"{SIM_LABELS[sim]}  τ={tau}", fontsize=8.5)
            ax.set_xlabel("x")
            ax.set_ylabel("tail probability")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    plt.close()


def _filter_t_values(df: pd.DataFrame, t_values: list[float] | None) -> pd.DataFrame:
    """Filter rows by selected t values with tolerance."""
    if not t_values:
        return df
    t_arr = np.asarray(t_values, dtype=float)
    mask = np.zeros(len(df), dtype=bool)
    for tv in t_arr:
        mask |= np.isclose(df["t_value"].values, tv, atol=1e-10, rtol=1e-8)
    return df[mask].copy()


def plot_tailprob_fixed_t(
    curve_path: str,
    macrorep: int,
    save_path: str | None = None,
    t_values: list[float] | None = None,
):
    """Plot P(Y>t|X=x) curves for fixed t values on one macrorep."""
    if not os.path.exists(curve_path):
        print(f"Tail-prob curve data not found: {curve_path}")
        return

    df = pd.read_csv(curve_path)
    df = df[df["macrorep"] == macrorep].copy()
    df = _filter_t_values(df, t_values)
    if df.empty:
        print(f"No tail-prob curve rows for macrorep_{macrorep}")
        return

    fig, axes = plt.subplots(len(SIMULATORS), 1, figsize=(10, 9), sharex=False)
    if len(SIMULATORS) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")

    for ax, sim in zip(axes, SIMULATORS):
        dsim = df[df["simulator"] == sim].copy()
        if dsim.empty:
            ax.set_visible(False)
            continue
        t_vals = sorted(dsim["t_value"].unique())
        for i, t_val in enumerate(t_vals):
            d = dsim[dsim["t_value"] == t_val].sort_values("x")
            color = cmap(i % 10)
            ax.plot(d["x"].values, d["tail_prob_true"].values, color=color, lw=1.6,
                    label=f"True t={t_val:.3g}")
            ax.plot(d["x"].values, d["tail_prob_ckme"].values, color=color, lw=1.6,
                    ls="--", label=f"CKME t={t_val:.3g}")

        ax.set_title(SIM_LABELS.get(sim, sim))
        ax.set_xlabel("x")
        ax.set_ylabel("P(Y > t | X=x)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle(f"Fixed-threshold tail probability curves (macrorep={macrorep})", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    plt.close()


def plot_tailprob_fixed_t_agg(
    curve_path: str,
    save_path: str | None = None,
    max_box_per_curve: int = 28,
    t_values: list[float] | None = None,
):
    """
    Aggregate fixed-threshold tail probability curves over macroreps.

    For each (simulator, t, x), we plot:
      - CKME small boxplots across macroreps at sampled x locations
      - True median curve
    """
    if not os.path.exists(curve_path):
        print(f"Tail-prob curve data not found: {curve_path}")
        return

    df = pd.read_csv(curve_path)
    df = _filter_t_values(df, t_values)
    if df.empty:
        print("Tail-prob curve data is empty.")
        return

    fig, axes = plt.subplots(len(SIMULATORS), 1, figsize=(10, 9), sharex=False)
    if len(SIMULATORS) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")

    for ax, sim in zip(axes, SIMULATORS):
        dsim = df[df["simulator"] == sim].copy()
        if dsim.empty:
            ax.set_visible(False)
            continue

        t_vals = sorted(dsim["t_value"].unique())
        for i, t_val in enumerate(t_vals):
            d = dsim[dsim["t_value"] == t_val].copy()
            # Aggregate true line at each x
            agg_true = d.groupby("x", as_index=False).agg(
                true_median=("tail_prob_true", "median"),
            ).sort_values("x")
            x = agg_true["x"].values
            color = cmap(i % 10)

            # True curve (median over macroreps)
            ax.plot(x, agg_true["true_median"].values, color=color, lw=1.8)

            # CKME small boxplots over x (sampled to avoid overplotting)
            unique_x = np.sort(d["x"].unique())
            if len(unique_x) == 0:
                continue
            step = max(1, int(np.ceil(len(unique_x) / max_box_per_curve)))
            x_sample = unique_x[::step]

            x_span = max(float(unique_x[-1] - unique_x[0]), 1e-8)
            width = 0.018 * x_span
            box_values = []
            box_positions = []
            for xi in x_sample:
                v = d.loc[np.isclose(d["x"].values, xi), "tail_prob_ckme"].values
                if len(v) > 0:
                    box_values.append(v)
                    box_positions.append(float(xi))

            if box_values:
                bp = ax.boxplot(
                    box_values,
                    positions=box_positions,
                    widths=width,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                )
                for patch in bp["boxes"]:
                    patch.set_facecolor(color)
                    patch.set_alpha(0.22)
                    patch.set_edgecolor(color)
                for whisk in bp["whiskers"]:
                    whisk.set_color(color)
                    whisk.set_alpha(0.35)
                for cap in bp["caps"]:
                    cap.set_color(color)
                    cap.set_alpha(0.35)
                for median in bp["medians"]:
                    median.set_color(color)
                    median.set_linewidth(1.2)
        ax.set_title(SIM_LABELS.get(sim, sim))
        ax.set_xlabel("x")
        ax.set_ylabel("P(Y > t | X=x)")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.2)

    fig.suptitle("Fixed-threshold tail probability: true line + CKME x-boxplots", fontsize=12)
    legend_handles = [
        Line2D([0], [0], color="black", lw=1.8, label="True line"),
        Patch(facecolor="steelblue", edgecolor="steelblue", alpha=0.22, label="CKME boxplot"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=[
            "bounds",
            "coverage",
            "sup_error",
            "mean_error",
            "tailprob",
            "tailprob_t",
            "tailprob_t_agg",
        ],
        default="bounds",
    )
    parser.add_argument("--macrorep", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        default=os.path.join(os.path.dirname(__file__), "output"),
    )
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save figure. If omitted, display interactively.")
    parser.add_argument(
        "--plot_t_values",
        type=str,
        default=None,
        help=(
            "Comma-separated t values to plot only in tailprob_t / tailprob_t_agg. "
            "Example: --plot_t_values 0.0,1.0"
        ),
    )
    args = parser.parse_args()
    selected_t = (
        [float(v.strip()) for v in args.plot_t_values.split(",")]
        if args.plot_t_values else None
    )

    if args.mode == "bounds":
        data = load_perpoint(args.output_dir, args.macrorep)
        if not data:
            print(f"No per-point data found in {args.output_dir}/macrorep_{args.macrorep}/")
            return
        save = args.save or os.path.join(args.output_dir, "onesided_bounds.png")
        plot_bounds(data, save_path=save)

    elif args.mode == "coverage":
        summary_path = os.path.join(args.output_dir, "onesided_summary.csv")
        save = args.save or os.path.join(args.output_dir, "onesided_coverage.png")
        plot_coverage(summary_path, save_path=save)
    elif args.mode == "sup_error":
        sup_error_path = os.path.join(args.output_dir, "onesided_quantile_sup_error.csv")
        save = args.save or os.path.join(args.output_dir, "onesided_sup_error_boxplot.png")
        plot_sup_error(sup_error_path, save_path=save)
    elif args.mode == "mean_error":
        mean_error_path = os.path.join(args.output_dir, "onesided_quantile_sup_error.csv")
        save = args.save or os.path.join(args.output_dir, "onesided_mean_error_boxplot.png")
        plot_mean_error(mean_error_path, save_path=save)
    elif args.mode == "tailprob":
        data = load_perpoint(args.output_dir, args.macrorep)
        if not data:
            print(f"No per-point data found in {args.output_dir}/macrorep_{args.macrorep}/")
            return
        save = args.save or os.path.join(args.output_dir, "onesided_tailprob.png")
        plot_tailprob(data, save_path=save)
    elif args.mode == "tailprob_t":
        curve_path = os.path.join(args.output_dir, "onesided_tailprob_curve.csv")
        save = args.save or os.path.join(args.output_dir, "onesided_tailprob_fixed_t.png")
        plot_tailprob_fixed_t(curve_path, args.macrorep, save_path=save, t_values=selected_t)
    elif args.mode == "tailprob_t_agg":
        curve_path = os.path.join(args.output_dir, "onesided_tailprob_curve.csv")
        save = args.save or os.path.join(args.output_dir, "onesided_tailprob_fixed_t_agg.png")
        plot_tailprob_fixed_t_agg(curve_path, save_path=save, t_values=selected_t)


if __name__ == "__main__":
    main()
