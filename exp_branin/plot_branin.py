"""
plot_branin.py

Visualize exp_branin results: CKME vs DCP-DR vs hetGP on 2D Branin-Hoo.

Plot types (--plot_type):
  boxplot  (default): per-simulator boxplot figure (coverage / width / IS across macroreps).
                      Produces two separate figures: one for branin_gauss, one for branin_student.
                      Use --save as a path prefix: e.g. --save exp_branin/output/branin
                      → saves branin_gauss.png and branin_student.png.
  noise             : Oracle noise characteristics — sigma(x1) and nu(x1) vs x1.
  summary           : Bar chart from branin_compare_summary.csv.

A detailed summary table is always written to <output_dir>/branin_detailed_summary.csv
when running --plot_type boxplot.

Usage (from project root):
    python exp_branin/plot_branin.py --save exp_branin/output/branin
    python exp_branin/plot_branin.py --plot_type noise --save exp_branin/output/noise.png
    python exp_branin/plot_branin.py --plot_type summary --save exp_branin/output/summary.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

SIMULATORS = ["branin_gauss", "branin_student"]
SIM_LABELS = {
    "branin_gauss":   r"Branin-Gauss  ($\epsilon\sim\mathcal{N}$)",
    "branin_student": r"Branin-Student  ($\epsilon\sim t_{\nu(x)}$)",
}

METHODS_ORDER = ["CKME", "DCP-DR", "hetGP"]
COL_MAP = {
    "CKME":   {"cov": "covered_interval",       "width": "width",       "is": "interval_score"},
    "DCP-DR": {"cov": "covered_interval_dr",    "width": "width_dr",    "is": "interval_score_dr"},
    "hetGP":  {"cov": "covered_interval_hetgp", "width": "width_hetgp", "is": "interval_score_hetgp"},
}
# Score-based coverage fallback for CKME
SCORE_COV = {
    "CKME":   "covered_score",
    "DCP-DR": "covered_score_dr",
    "hetGP":  "covered_interval_hetgp",
}

METHOD_COLORS = {"CKME": "#2166ac", "DCP-DR": "#d6604d", "hetGP": "#4dac26"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_point(output_dir: Path, sim: str, site_method: str) -> list[pd.DataFrame]:
    paths = sorted(output_dir.glob(f"macrorep_*/case_{sim}_{site_method}/per_point.csv"))
    if not paths:
        print(f"  Warning: no per_point.csv for {sim}/{site_method}", file=sys.stderr)
        return []
    return [pd.read_csv(p) for p in paths]


def _resolve_cov_col(mname: str, df: pd.DataFrame) -> str:
    """Return coverage column for method, falling back to score-based."""
    col = COL_MAP[mname]["cov"]
    if col not in df.columns:
        col = SCORE_COV.get(mname, col)
    return col


def collect_macrorep_means(dfs: list[pd.DataFrame]) -> dict[str, dict[str, list[float]]]:
    """
    For each macrorep and each method, compute mean coverage / width / IS
    over all test points in that macrorep.

    Returns: {method: {"cov": [...], "width": [...], "is": [...]}}
    """
    results: dict[str, dict[str, list]] = {m: {"cov": [], "width": [], "is": []} for m in METHODS_ORDER}
    for df in dfs:
        for mname in METHODS_ORDER:
            cov_col   = _resolve_cov_col(mname, df)
            width_col = COL_MAP[mname]["width"]
            is_col    = COL_MAP[mname]["is"]
            results[mname]["cov"].append(df[cov_col].mean()   if cov_col   in df.columns else np.nan)
            results[mname]["width"].append(df[width_col].mean() if width_col in df.columns else np.nan)
            results[mname]["is"].append(df[is_col].mean()     if is_col    in df.columns else np.nan)
    return results


# ---------------------------------------------------------------------------
# Summary table (CSV + figure)
# ---------------------------------------------------------------------------

def build_summary_table(out_dir: Path, site_method: str) -> pd.DataFrame:
    """
    Aggregate macrorep-level means into a summary table:
    simulator × method × {mean, sd} for coverage / width / IS.
    Saved to <out_dir>/branin_detailed_summary.csv.
    """
    rows = []
    for sim in SIMULATORS:
        dfs = load_per_point(out_dir, sim, site_method)
        if not dfs:
            continue
        data = collect_macrorep_means(dfs)
        for mname in METHODS_ORDER:
            for metric, key in [("coverage", "cov"), ("width", "width"), ("interval_score", "is")]:
                vals = np.array(data[mname][key], dtype=float)
                vals = vals[~np.isnan(vals)]
                rows.append({
                    "simulator":   sim,
                    "method":      mname,
                    "metric":      metric,
                    "n_macroreps": len(vals),
                    "mean":        np.mean(vals)           if len(vals) else np.nan,
                    "sd":          np.std(vals, ddof=1)    if len(vals) > 1 else np.nan,
                    "median":      np.median(vals)         if len(vals) else np.nan,
                    "q25":         np.percentile(vals, 25) if len(vals) else np.nan,
                    "q75":         np.percentile(vals, 75) if len(vals) else np.nan,
                })
    df = pd.DataFrame(rows)
    out_path = out_dir / "branin_detailed_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved summary table to {out_path}")
    return df


def plot_table_figure(df: pd.DataFrame, alpha: float, save: str | None):
    """
    Render the summary table as a matplotlib figure.

    Layout: rows = simulator × metric combinations,
            cols = Method | Coverage mean±SD | Width mean±SD | IS mean±SD
    One row group per simulator, separated by a blank row.
    """
    METRICS = [("coverage", "Coverage", f"target {1-alpha:.0%}"),
               ("width",    "Width",    ""),
               ("interval_score", "Interval Score", "")]

    col_headers = ["Simulator", "Method", "Coverage\nmean ± SD", "Width\nmean ± SD",
                   "Interval Score\nmean ± SD"]

    table_rows = []
    for sim_idx, sim in enumerate(SIMULATORS):
        sub = df[df["simulator"] == sim]
        if sub.empty:
            continue
        for m_idx, mname in enumerate(METHODS_ORDER):
            row_data = [SIM_LABELS[sim] if m_idx == 0 else "", mname]
            for metric, _, _ in METRICS:
                cell = sub[(sub["method"] == mname) & (sub["metric"] == metric)]
                if cell.empty:
                    row_data.append("—")
                else:
                    mean = cell["mean"].values[0]
                    sd   = cell["sd"].values[0]
                    if np.isnan(mean):
                        row_data.append("—")
                    elif np.isnan(sd):
                        row_data.append(f"{mean:.3f}")
                    else:
                        row_data.append(f"{mean:.3f} ± {sd:.3f}")
            table_rows.append(row_data)
        # blank separator between simulators
        if sim_idx < len(SIMULATORS) - 1:
            table_rows.append([""] * len(col_headers))

    n_rows = len(table_rows)
    fig, ax = plt.subplots(figsize=(11, 0.45 * n_rows + 1.5))
    ax.axis("off")
    fig.suptitle("Branin-Hoo: Summary Table (mean ± SD across macroreps)", fontsize=12, y=0.98)

    tbl = ax.table(
        cellText=table_rows,
        colLabels=col_headers,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width(list(range(len(col_headers))))

    # Style header row
    for j in range(len(col_headers)):
        tbl[(0, j)].set_facecolor("#404040")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Color-code method cells and alternating simulator blocks
    sim_colors = {"branin_gauss": "#dceeff", "branin_student": "#fff3dc"}
    row_idx = 1
    for sim_idx, sim in enumerate(SIMULATORS):
        sub = df[df["simulator"] == sim]
        if sub.empty:
            continue
        n_method_rows = len(METHODS_ORDER)
        bg = sim_colors.get(sim, "#ffffff")
        for i in range(n_method_rows):
            for j in range(len(col_headers)):
                tbl[(row_idx, j)].set_facecolor(bg)
            # Bold the method name cell
            tbl[(row_idx, 1)].set_text_props(
                color=METHOD_COLORS.get(METHODS_ORDER[i], "black"), fontweight="bold"
            )
            row_idx += 1
        row_idx += 1  # blank separator row

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Boxplot (one figure per simulator)
# ---------------------------------------------------------------------------

def plot_boxplot_one(sim: str, dfs: list[pd.DataFrame], alpha: float,
                     save: str | None):
    """3-row boxplot figure for one simulator: coverage / width / IS."""
    if not dfs:
        print(f"  No data for {sim}, skipping.", file=sys.stderr)
        return

    data = collect_macrorep_means(dfs)
    n_macro = len(dfs)

    fig, axes = plt.subplots(3, 1, figsize=(6, 10))
    fig.suptitle(f"{SIM_LABELS[sim]}\n({n_macro} macroreps)", fontsize=12)

    metrics = [
        ("cov",   "Coverage",       1 - alpha),
        ("width", "Width",          None),
        ("is",    "Interval Score", None),
    ]

    for ax, (key, ylabel, ref) in zip(axes, metrics):
        boxes_data = [data[m][key] for m in METHODS_ORDER]
        bp = ax.boxplot(
            boxes_data,
            tick_labels=METHODS_ORDER,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 2},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
            flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
        )
        for patch, mname in zip(bp["boxes"], METHODS_ORDER):
            patch.set_facecolor(METHOD_COLORS[mname])
            patch.set_alpha(0.7)

        if ref is not None:
            ax.axhline(ref, color="black", linestyle=":", linewidth=1.2,
                       label=f"target ({ref:.0%})")
            ax.legend(fontsize=9)
            ax.set_ylim(0, 1.05)

        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if save:
        suffix = "_gauss.png" if "gauss" in sim else "_student.png"
        save_path = Path(save).parent / (Path(save).name + suffix)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_boxplots(out_dir: Path, alpha: float, site_method: str, save: str | None):
    """Produce one boxplot figure per simulator and write summary table."""
    build_summary_table(out_dir, site_method)
    for sim in SIMULATORS:
        dfs = load_per_point(out_dir, sim, site_method)
        plot_boxplot_one(sim, dfs, alpha, save)


def plot_table(out_dir: Path, alpha: float, site_method: str, save: str | None):
    """Build summary table and render it as a figure."""
    df = build_summary_table(out_dir, site_method)
    plot_table_figure(df, alpha, save)


# ---------------------------------------------------------------------------
# Noise oracle plot
# ---------------------------------------------------------------------------

def plot_noise(save: str | None):
    x1 = np.linspace(-5, 10, 300)
    x1_scaled = (x1 - (-5)) / 15.0
    sigma = 0.4 * (4.0 * x1_scaled + 1.0)
    nu    = np.maximum(2.0, 6.0 - 4.0 * x1_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Branin-Hoo: Oracle noise characteristics", fontsize=12)

    axes[0].plot(x1, sigma, color="#2166ac", linewidth=2)
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$\sigma(x_1)$")
    axes[0].set_title(r"$\sigma(x_1)=0.4\,(4\,x_{1,\mathrm{sc}}+1)$  [both simulators]")
    axes[0].set_xlim(-5, 10)
    axes[0].axhline(0.4, color="gray", linestyle=":", linewidth=1, label=r"$\sigma=0.4$ at $x_1=-5$")
    axes[0].axhline(2.0, color="gray", linestyle="--", linewidth=1, label=r"$\sigma=2.0$ at $x_1=10$")
    axes[0].legend(fontsize=9)

    axes[1].plot(x1, nu, color="#d6604d", linewidth=2)
    axes[1].set_xlabel(r"$x_1$")
    axes[1].set_ylabel(r"$\nu(x_1)$")
    axes[1].set_title(r"$\nu(x_1)=\max(2,\;6-4\,x_{1,\mathrm{sc}})$  [branin_student only]")
    axes[1].set_xlim(-5, 10)
    axes[1].axhline(2.0, color="black", linestyle=":", linewidth=1.2,
                    label=r"$\nu=2$ (infinite variance)")
    axes[1].axhline(6.0, color="gray", linestyle="--", linewidth=1, label=r"$\nu=6$ at $x_1=-5$")
    axes[1].set_ylim(1.5, 7)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def plot_summary(out_dir: Path, alpha: float, save: str | None):
    summary_path = out_dir / "branin_compare_summary.csv"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found. Run run_branin_compare.py first.", file=sys.stderr)
        return

    df = pd.read_csv(summary_path)
    x = np.arange(len(METHODS_ORDER))
    bar_width = 0.5

    fig, axes = plt.subplots(3, 2, figsize=(10, 10))
    fig.suptitle("Branin-Hoo: Summary (mean ± SD across macroreps)", fontsize=13)
    metrics = [
        ("mean_coverage",       "sd_coverage",       "Coverage",       1 - alpha),
        ("mean_width",          "sd_width",           "Width",          None),
        ("mean_interval_score", "sd_interval_score",  "Interval Score", None),
    ]

    for row_idx, (mean_col, sd_col, label, ref) in enumerate(metrics):
        for col_idx, sim in enumerate(SIMULATORS):
            ax = axes[row_idx, col_idx]
            sub = df[df["simulator"] == sim].set_index("method")
            means  = [sub.loc[m, mean_col] if m in sub.index else np.nan for m in METHODS_ORDER]
            sds    = [sub.loc[m, sd_col]   if m in sub.index else 0.0   for m in METHODS_ORDER]
            colors = [METHOD_COLORS[m] for m in METHODS_ORDER]
            ax.bar(x, means, bar_width, yerr=sds, color=colors, alpha=0.8,
                   capsize=5, error_kw={"linewidth": 1.4})
            ax.set_xticks(x)
            ax.set_xticklabels(METHODS_ORDER, fontsize=9)
            ax.set_ylabel(label)
            if ref is not None:
                ax.axhline(ref, color="black", linestyle=":", linewidth=1.2,
                           label=f"target ({ref:.0%})")
                ax.legend(fontsize=8)
            if row_idx == 0:
                ax.set_title(SIM_LABELS[sim], fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    _save_or_show(fig, save)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig, save: str | None):
    if save:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot exp_branin results")
    parser.add_argument("--output_dir",  type=str,   default=None)
    parser.add_argument("--plot_type",   type=str,   default="boxplot",
                        choices=("boxplot", "table", "noise", "summary"))
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--site_method", type=str,   default="lhs")
    parser.add_argument("--save",        type=str,   default=None,
                        help="For boxplot: path prefix (e.g. exp_branin/output/branin). "
                             "For noise/summary: full file path.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else _root / "exp_branin" / "output"

    if args.plot_type == "boxplot":
        plot_boxplots(out_dir, args.alpha, args.site_method, args.save)
    elif args.plot_type == "table":
        plot_table(out_dir, args.alpha, args.site_method, args.save)
    elif args.plot_type == "noise":
        plot_noise(args.save)
    elif args.plot_type == "summary":
        plot_summary(out_dir, args.alpha, args.save)


if __name__ == "__main__":
    main()
