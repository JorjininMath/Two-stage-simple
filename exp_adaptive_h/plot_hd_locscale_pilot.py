"""Plot diagnostics for the high-dimensional location-scale pilot."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/or_figure_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/or_figure_cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARM_ORDER = ["fixed", "plugin", "oracle"]
ARM_LABEL = {
    "fixed": "Fixed h",
    "plugin": "Plugin h(x)",
    "oracle": "Oracle h(x)",
}
ARM_COLOR = {
    "fixed": "#4C78A8",
    "plugin": "#F58518",
    "oracle": "#54A24B",
}


def _set_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["figure.dpi"] = 150


def _export(fig, outdir: Path, name: str) -> list[Path]:
    paths = []
    for ext in ["svg", "pdf", "png", "tiff"]:
        path = outdir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _arm_from_path(path: Path, simulator: str) -> str:
    prefix = f"case_{simulator}_"
    name = path.parent.name
    if not name.startswith(prefix):
        raise ValueError(f"Unexpected case directory name: {name}")
    return name[len(prefix):]


def load_per_point(root: Path, simulator: str, budget: int) -> pd.DataFrame:
    rows = []
    pattern = f"macrorep_*/budget_{budget}/case_{simulator}_*/per_point.csv"
    for path in sorted(root.glob(pattern)):
        macrorep = int(path.parts[-4].split("_")[1])
        arm = _arm_from_path(path, simulator)
        df = pd.read_csv(path)
        df["macrorep"] = macrorep
        df["arm"] = arm
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No per_point.csv files found under {root} with pattern {pattern}")
    return pd.concat(rows, ignore_index=True)


def plot_summary(
    summary: pd.DataFrame,
    outdir: Path,
    target: float,
    simulator: str,
    budget: int,
) -> list[Path]:
    metrics = [
        ("mean_coverage", "sd_coverage", "Coverage", True),
        ("mean_width", "sd_width", "Mean width", False),
        ("mean_interval_score", "sd_interval_score", "Interval score", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
    for ax, (mean_col, sd_col, ylabel, is_cov) in zip(axes, metrics):
        vals = [float(summary.loc[summary["arm"] == arm, mean_col].iloc[0]) for arm in ARM_ORDER]
        errs = [float(summary.loc[summary["arm"] == arm, sd_col].iloc[0]) for arm in ARM_ORDER]
        x = np.arange(len(ARM_ORDER))
        ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=3,
            color=[ARM_COLOR[a] for a in ARM_ORDER],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER], rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        if is_cov:
            ax.axhline(target, color="black", linestyle="--", linewidth=1.0)
            ax.set_ylim(min(0.84, min(vals) - 0.03), max(0.96, max(vals) + 0.03))
        else:
            ax.set_ylim(0, max(vals) * 1.25)
    n_macro = int(summary["n_macroreps"].iloc[0])
    fig.suptitle(f"{simulator} pilot, B={budget}, {n_macro} macroreps", y=1.04, fontsize=11)
    fig.tight_layout()
    return _export(fig, outdir, "hd_locscale_d5_summary_metrics")


def _add_bins(per_point: pd.DataFrame, n_bins: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed_x = per_point.loc[per_point["arm"] == "fixed", "x0"].to_numpy()
    _, edges = pd.qcut(fixed_x, q=n_bins, retbins=True, duplicates="drop")
    edges[0] = -np.inf
    edges[-1] = np.inf
    out = per_point.copy()
    out["bin"] = pd.cut(out["x0"], bins=edges, labels=False, include_lowest=True)
    mids = (
        out.groupby("bin", observed=True)
        .agg(x_mid=("x0", "mean"), x_lo=("x0", "min"), x_hi=("x0", "max"))
        .reset_index()
    )
    return out, mids


def plot_bin_coverage(per_point: pd.DataFrame, outdir: Path, target: float, n_bins: int) -> list[Path]:
    df, mids = _add_bins(per_point, n_bins)
    by_rep = (
        df.groupby(["macrorep", "arm", "bin"], observed=True)
        .agg(coverage=("covered_score", "mean"))
        .reset_index()
    )
    agg = (
        by_rep.groupby(["arm", "bin"], observed=True)
        .agg(mean_coverage=("coverage", "mean"), sd_coverage=("coverage", "std"))
        .reset_index()
        .merge(mids, on="bin", how="left")
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for arm in ARM_ORDER:
        g = agg[agg["arm"] == arm].sort_values("x_mid")
        ax.plot(g["x_mid"], g["mean_coverage"], marker="o", linewidth=1.8,
                color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        ax.fill_between(
            g["x_mid"].to_numpy(),
            (g["mean_coverage"] - g["sd_coverage"].fillna(0)).to_numpy(),
            (g["mean_coverage"] + g["sd_coverage"].fillna(0)).to_numpy(),
            color=ARM_COLOR[arm],
            alpha=0.15,
            linewidth=0,
        )
    ax.axhline(target, color="black", linestyle="--", linewidth=1.0, label="Target")
    ax.set_xlabel("x0 bin midpoint")
    ax.set_ylabel("Score coverage")
    ax.set_ylim(0.75, 1.0)
    ax.legend(ncol=2, loc="lower right")
    ax.set_title("Binwise coverage by informative coordinate")
    fig.tight_layout()
    return _export(fig, outdir, "hd_locscale_d5_binwise_coverage")


def plot_bin_width(per_point: pd.DataFrame, outdir: Path, n_bins: int) -> list[Path]:
    df, mids = _add_bins(per_point, n_bins)
    by_rep = (
        df.groupby(["macrorep", "arm", "bin"], observed=True)
        .agg(width=("width", "mean"))
        .reset_index()
    )
    agg = (
        by_rep.groupby(["arm", "bin"], observed=True)
        .agg(mean_width=("width", "mean"), sd_width=("width", "std"))
        .reset_index()
        .merge(mids, on="bin", how="left")
    )
    by_rep.merge(mids, on="bin", how="left").to_csv(
        outdir / "hd_locscale_d5_binwise_width_by_macrorep.csv",
        index=False,
    )
    agg.to_csv(outdir / "hd_locscale_d5_binwise_width.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for arm in ARM_ORDER:
        g = agg[agg["arm"] == arm].sort_values("x_mid")
        ax.plot(g["x_mid"], g["mean_width"], marker="o", linewidth=1.8,
                color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        ax.fill_between(
            g["x_mid"].to_numpy(),
            (g["mean_width"] - g["sd_width"].fillna(0)).to_numpy(),
            (g["mean_width"] + g["sd_width"].fillna(0)).to_numpy(),
            color=ARM_COLOR[arm],
            alpha=0.15,
            linewidth=0,
        )
    ax.set_xlabel("x0 bin midpoint")
    ax.set_ylabel("Mean interval width")
    ax.set_title("Binwise interval width by informative coordinate")
    ax.legend(ncol=3, loc="upper center")
    fig.tight_layout()
    return _export(fig, outdir, "hd_locscale_d5_binwise_width")


def plot_bandwidth(per_point: pd.DataFrame, outdir: Path, n_bins: int) -> list[Path]:
    df, mids = _add_bins(per_point, n_bins)
    by_rep = (
        df.groupby(["macrorep", "arm", "bin"], observed=True)
        .agg(mean_h=("h_query", "mean"))
        .reset_index()
    )
    agg = (
        by_rep.groupby(["arm", "bin"], observed=True)
        .agg(mean_h=("mean_h", "mean"), sd_h=("mean_h", "std"))
        .reset_index()
        .merge(mids, on="bin", how="left")
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for arm in ARM_ORDER:
        g = agg[agg["arm"] == arm].sort_values("x_mid")
        ax.plot(g["x_mid"], g["mean_h"], marker="o", linewidth=1.8,
                color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.set_xlabel("x0 bin midpoint")
    ax.set_ylabel("Mean h query")
    ax.set_title("Bandwidth used by each arm")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.36))
    fig.tight_layout()
    return _export(fig, outdir, "hd_locscale_d5_bandwidth_by_x0")


def plot_clipping(per_point: pd.DataFrame, outdir: Path) -> list[Path]:
    by_rep = (
        per_point.groupby(["macrorep", "arm"], observed=True)
        .agg(
            y_out_grid=("y_in_grid", lambda s: 1.0 - float(np.mean(s))),
            L_at_lo=("L_at_grid_lo", "mean"),
            U_at_hi=("U_at_grid_hi", "mean"),
        )
        .reset_index()
    )
    agg = by_rep.groupby("arm", observed=True).mean(numeric_only=True).reset_index()
    metrics = ["y_out_grid", "L_at_lo", "U_at_hi"]
    labels = ["Y outside grid", "L at grid low", "U at grid high"]
    x = np.arange(len(metrics))
    width = 0.24
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, arm in enumerate(ARM_ORDER):
        vals = [float(agg.loc[agg["arm"] == arm, m].iloc[0]) for m in metrics]
        ax.bar(x + (i - 1) * width, vals, width=width, color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, max(0.1, float(agg[metrics].to_numpy().max()) * 1.25))
    ax.set_title("Grid clipping diagnostics")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    return _export(fig, outdir, "hd_locscale_d5_grid_clipping")


def write_qa(
    outdir: Path,
    root: Path,
    summary: pd.DataFrame,
    per_point: pd.DataFrame,
    exports: list[Path],
    target: float,
    n_bins: int,
    simulator: str,
    budget: int,
) -> None:
    clip = (
        per_point.groupby("arm", observed=True)
        .agg(
            y_out_grid=("y_in_grid", lambda s: 1.0 - float(np.mean(s))),
            L_at_lo=("L_at_grid_lo", "mean"),
            U_at_hi=("U_at_grid_hi", "mean"),
        )
        .reset_index()
    )
    lines = [
        "# High-D Pilot Figure QA",
        "",
        f"Input root: `{root}`",
        "Input files:",
        "- `exp4_summary.csv`",
        "- `exp4_per_arm.csv`",
        f"- `macrorep_*/budget_{budget}/case_{simulator}_*/per_point.csv`",
        "",
        f"Summary rows: {len(summary)}",
        f"Per-point rows: {len(per_point)}",
        f"Target coverage: {target}",
        f"Coverage bins: {n_bins}",
        "Statistical unit: macroreplication; shaded bands use macroreplication SD.",
        "",
        "Exports:",
    ]
    lines.extend(f"- `{p.name}`" for p in exports)
    lines.extend([
        "",
        "Binwise width outputs:",
        "- `hd_locscale_d5_binwise_width.csv`",
        "- `hd_locscale_d5_binwise_width_by_macrorep.csv`",
        "",
        "Clipping diagnostics by arm:",
        "",
        clip.to_markdown(index=False),
        "",
        "Warnings:",
        "- This is a 5-macrorep pilot, not a manuscript-strength result.",
        "- Check the matching pretrained-parameter JSON before treating this as a final fixed-h baseline.",
        "- Nonzero `U_at_hi` means upper interval clipping remains a diagnostic risk.",
        "",
    ])
    (outdir / "hd_locscale_d5_figures_qa.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="exp_adaptive_h/output_hd_locscale_d5_pilot_b250")
    parser.add_argument("--simulator", type=str, default="hd_locscale_d5")
    parser.add_argument("--budget", type=int, default=250)
    parser.add_argument("--target", type=float, default=0.9)
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    _set_style()
    root = Path(args.output_dir)
    outdir = root / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(root / "exp4_summary.csv")
    summary = summary[(summary["simulator"] == args.simulator) & (summary["budget"] == args.budget)]
    if summary.empty:
        raise ValueError(f"No summary rows found for {args.simulator}, budget={args.budget}")
    per_point = load_per_point(root, args.simulator, args.budget)

    exports: list[Path] = []
    exports.extend(plot_summary(summary, outdir, args.target, args.simulator, args.budget))
    exports.extend(plot_bin_coverage(per_point, outdir, args.target, args.n_bins))
    exports.extend(plot_bin_width(per_point, outdir, args.n_bins))
    exports.extend(plot_bandwidth(per_point, outdir, args.n_bins))
    exports.extend(plot_clipping(per_point, outdir))
    write_qa(
        outdir,
        root,
        summary,
        per_point,
        exports,
        args.target,
        args.n_bins,
        args.simulator,
        args.budget,
    )
    print(f"Wrote figures to {outdir}")


if __name__ == "__main__":
    main()
