"""Create OR-style figures for the NNPI repo-level reproduction.

Outputs:
- nnpi_example_data_overview.{svg,pdf,png,tiff}
- nnpi_repo_reproduction_results.{svg,pdf,png,tiff}
- nnpi_or_figure_qa.md
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

import numpy as np


def _configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "or_figure_mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "or_figure_cache"))
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["axes.grid"] = False
    return plt


def _read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_per_slice_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _export_bundle(fig, outdir: Path, stem: str) -> list[Path]:
    paths = []
    for ext, kwargs in [
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 220}),
        ("tiff", {"dpi": 300}),
    ]:
        path = outdir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        paths.append(path)
    return paths


def _site_summary(raw: np.ndarray):
    x = raw[:, 1:5]
    y_ms = raw[:, 5] * 1000.0
    unique_x, inv = np.unique(x, axis=0, return_inverse=True)
    mean = np.zeros(len(unique_x))
    q025 = np.zeros(len(unique_x))
    q975 = np.zeros(len(unique_x))
    n_rep = np.zeros(len(unique_x), dtype=int)
    for idx in range(len(unique_x)):
        vals = y_ms[inv == idx]
        n_rep[idx] = len(vals)
        mean[idx] = np.mean(vals)
        q025[idx] = np.quantile(vals, 0.025)
        q975[idx] = np.quantile(vals, 0.975)
    return x, y_ms, unique_x, n_rep, mean, q025, q975


def plot_overview(test_csv: Path, outdir: Path) -> tuple[list[Path], dict[str, float]]:
    plt = _configure_matplotlib()
    from matplotlib.patches import Circle

    raw = np.genfromtxt(test_csv, delimiter=",", skip_header=1)
    x, y_ms, unique_x, n_rep, site_mean, site_q025, site_q975 = _site_summary(raw)
    labels = ["message mean x", "lambda_12", "lambda_13", "lambda_14"]

    site_csv = outdir / "nnpi_example_site_summary.csv"
    np.savetxt(
        site_csv,
        np.column_stack([unique_x, n_rep, site_mean, site_q025, site_q975]),
        delimiter=",",
        header="message_mean_x,lambda_12,lambda_13,lambda_14,n_rep,mean_delay_ms,q025_delay_ms,q975_delay_ms",
        comments="",
    )

    rng = np.random.default_rng(20260629)
    sample_idx = rng.choice(len(y_ms), size=min(4500, len(y_ms)), replace=False)

    fig = plt.figure(figsize=(12, 8), dpi=170)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], wspace=0.26, hspace=0.35)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("A. Communication-network simulation")
    node_pos = {
        "Node 1": (0.08, 0.50),
        "Node 2": (0.50, 0.82),
        "Node 3": (0.50, 0.20),
        "Node 4": (0.92, 0.50),
    }
    edges = [
        ("Node 1", "Node 2", "Channel 1"),
        ("Node 2", "Node 3", "Channel 2"),
        ("Node 2", "Node 4", "Channel 3"),
        ("Node 3", "Node 4", "Channel 4"),
    ]
    for node_a, node_b, channel in edges:
        xa, ya = node_pos[node_a]
        xb, yb = node_pos[node_b]
        ax0.plot([xa, xb], [ya, yb], color="#555555", lw=1.2, zorder=1)
        ax0.text(
            (xa + xb) / 2,
            (ya + yb) / 2 + (0.045 if channel != "Channel 2" else 0),
            channel,
            ha="center",
            va="center",
            color="#333333",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
    for node, (x_coord, y_coord) in node_pos.items():
        ax0.add_patch(Circle((x_coord, y_coord), 0.095, facecolor="#EEF3F8", edgecolor="#2F3A45", lw=1.1, zorder=2))
        ax0.text(x_coord, y_coord, node, ha="center", va="center", fontsize=9, zorder=3)
    ax0.text(0.02, 0.02, "Output: average delay of first 30 messages", ha="left", va="bottom", color="#333333")
    ax0.set_xlim(-0.05, 1.05)
    ax0.set_ylim(0.02, 0.98)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("B. Released 4-input test design ranges")
    ypos = np.arange(len(labels))[::-1]
    mins = x.min(axis=0)
    maxs = x.max(axis=0)
    for yi, label, lo, hi in zip(ypos, labels, mins, maxs):
        ax1.plot([lo, hi], [yi, yi], color="#4C78A8", lw=8, solid_capstyle="round")
        ax1.scatter([lo, hi], [yi, yi], color="#1F3B57", s=24, zorder=3)
        ax1.text(lo, yi + 0.18, f"{lo:.1f}", ha="center", fontsize=8)
        ax1.text(hi, yi + 0.18, f"{hi:.1f}", ha="center", fontsize=8)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("input value")
    ax1.grid(axis="x", color="#E0E0E0", lw=0.6)
    ax1.set_ylim(-0.6, len(labels) - 0.25)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_title("C. Test output distribution")
    ax2.hist(y_ms, bins=55, color="#6E95BF", edgecolor="white", alpha=0.9)
    for q, text, color, ls in [(0.025, "2.5%", "#B94D4D", "--"), (0.5, "median", "#222222", "-"), (0.975, "97.5%", "#B94D4D", "--")]:
        val = np.quantile(y_ms, q)
        ax2.axvline(val, color=color, linestyle=ls, lw=1.4)
        ax2.text(val, ax2.get_ylim()[1] * 0.95, text, rotation=90, va="top", ha="right" if q < 0.5 else "left", color=color, fontsize=8)
    ax2.set_xlabel("delay Y (milliseconds)")
    ax2.set_ylabel("test samples")
    ax2.grid(axis="y", color="#E0E0E0", lw=0.6)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_title("D. Message length mean vs. delay")
    ax3.scatter(x[sample_idx, 0], y_ms[sample_idx], s=6, alpha=0.10, color="#4C78A8", linewidth=0, label="simulation replications")
    order = np.argsort(unique_x[:, 0])
    ux = unique_x[order, 0]
    ax3.vlines(ux, site_q025[order], site_q975[order], color="#D95F02", alpha=0.23, lw=1.0, label="site 95% range")
    ax3.scatter(ux, site_mean[order], s=20, color="#D95F02", edgecolor="white", linewidth=0.3, alpha=0.85, label="site mean")
    ax3.set_xlabel("message length mean x")
    ax3.set_ylabel("delay Y (milliseconds)")
    ax3.grid(color="#E0E0E0", lw=0.6)
    ax3.legend(loc="upper left", fontsize=8)

    fig.suptitle("NNPI released data: 4-input communication-network simulation example", y=0.985, fontsize=13)
    paths = _export_bundle(fig, outdir, "nnpi_example_data_overview")
    plt.close(fig)

    stats = {
        "n_test_rows": float(len(y_ms)),
        "n_sites": float(len(unique_x)),
        "replications_per_site_min": float(n_rep.min()),
        "replications_per_site_max": float(n_rep.max()),
        "delay_ms_mean": float(np.mean(y_ms)),
        "delay_ms_median": float(np.median(y_ms)),
        "delay_ms_q025": float(np.quantile(y_ms, 0.025)),
        "delay_ms_q975": float(np.quantile(y_ms, 0.975)),
    }
    return paths + [site_csv], stats


def plot_results(per_slice_csv: Path, summary_csv: Path, outdir: Path) -> tuple[list[Path], list[dict[str, str]]]:
    plt = _configure_matplotlib()
    rows = _read_per_slice_csv(per_slice_csv)
    summary = _read_summary_csv(summary_csv)
    methods = ["NNVA", "NNGN", "NNGU"]
    colors = {"NNVA": "#B279A2", "NNGN": "#F58518", "NNGU": "#E45756"}

    ep = {row["method"]: float(row["EP"]) for row in summary}
    iw = {row["method"]: float(row["IW"]) for row in summary}
    mean_cov = {row["method"]: float(row["mean_test_cov"]) for row in summary}
    widths = {
        method: np.asarray([float(row["selected_test_width"]) for row in rows if row["method"] == method], dtype=float)
        for method in methods
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=170)
    x = np.arange(len(methods))

    ax0 = axes[0]
    ax0.bar(x, [ep[m] for m in methods], color=[colors[m] for m in methods], alpha=0.86, width=0.62)
    ax0.axhline(0.95, color="#333333", linestyle="--", lw=1.1)
    ax0.text(len(methods) - 0.15, 0.955, "0.95 target", ha="right", va="bottom", fontsize=8)
    for idx, method in enumerate(methods):
        ax0.text(idx, ep[method] + 0.025, f"mean cov {mean_cov[method]:.3f}", ha="center", va="bottom", fontsize=8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(methods)
    ax0.set_ylim(0, 1.08)
    ax0.set_ylabel("EP: fraction of slices with test coverage >= 0.95")
    ax0.set_title("A. Coverage-attainment reliability")
    ax0.grid(axis="y", color="#E0E0E0", lw=0.6)

    ax1 = axes[1]
    rng = np.random.default_rng(20260629)
    box_data = [widths[m] for m in methods]
    bp = ax1.boxplot(box_data, positions=x, widths=0.46, patch_artist=True, showfliers=False, medianprops={"color": "#222222", "lw": 1.2})
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(colors[method])
        patch.set_alpha(0.26)
        patch.set_edgecolor(colors[method])
    for idx, method in enumerate(methods):
        jitter = rng.normal(0.0, 0.035, size=len(widths[method]))
        ax1.scatter(np.full(len(widths[method]), idx) + jitter, widths[method], s=14, color=colors[method], alpha=0.55, linewidth=0)
        ax1.scatter([idx], [iw[method]], s=48, marker="D", color="#222222", zorder=4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylabel("selected interval width (milliseconds)")
    ax1.set_title("B. Efficiency after validation")
    ax1.grid(axis="y", color="#E0E0E0", lw=0.6)

    fig.suptitle("NNPI repo-level reproduction: validity-width tradeoff", y=1.03, fontsize=13)
    fig.tight_layout()
    paths = _export_bundle(fig, outdir, "nnpi_repo_reproduction_results")
    plt.close(fig)
    return paths, summary


def write_qa(
    outdir: Path,
    test_csv: Path,
    per_slice_csv: Path,
    summary_csv: Path,
    overview_paths: list[Path],
    results_paths: list[Path],
    overview_stats: dict[str, float],
    summary_rows: list[dict[str, str]],
) -> Path:
    qa_path = outdir / "nnpi_or_figure_qa.md"
    lines = [
        "# NNPI OR Figure QA",
        "",
        "## Figure Contract",
        "",
        "Core conclusion: The released NNPI data form a 4-input communication-network simulation with right-skewed delays; the repo-level reproduction shows Gaussian validation improves target-attainment reliability at the cost of wider intervals.",
        "Primary comparison: NNVA vs NNGN vs NNGU on the public 4-input NNPI reproduction.",
        f"Data source: `{test_csv}`, `{per_slice_csv}`, `{summary_csv}`.",
        "Rows / statistical unit: overview uses 20,000 test replications and 200 test sites; results use 50 validation slices.",
        "Panels: overview explains the example and raw data; results separates EP validity from interval-width efficiency.",
        "Target coverage / threshold: 0.95.",
        "Uncertainty display: selected-width distributions are shown over validation slices; no fabricated CIs are added.",
        "Export formats: SVG, PDF, PNG, TIFF.",
        "Reviewer risk: This is repo-level NN-only reproduction, not full paper-level baseline reproduction.",
        "",
        "## Input Checks",
        "",
    ]
    for key, value in overview_stats.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Summary Results", ""])
    for row in summary_rows:
        lines.append(
            f"- {row['method']}: EP={float(row['EP']):.3f}, "
            f"IW={float(row['IW']):.3f}, mean_test_cov={float(row['mean_test_cov']):.3f}"
        )
    lines.extend(["", "## Exports", ""])
    for path in overview_paths + results_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Visual QA",
            "",
            "- Axes include units where applicable.",
            "- Coverage target is shown in the EP panel.",
            "- Results figure pairs validity (EP) with efficiency (interval width).",
            "- Method labels are stable across panels.",
            "- Source CSVs are listed above.",
        ]
    )
    qa_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return qa_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", default="/private/tmp/NNPI/truth_4input_te.csv")
    parser.add_argument("--per-slice-csv", default="analysis/nnpi_repro/output_repo_level_full/per_slice_results.csv")
    parser.add_argument("--summary-csv", default="analysis/nnpi_repro/output_repo_level_full/summary_results.csv")
    parser.add_argument("--outdir", default="analysis/nnpi_repro/output_or_figures")
    args = parser.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    test_csv = Path(args.test_csv).resolve()
    per_slice_csv = Path(args.per_slice_csv).resolve()
    summary_csv = Path(args.summary_csv).resolve()

    overview_paths, overview_stats = plot_overview(test_csv, outdir)
    results_paths, summary_rows = plot_results(per_slice_csv, summary_csv, outdir)
    qa_path = write_qa(outdir, test_csv, per_slice_csv, summary_csv, overview_paths, results_paths, overview_stats, summary_rows)

    print("wrote:")
    for path in overview_paths + results_paths + [qa_path]:
        print(path)


if __name__ == "__main__":
    main()
