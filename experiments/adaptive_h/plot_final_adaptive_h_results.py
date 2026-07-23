"""Create publication-ready figures from final adaptive-h benchmark outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parents[2]
for _import_path in (_ROOT / "src", _ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from Two_stage.sim_functions import get_experiment_config
from experiments.adaptive_h.adaptive_bandwidth import ORACLE_SCALE

DEFAULT_OUTPUT = Path(__file__).with_name("output_final_adaptive_h")
SIMULATORS = ("mm1_sojourn", "raised_floor_gauss", "raised_floor_t3")
ARMS = ("fixed", "plugin_sd_nw", "oracle")
ARM_LABELS = {
    "fixed": "Fixed",
    "plugin_sd_nw": "Sample-SD + NW",
    "oracle": "Oracle",
}
DGP_LABELS = {
    "mm1_sojourn": "M/M/1 sojourn time",
    "raised_floor_gauss": "Raised-floor Gaussian",
    "raised_floor_t3": r"Raised-floor Student-$t_3$",
}
COLORS = {
    "fixed": "#3B3B3B",
    "plugin_sd_nw": "#D55E00",
    "oracle": "#0072B2",
}
MARKERS = {"fixed": "o", "plugin_sd_nw": "s", "oracle": "^"}
FIGURE_STEMS = (
    "scale_functions",
    "plugin_oracle_budget_gap",
    "raw_score_homogeneity",
    "binwise_coverage",
    "effective_bandwidth_ratio",
)
FIGURE_SUFFIXES = (".pdf", ".svg", ".png", ".tiff")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_figure(fig: mpl.figure.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in FIGURE_SUFFIXES:
        path = stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight"}
        if suffix in {".png", ".tiff"}:
            kwargs["dpi"] = 400
        if suffix == ".tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def _mc_summary(
    frame: pd.DataFrame,
    group_columns: list[str],
    value: str,
) -> pd.DataFrame:
    result = (
        frame.groupby(group_columns, as_index=False)[value]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    result["mcse"] = result["std"] / np.sqrt(result["count"])
    return result


def plot_scale_functions(fig_dir: Path, data_dir: Path) -> list[Path]:
    rows: list[dict[str, float | str]] = []
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    for axis, simulator in zip(axes, SIMULATORS):
        config = get_experiment_config(simulator)
        lower, upper = config["bounds"]
        x = np.linspace(float(lower[0]), float(upper[0]), 400)
        scale = np.asarray(ORACLE_SCALE[simulator](x[:, None])).ravel()
        axis.plot(x, scale, color="#0072B2")
        axis.set_title(DGP_LABELS[simulator])
        axis.set_xlabel(r"$\rho$" if simulator == "mm1_sojourn" else r"$x$")
        for x_value, s_value in zip(x, scale):
            rows.append(
                {"simulator": simulator, "x0": x_value, "s_oracle": s_value}
            )
    axes[0].set_ylabel(r"Conditional SD $s(x)$")
    fig.tight_layout()
    pd.DataFrame(rows).to_csv(data_dir / "scale_functions.csv", index=False)
    return _save_figure(fig, fig_dir / "scale_functions")


def plot_budget_gap(
    output_dir: Path, fig_dir: Path, data_dir: Path
) -> list[Path]:
    paired = pd.read_csv(output_dir / "paired_deltas.csv")
    paired = paired.loc[
        paired["comparison"] == "plugin_sd_nw_minus_oracle"
    ].copy()
    metrics = [
        ("delta_interval_score", "Interval-score gap"),
        ("delta_worst_group_coverage_gap", "Worst-group-gap difference"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65))
    plot_rows: list[pd.DataFrame] = []
    for axis, (metric, label) in zip(axes, metrics):
        summary = _mc_summary(
            paired, ["simulator", "budget"], metric
        )
        summary["metric"] = metric
        plot_rows.append(summary)
        for simulator in SIMULATORS:
            subset = summary.loc[summary["simulator"] == simulator]
            axis.errorbar(
                subset["budget"],
                subset["mean"],
                yerr=subset["mcse"],
                marker=MARKERS[ARMS[SIMULATORS.index(simulator)]],
                capsize=2.5,
                label=DGP_LABELS[simulator],
            )
        axis.axhline(0.0, color="0.45", linewidth=0.9, linestyle="--")
        axis.set_xscale("log")
        axis.set_xticks(sorted(paired["budget"].unique()))
        axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axis.set_xlabel(r"Stage-1 budget $B=n_0r_0$")
        axis.set_ylabel(label)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    pd.concat(plot_rows, ignore_index=True).to_csv(
        data_dir / "plugin_oracle_budget_gap.csv", index=False
    )
    return _save_figure(fig, fig_dir / "plugin_oracle_budget_gap")


def plot_score_homogeneity(
    output_dir: Path, fig_dir: Path, data_dir: Path
) -> list[Path]:
    per_arm = pd.read_csv(output_dir / "score_homogeneity_per_arm.csv")
    summary = _mc_summary(
        per_arm,
        ["simulator", "budget", "arm"],
        "max_pairwise_score_ks",
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), sharey=True)
    for axis, simulator in zip(axes, SIMULATORS):
        for arm in ARMS:
            subset = summary.loc[
                (summary["simulator"] == simulator)
                & (summary["arm"] == arm)
            ]
            axis.errorbar(
                subset["budget"],
                subset["mean"],
                yerr=subset["mcse"],
                color=COLORS[arm],
                marker=MARKERS[arm],
                markersize=3.5,
                capsize=2,
                label=ARM_LABELS[arm],
            )
        axis.set_xscale("log")
        axis.set_xticks(sorted(per_arm["budget"].unique()))
        axis.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        axis.set_title(DGP_LABELS[simulator])
        axis.set_xlabel(r"Stage-1 budget $B=n_0r_0$")
    axes[0].set_ylabel("Maximum pairwise score KS")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    summary.to_csv(data_dir / "raw_score_homogeneity.csv", index=False)
    return _save_figure(fig, fig_dir / "raw_score_homogeneity")


def _collect_binwise(
    output_dir: Path,
    main_budget: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_rows: list[dict] = []
    ratio_rows: list[dict] = []
    paths = sorted(
        (output_dir / "jobs").glob(
            f"*/budget_{main_budget}/macrorep_*/per_point_*.csv"
        )
    )
    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=[
                "macrorep",
                "simulator",
                "arm",
                "x0",
                "group_bin",
                "covered_score",
                "h_over_s",
            ],
        )
        grouped = frame.groupby("group_bin", as_index=False).agg(
            x_mean=("x0", "mean"),
            coverage=("covered_score", "mean"),
            h_over_s=("h_over_s", "mean"),
        )
        metadata = frame.iloc[0]
        for row in grouped.itertuples(index=False):
            base = {
                "macrorep": int(metadata["macrorep"]),
                "simulator": metadata["simulator"],
                "arm": metadata["arm"],
                "group_bin": int(row.group_bin),
                "x_mean": float(row.x_mean),
            }
            coverage_rows.append({**base, "coverage": float(row.coverage)})
            ratio_rows.append({**base, "h_over_s": float(row.h_over_s)})
    return pd.DataFrame(coverage_rows), pd.DataFrame(ratio_rows)


def _plot_faceted_lines(
    raw: pd.DataFrame,
    *,
    value: str,
    ylabel: str,
    target: float | None,
    stem: str,
    fig_dir: Path,
    data_dir: Path,
) -> list[Path]:
    x_summary = raw.groupby(
        ["simulator", "arm", "group_bin"], as_index=False
    ).agg(x_mean=("x_mean", "mean"))
    values = _mc_summary(
        raw, ["simulator", "arm", "group_bin"], value
    )
    summary = values.merge(
        x_summary, on=["simulator", "arm", "group_bin"], how="left"
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), sharey=False)
    for axis, simulator in zip(axes, SIMULATORS):
        for arm in ARMS:
            subset = summary.loc[
                (summary["simulator"] == simulator)
                & (summary["arm"] == arm)
            ]
            axis.errorbar(
                subset["x_mean"],
                subset["mean"],
                yerr=subset["mcse"],
                color=COLORS[arm],
                marker=MARKERS[arm],
                markersize=3.5,
                capsize=2,
                label=ARM_LABELS[arm],
            )
        if target is not None:
            axis.axhline(
                target, color="0.45", linewidth=0.9, linestyle="--"
            )
        axis.set_title(DGP_LABELS[simulator])
        axis.set_xlabel(r"$\rho$" if simulator == "mm1_sojourn" else r"$x$")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    summary.to_csv(data_dir / f"{stem}.csv", index=False)
    return _save_figure(fig, fig_dir / stem)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def figure_qa(paths: list[Path], fig_dir: Path) -> dict:
    expected_names = {
        f"{stem}{suffix}"
        for stem in FIGURE_STEMS
        for suffix in FIGURE_SUFFIXES
    }
    actual_names = {path.name for path in paths}
    records = []
    for path in paths:
        record = {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": _hash(path),
            "status": "pass" if path.stat().st_size > 1000 else "fail",
        }
        if path.suffix in {".png", ".tiff"}:
            with Image.open(path) as image:
                record["width_px"] = image.width
                record["height_px"] = image.height
                record["mode"] = image.mode
                if min(image.width, image.height) < 800:
                    record["status"] = "fail"
        elif path.suffix == ".pdf":
            payload = path.read_bytes()
            record["type3_font"] = b"/Subtype /Type3" in payload
            if record["type3_font"]:
                record["status"] = "fail"
        elif path.suffix == ".svg":
            payload = path.read_text(encoding="utf-8")
            record["text_elements"] = payload.count("<text")
            if record["text_elements"] == 0:
                record["status"] = "fail"
        records.append(record)
    complete_set = actual_names == expected_names
    report = {
        "status": (
            "pass"
            if complete_set
            and all(row["status"] == "pass" for row in records)
            else "fail"
        ),
        "expected_files": sorted(expected_names),
        "missing_files": sorted(expected_names - actual_names),
        "unexpected_files": sorted(actual_names - expected_names),
        "files": records,
    }
    (fig_dir / "figure_qa.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--main-budget", "--main_budget", type=int, default=1000)
    args = parser.parse_args()
    _style()
    output_dir = _resolve(args.output_dir)
    fig_dir = output_dir / "figures"
    data_dir = output_dir / "plot_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    paths.extend(plot_scale_functions(fig_dir, data_dir))
    paths.extend(plot_budget_gap(output_dir, fig_dir, data_dir))
    paths.extend(plot_score_homogeneity(output_dir, fig_dir, data_dir))
    coverage, ratio = _collect_binwise(output_dir, args.main_budget)
    if coverage.empty:
        raise FileNotFoundError(
            f"No B={args.main_budget} per-point files under {output_dir}"
        )
    paths.extend(
        _plot_faceted_lines(
            coverage,
            value="coverage",
            ylabel="Raw-score coverage",
            target=0.90,
            stem="binwise_coverage",
            fig_dir=fig_dir,
            data_dir=data_dir,
        )
    )
    paths.extend(
        _plot_faceted_lines(
            ratio,
            value="h_over_s",
            ylabel=r"Effective ratio $h(x)/s(x)$",
            target=1.0,
            stem="effective_bandwidth_ratio",
            fig_dir=fig_dir,
            data_dir=data_dir,
        )
    )
    report = figure_qa(paths, fig_dir)
    print(f"Created {len(paths)} figure files; QA={report['status']}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
