"""Plot KME feasibility metrics and example CDF curves."""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--predictions", default="")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", default="experiment_logs/ckme_dcp_mm1")
    return parser


def write_png(path: Path, pixels: np.ndarray) -> None:
    """Write an RGB image array as PNG without optional plotting libraries."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.asarray(pixels, dtype=np.uint8)
    height, width, channels = pixels.shape
    if channels != 3:
        raise ValueError("pixels must have shape (height, width, 3)")

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw, level=6))
    data += chunk(b"IEND", b"")
    path.write_bytes(data)


def draw_line(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], thickness: int = 1) -> None:
    """Draw a simple Bresenham line."""

    height, width, _ = img.shape
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= x < width and 0 <= y < height:
            img[max(0, y - thickness + 1) : min(height, y + thickness), max(0, x - thickness + 1) : min(width, x + thickness)] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def draw_axes(img: np.ndarray, margin: int = 52) -> tuple[int, int, int, int]:
    height, width, _ = img.shape
    left, right = margin, width - 24
    top, bottom = 24, height - margin
    img[:] = 255
    draw_line(img, left, bottom, right, bottom, (35, 35, 35), thickness=2)
    draw_line(img, left, bottom, left, top, (35, 35, 35), thickness=2)
    for frac in np.linspace(0.0, 1.0, 6):
        y = int(bottom - frac * (bottom - top))
        draw_line(img, left, y, right, y, (230, 230, 230), thickness=1)
    return left, right, top, bottom


def scale_points(x: np.ndarray, y: np.ndarray, box: tuple[int, int, int, int], y_max: float | None = None) -> list[tuple[int, int]]:
    left, right, top, bottom = box
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max is None:
        y_max = float(np.max(y)) if y.size else 1.0
    y_max = max(y_max, 1e-12)
    points = []
    for xi, yi in zip(x, y):
        px = int(left + (xi - x_min) / (x_max - x_min) * (right - left))
        py = int(bottom - max(0.0, yi) / y_max * (bottom - top))
        points.append((px, py))
    return points


def fallback_bar_png(path: Path, values: list[float], color: tuple[int, int, int]) -> None:
    img = np.full((420, 720, 3), 255, dtype=np.uint8)
    left, right, top, bottom = draw_axes(img)
    clean = [0.0 if not np.isfinite(v) else float(v) for v in values]
    y_max = max(max(clean), 1e-12) * 1.15
    n = max(len(clean), 1)
    slot = (right - left) / n
    for idx, value in enumerate(clean):
        x0 = int(left + idx * slot + slot * 0.18)
        x1 = int(left + (idx + 1) * slot - slot * 0.18)
        y = int(bottom - value / y_max * (bottom - top))
        img[y:bottom, x0:x1] = color
    write_png(path, img)


def fallback_line_png(path: Path, x_values: list[float], y_values: list[float], color: tuple[int, int, int]) -> None:
    img = np.full((420, 720, 3), 255, dtype=np.uint8)
    box = draw_axes(img)
    clean_y = np.array([0.0 if not np.isfinite(v) else float(v) for v in y_values], dtype=float)
    points = scale_points(np.arange(len(x_values), dtype=float), clean_y, box, y_max=max(float(np.max(clean_y)) * 1.15, 1e-12))
    for p0, p1 in zip(points[:-1], points[1:]):
        draw_line(img, p0[0], p0[1], p1[0], p1[1], color, thickness=2)
    for x, y in points:
        img[max(0, y - 4) : y + 5, max(0, x - 4) : x + 5] = color
    write_png(path, img)


def fallback_cdf_png(path: Path, predictions: pd.DataFrame, seed: int | None) -> None:
    if predictions.empty:
        return
    if seed is None:
        seed = int(predictions["seed"].iloc[0])
    subset = predictions[predictions["seed"] == seed]
    scenario_ids = sorted(subset["scenario_id"].unique())[:5]
    if not scenario_ids:
        return
    panel_h = 240
    img = np.full((panel_h * len(scenario_ids), 760, 3), 255, dtype=np.uint8)
    for panel, scenario_id in enumerate(scenario_ids):
        y_offset = panel * panel_h
        canvas = img[y_offset : y_offset + panel_h]
        box = draw_axes(canvas, margin=42)
        rows = subset[subset["scenario_id"] == scenario_id].sort_values("grid_index")
        x = rows["t"].to_numpy(dtype=float)
        pred_points = scale_points(x, rows["pred_cdf"].to_numpy(dtype=float), box, y_max=1.0)
        oracle_points = scale_points(x, rows["oracle_cdf"].to_numpy(dtype=float), box, y_max=1.0)
        for p0, p1 in zip(pred_points[:-1], pred_points[1:]):
            draw_line(canvas, p0[0], p0[1], p1[0], p1[1], (78, 121, 167), thickness=2)
        for p0, p1 in zip(oracle_points[:-1], oracle_points[1:]):
            draw_line(canvas, p0[0], p0[1], p1[0], p1[1], (0, 0, 0), thickness=1)
    write_png(path, img)


def fallback_scatter_png(path: Path, diagnostics: pd.DataFrame) -> None:
    if diagnostics.empty:
        return
    img = np.full((420, 720, 3), 255, dtype=np.uint8)
    box = draw_axes(img)
    colors = {
        20: (78, 121, 167),
        50: (242, 142, 43),
        200: (89, 161, 79),
        1000: (225, 87, 89),
    }
    rho = diagnostics["true_rho"].to_numpy(dtype=float)
    sup_error = diagnostics["sup_error"].to_numpy(dtype=float)
    points = scale_points(rho, sup_error, box, y_max=max(float(np.max(sup_error)) * 1.15, 1e-12))
    for (x, y), n1 in zip(points, diagnostics["n1"].to_numpy(dtype=int)):
        color = colors.get(int(n1), (85, 85, 85))
        img[max(0, y - 4) : min(img.shape[0], y + 5), max(0, x - 4) : min(img.shape[1], x + 5)] = color
    write_png(path, img)


def prediction_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"seed", "scenario_id", "pred_cdf", "oracle_cdf", "n1", "rho_hat", "true_rho"}
    if predictions.empty or not required.issubset(predictions.columns):
        return pd.DataFrame()
    rows: list[dict[str, float | int]] = []
    for (seed, scenario_id), group in predictions.groupby(["seed", "scenario_id"], sort=True):
        diff = group["pred_cdf"].to_numpy(dtype=float) - group["oracle_cdf"].to_numpy(dtype=float)
        rho_hat_value = float(group["rho_hat"].iloc[0])
        true_rho_value = float(group["true_rho"].iloc[0])
        rows.append(
            {
                "seed": int(seed),
                "scenario_id": int(scenario_id),
                "n1": int(group["n1"].iloc[0]),
                "true_rho": true_rho_value,
                "rho_hat": rho_hat_value,
                "rho_gap": rho_hat_value - true_rho_value,
                "sup_error": float(np.max(np.abs(diff))),
                "mean_abs_error": float(np.mean(np.abs(diff))),
            }
        )
    return pd.DataFrame(rows)


def save_metric_plots(metrics: pd.DataFrame, out_dir: Path, pred_diag: pd.DataFrame | None = None) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        out_dir.mkdir(parents=True, exist_ok=True)
        fallback_bar_png(out_dir / "mean_ise_by_seed.png", metrics["ise_mean"].tolist(), (78, 121, 167))
        input_sizes = [20, 50, 200, 1000]
        ise_means = [metrics[f"ise_n{n}_mean"].mean() for n in input_sizes if f"ise_n{n}_mean" in metrics]
        fallback_line_png(out_dir / "ise_by_input_size.png", input_sizes[: len(ise_means)], ise_means, (89, 161, 79))
        fallback_bar_png(
            out_dir / "quantile_error.png",
            [metrics[c].mean() for c in ["qerr_0.1_mean", "qerr_0.5_mean", "qerr_0.9_mean"]],
            (242, 142, 43),
        )
        if pred_diag is not None and not pred_diag.empty:
            fallback_scatter_png(out_dir / "sup_error_vs_rho.png", pred_diag)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    run_labels = [str(i + 1) for i in range(len(metrics))]
    ax.bar(run_labels, metrics["ise_mean"], color="#4E79A7")
    ax.set_xlabel("run")
    ax.set_ylabel("mean ISE")
    ax.set_title("Mean CDF ISE by run")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "mean_ise_by_seed.png", dpi=200)
    plt.close(fig)

    input_sizes = [20, 50, 200, 1000]
    ise_means = [metrics[f"ise_n{n}_mean"].mean() for n in input_sizes if f"ise_n{n}_mean" in metrics]
    shown_sizes = [n for n in input_sizes if f"ise_n{n}_mean" in metrics]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([str(n) for n in shown_sizes], ise_means, marker="o", color="#59A14F")
    ax.set_xlabel("input sample size n1")
    ax.set_ylabel("mean ISE")
    ax.set_title("CDF ISE by input sample size")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "ise_by_input_size.png", dpi=200)
    plt.close(fig)

    q_cols = ["qerr_0.1_mean", "qerr_0.5_mean", "qerr_0.9_mean"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["0.1", "0.5", "0.9"], [metrics[c].mean() for c in q_cols], color="#F28E2B")
    ax.set_xlabel("quantile level")
    ax.set_ylabel("mean absolute quantile error")
    ax.set_title("Quantile error")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "quantile_error.png", dpi=200)
    plt.close(fig)

    if pred_diag is not None and not pred_diag.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {20: "#4E79A7", 50: "#F28E2B", 200: "#59A14F", 1000: "#E15759"}
        for n1, group in pred_diag.groupby("n1", sort=True):
            ax.scatter(
                group["true_rho"],
                group["sup_error"],
                s=22,
                alpha=0.72,
                color=colors.get(int(n1), "#777777"),
                label=f"n1={int(n1)}",
            )
        ax.set_xlabel("true rho")
        ax.set_ylabel("sup CDF error")
        ax.set_title("Sup CDF error vs traffic intensity")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "sup_error_vs_rho.png", dpi=200)
        plt.close(fig)


def save_example_cdf_plot(predictions: pd.DataFrame, out_dir: Path, seed: int | None) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError:
        fallback_cdf_png(out_dir / "example_cdf_curves.png", predictions, seed)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if predictions.empty or "pred_cdf" not in predictions:
        return
    if seed is None:
        seed = int(predictions["seed"].iloc[0])
    subset = predictions[predictions["seed"] == seed]
    scenario_ids = sorted(subset["scenario_id"].unique())[:5]
    if not scenario_ids:
        return
    fig, axes = plt.subplots(len(scenario_ids), 1, figsize=(7, 2.4 * len(scenario_ids)), sharex=True)
    if len(scenario_ids) == 1:
        axes = [axes]
    for ax, scenario_id in zip(axes, scenario_ids):
        rows = subset[subset["scenario_id"] == scenario_id].sort_values("grid_index")
        ax.plot(rows["t"], rows["pred_cdf"], color="#4E79A7", linewidth=2.0, label="predicted CDF")
        ax.plot(rows["t"], rows["oracle_cdf"], color="black", linewidth=1.6, linestyle="--", label="oracle CDF")
        n1 = int(rows["n1"].iloc[0])
        rho = float(rows["true_rho"].iloc[0])
        ax.set_title(f"scenario={scenario_id}, n1={n1}, true rho={rho:.2f}")
        ax.set_ylabel("CDF")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("average sojourn time")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "example_cdf_curves.png", dpi=200)
    plt.close(fig)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_log_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def format_float(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.6g}"


def mean_metric(metrics: pd.DataFrame, column: str) -> str:
    if column not in metrics:
        return "NA"
    return format_float(metrics[column].mean())


def maybe_load_config(metrics_path: Path) -> dict[str, object]:
    if metrics_path.name.endswith("_metrics.csv"):
        config_path = metrics_path.with_name(metrics_path.name.replace("_metrics.csv", "_config.json"))
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def metric_line(metrics: pd.DataFrame, column: str, label: str) -> str:
    return f"| {label} | {mean_metric(metrics, column)} |"


def figure_manifest(out_dir: Path, has_predictions: bool, has_prediction_diagnostics: bool) -> list[tuple[Path, str]]:
    figures = [
        (
            out_dir / "mean_ise_by_seed.png",
            "Mean integrated squared CDF error for each run. The x-axis uses run index rather than the random seed.",
        ),
        (
            out_dir / "ise_by_input_size.png",
            "Mean CDF ISE grouped by finite inter-arrival sample size n1. Lower values for larger n1 indicate less input-sample uncertainty.",
        ),
        (
            out_dir / "quantile_error.png",
            "Mean absolute quantile error at levels 0.1, 0.5, and 0.9, computed by inverting predicted and oracle CDFs on the same grid.",
        ),
    ]
    if has_predictions:
        figures.append(
            (
                out_dir / "example_cdf_curves.png",
                "Example predicted CDF curves against oracle Monte Carlo CDF curves for representative test scenarios.",
            )
        )
    if has_prediction_diagnostics:
        figures.append(
            (
                out_dir / "sup_error_vs_rho.png",
                "Scenario-level sup CDF error plotted against true traffic intensity, with color indicating finite input sample size n1.",
            )
        )
    return figures


def summary_stats(values: pd.Series) -> tuple[str, str, str]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return "NA", "NA", "NA"
    return format_float(clean.mean()), format_float(clean.min()), format_float(clean.max())


def prediction_diagnostics_tables(pred_diag: pd.DataFrame) -> tuple[str, str, str]:
    if pred_diag.empty:
        return "No prediction diagnostics were available.", "", ""

    sup_mean, sup_min, sup_max = summary_stats(pred_diag["sup_error"])
    gap_mean, gap_min, gap_max = summary_stats(pred_diag["rho_gap"])
    mae_mean, mae_min, mae_max = summary_stats(pred_diag["mean_abs_error"])
    overall = "\n".join(
        [
            "| diagnostic | mean | min | max |",
            "| --- | ---: | ---: | ---: |",
            f"| sup CDF error | {sup_mean} | {sup_min} | {sup_max} |",
            f"| mean absolute CDF error | {mae_mean} | {mae_min} | {mae_max} |",
            f"| rho_hat - true rho | {gap_mean} | {gap_min} | {gap_max} |",
        ]
    )

    by_n1_rows = ["| n1 | count | mean sup error | mean rho_hat - true rho |", "| ---: | ---: | ---: | ---: |"]
    for n1, group in pred_diag.groupby("n1", sort=True):
        by_n1_rows.append(
            f"| {int(n1)} | {len(group)} | {format_float(group['sup_error'].mean())} | {format_float(group['rho_gap'].mean())} |"
        )

    by_traffic = pred_diag.assign(traffic=np.where(pred_diag["true_rho"] > 0.75, "heavy", "light"))
    traffic_rows = ["| traffic group | count | mean sup error | mean rho_hat - true rho |", "| --- | ---: | ---: | ---: |"]
    for label, group in by_traffic.groupby("traffic", sort=True):
        traffic_rows.append(
            f"| {label} | {len(group)} | {format_float(group['sup_error'].mean())} | {format_float(group['rho_gap'].mean())} |"
        )
    return overall, "\n".join(by_n1_rows), "\n".join(traffic_rows)


def write_experiment_log(
    log_dir: Path,
    metrics_path: Path,
    predictions_path: Path | None,
    out_dir: Path,
    metrics: pd.DataFrame,
    config: dict[str, object],
    selected_seed: int | None,
    pred_diag: pd.DataFrame,
) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}_kme_feasibility.md"
    figure_rows = "\n".join(
        f"| `{path}` | {meaning} |"
        for path, meaning in figure_manifest(out_dir, predictions_path is not None, not pred_diag.empty)
    )
    settings = {
        "dgp": config.get("dgp", metrics["dgp"].iloc[0] if "dgp" in metrics else "NA"),
        "n_fit": config.get("n_fit", metrics["n_fit"].iloc[0] if "n_fit" in metrics else "NA"),
        "n_val": config.get("n_val", metrics["n_val"].iloc[0] if "n_val" in metrics else "NA"),
        "n_test": config.get("n_test", metrics["n_test"].iloc[0] if "n_test" in metrics else "NA"),
        "r_train": config.get("r_train", metrics["r_train"].iloc[0] if "r_train" in metrics else "NA"),
        "r_oracle": config.get("r_oracle", metrics["r_oracle"].iloc[0] if "r_oracle" in metrics else "NA"),
        "rff_dim": config.get("rff_dim", metrics["rff_dim"].iloc[0] if "rff_dim" in metrics else "NA"),
        "grid_size": config.get("grid_size", metrics["grid_size"].iloc[0] if "grid_size" in metrics else "NA"),
        "n_seeds": config.get("n_seeds", len(metrics)),
        "seed_start": config.get("seed", metrics["seed"].iloc[0] if "seed" in metrics else "NA"),
    }
    settings_rows = "\n".join(f"- {key}: {value}" for key, value in settings.items())
    input_size_rows = "\n".join(
        metric_line(metrics, f"ise_n{n}_mean", f"n1={n}") for n in (20, 50, 200, 1000)
    )
    traffic_rows = "\n".join(
        metric_line(metrics, f"ise_{group}_mean", group) for group in ("light", "heavy")
    )
    pred_overall, pred_by_n1, pred_by_traffic = prediction_diagnostics_tables(pred_diag)
    selected_seed_text = "first available run" if selected_seed is None else str(selected_seed)
    predictions_text = f"`{predictions_path}`" if predictions_path is not None else "not provided"
    content = f"""# KME/CKME Feasibility Experiment Log

Date: {datetime.now().astimezone().isoformat()}

## Goal

This is a feasibility-only KME/CKME conditional CDF experiment for M/M/1 input
uncertainty. It evaluates predicted CDF curves against oracle Monte Carlo CDFs.
It does not run conformal prediction and does not report coverage guarantees.

## Settings

{settings_rows}
- example CDF figure run: {selected_seed_text}

## Source Files

- metrics CSV: `{metrics_path}`
- predictions CSV: {predictions_text}
- figure directory: `{out_dir}`

## Main Results

| metric | mean over runs |
| --- | ---: |
{metric_line(metrics, "ise_mean", "ISE")}
{metric_line(metrics, "iae_mean", "IAE")}
{metric_line(metrics, "qerr_0.1_mean", "absolute quantile error, q=0.1")}
{metric_line(metrics, "qerr_0.5_mean", "absolute quantile error, q=0.5")}
{metric_line(metrics, "qerr_0.9_mean", "absolute quantile error, q=0.9")}
{metric_line(metrics, "mono_violation_mean", "raw monotonicity violation")}
{metric_line(metrics, "val_mse", "validation CDF MSE")}

## CDF ISE By Input Sample Size

| group | mean over runs |
| --- | ---: |
{input_size_rows}

## CDF ISE By Traffic Group

| group | mean over runs |
| --- | ---: |
{traffic_rows}

## Sup Error And Rho-Hat Diagnostics

{pred_overall}

## Sup Error By Input Sample Size

{pred_by_n1}

## Sup Error By Traffic Group

{pred_by_traffic}

## Figures

| figure | meaning |
| --- | --- |
{figure_rows}

## Interpretation

- Smaller CDF ISE and IAE indicate closer agreement between the learned CKME CDF
  and the oracle Monte Carlo CDF.
- The input-size plot checks whether finite input uncertainty is visible: errors
  should generally be larger for smaller input sample sizes.
- The quantile-error plot shows where CDF error matters for distributional
  summaries, especially the high quantile.
- The sup-error scatter checks whether heavy traffic is associated with larger
  worst-case CDF discrepancies and whether small input samples are concentrated
  among the hard cases.
- The rho_hat minus true rho summary helps distinguish model bias from finite
  input samples that make a scenario look lighter or heavier than it really is.
- The example CDF plot is qualitative: it compares predicted CDF shape against
  oracle CDF shape on selected scenarios without showing seed labels in the
  figure.
"""
    log_path.write_text(content, encoding="utf-8")
    return log_path


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    metrics_path = Path(args.metrics)
    metrics = pd.read_csv(metrics_path)
    pred_diag = pd.DataFrame()
    predictions_path = None
    if args.predictions:
        predictions_path = Path(args.predictions)
        if predictions_path.exists():
            predictions = pd.read_csv(predictions_path)
            pred_diag = prediction_diagnostics(predictions)
            save_example_cdf_plot(predictions, out_dir, args.seed)
        else:
            predictions_path = None
    save_metric_plots(metrics, out_dir, pred_diag)
    log_path = write_experiment_log(
        resolve_log_path(args.log_dir),
        metrics_path,
        predictions_path,
        out_dir,
        metrics,
        maybe_load_config(metrics_path),
        args.seed,
        pred_diag,
    )
    print(f"Wrote figures to: {out_dir}")
    print(f"Wrote experiment log: {log_path}")


if __name__ == "__main__":
    main()
