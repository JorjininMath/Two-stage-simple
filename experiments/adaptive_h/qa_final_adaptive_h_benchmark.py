"""Quality-assurance checks for the final adaptive-h benchmark outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
for _import_path in (_ROOT / "src", _ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

import numpy as np
import pandas as pd

REQUIRED_POINT_COLUMNS = {
    "macrorep",
    "simulator",
    "budget",
    "arm",
    "test_index",
    "x0",
    "y",
    "L",
    "U",
    "covered_interval",
    "covered_score",
    "raw_score",
    "width",
    "interval_score",
    "h_query",
    "s_oracle",
    "s_hat",
    "h_over_s",
    "group_bin",
    "y_in_grid",
    "L_at_grid_lo",
    "U_at_grid_hi",
}
ARMS = {"fixed", "plugin_sd_nw", "oracle"}
SCORE_METRICS = (
    "max_pairwise_score_ks",
    "mean_pairwise_score_ks",
    "bin_mean_score_range",
)
SCORE_COMPARISONS = {
    "plugin_sd_nw_minus_oracle",
    "plugin_sd_nw_minus_fixed",
    "oracle_minus_fixed",
}
FINAL_FIGURE_STEMS = (
    "scale_functions",
    "plugin_oracle_budget_gap",
    "raw_score_homogeneity",
    "binwise_coverage",
    "effective_bandwidth_ratio",
)
DEFAULT_OUTPUT = Path(__file__).with_name("output_final_adaptive_h")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple]:
    normalized = frame[columns].copy()
    for column in ("macrorep", "budget"):
        if column in normalized:
            normalized[column] = normalized[column].astype(int)
    for column in ("simulator", "arm", "comparison"):
        if column in normalized:
            normalized[column] = normalized[column].astype(str)
    return set(normalized.itertuples(index=False, name=None))


def _record(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    severity: str = "error",
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
        }
    )


def run_qa(
    output_dir: Path,
    *,
    require_final: bool,
    max_grid_clip_rate: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        _record(checks, "manifest_exists", False, _display_path(manifest_path))
        return {"checks": checks}
    manifest = json.loads(manifest_path.read_text())
    config = manifest.get("resolved_config", {})

    _record(
        checks,
        "manifest_complete",
        manifest.get("status") == "complete",
        f"status={manifest.get('status')}",
    )
    _record(
        checks,
        "no_s0",
        manifest.get("uses_s0") is False,
        f"uses_s0={manifest.get('uses_s0')}",
    )
    _record(
        checks,
        "iid_protocol",
        config.get("calibration_method") == "iid"
        and int(config.get("r_cal", -1)) == 1
        and config.get("test_method") == "iid"
        and int(config.get("r_test", -1)) == 1,
        (
            f"calibration={config.get('calibration_method')}/"
            f"r={config.get('r_cal')}, test={config.get('test_method')}/"
            f"r={config.get('r_test')}"
        ),
    )
    expected_jobs = (
        len(config.get("simulators", []))
        * len(config.get("budgets", []))
        * int(config.get("n_macro", 0))
    )
    per_arm_path = output_dir / "per_arm.csv"
    if not per_arm_path.exists():
        _record(checks, "per_arm_exists", False, _display_path(per_arm_path))
        return {"manifest": manifest, "checks": checks}
    per_arm = pd.read_csv(per_arm_path)
    actual_jobs = per_arm[
        ["macrorep", "simulator", "budget"]
    ].drop_duplicates().shape[0]
    _record(
        checks,
        "job_count",
        actual_jobs == expected_jobs,
        f"actual={actual_jobs}, expected={expected_jobs}",
    )
    macro_count = int(per_arm["macrorep"].nunique())
    _record(
        checks,
        "paper_macrorep_threshold",
        (not require_final) or macro_count >= 50,
        f"unique_macroreps={macro_count}, require_final={require_final}",
    )
    arm_sets = per_arm.groupby(
        ["macrorep", "simulator", "budget"]
    )["arm"].agg(set)
    bad_arms = int(sum(value != ARMS for value in arm_sets))
    _record(
        checks,
        "paired_arms_complete",
        bad_arms == 0,
        f"jobs_with_incomplete_arms={bad_arms}",
    )
    numeric_metrics = [
        "coverage",
        "coverage_interval",
        "width",
        "interval_score",
        "q_hat",
    ]
    finite = np.isfinite(per_arm[numeric_metrics].to_numpy(dtype=float)).all()
    _record(checks, "aggregate_metrics_finite", finite, str(numeric_metrics))

    point_paths = sorted(
        (output_dir / "jobs").glob(
            "*/budget_*/macrorep_*/per_point_*.csv"
        )
    )
    expected_point_files = expected_jobs * len(ARMS)
    _record(
        checks,
        "per_point_file_count",
        len(point_paths) == expected_point_files,
        f"actual={len(point_paths)}, expected={expected_point_files}",
    )

    missing_columns: dict[str, list[str]] = {}
    nonfinite_files: list[str] = []
    pairing_failures: list[str] = []
    interval_clip_rates: list[float] = []
    outside_grid_rates: list[float] = []
    max_clip = 0.0
    max_outside = 0.0
    jobs_root = output_dir / "jobs"
    for job_dir in sorted(jobs_root.glob("*/budget_*/macrorep_*")):
        frames: dict[str, pd.DataFrame] = {}
        for arm in sorted(ARMS):
            path = job_dir / f"per_point_{arm}.csv"
            if not path.exists():
                continue
            frame = pd.read_csv(path)
            frames[arm] = frame
            missing = sorted(REQUIRED_POINT_COLUMNS - set(frame.columns))
            if missing:
                missing_columns[str(path.relative_to(output_dir))] = missing
                continue
            numeric = frame[
                [
                    "x0",
                    "y",
                    "L",
                    "U",
                    "raw_score",
                    "width",
                    "interval_score",
                    "h_query",
                    "s_oracle",
                    "s_hat",
                    "h_over_s",
                ]
            ].to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                nonfinite_files.append(str(path.relative_to(output_dir)))
            file_clip_rate = max(
                float(frame["L_at_grid_lo"].mean()),
                float(frame["U_at_grid_hi"].mean()),
            )
            file_outside_rate = float(1.0 - frame["y_in_grid"].mean())
            interval_clip_rates.append(file_clip_rate)
            outside_grid_rates.append(file_outside_rate)
            max_clip = max(max_clip, file_clip_rate)
            max_outside = max(max_outside, file_outside_rate)
        if len(frames) == len(ARMS):
            reference = frames["fixed"][["test_index", "x0", "y"]]
            for arm in ("plugin_sd_nw", "oracle"):
                candidate = frames[arm][["test_index", "x0", "y"]]
                if not np.allclose(
                    reference.to_numpy(dtype=float),
                    candidate.to_numpy(dtype=float),
                    rtol=0.0,
                    atol=0.0,
                ):
                    pairing_failures.append(
                        f"{job_dir.relative_to(output_dir)}:{arm}"
                    )
    _record(
        checks,
        "per_point_schema",
        not missing_columns,
        (
            "all required columns present"
            if not missing_columns
            else json.dumps(missing_columns)[:1000]
        ),
    )
    _record(
        checks,
        "per_point_values_finite",
        not nonfinite_files,
        (
            "all finite"
            if not nonfinite_files
            else ", ".join(nonfinite_files[:10])
        ),
    )
    _record(
        checks,
        "paired_test_data_identical",
        not pairing_failures,
        (
            "all paired arms share test data"
            if not pairing_failures
            else ", ".join(pairing_failures[:10])
        ),
    )
    _record(
        checks,
        "grid_boundary_rate",
        max(max_clip, max_outside) <= max_grid_clip_rate,
        (
            f"max_interval_clip={max_clip:.4f}, "
            f"max_y_outside={max_outside:.4f}, "
            "arm_files_over_threshold="
            f"{sum(value > max_grid_clip_rate for value in interval_clip_rates)}"
            f"/{len(interval_clip_rates)}, "
            f"mean_interval_clip={np.mean(interval_clip_rates):.6f}, "
            f"mean_y_outside={np.mean(outside_grid_rates):.6f}, "
            f"threshold={max_grid_clip_rate:.4f}"
        ),
        severity="warning",
    )

    summary_path = output_dir / "summary.csv"
    paired_path = output_dir / "paired_deltas.csv"
    scale_summary_path = output_dir / "scale_diagnostics_summary.csv"
    _record(
        checks,
        "summary_exists",
        summary_path.exists(),
        _display_path(summary_path),
    )
    _record(
        checks,
        "paired_deltas_exists",
        paired_path.exists(),
        _display_path(paired_path),
    )
    _record(
        checks,
        "scale_diagnostics_summary_exists",
        scale_summary_path.exists(),
        _display_path(scale_summary_path),
    )
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        target = 1.0 - float(config.get("alpha", 0.1))
        max_z = 0.0
        for row in summary.itertuples(index=False):
            mcse = float(row.mcse_coverage)
            if math.isfinite(mcse) and mcse > 0:
                max_z = max(
                    max_z, abs(float(row.mean_coverage) - target) / mcse
                )
        _record(
            checks,
            "coverage_sanity",
            max_z <= 4.0,
            f"maximum marginal-coverage deviation={max_z:.2f} MCSE",
            severity="warning",
        )
    if scale_summary_path.exists():
        scale_summary = pd.read_csv(scale_summary_path)
        scale_keys = ["simulator", "budget"]
        scale_value_columns = [
            column
            for column in scale_summary
            if column.startswith(("mean_", "sd_", "mcse_"))
        ]
        expected_scale_rows = (
            len(config.get("simulators", []))
            * len(config.get("budgets", []))
        )
        scale_complete = (
            {"simulator", "budget", "n_macroreps"}.issubset(scale_summary)
            and len(scale_summary) == expected_scale_rows
            and not scale_summary.duplicated(scale_keys).any()
            and (
                scale_summary["n_macroreps"].astype(int)
                == int(config.get("n_macro", -1))
            ).all()
            and bool(scale_value_columns)
            and np.isfinite(
                scale_summary[scale_value_columns].to_numpy(dtype=float)
            ).all()
        )
        _record(
            checks,
            "scale_diagnostics_summary_complete",
            scale_complete,
            (
                f"rows={len(scale_summary)}, "
                f"expected_rows={expected_scale_rows}, "
                f"value_columns={len(scale_value_columns)}"
            ),
        )

    score_per_arm_path = output_dir / "score_homogeneity_per_arm.csv"
    score_summary_path = output_dir / "score_homogeneity_summary.csv"
    score_paired_path = output_dir / "score_homogeneity_paired_deltas.csv"
    _record(
        checks,
        "score_homogeneity_per_arm_exists",
        score_per_arm_path.exists(),
        _display_path(score_per_arm_path),
    )
    _record(
        checks,
        "score_homogeneity_summary_exists",
        score_summary_path.exists(),
        _display_path(score_summary_path),
    )
    _record(
        checks,
        "score_homogeneity_paired_exists",
        score_paired_path.exists(),
        _display_path(score_paired_path),
    )
    score_outputs = [
        score_per_arm_path,
        score_summary_path,
        score_paired_path,
    ]
    if all(path.exists() for path in score_outputs):
        score_per_arm = pd.read_csv(score_per_arm_path)
        score_summary = pd.read_csv(score_summary_path)
        score_paired = pd.read_csv(score_paired_path)
        arm_key_columns = ["macrorep", "simulator", "budget", "arm"]
        expected_arm_keys = _key_set(per_arm, arm_key_columns)
        required_per_arm_columns = {
            *arm_key_columns,
            "n_groups",
            *SCORE_METRICS,
        }
        score_schema_ok = required_per_arm_columns.issubset(score_per_arm)
        _record(
            checks,
            "score_homogeneity_per_arm_schema",
            score_schema_ok,
            (
                "all required columns present"
                if score_schema_ok
                else (
                    "missing="
                    + str(
                        sorted(
                            required_per_arm_columns - set(score_per_arm)
                        )
                    )
                )
            ),
        )
        score_arm_keys = (
            _key_set(score_per_arm, arm_key_columns)
            if score_schema_ok
            else set()
        )
        _record(
            checks,
            "score_homogeneity_key_match",
            score_schema_ok
            and not score_per_arm.duplicated(arm_key_columns).any()
            and score_arm_keys == expected_arm_keys,
            (
                f"actual_unique={len(score_arm_keys)}, "
                f"expected_unique={len(expected_arm_keys)}"
            ),
        )
        expected_group_count = int(config.get("group_bins", -1))
        group_counts_ok = (
            score_schema_ok
            and len(score_per_arm) == len(expected_arm_keys)
            and (score_per_arm["n_groups"].astype(int) == expected_group_count).all()
        )
        _record(
            checks,
            "score_homogeneity_group_count",
            group_counts_ok,
            (
                f"rows={len(score_per_arm)}, "
                f"expected_rows={len(expected_arm_keys)}, "
                f"expected_groups={expected_group_count}"
            ),
        )
        score_values = (
            score_per_arm[list(SCORE_METRICS)].to_numpy(dtype=float)
            if score_schema_ok
            else np.empty((0, 0), dtype=float)
        )
        _record(
            checks,
            "score_homogeneity_per_arm_values",
            bool(score_values.size) and np.isfinite(score_values).all(),
            (
                f"rows={len(score_per_arm)}, "
                f"metrics={list(SCORE_METRICS)}"
            ),
        )
        ks_values_ok = (
            bool(score_values.size)
            and score_per_arm[
                [
                    "max_pairwise_score_ks",
                    "mean_pairwise_score_ks",
                ]
            ]
            .ge(0.0)
            .all()
            .all()
            and score_per_arm[
                [
                    "max_pairwise_score_ks",
                    "mean_pairwise_score_ks",
                ]
            ]
            .le(1.0)
            .all()
            .all()
            and score_per_arm["bin_mean_score_range"].ge(0.0).all()
        )
        _record(
            checks,
            "score_homogeneity_metric_ranges",
            ks_values_ok,
            "KS in [0,1] and bin-mean range nonnegative",
        )

        summary_key_columns = ["simulator", "budget", "arm"]
        summary_metric_columns = [
            f"{prefix}_{metric}"
            for metric in SCORE_METRICS
            for prefix in ("mean", "sd", "mcse")
        ]
        required_summary_columns = {
            *summary_key_columns,
            "n_macroreps",
            *summary_metric_columns,
        }
        summary_schema_ok = required_summary_columns.issubset(score_summary)
        expected_summary_keys = _key_set(
            per_arm, summary_key_columns
        )
        actual_summary_keys = (
            _key_set(score_summary, summary_key_columns)
            if summary_schema_ok
            else set()
        )
        summary_values = (
            score_summary[summary_metric_columns].to_numpy(dtype=float)
            if summary_schema_ok
            else np.empty((0, 0), dtype=float)
        )
        summary_complete = (
            summary_schema_ok
            and not score_summary.duplicated(summary_key_columns).any()
            and actual_summary_keys == expected_summary_keys
            and len(score_summary) == len(expected_summary_keys)
            and (
                score_summary["n_macroreps"].astype(int)
                == int(config.get("n_macro", -1))
            ).all()
            and bool(summary_values.size)
            and np.isfinite(summary_values).all()
        )
        _record(
            checks,
            "score_homogeneity_summary_complete",
            summary_complete,
            (
                f"rows={len(score_summary)}, "
                f"expected_rows={len(expected_summary_keys)}, "
                f"n_macro={config.get('n_macro')}"
            ),
        )

        paired_key_columns = [
            "macrorep",
            "simulator",
            "budget",
            "comparison",
        ]
        paired_metric_columns = [
            f"delta_{metric}" for metric in SCORE_METRICS
        ]
        required_paired_columns = {
            *paired_key_columns,
            *paired_metric_columns,
        }
        paired_schema_ok = required_paired_columns.issubset(score_paired)
        expected_pair_rows = expected_jobs * len(SCORE_COMPARISONS)
        comparison_sets = (
            score_paired.groupby(
                ["macrorep", "simulator", "budget"]
            )["comparison"].agg(set)
            if paired_schema_ok
            else pd.Series(dtype=object)
        )
        paired_values = (
            score_paired[paired_metric_columns].to_numpy(dtype=float)
            if paired_schema_ok
            else np.empty((0, 0), dtype=float)
        )
        paired_complete = (
            paired_schema_ok
            and not score_paired.duplicated(paired_key_columns).any()
            and len(score_paired) == expected_pair_rows
            and len(comparison_sets) == expected_jobs
            and all(value == SCORE_COMPARISONS for value in comparison_sets)
            and bool(paired_values.size)
            and np.isfinite(paired_values).all()
        )
        _record(
            checks,
            "score_homogeneity_paired_complete",
            paired_complete,
            (
                f"rows={len(score_paired)}, "
                f"expected_rows={expected_pair_rows}, "
                f"comparisons={sorted(SCORE_COMPARISONS)}"
            ),
        )

        latest_point_mtime = max(
            (path.stat().st_mtime_ns for path in point_paths),
            default=0,
        )
        score_fresh = all(
            path.stat().st_mtime_ns >= latest_point_mtime
            for path in score_outputs
        )
        _record(
            checks,
            "score_homogeneity_outputs_fresh",
            score_fresh,
            "score outputs are no older than the latest per-point file",
        )

    figure_qa_path = output_dir / "figures" / "figure_qa.json"
    figure_status = None
    if figure_qa_path.exists():
        figure_status = json.loads(figure_qa_path.read_text()).get("status")
    _record(
        checks,
        "figure_qa_passes",
        figure_status == "pass",
        f"path={_display_path(figure_qa_path)}, status={figure_status}",
    )
    if figure_qa_path.exists() and all(
        path.exists() for path in score_outputs
    ):
        plot_inputs = [
            summary_path,
            paired_path,
            *score_outputs,
        ]
        latest_plot_input = max(
            path.stat().st_mtime_ns
            for path in plot_inputs
            if path.exists()
        )
        _record(
            checks,
            "figure_outputs_fresh",
            figure_qa_path.stat().st_mtime_ns >= latest_plot_input,
            "figure QA is no older than all summary and score inputs",
        )

    main_budget = max(
        (int(value) for value in config.get("budgets", [])),
        default=0,
    )
    artifact_paths = [
        manifest_path,
        summary_path,
        paired_path,
        scale_summary_path,
        output_dir / f"main_table_B{main_budget}.tex",
        *score_outputs,
        figure_qa_path,
        *[
            output_dir / "figures" / f"{stem}.pdf"
            for stem in FINAL_FIGURE_STEMS
        ],
        *[
            output_dir / "plot_data" / f"{stem}.csv"
            for stem in FINAL_FIGURE_STEMS
        ],
    ]
    missing_artifacts = [
        _display_path(path) for path in artifact_paths if not path.is_file()
    ]
    _record(
        checks,
        "final_asset_set_complete",
        not missing_artifacts,
        (
            f"verified_artifacts={len(artifact_paths)}"
            if not missing_artifacts
            else f"missing={missing_artifacts}"
        ),
    )
    artifact_sha256 = {
        _display_path(path): _sha256(path)
        for path in artifact_paths
        if path.is_file()
    }
    errors = [
        check
        for check in checks
        if not check["passed"] and check["severity"] == "error"
    ]
    warnings = [
        check
        for check in checks
        if not check["passed"] and check["severity"] == "warning"
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": _display_path(output_dir),
        "status": "pass" if not errors else "fail",
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "artifact_sha256": artifact_sha256,
        "checks": checks,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Final Adaptive-h Benchmark QA",
        "",
        f"- Status: **{report.get('status', 'fail').upper()}**",
        f"- Errors: {report.get('n_errors', 0)}",
        f"- Warnings: {report.get('n_warnings', 0)}",
        "",
        "| Check | Result | Severity | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for check in report.get("checks", []):
        result = "PASS" if check["passed"] else "FAIL"
        detail = str(check["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {check['name']} | {result} | {check['severity']} | {detail} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", "--output_dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--require-final", action="store_true")
    parser.add_argument("--max-grid-clip-rate", type=float, default=0.02)
    args = parser.parse_args()
    output_dir = _resolve(args.output_dir)
    report = run_qa(
        output_dir,
        require_final=args.require_final,
        max_grid_clip_rate=args.max_grid_clip_rate,
    )
    markdown_path = output_dir / "qa_report.md"
    markdown_path.write_text(_markdown(report))
    report["artifact_sha256"][_display_path(markdown_path)] = _sha256(
        markdown_path
    )
    (output_dir / "qa_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(_markdown(report))
    if report.get("status") != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
