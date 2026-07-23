"""Quality-assurance checks for the final adaptive-h benchmark outputs."""
from __future__ import annotations

import argparse
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
DEFAULT_OUTPUT = Path(__file__).with_name("output_final_adaptive_h")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (_ROOT / path).resolve()


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
        _record(checks, "manifest_exists", False, str(manifest_path))
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
        _record(checks, "per_arm_exists", False, str(per_arm_path))
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
            max_clip = max(
                max_clip,
                float(frame["L_at_grid_lo"].mean()),
                float(frame["U_at_grid_hi"].mean()),
            )
            max_outside = max(
                max_outside, float(1.0 - frame["y_in_grid"].mean())
            )
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
            f"threshold={max_grid_clip_rate:.4f}"
        ),
        severity="warning",
    )

    summary_path = output_dir / "summary.csv"
    paired_path = output_dir / "paired_deltas.csv"
    _record(checks, "summary_exists", summary_path.exists(), str(summary_path))
    _record(checks, "paired_deltas_exists", paired_path.exists(), str(paired_path))
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
        "output_dir": str(output_dir),
        "status": "pass" if not errors else "fail",
        "n_errors": len(errors),
        "n_warnings": len(warnings),
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
    (output_dir / "qa_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "qa_report.md").write_text(_markdown(report))
    print(_markdown(report))
    if report.get("status") != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
