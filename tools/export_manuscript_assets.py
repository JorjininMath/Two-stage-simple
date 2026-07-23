#!/usr/bin/env python3
"""Export a small, auditable set of adaptive-h evidence assets.

The allowlist is intentionally explicit. It contains the historical non-IQR
Exp1--3 mechanism snapshot plus the compact, QA-checked products of the final
iid-calibrated sample-SD/NW benchmark. Retired IQR response-scale artifacts are
never eligible for export.

Each copy is hash-checked and recorded in a JSON provenance manifest. Relative
paths are resolved from the repository root, so the script does not embed a
machine-specific absolute path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "manuscript/generated/adaptive_h_assets_manifest.json"
FINAL_QA_PATH = (
    REPO_ROOT
    / "experiments/adaptive_h/output_final_adaptive_h/qa_report.json"
)


@dataclass(frozen=True)
class Asset:
    source: str
    destination: str
    evidence_id: str
    role: str


# Keep this list small and reviewable. Adding an asset is a claim-management
# decision, not a recursive file-copy operation.
ASSETS = (
    Asset(
        source="experiments/adaptive_h/output_exp1/exp1_table.tex",
        destination="manuscript/generated/tables/adaptive_h/exp1_fixed_reference_table.tex",
        evidence_id="adaptive_h_exp1_fixed_reference",
        role="Fixed-bandwidth mechanism reference table",
    ),
    Asset(
        source="experiments/adaptive_h/output_exp1/exp1_coverage_curves.png",
        destination="manuscript/generated/figures/adaptive_h/exp1_fixed_reference_coverage.png",
        evidence_id="adaptive_h_exp1_fixed_reference",
        role="Fixed-bandwidth binwise coverage reference",
    ),
    Asset(
        source="experiments/adaptive_h/output_exp2/exp2_table.tex",
        destination="manuscript/generated/tables/adaptive_h/exp2_oracle_vs_fixed_table.tex",
        evidence_id="adaptive_h_exp2_oracle_vs_fixed",
        role="Oracle versus fixed paired comparison table",
    ),
    Asset(
        source="experiments/adaptive_h/output_exp2/exp2_coverage_curves.png",
        destination="manuscript/generated/figures/adaptive_h/exp2_oracle_vs_fixed_coverage.png",
        evidence_id="adaptive_h_exp2_oracle_vs_fixed",
        role="Oracle versus fixed binwise coverage curves",
    ),
    Asset(
        source="experiments/adaptive_h/output_exp3/exp3_table.tex",
        destination="manuscript/generated/tables/adaptive_h/exp3_oracle_c_sweep_table.tex",
        evidence_id="adaptive_h_exp3_multiplier_sweep",
        role="Oracle bandwidth-multiplier sensitivity table",
    ),
    Asset(
        source="experiments/adaptive_h/output_exp3/exp3_metric_vs_c.png",
        destination="manuscript/generated/figures/adaptive_h/exp3_oracle_c_sweep_metrics.png",
        evidence_id="adaptive_h_exp3_multiplier_sweep",
        role="Oracle bandwidth-multiplier sensitivity figure",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/main_table_B1000.tex",
        destination="manuscript/generated/tables/adaptive_h/final_adaptive_h_B1000_table.tex",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final B=1000 fixed, sample-SD/NW, and oracle comparison",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/figures/scale_functions.pdf",
        destination="manuscript/generated/figures/adaptive_h/final_scale_functions.pdf",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final benchmark response-scale functions",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/figures/plugin_oracle_budget_gap.pdf",
        destination="manuscript/generated/figures/adaptive_h/final_plugin_oracle_budget_gap.pdf",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Paired plug-in minus oracle budget trends",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/figures/binwise_coverage.pdf",
        destination="manuscript/generated/figures/adaptive_h/final_binwise_coverage.pdf",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final raw-score binwise coverage",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/figures/effective_bandwidth_ratio.pdf",
        destination="manuscript/generated/figures/adaptive_h/final_effective_bandwidth_ratio.pdf",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final effective bandwidth ratio diagnostic",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/figures/raw_score_homogeneity.pdf",
        destination="manuscript/generated/figures/adaptive_h/final_raw_score_homogeneity.pdf",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final raw-score homogeneity diagnostic",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/plot_data/scale_functions.csv",
        destination="analysis/adaptive_h/plot_data/final_scale_functions.csv",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Compact source data for the final response-scale figure",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/plot_data/plugin_oracle_budget_gap.csv",
        destination="analysis/adaptive_h/plot_data/final_plugin_oracle_budget_gap.csv",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Compact source data for the paired budget-trend figure",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/plot_data/binwise_coverage.csv",
        destination="analysis/adaptive_h/plot_data/final_binwise_coverage.csv",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Compact source data for the final binwise-coverage figure",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/plot_data/effective_bandwidth_ratio.csv",
        destination="analysis/adaptive_h/plot_data/final_effective_bandwidth_ratio.csv",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Compact source data for the effective-bandwidth-ratio figure",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/plot_data/raw_score_homogeneity.csv",
        destination="analysis/adaptive_h/plot_data/final_raw_score_homogeneity.csv",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Compact source data for the raw-score-homogeneity figure",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/manifest.json",
        destination="analysis/adaptive_h/final_run_manifest.json",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final scientific configuration and source provenance",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/summary.csv",
        destination="analysis/adaptive_h/final_summary.csv",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final arm-level means, standard deviations, and Monte Carlo SEs",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/paired_deltas.csv",
        destination="analysis/adaptive_h/final_paired_deltas.csv",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final macrorep-paired method differences",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/scale_diagnostics_summary.csv",
        destination="analysis/adaptive_h/final_scale_diagnostics_summary.csv",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final sample-SD/NW scale-learning diagnostics by budget",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/score_homogeneity_summary.csv",
        destination="analysis/adaptive_h/final_score_homogeneity_summary.csv",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final raw-score distribution homogeneity summary",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/score_homogeneity_paired_deltas.csv",
        destination="analysis/adaptive_h/final_score_homogeneity_paired_deltas.csv",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final paired raw-score homogeneity differences",
    ),
    Asset(
        source="experiments/adaptive_h/output_final_adaptive_h/qa_report.md",
        destination="analysis/adaptive_h/final_qa_report.md",
        evidence_id="adaptive_h_final_sample_sd_nw",
        role="Final benchmark QA report",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def qa_artifact_hashes_match(
    report: dict[str, object],
    *,
    assets: tuple[Asset, ...] = ASSETS,
    repo_root: Path = REPO_ROOT,
) -> bool:
    if report.get("status") != "pass":
        return False
    verified_hashes = report.get("artifact_sha256", {})
    if not isinstance(verified_hashes, dict):
        return False
    final_assets = [
        asset
        for asset in assets
        if asset.evidence_id == "adaptive_h_final_sample_sd_nw"
    ]
    for asset in final_assets:
        source = repo_root / asset.source
        if not source.is_file():
            return False
        if verified_hashes.get(asset.source) != sha256(source):
            return False
    return True


def final_qa_passed() -> bool:
    if not FINAL_QA_PATH.is_file():
        return False
    try:
        report = json.loads(FINAL_QA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return qa_artifact_hashes_match(report)


def expected_record(asset: Asset) -> dict[str, object]:
    source = REPO_ROOT / asset.source
    destination = REPO_ROOT / asset.destination
    stat = source.stat()
    return {
        "evidence_id": asset.evidence_id,
        "role": asset.role,
        "source": asset.source,
        "destination": asset.destination,
        "source_modified_at": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "size_bytes": stat.st_size,
        "sha256": sha256(source),
        "destination_exists": destination.exists(),
    }


def dry_run() -> int:
    missing = []
    for asset in ASSETS:
        source = REPO_ROOT / asset.source
        marker = "OK" if source.is_file() else "MISSING"
        print(f"{marker}: {asset.source} -> {asset.destination}")
        if not source.is_file():
            missing.append(asset.source)
    if missing:
        print(f"Dry run found {len(missing)} missing source(s).", file=sys.stderr)
        return 1
    print(f"Dry run verified {len(ASSETS)} allowlisted source assets.")
    return 0


def export() -> int:
    missing = [asset.source for asset in ASSETS if not (REPO_ROOT / asset.source).is_file()]
    if missing:
        for path in missing:
            print(f"MISSING: {path}", file=sys.stderr)
        return 1

    if not final_qa_passed():
        print(
            "ERROR: Final adaptive-h QA report is missing or not pass.",
            file=sys.stderr,
        )
        return 1

    records = []
    for asset in ASSETS:
        source = REPO_ROOT / asset.source
        destination = REPO_ROOT / asset.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        record = expected_record(asset)
        destination_hash = sha256(destination)
        if destination_hash != record["sha256"]:
            raise RuntimeError(f"Hash mismatch after copying {relative(destination)}")
        record["destination_exists"] = True
        records.append(record)
        print(f"COPIED: {asset.source} -> {asset.destination}")

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": "tools/export_manuscript_assets.py",
        "evidence_scope": (
            "Non-IQR Exp1-Exp3 mechanism assets and final iid-calibrated "
            "sample-SD/NW benchmark evidence"
        ),
        "final_protocol_evidence": True,
        "explicit_exclusions": [
            "All pre-protocol Exp4 artifacts",
            "All retired IQR response-scale plug-in artifacts",
        ],
        "assets": records,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE: {relative(MANIFEST_PATH)}")
    return 0


def check() -> int:
    errors = []
    if not final_qa_passed():
        errors.append("Final adaptive-h QA report is missing or not pass.")
    if not MANIFEST_PATH.is_file():
        errors.append(f"Missing manifest: {relative(MANIFEST_PATH)}")
        manifest = {}
    else:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    manifest_records = {
        item.get("destination"): item for item in manifest.get("assets", [])
    }
    for asset in ASSETS:
        source = REPO_ROOT / asset.source
        destination = REPO_ROOT / asset.destination
        if not source.is_file():
            errors.append(f"Missing source: {asset.source}")
            continue
        if not destination.is_file():
            errors.append(f"Missing destination: {asset.destination}")
            continue
        source_hash = sha256(source)
        destination_hash = sha256(destination)
        if source_hash != destination_hash:
            errors.append(f"Stale destination: {asset.destination}")
        record = manifest_records.get(asset.destination)
        if record is None:
            errors.append(f"Missing manifest record: {asset.destination}")
        elif record.get("sha256") != source_hash:
            errors.append(f"Stale manifest hash: {asset.destination}")

    unexpected = sorted(set(manifest_records) - {asset.destination for asset in ASSETS})
    for destination in unexpected:
        errors.append(f"Manifest contains non-allowlisted destination: {destination}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"CHECK OK: {len(ASSETS)} assets match sources and manifest hashes.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify copies and manifest without writing.")
    mode.add_argument("--dry-run", action="store_true", help="Show the allowlist and verify source existence.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check:
        return check()
    if args.dry_run:
        return dry_run()
    return export()


if __name__ == "__main__":
    raise SystemExit(main())
