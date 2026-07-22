#!/usr/bin/env python3
"""Export a small, auditable set of adaptive-h manuscript assets.

The allowlist is intentionally explicit. It contains only the non-IQR Exp1--3
snapshot artifacts used by the paper-facing adaptive-h write-up. Exp4 is not
included because the final sample-SD/NW run is still pending, and retired IQR
response-scale artifacts are never eligible for export.

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
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


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
        "evidence_scope": "Non-IQR Exp1-Exp3 manuscript snapshot assets",
        "final_protocol_evidence": False,
        "explicit_exclusions": [
            "All Exp4 artifacts until the locked-protocol sample-SD/NW run exists",
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
