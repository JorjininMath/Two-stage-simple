#!/usr/bin/env python3
"""Export stable PROJECT_STATUS.md blocks as one-way Career OS JSON.

This script never edits Career OS or PROJECT_STATUS.md.  It gives an external
tool a small, versioned interface while the Markdown page remains authoritative.
By default JSON is written to stdout; ``--output`` may write a project-relative
or absolute file when a caller explicitly wants a handoff artifact.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = PROJECT_ROOT / "PROJECT_STATUS.md"
PROJECT_ID = "wk_ckme_ext"
FIELD_NAMES = (
    "latestResult",
    "bottleneck",
    "nextAction",
    "advisorAsk",
    "nextDecision",
)


def extract_status(markdown: str) -> dict[str, object]:
    """Parse the five stable marker blocks and the page's update date."""

    fields: dict[str, str] = {}
    for name in FIELD_NAMES:
        pattern = re.compile(
            rf"<!-- career-os:{name}:start -->\s*(.*?)\s*"
            rf"<!-- career-os:{name}:end -->",
            re.DOTALL,
        )
        matches = pattern.findall(markdown)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one career-os:{name} marker block; "
                f"found {len(matches)}"
            )
        value = matches[0].strip()
        if not value:
            raise ValueError(f"career-os:{name} marker block is empty")
        fields[name] = value

    date_match = re.search(r"^Last updated:\s*(\d{4}-\d{2}-\d{2})\s*$", markdown, re.MULTILINE)
    if date_match is None:
        raise ValueError("PROJECT_STATUS.md has no valid 'Last updated' date")

    return {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "source": "PROJECT_STATUS.md",
        "last_updated": date_match.group(1),
        "fields": fields,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate markers without emitting the JSON payload.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path; relative paths use the project root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = extract_status(STATUS_PATH.read_text(encoding="utf-8"))
    if args.check:
        print(f"CHECK OK: {len(FIELD_NAMES)} Career OS status blocks")
        return 0

    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
        return 0

    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"WROTE: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
