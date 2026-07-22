"""Compatibility-only entry point for the sample-SD/NW Exp 4 summary.

Canonical module: ``experiments.adaptive_h.summarize_exp4_sample_sd_nw``.
"""

from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for import_path in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


if __name__ == "__main__":
    runpy.run_module(
        "experiments.adaptive_h.summarize_exp4_sample_sd_nw",
        run_name="__main__",
    )
