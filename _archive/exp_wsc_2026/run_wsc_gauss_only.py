"""
run_wsc_gauss_only.py

Standalone subset runner: wsc_gauss DGP only, single Stage-2 budget
(n_1=500, r_1=10), single method (lhs). Produces per_point.csv files in the
same on-disk format as run_wsc_compare.py so plotting code can read them
directly. Used to support a per-x coverage comparison plot of CKME against
DCP-DR and hetGP for the wsc_gauss DGP.

Usage (from project root):
    python exp_wsc/pretrain_params.py --sims wsc_gauss
    python exp_wsc/run_wsc_gauss_only.py --n_macro 11 --n_workers 4
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import pandas as pd

from CKME.parameters import Params
from exp_wsc.config_utils import get_config, load_config_from_file

DGPS_SUBSET = [("wsc_gauss", "Exp1_Gaussian")]
STAGE2_CASES_SUBSET = [(500, 10)]
METHODS_SUBSET = ["lhs"]


def _macrorep_worker(macrorep_id, base_seed, config, out_dir, n_grid, pretrained):
    """Wrapper that monkeypatches the run_wsc_compare globals before delegating.

    Required because ProcessPoolExecutor on macOS uses spawn: each worker
    re-imports the module fresh, so subset constants set in the parent process
    are NOT propagated. We re-set them here in the worker.
    """
    from exp_wsc import run_wsc_compare as base
    base.DGPS = DGPS_SUBSET
    base.STAGE2_CASES = STAGE2_CASES_SUBSET
    base.METHODS = METHODS_SUBSET
    return base.run_one_macrorep(macrorep_id, base_seed, config, out_dir, n_grid, pretrained)


def main():
    parser = argparse.ArgumentParser(
        description="WSC subset runner: wsc_gauss × (n_1=500, r_1=10) × lhs"
    )
    parser.add_argument("--config",     type=str, default="exp_wsc/config.txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_macro",    type=int, default=11)
    parser.add_argument("--base_seed",  type=int, default=42)
    parser.add_argument("--n_workers",  type=int, default=1)
    args = parser.parse_args()

    # Apply subset to the parent process too (for sequential mode).
    from exp_wsc import run_wsc_compare as base
    base.DGPS = DGPS_SUBSET
    base.STAGE2_CASES = STAGE2_CASES_SUBSET
    base.METHODS = METHODS_SUBSET

    config = get_config(load_config_from_file(_root / args.config), quick=False)
    n_grid = config.get("t_grid_size", 500)
    out_dir = (
        Path(args.output_dir) if args.output_dir
        else _root / "exp_wsc" / "output"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pretrained_path = _root / "exp_wsc" / "pretrained_params.json"
    if not pretrained_path.exists():
        print(
            f"ERROR: {pretrained_path} not found. Run "
            "'python exp_wsc/pretrain_params.py --sims wsc_gauss' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(pretrained_path) as f:
        raw = json.load(f)
    if "wsc_gauss" not in raw:
        print(
            "ERROR: pretrained_params.json does not contain wsc_gauss; "
            "run 'python exp_wsc/pretrain_params.py --sims wsc_gauss' first.",
            file=sys.stderr,
        )
        sys.exit(1)
    pretrained = {"wsc_gauss": Params(**raw["wsc_gauss"])}
    print(f"Loaded pretrained params for wsc_gauss: {raw['wsc_gauss']}")

    if not base.R_SCRIPT.exists():
        print(
            f"ERROR: R script not found at {base.R_SCRIPT}; needed for "
            "DCP-DR/hetGP benchmarks.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"Subset: DGPS={DGPS_SUBSET}, STAGE2_CASES={STAGE2_CASES_SUBSET}, "
        f"METHODS={METHODS_SUBSET}"
    )
    print(f"n_macro={args.n_macro}, n_workers={args.n_workers}, output_dir={out_dir}")

    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futs = {
                pool.submit(
                    _macrorep_worker, k, args.base_seed, config, out_dir, n_grid, pretrained
                ): k
                for k in range(args.n_macro)
            }
            result_map: dict[int, list] = {}
            for fut in as_completed(futs):
                k = futs[fut]
                result_map[k] = fut.result()
                print(f"  macrorep {k} done")
        all_rows = [row for k in range(args.n_macro) for row in result_map[k]]
    else:
        all_rows = []
        for k in range(args.n_macro):
            print(f"  macrorep {k} ...")
            all_rows.extend(
                _macrorep_worker(k, args.base_seed, config, out_dir, n_grid, pretrained)
            )

    per_rep = pd.DataFrame(all_rows)
    per_rep.to_csv(out_dir / "wsc_gauss_only_per_macrorep.csv", index=False)
    print(f"\nWrote {out_dir / 'wsc_gauss_only_per_macrorep.csv'}")


if __name__ == "__main__":
    main()
