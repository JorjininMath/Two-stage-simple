"""
Saturation sweep: Δ IS = IS(lhs) − IS(adaptive_tail) vs B_total.

Produces the main-paper "Adaptive Gain Curve" for Topic 1 (Allocation
Design). For a given DGP, sweeps total budget `B_total`, splits it
balanced (B1:B2 = 1:1), and runs both LHS and S^0-tail-adaptive site
selection. Output: one summary.csv per DGP with columns
(B_total, label, coverage, width, interval_score, ...).

Split convention (balanced, "5/5"):
  n_0 * r_0 = n_1 * r_1 = B_total / 2
  r_0 = r_1 = r_fixed (default 10)
  => n_0 = n_1 = B_total / (2 * r_fixed)

Usage (from project root):
  # Test on D2 (exp2 Student-t heavy, nongauss_A1L_raw) with 5 macroreps
  python exp_design/run_saturation_sweep.py \\
      --dgp nongauss_A1L_raw \\
      --b_total_list 500 1000 2000 5000 10000 \\
      --n_macro 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd

from CKME.parameters import Params
from exp_design.run_design_compare import (
    load_dgp,
    run_one_macrorep,
    _register_dgp,
    PARAM_GRID,
)

MIXED_RATIO = 0.7


def build_cases_saturation(
    b_total_list: list[int],
    r_fixed: int,
    methods: list[tuple[str, str]],
) -> list[dict]:
    """For each B_total, balanced split; for each method, emit a case."""
    cases = []
    for B in b_total_list:
        half = B // 2
        if half % r_fixed != 0:
            raise ValueError(
                f"B_total/2={half} not divisible by r_fixed={r_fixed}. "
                f"Choose B_total such that B_total/(2*r) is integer."
            )
        n_per_stage = half // r_fixed
        for method, s0_type in methods:
            if method == "lhs":
                mlabel = "lhs"
            else:
                mlabel = f"{method}_{s0_type}"
            cases.append({
                "n_0": n_per_stage, "r_0": r_fixed,
                "n_1": n_per_stage, "r_1": r_fixed,
                "method": method,
                "s0_score_type": s0_type,
                "label": f"B{B}_{mlabel}",
                "B_total": B,
            })
    return cases


def main():
    parser = argparse.ArgumentParser(description="Saturation sweep for Adaptive Gain Curve")
    parser.add_argument("--dgp", type=str, required=True,
                        help="DGP name (registered in Two_stage or exp_design._LOCAL_DGPS)")
    parser.add_argument("--b_total_list", type=int, nargs="+",
                        default=[500, 1000, 2000, 5000, 10000],
                        help="Total budget values to sweep")
    parser.add_argument("--r_fixed", type=int, default=10,
                        help="Fixed replication count (r_0 = r_1 = r_fixed)")
    parser.add_argument("--n_macro", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--methods", type=str, nargs="+",
        default=["lhs:tail", "sampling:tail"],
        help="Methods to compare, format 'method:s0_type'. Default: lhs vs adaptive_tail.",
    )
    args = parser.parse_args()

    # Parse methods
    methods_parsed: list[tuple[str, str]] = []
    for m in args.methods:
        if ":" in m:
            method, s0 = m.split(":", 1)
        else:
            method, s0 = m, "tail"
        methods_parsed.append((method, s0))

    # Load DGP
    dgp_config = load_dgp(args.dgp)
    _register_dgp(args.dgp, dgp_config)

    # Build cases
    cases = build_cases_saturation(args.b_total_list, args.r_fixed, methods_parsed)

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent / "output" / "saturation" / args.dgp

    alpha = 0.1
    n_test = 1000
    n_cand = 1000
    t_grid_size = 500

    # Pretrained params
    params_file = Path(__file__).parent / "pretrained_params.json"
    params = None
    if params_file.exists():
        with open(params_file) as f:
            all_params = json.load(f)
        if args.dgp in all_params:
            p = all_params[args.dgp]
            params = Params(ell_x=p["ell_x"], lam=p["lam"], h=p["h"])
            print(f"Using pretrained params for {args.dgp}: {params}")
        else:
            print(f"No pretrained params for {args.dgp}; CV tuning per macrorep "
                  f"with grid {PARAM_GRID}")

    print(f"Saturation sweep | DGP: {args.dgp} | "
          f"B_total: {args.b_total_list} | r_fixed={args.r_fixed} | "
          f"{len(cases)} cases | {args.n_macro} macroreps")
    for c in cases:
        print(f"  {c['label']}: n0={c['n_0']} r0={c['r_0']} "
              f"n1={c['n_1']} r1={c['r_1']} method={c['method']}/{c['s0_score_type']}")

    all_rows = []
    for k in range(args.n_macro):
        print(f"  macrorep {k}/{args.n_macro} ...", flush=True)
        rows = run_one_macrorep(
            k, args.base_seed, args.dgp, dgp_config, cases,
            alpha, n_test, n_cand, t_grid_size, out_dir, params,
        )
        all_rows.extend(rows)

    # Save full summary
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "summary.csv", index=False)

    # Derive variant from label (e.g., "B500_lhs" -> "lhs",
    # "B500_sampling_tail" -> "sampling_tail")
    df["variant"] = df["label"].str.split("_", n=1).str[1]

    # Aggregate over macroreps per (B_total, variant)
    agg = df.groupby(["B_total", "variant", "method"]).agg(
        coverage=("coverage", "mean"),
        coverage_se=("coverage", "sem"),
        width=("width", "mean"),
        width_se=("width", "sem"),
        IS=("interval_score", "mean"),
        IS_se=("interval_score", "sem"),
    ).round(4).reset_index()
    agg.to_csv(out_dir / "summary_agg.csv", index=False)

    # Δ IS = IS(lhs) − IS(variant) per B_total, for each non-lhs variant
    pivot = agg.pivot_table(
        index="B_total", columns="variant", values="IS", aggfunc="first",
    )
    if "lhs" in pivot.columns:
        # SE pivot for ± band on Δ IS
        se_pivot = agg.pivot_table(
            index="B_total", columns="variant", values="IS_se", aggfunc="first",
        )
        for col in list(pivot.columns):
            if col == "lhs":
                continue
            pivot[f"delta_{col}"] = pivot["lhs"] - pivot[col]
            # Independent-macrorep SE assumption
            pivot[f"delta_{col}_se"] = np.sqrt(
                se_pivot["lhs"] ** 2 + se_pivot[col] ** 2
            )
        print("\n=== Adaptive Gain Curve (Δ IS = IS_lhs − IS_variant, ± SE) ===")
        print(pivot.round(4).to_string())
        pivot.to_csv(out_dir / "delta_is_curve.csv")

    print("\n=== Per-case aggregate ===")
    print(agg.to_string(index=False))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
