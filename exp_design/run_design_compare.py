"""
exp_design: Stage design experiments for CKME-DCP.

Studies budget allocation, site selection, and Stage 1/2 ratio.
Uses DGPs from examples/ folder.

Usage (from project root):
  # Exp A: n_1 vs r_1 allocation (default)
  python exp_design/run_design_compare.py --exp A --dgp exp2_gauss_low --n_macro 5

  # Exp B: S^0 vs LHS
  python exp_design/run_design_compare.py --exp B --dgp exp2_gauss_high --n_macro 20

  # Exp C: B1/B2 ratio
  python exp_design/run_design_compare.py --exp C --dgp exp2_gauss_low --n_macro 20
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

import numpy as np
import pandas as pd

from CKME.parameters import Params, ParamGrid
from Two_stage import run_stage1_train, run_stage2
from Two_stage.evaluation import evaluate_per_point
from Two_stage.test_data import generate_test_data
from Two_stage.design import generate_space_filling_design

# ---------------------------------------------------------------------------
# DGP loading from examples/
# ---------------------------------------------------------------------------

def _make_a1_raw(nu: float) -> dict:
    """A1-Student-t without variance normalization (scale = sigma_tar(x) directly).

    Unlike Two_stage.sim_functions.sim_nongauss_A1 which enforces Var(Y|x)=sigma_tar^2
    via sqrt((nu-2)/nu), this variant lets variance grow with heavy tails:
      nu=3  -> Var = 3 * sigma_tar^2
      nu=10 -> Var = 1.25 * sigma_tar^2
    Local to exp_design only; does not affect exp_nongauss results.
    """
    from Two_stage.sim_functions.exp2 import exp2_true_function, EXP2_X_BOUNDS

    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.asarray(x, dtype=float)
        sigma_tar = 0.1 + 0.1 * (x - np.pi) ** 2
        noise = rng.standard_t(df=nu, size=x.shape) * sigma_tar
        return exp2_true_function(x) + noise

    return {"simulator": simulator, "bounds": EXP2_X_BOUNDS, "d": 1}


def _make_gibbs_s1_d(d: int) -> dict:
    """RLCP Setting 1 extended to d dims: X in [-3,3]^d.

      Y = 0.5 * mean(X) + |sin(X_1)| * N(0,1)

    sigma depends only on X_1 → irrelevant directions X_2,...,X_d let us test
    whether S^0-guided sampling actually concentrates budget in the
    informative direction.
    """
    lo = np.full(d, -3.0)
    hi = np.full(d, 3.0)

    def simulator(x, random_state=None):
        rng = np.random.default_rng(random_state)
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != d:
            x = x.reshape(-1, d)
        sigma = np.abs(np.sin(x[:, 0]))
        return 0.5 * x.mean(axis=1) + sigma * rng.standard_normal(x.shape[0])

    return {"simulator": simulator, "bounds": (lo, hi), "d": d}


_LOCAL_DGPS = {
    "nongauss_A1S_raw": _make_a1_raw(10.0),
    "nongauss_A1L_raw": _make_a1_raw(3.0),
    "gibbs_s1_d5": _make_gibbs_s1_d(5),
}


def load_dgp(dgp_name: str) -> dict:
    """Load DGP from examples/ or Two_stage registry. Returns dict with simulator, bounds, d."""
    from examples import exp2_gauss
    from Two_stage.sim_functions import _EXPERIMENT_REGISTRY
    all_dgps = {}
    all_dgps.update(_EXPERIMENT_REGISTRY)
    all_dgps.update(exp2_gauss.REGISTRY)
    all_dgps.update(_LOCAL_DGPS)
    if dgp_name not in all_dgps:
        raise ValueError(f"Unknown DGP: {dgp_name}. Available: {list(all_dgps)}")
    return all_dgps[dgp_name]


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

# Exp A: Stage 2 internal allocation (n_1 vs r_1), fixed B_2 = 5000
EXP_A_STAGE1 = {"n_0": 100, "r_0": 10}
EXP_A_ALLOCATIONS = [(100, 50), (250, 20), (500, 10), (1000, 5)]
EXP_A_METHOD = "mixed"

# Exp B: S^0 vs LHS
EXP_B_STAGE1 = {"n_0": 100, "r_0": 10}
EXP_B_STAGE2 = {"n_1": 500, "r_1": 10}
EXP_B_METHODS = [
    ("lhs", "tail"),            # LHS (S^0 unused)
    ("sampling", "tail"),       # S^0 = tail quantile width
    ("sampling", "variance"),   # S^0 = sqrt(Var[Y|x]) via c(x)-weighted moments
    ("sampling", "epistemic"),  # S^0 = bootstrap CDF variance (K=30, fixed params)
]

# Exp C: B1/B2 ratio, fixed total ~ 5000
EXP_C_CONFIGS = [
    {"n_0": 400, "r_0": 10, "n_1": 200, "r_1": 5, "label": "more_s1"},
    {"n_0": 250, "r_0": 10, "n_1": 500, "r_1": 5, "label": "balanced"},
    {"n_0": 150, "r_0": 10, "n_1": 700, "r_1": 5, "label": "more_s2"},
]
EXP_C_METHOD = "mixed"

MIXED_RATIO = 0.7

# CV grid for hyperparameter tuning
PARAM_GRID = ParamGrid(
    ell_x_list=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0],
    lam_list=[1e-5, 1e-3, 1e-1],
    h_list=[0.05, 0.1, 0.2],
)


# ---------------------------------------------------------------------------
# Core: one macrorep
# ---------------------------------------------------------------------------

def run_one_macrorep(
    macrorep_id: int,
    base_seed: int,
    dgp_name: str,
    dgp_config: dict,
    cases: list[dict],
    alpha: float,
    n_test: int,
    n_cand: int,
    t_grid_size: int,
    out_dir: Path,
    params: Params | None = None,
) -> list[dict]:
    """Run one macrorep for all cases. Stage 1 is shared when n_0/r_0 match."""
    seed = base_seed + macrorep_id * 10000
    rng = np.random.default_rng(seed)

    simulator = dgp_config["simulator"]
    bounds = dgp_config["bounds"]
    d = dgp_config["d"]

    # Candidate pool for S^0 and test
    X_cand = generate_space_filling_design(
        n=n_cand, d=d, method="lhs", bounds=bounds, random_state=seed + 1,
    )

    # Cache Stage 1 results by (n_0, r_0) to avoid retraining
    stage1_cache = {}
    rows = []

    for case in cases:
        n_0, r_0 = case["n_0"], case["r_0"]
        n_1, r_1 = case["n_1"], case["r_1"]
        method = case["method"]
        s0_type = case.get("s0_score_type", "tail")
        label = case["label"]

        # Train Stage 1 (cached)
        s1_key = (n_0, r_0)
        if s1_key not in stage1_cache:
            stage1 = _train_stage1(
                dgp_name, dgp_config, n_0, r_0, t_grid_size,
                seed + 2, params,
            )
            stage1_cache[s1_key] = stage1
        stage1 = stage1_cache[s1_key]

        # Stage 2
        stage2 = run_stage2(
            stage1_result=stage1,
            X_cand=X_cand,
            n_1=n_1, r_1=r_1,
            simulator_func=dgp_name,
            method=method,
            alpha=alpha,
            mixed_ratio=MIXED_RATIO,
            random_state=seed + 3,
            verbose=False,
            s0_score_type=s0_type,
        )

        # Test data
        X_test = generate_space_filling_design(
            n=n_test, d=d, method="lhs", bounds=bounds, random_state=seed + 7,
        )
        Y_test = simulator(X_test.ravel() if d == 1 else X_test, random_state=seed + 8)

        # Evaluate
        eval_result = evaluate_per_point(stage2, X_test, Y_test)
        point_rows = eval_result["rows"]

        # Save per-point CSV
        case_dir = out_dir / f"macrorep_{macrorep_id}" / f"case_{label}"
        case_dir.mkdir(parents=True, exist_ok=True)
        df_points = pd.DataFrame(point_rows)
        df_points.to_csv(case_dir / "per_point.csv", index=False)

        # Aggregate metrics
        summary = {
            "dgp": dgp_name, "label": label,
            "n_0": n_0, "r_0": r_0, "n_1": n_1, "r_1": r_1,
            "method": method, "macrorep": macrorep_id,
            "B_train": n_0 * r_0, "B_cal": n_1 * r_1,
            "B_total": n_0 * r_0 + n_1 * r_1,
            "coverage": df_points["covered_score"].mean(),
            "width": df_points["width"].mean(),
            "interval_score": df_points["interval_score"].mean(),
        }

        # Group coverage (K=10 bins)
        if d == 1 and "x0" in df_points.columns:
            bins = pd.cut(df_points["x0"], bins=10)
            bin_cov = df_points.groupby(bins, observed=True)["covered_score"].mean()
            summary["max_bin_dev"] = float((bin_cov - (1 - alpha)).abs().max())
        rows.append(summary)

    return rows


def _train_stage1(dgp_name, dgp_config, n_0, r_0, t_grid_size, seed, params):
    """Train Stage 1. Uses examples/ simulator via custom data collection."""
    simulator = dgp_config["simulator"]
    bounds = dgp_config["bounds"]
    d = dgp_config["d"]

    # Generate design sites
    X_0 = generate_space_filling_design(
        n=n_0, d=d, method="lhs", bounds=bounds, random_state=seed,
    )

    # Collect replications
    X_all_list, Y_all_list = [], []
    for i in range(n_0):
        xi = X_0[i]
        xi_rep = np.tile(xi, (r_0, 1)) if d > 1 else np.full(r_0, xi.item())
        yi = simulator(xi_rep, random_state=seed + 100 + i)
        X_all_list.append(xi_rep.reshape(r_0, -1) if d > 1 else xi_rep.reshape(r_0, 1))
        Y_all_list.append(yi.ravel())

    X_all = np.vstack(X_all_list)
    Y_all = np.concatenate(Y_all_list)

    # Build t_grid: 1%/99% percentile base + 50% margin to cover heavy tails
    # (Student-t ν=3 and similar DGPs have extreme values that exceed the
    # empirical Y range at moderate n_0*r_0; the extended margin ensures
    # F(t_hi|x) ≥ 1-α/2 so quantile-based S^0 doesn't degenerate.)
    Y_lo = np.percentile(Y_all, 1.0)
    Y_hi = np.percentile(Y_all, 99.0)
    y_margin = 0.50 * (Y_hi - Y_lo)
    t_grid = np.linspace(Y_lo - y_margin, Y_hi + y_margin, t_grid_size)

    # Train CKME
    from CKME import CKMEModel
    from CKME.tuning import tune_ckme_params
    if params is not None:
        model = CKMEModel(indicator_type="logistic")
        model.fit(X_all, Y_all, params=params, r=r_0)
    else:
        best_params, _ = tune_ckme_params(
            X_all, Y_all, param_grid=PARAM_GRID, t_grid=t_grid,
            cv_folds=5, random_state=seed,
        )
        model = CKMEModel(indicator_type="logistic")
        model.fit(X_all, Y_all, params=best_params, r=r_0)
        params = best_params

    from Two_stage.stage1_train import Stage1TrainResult
    return Stage1TrainResult(
        model=model, t_grid=t_grid, X_0=X_0,
        X_all=X_all, Y_all=Y_all, params=params,
        n_0=n_0, r_0=r_0, d=d,
    )


# ---------------------------------------------------------------------------
# Build case lists for each experiment
# ---------------------------------------------------------------------------

def build_cases_A() -> list[dict]:
    cases = []
    for n_1, r_1 in EXP_A_ALLOCATIONS:
        cases.append({
            "n_0": EXP_A_STAGE1["n_0"], "r_0": EXP_A_STAGE1["r_0"],
            "n_1": n_1, "r_1": r_1,
            "method": EXP_A_METHOD,
            "label": f"n1_{n_1}_r1_{r_1}",
        })
    return cases


def build_cases_B() -> list[dict]:
    cases = []
    for method, s0_type in EXP_B_METHODS:
        if method == "lhs":
            label = "method_lhs"
        else:
            label = f"method_{method}_{s0_type}"
        cases.append({
            "n_0": EXP_B_STAGE1["n_0"], "r_0": EXP_B_STAGE1["r_0"],
            "n_1": EXP_B_STAGE2["n_1"], "r_1": EXP_B_STAGE2["r_1"],
            "method": method,
            "s0_score_type": s0_type,
            "label": label,
        })
    return cases


def build_cases_C() -> list[dict]:
    cases = []
    for cfg in EXP_C_CONFIGS:
        cases.append({
            "n_0": cfg["n_0"], "r_0": cfg["r_0"],
            "n_1": cfg["n_1"], "r_1": cfg["r_1"],
            "method": EXP_C_METHOD,
            "label": cfg["label"],
        })
    return cases


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage design experiments")
    parser.add_argument("--exp", type=str, default="A", choices=["A", "B", "C"],
                        help="Experiment: A=n1 vs r1, B=S0 vs LHS, C=B1/B2 ratio")
    parser.add_argument("--dgp", type=str, default="exp2_gauss_low",
                        help="DGP name from examples/")
    parser.add_argument("--n_macro", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_0", type=int, default=None,
                        help="Override Stage-1 n_0 (defaults: ExpA/B=100, ExpC=per-config)")
    parser.add_argument("--r_0", type=int, default=None,
                        help="Override Stage-1 r_0")
    args = parser.parse_args()

    dgp_config = load_dgp(args.dgp)
    # Register DGP in Two_stage so run_stage2's simulator lookup works
    _register_dgp(args.dgp, dgp_config)

    cases = {"A": build_cases_A, "B": build_cases_B, "C": build_cases_C}[args.exp]()

    if args.n_0 is not None or args.r_0 is not None:
        for c in cases:
            if args.n_0 is not None:
                c["n_0"] = args.n_0
            if args.r_0 is not None:
                c["r_0"] = args.r_0
            lbl = c["label"]
            c["label"] = f"{lbl}_n0{c['n_0']}r0{c['r_0']}"

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(__file__).parent / "output" / f"exp{args.exp}" / args.dgp

    alpha = 0.1
    n_test = 1000
    n_cand = 1000
    t_grid_size = 500

    # Load pretrained params if available
    params_file = Path(__file__).parent / "pretrained_params.json"
    params = None
    if params_file.exists():
        with open(params_file) as f:
            all_params = json.load(f)
        if args.dgp in all_params:
            p = all_params[args.dgp]
            params = Params(ell_x=p["ell_x"], lam=p["lam"], h=p["h"])
            print(f"Using pretrained params for {args.dgp}: {params}")

    print(f"Experiment {args.exp} | DGP: {args.dgp} | {len(cases)} cases | "
          f"{args.n_macro} macroreps")
    for c in cases:
        print(f"  {c['label']}: n0={c['n_0']} r0={c['r_0']} "
              f"n1={c['n_1']} r1={c['r_1']} method={c['method']}")

    all_rows = []

    if args.n_workers <= 1:
        for k in range(args.n_macro):
            print(f"  macrorep {k}/{args.n_macro} ...", flush=True)
            rows = run_one_macrorep(
                k, args.base_seed, args.dgp, dgp_config, cases,
                alpha, n_test, n_cand, t_grid_size, out_dir, params,
            )
            all_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {
                pool.submit(
                    run_one_macrorep, k, args.base_seed, args.dgp, dgp_config,
                    cases, alpha, n_test, n_cand, t_grid_size, out_dir, params,
                ): k for k in range(args.n_macro)
            }
            for fut in as_completed(futures):
                k = futures[fut]
                try:
                    rows = fut.result()
                    all_rows.extend(rows)
                    print(f"  macrorep {k} done", flush=True)
                except Exception as e:
                    print(f"  macrorep {k} FAILED: {e}", file=sys.stderr)

    # Save summary
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "summary.csv", index=False)

    # Print results
    print("\n=== Results ===")
    agg = df.groupby("label").agg(
        coverage=("coverage", "mean"),
        coverage_se=("coverage", "sem"),
        width=("width", "mean"),
        width_se=("width", "sem"),
        IS=("interval_score", "mean"),
        IS_se=("interval_score", "sem"),
    ).round(4)
    print(agg.to_string())
    print(f"\nSaved to {out_dir / 'summary.csv'}")


def _register_dgp(name: str, config: dict):
    """Register examples/ DGP in Two_stage registry so run_stage2 can find it."""
    from Two_stage.sim_functions import _EXPERIMENT_REGISTRY
    if name not in _EXPERIMENT_REGISTRY:
        _EXPERIMENT_REGISTRY[name] = config


if __name__ == "__main__":
    main()
