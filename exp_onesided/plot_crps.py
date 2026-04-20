"""
plot_crps.py

Compute and plot per-point CRPS of CKME on the one-sided experiment simulators.

CRPS measures the quality of the full conditional CDF estimate:
    CRPS_i = int [F_hat(t|x_i) - 1{Y_i <= t}]^2 dt   (approximated on t_grid)

QR only outputs a single quantile, not a full CDF, so CRPS is only computed for CKME.

Oracle CRPS uses the true Gaussian CDF (closed-form) as a lower-bound reference.

Output:
  - exp_onesided/output/crps_perpoint.csv   per-point CRPS for all macroreps
  - exp_onesided/output/crps_curve.png      CRPS(x) curve (median + IQR band)

Usage:
    python exp_onesided/plot_crps.py
    python exp_onesided/plot_crps.py --n_macro 5
    python exp_onesided/plot_crps.py --n_macro 5 --n_jobs 4 --save exp_onesided/output/crps_curve.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from CKME.ckme import CKMEModel
from CKME.parameters import Params
from Two_stage.sim_functions import get_experiment_config
from Two_stage.sim_functions.exp1 import exp1_true_function, exp1_noise_variance_function
from Two_stage.sim_functions.exp2 import exp2_true_function, exp2_noise_variance_function

_PRETRAINED_PATH = Path(__file__).resolve().parent / "pretrained_params.json"

SIMULATORS = ["exp2"]
SIM_LABELS = {
    "exp1": "exp1 (Gaussian)",
    "exp2": "exp2 (heteroscedastic Gaussian)",
    "nongauss_B2L": "nongauss_B2L (Gamma strong skew)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pretrained() -> dict:
    if not _PRETRAINED_PATH.exists():
        raise FileNotFoundError(
            f"Pretrained params not found at {_PRETRAINED_PATH}. "
            "Run pretrain_params.py first."
        )
    with open(_PRETRAINED_PATH) as f:
        raw = json.load(f)
    loaded = {}
    for sim in SIMULATORS:
        if sim not in raw:
            raise KeyError(
                f"Simulator '{sim}' not found in {_PRETRAINED_PATH}. "
                "Run pretrain_params.py to include it."
            )
        loaded[sim] = Params(**raw[sim])
    return loaded


def _compute_crps_perpoint(
    F_pred: np.ndarray,
    Y_true: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """
    Per-point CRPS using uniform quadrature on t_grid.

    Parameters
    ----------
    F_pred : (n, M)  predicted CDF values F_hat(t_m | x_i)
    Y_true : (n,)    observed responses
    t_grid : (M,)    threshold grid (must be sorted)

    Returns
    -------
    crps : (n,)  CRPS value for each test point
    """
    empirical = (Y_true[:, None] <= t_grid[None, :]).astype(float)  # (n, M)
    squared_diff = (F_pred - empirical) ** 2                         # (n, M)
    t_length = t_grid[-1] - t_grid[0]
    return np.mean(squared_diff, axis=1) * t_length                  # (n,)


def _oracle_crps_gaussian(
    x: np.ndarray,
    y: np.ndarray,
    true_fn,
    var_fn,
) -> np.ndarray:
    """
    Closed-form CRPS for a Gaussian conditional distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [2*phi(z) + z*(2*Phi(z)-1) - 1/sqrt(pi)]
    where z = (y - mu) / sigma.

    Parameters
    ----------
    x       : (n,)  test inputs
    y       : (n,)  observed responses
    true_fn : callable  x -> mu(x)
    var_fn  : callable  x -> sigma^2(x)

    Returns
    -------
    crps : (n,)
    """
    mu = true_fn(x)
    sigma = np.sqrt(var_fn(x))
    z = (y - mu) / sigma
    return sigma * (2.0 * stats.norm.pdf(z) + z * (2.0 * stats.norm.cdf(z) - 1.0) - 1.0 / np.sqrt(np.pi))


# Oracle functions keyed by simulator name (only Gaussian sims supported)
_ORACLE_FNS = {
    "exp1": (exp1_true_function, exp1_noise_variance_function),
    "exp2": (exp2_true_function, exp2_noise_variance_function),
}


# ---------------------------------------------------------------------------
# Per-macrorep computation
# ---------------------------------------------------------------------------

def _build_shared_t_grids(config: dict) -> dict[str, np.ndarray]:
    """
    Compute a single fixed t_grid per simulator using oracle quantiles (for Gaussian sims).
    Evaluates the true 0.5th / 99.5th conditional percentile over a dense x grid,
    takes the min/max across x, then adds a 10% margin.
    """
    t_grids = {}
    t_grid_size = config["t_grid_size"]
    x_dense = np.linspace(0, 2 * np.pi, 10_000)  # covers EXP2 / EXP1 domain

    for sim_name in SIMULATORS:
        if sim_name not in _ORACLE_FNS:
            raise ValueError(
                f"No oracle function for '{sim_name}'; cannot build oracle t_grid."
            )
        true_fn, var_fn = _ORACLE_FNS[sim_name]
        mu = true_fn(x_dense)
        sigma = np.sqrt(var_fn(x_dense))

        # Tightest interval that covers [0.5%, 99.5%] for every x
        Y_lo = float(np.min(mu + sigma * stats.norm.ppf(0.005)))
        Y_hi = float(np.max(mu + sigma * stats.norm.ppf(0.995)))
        y_margin = 0.10 * (Y_hi - Y_lo)
        t_grids[sim_name] = np.linspace(Y_lo - y_margin, Y_hi + y_margin, t_grid_size)
        print(f"  [t_grid oracle] {sim_name}: [{t_grids[sim_name][0]:.3f}, {t_grids[sim_name][-1]:.3f}]")

    return t_grids


def _run_one_macrorep(
    macro_k: int,
    seed: int,
    config: dict,
    pretrained: dict,
    shared_t_grids: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """
    Fit CKME and compute per-point CRPS (and oracle CRPS if available).

    Returns
    -------
    dict mapping sim_name -> (x_test, crps_vals, oracle_crps_vals or None)
    """
    rng = np.random.default_rng(seed)
    results = {}

    for sim_name in SIMULATORS:
        sim_cfg = get_experiment_config(sim_name)
        simulator = sim_cfg["simulator"]
        x_lo = float(sim_cfg["bounds"][0].item())
        x_hi = float(sim_cfg["bounds"][1].item())

        n_train = config["n_train"]
        r_train = config["r_train"]
        n_test = config["n_test"]
        params = pretrained[sim_name]

        # Training data
        X_sites = rng.uniform(x_lo, x_hi, size=(n_train, 1))
        Y_reps = [
            simulator(X_sites.ravel(), random_state=int(rng.integers(0, 2**31)))
            for _ in range(r_train)
        ]
        X_train = np.tile(X_sites, (r_train, 1))
        Y_train = np.concatenate(Y_reps)

        # Test data on uniform grid (clean x axis for plotting)
        X_test = np.linspace(x_lo, x_hi, n_test).reshape(-1, 1)
        Y_test = simulator(X_test.ravel(), random_state=int(rng.integers(0, 2**31)))

        # Shared t_grid (same across all macroreps)
        t_grid = shared_t_grids[sim_name]

        # Fit CKME
        ckme = CKMEModel(indicator_type="logistic")
        ckme.fit(X_train, Y_train, params=params)

        # Full CDF on test grid: (n_test, t_grid_size)
        F_pred = ckme.predict_cdf(X_test, t_grid=t_grid)

        # Per-point CRPS
        crps_vals = _compute_crps_perpoint(F_pred, Y_test, t_grid)

        # Oracle CRPS (closed-form Gaussian, if available)
        oracle_vals = None
        if sim_name in _ORACLE_FNS:
            true_fn, var_fn = _ORACLE_FNS[sim_name]
            oracle_vals = _oracle_crps_gaussian(X_test.ravel(), Y_test, true_fn, var_fn)

        results[sim_name] = (X_test.ravel(), crps_vals, oracle_vals)
        print(
            f"  [rep{macro_k}] {sim_name}: "
            f"mean CRPS = {crps_vals.mean():.4f}  "
            f"max CRPS = {crps_vals.max():.4f}"
            + (f"  oracle mean = {oracle_vals.mean():.4f}" if oracle_vals is not None else "")
        )

    return results


# ---------------------------------------------------------------------------
# Load accumulated data from CSV
# ---------------------------------------------------------------------------

def _load_from_csv(
    csv_path: str,
) -> tuple[dict, dict, dict]:
    df = pd.read_csv(csv_path)
    all_x: dict[str, np.ndarray] = {}
    all_crps: dict[str, list[np.ndarray]] = {sim: [] for sim in SIMULATORS}
    all_oracle: dict[str, list[np.ndarray]] = {sim: [] for sim in SIMULATORS}

    for sim_name in SIMULATORS:
        sub = df[df["simulator"] == sim_name]
        if sub.empty:
            continue
        for macro_k in sorted(sub["macrorep"].unique()):
            rep = sub[sub["macrorep"] == macro_k].sort_values("x")
            if all_x.get(sim_name) is None:
                all_x[sim_name] = rep["x"].to_numpy()
            all_crps[sim_name].append(rep["crps"].to_numpy())
            if "oracle_crps" in rep.columns and rep["oracle_crps"].notna().all():
                all_oracle[sim_name].append(rep["oracle_crps"].to_numpy())

    return all_x, all_crps, all_oracle


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------

def _plot(
    all_x: dict,
    all_crps: dict,
    all_oracle: dict,
    save_path: str,
):
    n_sims = len(SIMULATORS)
    fig, axes = plt.subplots(1, n_sims, figsize=(5 * n_sims, 4), sharey=False)
    if n_sims == 1:
        axes = [axes]

    for ax, sim_name in zip(axes, SIMULATORS):
        if sim_name not in all_x:
            continue
        x_test = all_x[sim_name]
        crps_stack = np.stack(all_crps[sim_name], axis=0)  # (n_macro, n_test)
        n_macro = crps_stack.shape[0]

        med = np.median(crps_stack, axis=0)
        ax.plot(x_test, med, color="steelblue", lw=1.8, label="CKME (median)")

        if all_oracle[sim_name]:
            oracle_stack = np.stack(all_oracle[sim_name], axis=0)
            oracle_med = np.median(oracle_stack, axis=0)
            ax.plot(x_test, oracle_med, color="tomato", lw=1.5,
                    linestyle="--", label="Oracle (median)")

        ax.set_title(SIM_LABELS.get(sim_name, sim_name), fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("Empirical CRPS")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(f"Empirical CRPS (n_macro={n_macro})", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot -> {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute and plot per-point CRPS for CKME")
    parser.add_argument("--from_csv", action="store_true",
                        help="Skip computation; load existing crps_perpoint.csv and plot only")
    parser.add_argument("--n_macro", type=int, default=1,
                        help="Number of macroreps (default: 1)")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Parallel jobs for macroreps (default: 1, -1 = all cores)")
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--r_train", type=int, default=5)
    parser.add_argument("--n_test", type=int, default=1000)
    parser.add_argument("--t_grid_size", type=int, default=500)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "output"),
    )
    parser.add_argument("--save", type=str, default=None,
                        help="Path to save figure. Defaults to output_dir/crps_curve.png")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "crps_perpoint.csv")
    save_path = args.save or os.path.join(args.output_dir, "crps_curve.png")

    if args.from_csv:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}. Run without --from_csv first.")
        print(f"Loading from {csv_path} ...")
        all_x, all_crps, all_oracle = _load_from_csv(csv_path)
        _plot(all_x, all_crps, all_oracle, save_path)
        return

    # --- Full run ---
    config = {
        "n_train": args.n_train,
        "r_train": args.r_train,
        "n_test": args.n_test,
        "t_grid_size": args.t_grid_size,
    }

    pretrained = _load_pretrained()
    print("Loaded pretrained params:")
    for sim in SIMULATORS:
        p = pretrained[sim]
        print(f"  {sim}: ell_x={p.ell_x}, lam={p.lam}, h={p.h}")

    print("\nBuilding shared t_grids (oracle) ...")
    shared_t_grids = _build_shared_t_grids(config)

    seeds = [args.base_seed + k for k in range(args.n_macro)]
    all_results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(_run_one_macrorep)(k, seed, config, pretrained, shared_t_grids)
        for k, seed in enumerate(seeds)
    )

    all_x: dict[str, np.ndarray] = {}
    all_crps: dict[str, list[np.ndarray]] = {sim: [] for sim in SIMULATORS}
    all_oracle: dict[str, list[np.ndarray]] = {sim: [] for sim in SIMULATORS}

    for results in all_results:
        for sim_name, (x_test, crps_vals, oracle_vals) in results.items():
            all_x[sim_name] = x_test
            all_crps[sim_name].append(crps_vals)
            if oracle_vals is not None:
                all_oracle[sim_name].append(oracle_vals)

    # Save CSV
    rows = []
    for sim_name in SIMULATORS:
        x_test = all_x[sim_name]
        for macro_k, crps_vals in enumerate(all_crps[sim_name]):
            oracle_list = all_oracle[sim_name]
            oracle_vals = oracle_list[macro_k] if macro_k < len(oracle_list) else None
            for idx, (xi, ci) in enumerate(zip(x_test, crps_vals)):
                row = {"simulator": sim_name, "macrorep": macro_k,
                       "x": float(xi), "crps": float(ci)}
                if oracle_vals is not None:
                    row["oracle_crps"] = float(oracle_vals[idx])
                rows.append(row)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nSaved per-point CRPS -> {csv_path}")

    _plot(all_x, all_crps, all_oracle, save_path)


if __name__ == "__main__":
    main()
