"""
plot_cdf_check.py

Visualize CKME conditional CDF estimates to inspect monotonicity.

For a chosen simulator and a few query x values, this script:
  1. Trains a small CKME model (configurable n_0, r_0)
  2. Plots F(t | x) curves over the t_grid
  3. Plots finite differences dF/dt to highlight non-monotone regions
  4. Marks where argmax(F >= tau) lands for tau in {0.05, 0.5, 0.95}

Usage (from project root):
    python exp_nongauss/plot_cdf_check.py
    python exp_nongauss/plot_cdf_check.py --sim nongauss_B2L --n_0 100 --r_0 5
    python exp_nongauss/plot_cdf_check.py --save exp_nongauss/output/cdf_check.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from CKME.ckme import CKMEModel
from CKME.parameters import Params
from Two_stage.sim_functions import get_experiment_config
from Two_stage.data_collection import collect_stage1_data

PRETRAINED = _root / "exp_nongauss" / "pretrained_params.json"
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
TAU_COLORS = ["#e41a1c", "#ff7f00", "#4daf4a", "#377eb8", "#984ea3"]


def load_params(sim: str) -> Params:
    if PRETRAINED.exists():
        with open(PRETRAINED) as f:
            d = json.load(f)
        if sim in d:
            return Params(**d[sim])
    # Fallback defaults
    return Params(ell_x=0.5, lam=0.001, h=0.1)


def train_model(sim: str, n_0: int, r_0: int, seed: int, h_override: float | None = None) -> tuple:
    """Train a CKME model and return (model, t_grid, bounds, Y_train)."""
    exp_cfg = get_experiment_config(sim)
    bounds = exp_cfg["bounds"]
    d = exp_cfg["d"]

    X_train, Y_train = collect_stage1_data(
        n_0=n_0, d=d, r_0=r_0, simulator_func=sim, random_state=seed
    )

    # t_grid covering Y range with some margin
    margin = 0.2 * (Y_train.max() - Y_train.min())
    t_grid = np.linspace(Y_train.min() - margin, Y_train.max() + margin, 500)

    params = load_params(sim)
    if h_override is not None:
        params = Params(ell_x=params.ell_x, lam=params.lam, h=h_override)
    model = CKMEModel(indicator_type="logistic")
    model.fit(X_train, Y_train, params=params)

    return model, t_grid, bounds, Y_train


def pick_query_points(bounds, n_q: int = 5) -> np.ndarray:
    """Return n_q evenly spaced query x values in bounds.

    bounds is (lo_array, hi_array) as returned by get_experiment_config.
    """
    lo_arr, hi_arr = bounds
    lo, hi = float(lo_arr[0]), float(hi_arr[0])
    xs = np.linspace(lo, hi, n_q + 2)[1:-1]   # exclude endpoints
    return xs.reshape(-1, 1)


def plot_cdf_check(
    sim: str, n_0: int, r_0: int, seed: int, n_q: int,
    h_list: list[float], save: str | None,
):
    """
    If h_list has one value: single-h plot (2 rows × n_q cols, original layout).
    If h_list has multiple values: multi-h comparison (2*n_h rows × n_q cols),
    top half = CDF curves, bottom half = finite diffs, one row-pair per h.
    """
    # ── Collect data once, reuse across h values ────────────────────────────
    # Train with first h just to get data + bounds; we'll re-fit per h below
    print(f"Collecting data for {sim} (n_0={n_0}, r_0={r_0}, seed={seed})...")
    _, t_grid, bounds, _ = train_model(sim, n_0, r_0, seed, h_override=h_list[0])
    X_query = pick_query_points(bounds, n_q)
    x_vals = X_query[:, 0]

    n_h = len(h_list)
    t_mid = 0.5 * (t_grid[:-1] + t_grid[1:])

    # ── Compute F and dF per h ───────────────────────────────────────────────
    results = []
    for h in h_list:
        print(f"  h={h}: training model...")
        model, _, _, _ = train_model(sim, n_0, r_0, seed, h_override=h)
        F = model.predict_cdf(X_query, t_grid)       # (n_q, M)
        dF = np.diff(F, axis=1)                       # (n_q, M-1)
        n_viol = (dF < 0).sum(axis=1)
        results.append({"h": h, "F": F, "dF": dF, "n_viol": n_viol, "model": model})

    # ── Layout: 2*n_h rows × n_q cols ───────────────────────────────────────
    row_h = 3.0
    fig = plt.figure(figsize=(3.5 * n_q, row_h * 2 * n_h))
    gs = gridspec.GridSpec(2 * n_h, n_q, hspace=0.5, wspace=0.3)

    for ri, res in enumerate(results):
        h = res["h"]
        F = res["F"]
        dF = res["dF"]
        n_viol = res["n_viol"]
        model = res["model"]
        cdf_row = 2 * ri
        diff_row = 2 * ri + 1

        for j in range(n_q):
            xj = x_vals[j]
            fj = F[j]
            dfj = dF[j]

            # ── CDF curve ────────────────────────────────────────────────
            ax_cdf = fig.add_subplot(gs[cdf_row, j])
            ax_cdf.plot(t_grid, fj, color="steelblue", lw=1.5)
            ax_cdf.axhline(0, color="gray", lw=0.5, ls="--")
            ax_cdf.axhline(1, color="gray", lw=0.5, ls="--")
            for tau, col in zip(TAUS, TAU_COLORS):
                mask = fj >= tau
                if mask.any():
                    idx = mask.argmax()
                    ax_cdf.axhline(tau, color=col, lw=0.6, ls=":", alpha=0.7)
                    ax_cdf.scatter([t_grid[idx]], [tau], color=col, s=20, zorder=5)
            ax_cdf.set_title(f"x={xj:.2f}  viol={n_viol[j]}", fontsize=8)
            ax_cdf.set_ylim(-0.05, 1.05)
            ax_cdf.tick_params(labelsize=6)
            if j == 0:
                ax_cdf.set_ylabel(f"h={h}\nF(t|x)", fontsize=8)

            # ── Finite diff ───────────────────────────────────────────────
            ax_df = fig.add_subplot(gs[diff_row, j])
            neg_mask = dfj < 0
            ax_df.plot(t_mid, dfj, color="steelblue", lw=1, alpha=0.8)
            ax_df.fill_between(t_mid, dfj, 0, where=neg_mask,
                               color="red", alpha=0.4)
            ax_df.axhline(0, color="black", lw=0.7)
            ax_df.tick_params(labelsize=6)
            if j == 0:
                ax_df.set_ylabel("ΔF/Δt", fontsize=8)

    fig.suptitle(
        f"CKME CDF — h comparison — {sim}   (n_0={n_0}, r_0={r_0})",
        fontsize=10, y=1.01,
    )

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\n{'h':>6}  {'x':>7}  {'violations':>12}  {'max_neg_dF':>12}")
    for res in results:
        for j in range(n_q):
            neg_vals = res["dF"][j][res["dF"][j] < 0]
            max_neg = neg_vals.min() if len(neg_vals) > 0 else 0.0
            print(f"{res['h']:6.3f}  {x_vals[j]:7.3f}  {res['n_viol'][j]:12d}  {max_neg:12.6f}")
        print()

    if save:
        out = Path(save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="CKME CDF monotonicity visualization")
    parser.add_argument("--sim",  default="nongauss_B2L",
                        help="Simulator name (default: nongauss_B2L)")
    parser.add_argument("--n_0",  type=int, default=80,
                        help="Number of design sites (default: 80)")
    parser.add_argument("--r_0",  type=int, default=5,
                        help="Reps per site (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n_q",  type=int, default=5,
                        help="Number of query points to inspect (default: 5)")
    parser.add_argument("--h", type=str, default="0.05,0.1,0.3,0.5",
                        help="Comma-separated h values to compare (default: 0.05,0.1,0.3,0.5)")
    parser.add_argument("--save", default=None,
                        help="If given, save figure to this path instead of showing")
    args = parser.parse_args()

    h_list = [float(v) for v in args.h.split(",")]

    plot_cdf_check(
        sim=args.sim,
        n_0=args.n_0,
        r_0=args.r_0,
        seed=args.seed,
        n_q=args.n_q,
        h_list=h_list,
        save=args.save,
    )


if __name__ == "__main__":
    main()
