"""
plot_adaptive_vs_fixed.py

Compare adaptive-h results against fixed-h baseline.

Generates two figures:
  Figure A: MAE-Cov(n) side-by-side: fixed h vs adaptive h, per simulator
  Figure B: Conditional coverage curves x -> Cov_n(x) for the largest n,
            fixed h (top row) vs adaptive h (bottom row), per simulator

Usage:
  python exp_conditional_coverage/plot_adaptive_vs_fixed.py
  python exp_conditional_coverage/plot_adaptive_vs_fixed.py --c_scale 2.0 --save
  python exp_conditional_coverage/plot_adaptive_vs_fixed.py --adaptive_dir output_adaptive_h_c2.00 --save
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_HERE = Path(__file__).parent

SIM_LABELS = {
    "exp1":         "exp1 (MG1 queue, Gaussian)",
    "exp2":         "exp2 (Gaussian hetero.)",
    "nongauss_B2L": "B2L (Gamma strong skew)",
}

_CMAP = cm.get_cmap("viridis_r")


def _n_colors(n_vals):
    idxs = np.linspace(0.1, 0.9, len(n_vals))
    return {n: _CMAP(i) for n, i in zip(n_vals, idxs)}


def load_results(out_dir: Path, sims: list[str]) -> tuple[dict, dict]:
    results, summaries = {}, {}
    for sim in sims:
        rp = out_dir / f"results_{sim}.csv"
        sp = out_dir / f"summary_{sim}.csv"
        if rp.exists():
            results[sim] = pd.read_csv(rp)
        if sp.exists():
            summaries[sim] = pd.read_csv(sp)
    return results, summaries


# ---------------------------------------------------------------------------
# Figure A: MAE-Cov vs n (fixed vs adaptive, per simulator)
# ---------------------------------------------------------------------------

def plot_mae_comparison(
    summaries_fixed: dict,
    summaries_adapt: dict,
    sims: list[str],
    alpha: float,
    save_path: Path | None,
    c_scale: float,
):
    n_sims = len(sims)
    fig, axes = plt.subplots(1, n_sims, figsize=(5.5 * n_sims, 4.5), squeeze=False)

    for col, sim in enumerate(sims):
        ax = axes[0, col]
        for (label, summaries, color, ls) in [
            ("Fixed h (CV-tuned)", summaries_fixed, "tab:blue",   "-"),
            (f"Adaptive h (c={c_scale})", summaries_adapt, "tab:red", "--"),
        ]:
            if sim not in summaries:
                continue
            df = summaries[sim].sort_values("n_0")
            n = df["n_0"].values
            m = df["mae_cov_mean"].values
            s = df["mae_cov_sd"].values if "mae_cov_sd" in df.columns else np.zeros_like(m)
            ax.plot(n, m, marker="o", color=color, lw=2, ms=6, ls=ls, label=label)
            ax.fill_between(n, m - s, m + s, alpha=0.15, color=color)

        ax.set_xlabel("n  (Stage 1 = Stage 2 sites)", fontsize=11)
        ax.set_ylabel("MAE-Cov", fontsize=11)
        ax.set_title(SIM_LABELS.get(sim, sim), fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    fig.suptitle("MAE-Cov vs n: Fixed h vs Adaptive h(x) = c·σ̂(x)", fontsize=13)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure B: Coverage curves for largest n (fixed vs adaptive)
# ---------------------------------------------------------------------------

def plot_coverage_curves_comparison(
    results_fixed: dict,
    results_adapt: dict,
    sims: list[str],
    alpha: float,
    n_show: int | None,
    save_path: Path | None,
    c_scale: float,
):
    """
    Two-row figure: row 0 = fixed h, row 1 = adaptive h.
    One column per simulator.
    """
    n_sims = len(sims)
    fig, axes = plt.subplots(2, n_sims, figsize=(6 * n_sims, 8), squeeze=False)

    for col, sim in enumerate(sims):
        for row, (label, results) in enumerate([
            ("Fixed h", results_fixed),
            (f"Adaptive h (c={c_scale})", results_adapt),
        ]):
            ax = axes[row, col]
            if sim not in results:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            df = results[sim]
            n_vals = sorted(df["n_0"].unique())
            if n_show is not None:
                # Show largest n >= n_show, or just the n_vals that are in that range
                n_vals = [n for n in n_vals if n >= n_show]
                if not n_vals:
                    n_vals = [max(sorted(df["n_0"].unique()))]

            colors = _n_colors(n_vals)
            for n in n_vals:
                g = df[df["n_0"] == n].groupby("x_eval")["cov_mc"].agg(
                    mean_cov=("mean"), sd_cov=("std")
                ).reset_index()
                ax.plot(g["x_eval"], g["mean_cov"], color=colors[n], label=f"n={n}", lw=1.8)
                ax.fill_between(
                    g["x_eval"],
                    g["mean_cov"] - g["sd_cov"],
                    g["mean_cov"] + g["sd_cov"],
                    alpha=0.15, color=colors[n],
                )

            ax.axhline(1 - alpha, color="k", ls="--", lw=1.2, label=f"target {1-alpha:.0%}")
            ax.set_xlabel("x", fontsize=10)
            ax.set_ylabel("Cov_n(x)", fontsize=10)
            title = f"{label}\n{SIM_LABELS.get(sim, sim)}"
            ax.set_title(title, fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Conditional Coverage Curves: Fixed h vs Adaptive h(x) = c·σ̂(x)", fontsize=13, y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Figure C: Adaptive h(x) profile across x (from results h_at_x column)
# ---------------------------------------------------------------------------

def plot_h_profile(
    results_adapt: dict,
    sims: list[str],
    save_path: Path | None,
    c_scale: float,
):
    """Show h(x) = c * sigma_hat(x) profile vs x for the largest n."""
    if not any(sim in results_adapt for sim in sims):
        return None
    if "h_at_x" not in next(iter(results_adapt.values())).columns:
        print("  h_at_x column not found; skipping h profile plot.")
        return None

    n_sims = sum(1 for sim in sims if sim in results_adapt)
    fig, axes = plt.subplots(1, n_sims, figsize=(5.5 * n_sims, 4), squeeze=False)
    col = 0
    for sim in sims:
        if sim not in results_adapt:
            continue
        ax = axes[0, col]
        df = results_adapt[sim]
        n_max = df["n_0"].max()
        g = df[df["n_0"] == n_max].groupby("x_eval")["h_at_x"].mean().reset_index()
        ax.plot(g["x_eval"], g["h_at_x"], color="tab:red", lw=2, label=f"adaptive h(x), n={n_max}")

        # Also overlay the global fixed h from results (would need to read from fixed results)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("h(x)", fontsize=11)
        ax.set_title(f"Adaptive bandwidth profile\n{SIM_LABELS.get(sim, sim)}", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        col += 1

    fig.suptitle(f"h(x) = {c_scale} × σ̂(x)  (kernel-weighted per-site std)", fontsize=12)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare adaptive-h vs fixed-h coverage results")
    parser.add_argument("--fixed_dir",   type=str, default=None,
                        help="Directory with fixed-h results (default: output/)")
    parser.add_argument("--adaptive_dir", type=str, default=None,
                        help="Directory with adaptive-h results (default: auto from c_scale)")
    parser.add_argument("--c_scale",     type=float, default=2.0)
    parser.add_argument("--simulators",  type=str, default="exp2,nongauss_B2L")
    parser.add_argument("--n_show",      type=int, default=200,
                        help="Minimum n to show in coverage curves (use 0 for all)")
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--save",        action="store_true")
    parser.add_argument("--save_dir",    type=str, default=None,
                        help="Directory to save comparison figures (default: adaptive_dir)")
    args = parser.parse_args()

    sims = [s.strip() for s in args.simulators.split(",")]
    c_scale = args.c_scale

    fixed_dir   = Path(args.fixed_dir)   if args.fixed_dir   else (_HERE / "output")
    adapt_dir   = Path(args.adaptive_dir) if args.adaptive_dir else (_HERE / f"output_adaptive_h_c{c_scale:.2f}")
    save_dir    = Path(args.save_dir)    if args.save_dir    else adapt_dir

    print(f"Fixed-h results  : {fixed_dir}")
    print(f"Adaptive results : {adapt_dir}")

    results_fixed, summaries_fixed = load_results(fixed_dir, sims)
    results_adapt, summaries_adapt = load_results(adapt_dir, sims)

    if not results_fixed and not results_adapt:
        print("No result files found. Run experiments first.")
        return

    n_show = args.n_show if args.n_show > 0 else None
    alpha  = args.alpha

    # Figure A: MAE-Cov comparison
    save_a = (save_dir / "fig_compare_mae_cov.png") if args.save else None
    print("Generating Figure A: MAE-Cov comparison...")
    plot_mae_comparison(summaries_fixed, summaries_adapt, sims, alpha, save_a, c_scale)

    # Figure B: Coverage curves comparison
    save_b = (save_dir / "fig_compare_coverage_curves.png") if args.save else None
    print("Generating Figure B: Coverage curves comparison...")
    plot_coverage_curves_comparison(results_fixed, results_adapt, sims, alpha, n_show, save_b, c_scale)

    # Figure C: h(x) profile
    save_c = (save_dir / "fig_h_profile.png") if args.save else None
    print("Generating Figure C: h(x) profile...")
    plot_h_profile(results_adapt, sims, save_c, c_scale)

    if not args.save:
        plt.show()

    print("Done.")


if __name__ == "__main__":
    main()
