"""
experiments/framing_validation / Gate 1: coverage redistribution vs noise scale s(x).

Post-hoc analysis of existing experiments/adaptive_h/output_exp2 per-point outputs
(50 macroreps, 4 DGPs, fixed vs oracle adaptive h). No new simulation.

Claim under test (paper framing):
    Under a FIXED bandwidth h, marginal split-CP redistributes coverage
    across x: conditional coverage varies systematically with the noise
    scale s(x). Oracle adaptive h(x) = c * s(x) flattens this profile.

Method:
    - Pool test points across macroreps per (DGP, arm).
    - 10 equal-count x-bins (shared edges across arms, from pooled x).
    - Per macrorep x bin: mean covered_score -> mean +/- SE across macroreps.
    - Spearman correlation between bin coverage deviation (cov - 0.9)
      and s(bin center), per DGP x arm.

Outputs (experiments/framing_validation/output_gate1/):
    gate1_bin_coverage.csv
    gate1_correlations.csv
    gate1_coverage_vs_scale.png

Usage (from project root):
    python experiments/framing_validation/gate1_coverage_vs_scale.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Direct script execution needs both the src layout and the experiment packages.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
for _import_path in (_SRC_DIR, _PROJECT_ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))
_root = _PROJECT_ROOT
_EXPERIMENT_DIR = Path(__file__).resolve().parent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from experiments.adaptive_h.adaptive_bandwidth import ORACLE_SCALE

SIMULATORS = ["wsc_gauss", "gibbs_s1", "exp1", "nongauss_A1L"]
ARMS = ["fixed", "oracle"]
N_BINS = 10
ALPHA = 0.1

_HERE = Path(__file__).resolve().parent
EXP2_DIR = _root / "experiments" / "adaptive_h" / "output_exp2"
OUT_DIR = _HERE / "output_gate1"


def load_arm(sim: str, arm: str) -> pd.DataFrame:
    """Concatenate per_point.csv over all macroreps for one (sim, arm)."""
    frames = []
    for mdir in sorted(EXP2_DIR.glob("macrorep_*")):
        f = mdir / f"case_{sim}_{arm}" / "per_point.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, usecols=["x0", "y", "covered_score", "covered_interval", "width"])
        df["macrorep"] = int(mdir.name.split("_")[1])
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No per_point.csv found for {sim}/{arm} under {EXP2_DIR}")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bin_rows: list[dict] = []
    corr_rows: list[dict] = []

    fig, axes = plt.subplots(2, len(SIMULATORS), figsize=(4.2 * len(SIMULATORS), 7.2))

    for j, sim in enumerate(SIMULATORS):
        data = {arm: load_arm(sim, arm) for arm in ARMS}
        n_macro = data["fixed"]["macrorep"].nunique()

        # Shared equal-count bin edges from pooled x across both arms
        x_pool = np.concatenate([data[arm]["x0"].to_numpy() for arm in ARMS])
        edges = np.quantile(x_pool, np.linspace(0, 1, N_BINS + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        centers = 0.5 * (edges[:-1] + edges[1:])
        s_fn = ORACLE_SCALE[sim]
        s_centers = np.asarray(s_fn(centers), dtype=float)

        cov_curves = {}
        for arm in ARMS:
            df = data[arm].copy()
            df["bin"] = pd.cut(df["x0"], bins=edges, labels=False)
            per_rep = (
                df.groupby(["macrorep", "bin"])["covered_score"].mean().unstack("bin")
            )
            mean_cov = per_rep.mean(axis=0).to_numpy()
            se_cov = per_rep.std(axis=0, ddof=1).to_numpy() / np.sqrt(len(per_rep))
            cov_curves[arm] = (mean_cov, se_cov)

            rho, pval = spearmanr(s_centers, mean_cov - (1 - ALPHA))
            corr_rows.append(
                {
                    "simulator": sim,
                    "arm": arm,
                    "spearman_rho": rho,
                    "p_value": pval,
                    "max_abs_dev": float(np.max(np.abs(mean_cov - (1 - ALPHA)))),
                    "range_cov": float(mean_cov.max() - mean_cov.min()),
                    "n_macroreps": int(n_macro),
                }
            )
            for b in range(N_BINS):
                bin_rows.append(
                    {
                        "simulator": sim,
                        "arm": arm,
                        "bin": b,
                        "x_center": centers[b],
                        "s_center": s_centers[b],
                        "mean_cov": mean_cov[b],
                        "se_cov": se_cov[b],
                    }
                )

        # --- Row 1: coverage vs x, with s(x) on twin axis ---
        ax = axes[0, j]
        colors = {"fixed": "tab:red", "oracle": "tab:blue"}
        for arm in ARMS:
            m, se = cov_curves[arm]
            ax.errorbar(
                centers, m, yerr=se, marker="o", ms=3.5, lw=1.4,
                color=colors[arm], label=f"{arm} h", capsize=2,
            )
        ax.axhline(1 - ALPHA, color="k", ls="--", lw=0.9, alpha=0.7)
        ax.set_title(sim)
        ax.set_xlabel("x")
        if j == 0:
            ax.set_ylabel("bin coverage (score)")
        ax2 = ax.twinx()
        xs = np.linspace(edges[0], edges[-1], 300)
        ax2.plot(xs, s_fn(xs), color="gray", lw=1.0, alpha=0.55)
        ax2.set_ylabel("s(x)", color="gray", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
        if j == 0:
            ax.legend(fontsize=8, loc="lower left")

        # --- Row 2: coverage deviation vs s(x) scatter ---
        ax = axes[1, j]
        for arm in ARMS:
            m, _ = cov_curves[arm]
            rho = next(
                r["spearman_rho"] for r in corr_rows
                if r["simulator"] == sim and r["arm"] == arm
            )
            ax.scatter(
                s_centers, m - (1 - ALPHA),
                color=colors[arm], s=28,
                facecolors=colors[arm] if arm == "fixed" else "none",
                label=f"{arm} (rho={rho:.2f})",
            )
        ax.axhline(0.0, color="k", ls="--", lw=0.9, alpha=0.7)
        ax.set_xlabel("s(x) at bin center")
        if j == 0:
            ax.set_ylabel("coverage - 0.9")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Gate 1: bin coverage vs noise scale s(x) — fixed h vs oracle h(x) "
        f"(exp2 outputs, {N_BINS} bins)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = OUT_DIR / "gate1_coverage_vs_scale.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    pd.DataFrame(bin_rows).to_csv(OUT_DIR / "gate1_bin_coverage.csv", index=False)
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "gate1_correlations.csv", index=False)

    print(f"Wrote {fig_path}")
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
