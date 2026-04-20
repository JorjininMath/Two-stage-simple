"""
plot_gibbs_compare.py

Analyze and plot CKME-CP results on Gibbs et al. (RLCP) DGPs.

Produces a 2x2 figure:
  Row 1: Local coverage vs x-center (± 1 sd band), with oracle σ(x) on right axis
  Row 2: Mean interval width vs x-center, with oracle σ(x) on right axis

Usage (from project root):
  python exp_gibbs_compare/plot_gibbs_compare.py
  python exp_gibbs_compare/plot_gibbs_compare.py --output_dir exp_gibbs_compare/output
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).parent
_RLCP_ROOT = _root / "Conditional_Coverage" / "reproduce_rlcp" / "RLCP" / "results"

SIMULATORS = ["gibbs_s1", "gibbs_s2"]
SIM_LABELS = {
    "gibbs_s1": r"Setting 1: $\sigma(x) = |\sin(x)|$",
    "gibbs_s2": r"Setting 2: $\sigma(x) = 2\,\varphi(x/1.5)$",
}
ALPHA = 0.1
_CENTERS = np.arange(-2.5, 2.5 + 1e-9, 0.25)   # 21 points
_RADIUS  = 0.4


# ---------------------------------------------------------------------------
# Load Gibbs et al. and RLCP reference data
# ---------------------------------------------------------------------------

def _load_reference_data(out_dir: Path | None = None) -> dict:
    """
    Returns dict with keys like 'gibbs_s1_high', 'gibbs_s1_low', etc.
    Each value is a dict with 'mean' and 'sd' arrays (length 21).
    Also loads RLCP reproduced data.
    """
    ref = {}

    # Prefer per-rep Gibbs results with sd (from run_gibbs_source.py)
    search_dir = out_dir if out_dir is not None else _RLCP_ROOT.parent.parent / "exp_gibbs_compare" / "output"
    gibbs_configs = [
        ("gibbs_s1_high", "gibbs_gibbs_s1_high_results.csv"),
        ("gibbs_s1_low",  "gibbs_gibbs_s1_low_results.csv"),
        ("gibbs_s2_high", "gibbs_gibbs_s2_high_results.csv"),
        ("gibbs_s2_low",  "gibbs_gibbs_s2_low_results.csv"),
    ]
    for key, fname in gibbs_configs:
        p = search_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            ref[key] = {"mean": df["mean_local_cov"].values, "sd": df["sd_local_cov"].values}

    # Fallback: original 4x21 matrix (no sd)
    if "gibbs_s1_high" not in ref:
        gpath = _RLCP_ROOT / "gibbs_et_al_results.csv"
        if gpath.exists():
            g = np.loadtxt(gpath, delimiter=",")
            ref["gibbs_s1_high"] = {"mean": g[0], "sd": None}
            ref["gibbs_s1_low"]  = {"mean": g[1], "sd": None}
            ref["gibbs_s2_high"] = {"mean": g[2], "sd": None}
            ref["gibbs_s2_low"]  = {"mean": g[3], "sd": None}

    # RLCP reproduced
    for fname, key in [
        ("setting_1_local_coverage_RLCP0.05.csv", "rlcp_s1_h005"),
        ("setting_1_local_coverage_RLCP0.1.csv",  "rlcp_s1_h01"),
    ]:
        p = _RLCP_ROOT / fname
        if p.exists():
            ref[key] = pd.read_csv(p)["x"].values
    return ref


# ---------------------------------------------------------------------------
# Oracle σ(x) for each Gibbs DGP
# ---------------------------------------------------------------------------

def _sigma_s1(x: np.ndarray) -> np.ndarray:
    return np.abs(np.sin(x))


def _sigma_s2(x: np.ndarray) -> np.ndarray:
    scale = 1.5
    coeff = 2.0 / (np.sqrt(2.0 * np.pi) * scale)
    return coeff * np.exp(-0.5 * (x / scale) ** 2)


_SIGMA_FUNC = {"gibbs_s1": _sigma_s1, "gibbs_s2": _sigma_s2}


# ---------------------------------------------------------------------------
# Aggregate width profile from per_point.csv files
# ---------------------------------------------------------------------------

def _load_width_profile(out_dir: Path, sim: str, adaptive: bool = False,
                        n_macro_max: int | None = None) -> pd.DataFrame:
    """
    Aggregate per_point.csv across all macroreps for sim.
    Returns DataFrame with columns: center, mean_width, sd_width.
    """
    width_col = "width_adaptive" if adaptive else "width"
    rows = []
    macro_dirs = sorted(out_dir.glob("macrorep_*"),
                        key=lambda p: int(p.name.split("_")[1]))
    if n_macro_max is not None:
        macro_dirs = macro_dirs[:n_macro_max]
    for macro_dir in macro_dirs:
        case_dir = macro_dir / f"case_{sim}_lhs"
        pp = case_dir / "per_point.csv"
        if not pp.exists():
            continue
        df = pd.read_csv(pp)
        if width_col not in df.columns:
            continue
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["center", "mean_width", "sd_width"])
    all_df = pd.concat(rows, ignore_index=True)

    result = []
    for c in _CENTERS:
        mask = np.abs(all_df["x"] - c) <= _RADIUS
        sub = all_df.loc[mask, width_col]
        if len(sub) == 0:
            result.append({"center": float(c), "mean_width": np.nan, "sd_width": np.nan})
        else:
            result.append({
                "center":     float(c),
                "mean_width": sub.mean(),
                "sd_width":   sub.std(ddof=1),
            })
    return pd.DataFrame(result)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot_analysis(out_dir: Path, save_path: Path, fixed_dir: Path | None = None,
                  n_macro_max: int | None = None):
    summary_path = out_dir / "gibbs_compare_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else None

    # Use fixed_dir for fixed-h data if provided, else same as out_dir
    fdir = fixed_dir if fixed_dir is not None else out_dir
    ref = _load_reference_data(out_dir=fdir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for col, sim in enumerate(SIMULATORS):
        ax_cov = axes[0, col]
        ax_wid = axes[1, col]
        sim_idx = col + 1   # 1 or 2

        # --- Load local coverage (from adaptive dir first, fallback to fixed) ---
        lc_path = out_dir / f"local_coverage_{sim}.csv"
        lc = pd.read_csv(lc_path) if lc_path.exists() else None

        # If adaptive dir has no fixed-h local coverage but fixed_dir does, load from there
        lc_fixed_path = fdir / f"local_coverage_{sim}.csv"
        lc_fixed = pd.read_csv(lc_fixed_path) if lc_fixed_path.exists() else None

        # --- Load RLCP results (our run) ---
        rlcp_path = fdir / f"rlcp_s{sim_idx}_results.csv"
        rlcp = pd.read_csv(rlcp_path) if rlcp_path.exists() else None

        # --- Load width profiles ---
        wp = _load_width_profile(fdir, sim, n_macro_max=n_macro_max)

        # ---- Panel 1: local coverage ----
        ax_cov.axhline(1 - ALPHA, color="gray", linestyle="--", linewidth=1.2,
                       label=f"target = {1-ALPHA:.0%}", zorder=3)

        # Gibbs et al. reference lines (with sd bands if available)
        sim_prefix = "gibbs_s1" if sim == "gibbs_s1" else "gibbs_s2"
        gibbs_keys = [
            (f"{sim_prefix}_high", "Gibbs et al. (high adapt.)", "#2ca02c", "-"),
            (f"{sim_prefix}_low",  "Gibbs et al. (low adapt.)",  "#98df8a", "--"),
        ]
        rlcp_keys = ([("rlcp_s1_h005", "RLCP h=0.05", "#ff7f0e", "-"),
                      ("rlcp_s1_h01",  "RLCP h=0.1",  "#ffbb78", "--")]
                     if sim == "gibbs_s1" else [])

        for rk, rlabel, rcolor, rls in gibbs_keys:
            if rk in ref:
                d = ref[rk]
                ax_cov.plot(_CENTERS, d["mean"], color=rcolor, linewidth=1.5,
                            linestyle=rls, alpha=0.85, label=rlabel, zorder=3)
                if d["sd"] is not None:
                    ax_cov.fill_between(_CENTERS, d["mean"] - d["sd"], d["mean"] + d["sd"],
                                        color=rcolor, alpha=0.12)
        for rk, rlabel, rcolor, rls in rlcp_keys:
            if rk in ref:
                ax_cov.plot(_CENTERS, ref[rk], color=rcolor, linewidth=1.5,
                            linestyle=rls, alpha=0.85, label=rlabel, zorder=3)

        # RLCP (our run)
        if rlcp is not None:
            ax_cov.plot(rlcp["center"], rlcp["mean_local_cov"],
                        "^--", color="#ff7f0e", linewidth=1.8, markersize=4,
                        label="RLCP h=0.05 (reproduced)", zorder=4)
            ax_cov.fill_between(rlcp["center"],
                                rlcp["mean_local_cov"] - rlcp["sd_local_cov"],
                                rlcp["mean_local_cov"] + rlcp["sd_local_cov"],
                                alpha=0.15, color="#ff7f0e")

        # CKME-CP (fixed h) — from fixed_dir or out_dir
        lc_f = lc_fixed if lc_fixed is not None else lc
        if lc_f is not None and "mean_local_cov" in lc_f.columns:
            cov = lc_f["mean_local_cov"].values
            sd  = lc_f["sd_local_cov"].values
            ax_cov.plot(_CENTERS, cov, "o-", color="#1f77b4", linewidth=2.0,
                        markersize=4, label="CKME-CP (fixed h)", zorder=5)
            ax_cov.fill_between(_CENTERS, cov - sd, cov + sd,
                                alpha=0.18, color="#1f77b4")

        # CKME-CP (adaptive h) — from out_dir
        if lc is not None and "mean_local_cov_adaptive" in lc.columns:
            cov_a = lc["mean_local_cov_adaptive"].values
            sd_a  = lc["sd_local_cov_adaptive"].values
            ax_cov.plot(_CENTERS, cov_a, "s-", color="#e66101", linewidth=2.0,
                        markersize=4, label=r"CKME-CP (adaptive $\hat{h}$)", zorder=5)
            ax_cov.fill_between(_CENTERS, cov_a - sd_a, cov_a + sd_a,
                                alpha=0.18, color="#e66101")

        ax_cov.set_ylim(0.70, 1.05)
        ax_cov.set_ylabel("Local coverage", fontsize=11)
        ax_cov.set_xlabel(r"$x$ center", fontsize=10)
        ax_cov.set_title(SIM_LABELS[sim], fontsize=11)
        ax_cov.grid(True, alpha=0.25)

        # Marginal coverage annotation
        if summary is not None:
            row = summary[summary["simulator"] == sim]
            if len(row):
                mc = row["mean_coverage"].values[0]
                nm = int(row["n_macroreps"].values[0])
                ax_cov.text(0.03, 0.05,
                            f"CKME marginal cov = {mc:.1%}  (n_macro={nm})",
                            transform=ax_cov.transAxes, fontsize=8,
                            color="navy",
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="navy", alpha=0.8))

        ax_cov.legend(fontsize=7.5, loc="lower center", ncol=2, framealpha=0.85)

        # ---- Panel 2: width profile ----
        if rlcp is not None:
            ax_wid.plot(rlcp["center"], rlcp["mean_width"],
                        "^--", color="#ff7f0e", linewidth=1.8, markersize=4,
                        label="RLCP h=0.05 (reproduced)")
            ax_wid.fill_between(rlcp["center"],
                                rlcp["mean_width"] - rlcp["sd_width"],
                                rlcp["mean_width"] + rlcp["sd_width"],
                                alpha=0.15, color="#ff7f0e")

        if len(wp) > 0 and not wp["mean_width"].isna().all():
            ax_wid.plot(wp["center"], wp["mean_width"], "o-", color="#1f77b4",
                        linewidth=1.8, markersize=4, label="CKME-CP (fixed h)")
            ax_wid.fill_between(wp["center"],
                                wp["mean_width"] - wp["sd_width"],
                                wp["mean_width"] + wp["sd_width"],
                                alpha=0.20, color="#1f77b4")

        # Adaptive width profile
        wp_a = _load_width_profile(out_dir, sim, adaptive=True, n_macro_max=n_macro_max)
        if len(wp_a) > 0 and not wp_a["mean_width"].isna().all():
            ax_wid.plot(wp_a["center"], wp_a["mean_width"], "s-", color="#e66101",
                        linewidth=1.8, markersize=4, label=r"CKME-CP (adaptive $\hat{h}$)")
            ax_wid.fill_between(wp_a["center"],
                                wp_a["mean_width"] - wp_a["sd_width"],
                                wp_a["mean_width"] + wp_a["sd_width"],
                                alpha=0.20, color="#e66101")

        ax_wid.set_ylabel("Mean interval width", fontsize=11)
        ax_wid.set_xlabel(r"$x$ center", fontsize=10)
        ax_wid.set_title(f"{SIM_LABELS[sim]} — width profile", fontsize=11)
        ax_wid.grid(True, alpha=0.25)

        ax_wid.legend(fontsize=8, loc="upper right")

    fig.suptitle("CKME-CP: Local Coverage & Width  (radius=0.4)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default=str(_HERE / "output"))
    parser.add_argument("--fixed_dir", type=str, default=None,
                        help="Directory with fixed-h results (default: same as output_dir)")
    parser.add_argument("--n_macro_max", type=int, default=None,
                        help="Max macroreps to use from fixed_dir")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    fixed_dir = Path(args.fixed_dir) if args.fixed_dir else None
    plot_analysis(out_dir, out_dir / "fig_gibbs_analysis.png",
                  fixed_dir=fixed_dir, n_macro_max=args.n_macro_max)


if __name__ == "__main__":
    main()
