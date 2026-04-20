"""
run_diag_postprocess.py

Pure post-processing of existing results_{exp1,exp2}.csv from
  ../output_consistency_fixed/
  ../output_consistency_adaptive_c2.00/

Produces three diagnostics for the over-smoothing hypothesis:

  D1: h(x) vs the spread of Y|x  (h_at_x vs oracle inter-quantile gap)
  D2: q_hat_tau(x) vs q_tau(x), comparing n=64 and n=8192, fixed vs adaptive
  D4: bias / variance decomposition of q_hat_tau across macroreps, vs n

Outputs (in ./output/):
  diag1_h_vs_spread.png   diag1_h_vs_spread.csv
  diag2_quantile_curves.png
  diag4_bias_var.png      diag4_bias_var.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_EXP_DIR = _HERE.parent
FIXED_DIR = _EXP_DIR / "output_consistency_fixed"
ADAPT_DIR = _EXP_DIR / "output_consistency_adaptive_c2.00"
OUT_DIR = _HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIMS = ["exp1", "exp2"]
H_MODES = {
    "fixed":    FIXED_DIR,
    "adaptive": ADAPT_DIR,
}
TAU_LO = 0.05  # alpha=0.1
TAU_HI = 0.95


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_results(sim: str) -> dict[str, pd.DataFrame]:
    out = {}
    for mode, d in H_MODES.items():
        f = d / f"results_{sim}.csv"
        if f.exists():
            out[mode] = pd.read_csv(f)
    return out


# ===========================================================================
# Diagnostic 1: h(x) vs spread of Y|x
# ===========================================================================
def diag1(results_by_sim: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    For each (sim, mode), at the largest n and macrorep=0:
      spread(x) = q_hi_oracle - q_lo_oracle
      h_at_x    = recorded bandwidth
      ratio     = h_at_x / spread(x)
    Plot two rows (sim) x two panels (h(x) and ratio).
    """
    rows = []
    fig, axes = plt.subplots(len(SIMS), 2, figsize=(11, 4.0 * len(SIMS)))
    if len(SIMS) == 1:
        axes = axes[None, :]

    for i, sim in enumerate(SIMS):
        ax_h, ax_r = axes[i]
        for mode, df in results_by_sim[sim].items():
            n_max = df["n_0"].max()
            sub = df[(df["n_0"] == n_max) & (df["macrorep"] == df["macrorep"].min())].sort_values("x_eval")
            x = sub["x_eval"].values
            spread = (sub["q_hi_oracle"] - sub["q_lo_oracle"]).values
            h = sub["h_at_x"].values
            ratio = h / spread

            color = "tab:blue" if mode == "fixed" else "tab:orange"
            ax_h.plot(x, h, "-",  color=color, label=f"{mode}: h(x)")
            ax_h.plot(x, spread, "--", color=color, alpha=0.6,
                      label=f"{mode}: spread q_hi−q_lo")
            ax_r.plot(x, ratio, "-", color=color, label=mode)

            for j in range(len(x)):
                rows.append({
                    "sim": sim, "mode": mode, "n": int(n_max),
                    "x": float(x[j]),
                    "h_at_x": float(h[j]),
                    "spread": float(spread[j]),
                    "ratio": float(ratio[j]),
                })

        ax_h.set_xlabel("x"); ax_h.set_ylabel("magnitude")
        ax_h.set_title(f"{sim}: bandwidth h(x) vs conditional spread")
        ax_h.grid(True, linestyle=":", alpha=0.5)
        ax_h.legend(fontsize=8)

        ax_r.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        ax_r.set_xlabel("x"); ax_r.set_ylabel("h(x) / (q_hi − q_lo)")
        ax_r.set_title(f"{sim}: ratio (>1 means kernel covers entire CI)")
        ax_r.grid(True, linestyle=":", alpha=0.5)
        ax_r.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag1_h_vs_spread.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / "diag1_h_vs_spread.csv", index=False)
    print(f"D1: saved diag1_h_vs_spread.{{png,csv}}")
    return df_out


# ===========================================================================
# Diagnostic 2: q_hat_tau(x) vs q_tau(x), n=64 and n=8192
# ===========================================================================
def diag2(results_by_sim: dict[str, dict[str, pd.DataFrame]]) -> None:
    """
    2 rows (sim) x 2 cols (q_lo, q_hi).
    Each panel shows oracle (black), fixed at small/large n, adaptive at small/large n.
    Curves are macrorep-mean.
    """
    fig, axes = plt.subplots(len(SIMS), 2, figsize=(12, 4.2 * len(SIMS)))
    if len(SIMS) == 1:
        axes = axes[None, :]

    for i, sim in enumerate(SIMS):
        ax_lo, ax_hi = axes[i]
        # Use first available mode to read oracle (oracle is the same across modes)
        any_df = next(iter(results_by_sim[sim].values()))
        oracle = (any_df.groupby("x_eval")[["q_lo_oracle", "q_hi_oracle"]]
                        .first().reset_index().sort_values("x_eval"))
        ax_lo.plot(oracle["x_eval"], oracle["q_lo_oracle"], "k-",
                   linewidth=2, label="oracle q_0.05")
        ax_hi.plot(oracle["x_eval"], oracle["q_hi_oracle"], "k-",
                   linewidth=2, label="oracle q_0.95")

        styles = {
            ("fixed",    "small"): ("tab:blue",   "--", "fixed n=64"),
            ("fixed",    "large"): ("tab:blue",   "-",  "fixed n=8192"),
            ("adaptive", "small"): ("tab:orange", "--", "adaptive n=64"),
            ("adaptive", "large"): ("tab:orange", "-",  "adaptive n=8192"),
        }

        for mode, df in results_by_sim[sim].items():
            ns = sorted(df["n_0"].unique())
            n_small, n_large = ns[0], ns[-1]
            for tag, n in [("small", n_small), ("large", n_large)]:
                sub = df[df["n_0"] == n]
                grp = (sub.groupby("x_eval")[["q_lo_hat", "q_hi_hat"]]
                          .mean().reset_index().sort_values("x_eval"))
                color, ls, lbl = styles[(mode, tag)]
                ax_lo.plot(grp["x_eval"], grp["q_lo_hat"], ls, color=color, label=lbl)
                ax_hi.plot(grp["x_eval"], grp["q_hi_hat"], ls, color=color, label=lbl)

        ax_lo.set_xlabel("x"); ax_lo.set_ylabel(r"$\hat q_{0.05}(x)$")
        ax_lo.set_title(f"{sim}: lower quantile estimate vs oracle")
        ax_lo.grid(True, linestyle=":", alpha=0.5)
        ax_lo.legend(fontsize=7, loc="best")

        ax_hi.set_xlabel("x"); ax_hi.set_ylabel(r"$\hat q_{0.95}(x)$")
        ax_hi.set_title(f"{sim}: upper quantile estimate vs oracle")
        ax_hi.grid(True, linestyle=":", alpha=0.5)
        ax_hi.legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag2_quantile_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("D2: saved diag2_quantile_curves.png")


# ===========================================================================
# Diagnostic 4: bias / variance decomposition
# ===========================================================================
def diag4(results_by_sim: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    For each (sim, mode, n, x_eval, tau in {lo, hi}):
      mean_k(q_hat) -> bias = mean_k(q_hat) - q_oracle
      var_k(q_hat)
    Then average over x:
      bias2(n) = mean_x bias^2
      var(n)   = mean_x var
    Plot 2 rows (sim) x 2 cols (tau lo, tau hi); fixed and adaptive overlaid.
    """
    rows = []
    for sim, by_mode in results_by_sim.items():
        for mode, df in by_mode.items():
            for n in sorted(df["n_0"].unique()):
                sub = df[df["n_0"] == n]
                # group by x_eval, compute mean and var across macroreps
                grp = sub.groupby("x_eval").agg(
                    q_lo_mean   = ("q_lo_hat",    "mean"),
                    q_lo_var    = ("q_lo_hat",    "var"),
                    q_hi_mean   = ("q_hi_hat",    "mean"),
                    q_hi_var    = ("q_hi_hat",    "var"),
                    q_lo_oracle = ("q_lo_oracle", "first"),
                    q_hi_oracle = ("q_hi_oracle", "first"),
                ).reset_index()
                bias_lo2 = ((grp["q_lo_mean"] - grp["q_lo_oracle"]) ** 2).mean()
                var_lo   = grp["q_lo_var"].mean()
                bias_hi2 = ((grp["q_hi_mean"] - grp["q_hi_oracle"]) ** 2).mean()
                var_hi   = grp["q_hi_var"].mean()
                rows.append({
                    "sim": sim, "mode": mode, "n": int(n),
                    "bias2_lo": float(bias_lo2), "var_lo": float(var_lo),
                    "bias2_hi": float(bias_hi2), "var_hi": float(var_hi),
                })

    df_bv = pd.DataFrame(rows)
    df_bv.to_csv(OUT_DIR / "diag4_bias_var.csv", index=False)

    fig, axes = plt.subplots(len(SIMS), 2, figsize=(11, 4.0 * len(SIMS)))
    if len(SIMS) == 1:
        axes = axes[None, :]

    for i, sim in enumerate(SIMS):
        ax_lo, ax_hi = axes[i]
        for ax, b_col, v_col, tau_lbl in [
            (ax_lo, "bias2_lo", "var_lo", "0.05"),
            (ax_hi, "bias2_hi", "var_hi", "0.95"),
        ]:
            sub = df_bv[df_bv["sim"] == sim]
            for mode, color in [("fixed", "tab:blue"), ("adaptive", "tab:orange")]:
                d = sub[sub["mode"] == mode].sort_values("n")
                if d.empty:
                    continue
                ax.loglog(d["n"], d[b_col], "o-",  color=color, label=f"{mode} bias²")
                ax.loglog(d["n"], d[v_col], "s--", color=color, alpha=0.7,
                          label=f"{mode} var")
            ax.set_xlabel("n"); ax.set_ylabel("error")
            ax.set_title(f"{sim}: bias²/var of $\\hat q_{{{tau_lbl}}}(x)$")
            ax.grid(True, which="both", linestyle=":", alpha=0.5)
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "diag4_bias_var.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("D4: saved diag4_bias_var.{png,csv}")
    return df_bv


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    results_by_sim = {sim: load_results(sim) for sim in SIMS}
    for sim, by_mode in results_by_sim.items():
        for mode, df in by_mode.items():
            print(f"  {sim:5s} {mode:8s}: n_vals={sorted(df['n_0'].unique())}, "
                  f"macroreps={sorted(df['macrorep'].unique())}")

    print("\n--- Diagnostic 1: h(x) vs spread ---")
    d1 = diag1(results_by_sim)
    print(d1.groupby(["sim", "mode"])[["h_at_x", "spread", "ratio"]].agg(["min", "median", "max"]))

    print("\n--- Diagnostic 2: quantile curves ---")
    diag2(results_by_sim)

    print("\n--- Diagnostic 4: bias/variance decomposition ---")
    d4 = diag4(results_by_sim)
    print(d4.to_string(index=False))

    print("\nAll diagnostics done.")


if __name__ == "__main__":
    main()
