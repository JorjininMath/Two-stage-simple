"""
plot_cscan.py — visualise the c-scan diagnostic.

Inputs (from output_cscan/c_<c>/summary_{exp1,exp2}.csv):
  c_scale -> mae_cov_mean (L3), mae_q_lo_mean / mae_q_hi_mean (L1)
Outputs:
  output/diag3_cscan.png
  output/diag3_cscan.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).parent
SCAN_DIR = _HERE / "output_cscan"
OUT_DIR = _HERE / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]
SIMS = ["exp1", "exp2"]

rows = []
for c in C_VALUES:
    for sim in SIMS:
        f = SCAN_DIR / f"c_{c}" / f"summary_{sim}.csv"
        if not f.exists():
            print(f"missing {f}")
            continue
        d = pd.read_csv(f).iloc[0]
        rows.append({
            "sim": sim, "c": c,
            "mae_cov":  float(d["mae_cov_mean"]),
            "mae_q_lo": float(d["mae_q_lo_mean"]),
            "mae_q_hi": float(d["mae_q_hi_mean"]),
        })
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "diag3_cscan.csv", index=False)
print(df.to_string(index=False))

fig, axes = plt.subplots(len(SIMS), 2, figsize=(11, 4.0 * len(SIMS)))
for i, sim in enumerate(SIMS):
    sub = df[df["sim"] == sim].sort_values("c")
    ax_l1, ax_l3 = axes[i]
    ax_l1.plot(sub["c"], sub["mae_q_lo"], "o-", color="tab:blue",   label=r"L1: MAE $\hat q_{0.05}$")
    ax_l1.plot(sub["c"], sub["mae_q_hi"], "s-", color="tab:orange", label=r"L1: MAE $\hat q_{0.95}$")
    ax_l1.set_xscale("log"); ax_l1.set_xlabel("c (h(x) = c·σ(x))"); ax_l1.set_ylabel("L1 quantile MAE")
    ax_l1.set_title(f"{sim}: L1 vs c (n=512, 5 macroreps)")
    ax_l1.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_l1.legend(fontsize=8)

    ax_l3.plot(sub["c"], sub["mae_cov"], "o-", color="tab:green", label="L3: MAE conditional coverage")
    ax_l3.set_xscale("log"); ax_l3.set_xlabel("c (h(x) = c·σ(x))"); ax_l3.set_ylabel("L3 coverage MAE")
    ax_l3.set_title(f"{sim}: L3 vs c (n=512, 5 macroreps)")
    ax_l3.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_l3.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_DIR / "diag3_cscan.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved diag3_cscan.{png,csv}")
