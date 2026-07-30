"""Figure for the PCP pilot: conditional-coverage tilt and set sizes."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "pilot_pcp_n50.csv")
bcols = [f"b{i}" for i in range(10)]
xc = (np.arange(10) + 0.5) * (2 * np.pi / 10)
COLOR = {"ET": "#4878d0", "pcp": "#d65f5f", "pcp_scaled": "#6acc64"}
LABEL = {"ET": "equal-tailed DCP (fixed h)",
         "pcp": "PCP (raw score $E$)",
         "pcp_scaled": "PCP scaled ($E/\\hat s(x)$, ours)"}

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

# (a) bin-coverage profiles on hetero_gauss
ax = axes[0]
sub = df[df.dgp == "hetero_gauss"]
for arm in ["ET", "pcp", "pcp_scaled"]:
    prof = sub[sub.arm == arm][bcols]
    m = prof.mean().values
    se = prof.std().values / np.sqrt(len(prof))
    ax.errorbar(xc, m, yerr=se, marker="o", ms=4, lw=1.6,
                color=COLOR[arm], capsize=3)
ax.axhline(0.9, color="gray", ls="--", lw=1.0)
ax2 = ax.twinx()
s_het = 0.10 + 0.20 * (xc - np.pi) ** 2
ax2.plot(xc, s_het, color="dimgray", ls=":", lw=1.2)
ax2.set_ylabel("noise scale s(x)", color="dimgray")
ax2.tick_params(axis="y", colors="dimgray")
ax2.set_ylim(0, 4.4)
ax.set_xlabel("x (bin centers, 10 equal-width bins)")
ax.set_ylabel("bin coverage of the conformal set")
ax.set_ylim(0.66, 1.02)
ax.set_title("(a) Conditional-coverage tilt, heteroscedastic Gaussian DGP")
handles = [Line2D([], [], marker="o", ms=4, lw=1.6, color=COLOR[a],
                  label=LABEL[a]) for a in ["ET", "pcp", "pcp_scaled"]]
handles += [Line2D([], [], color="gray", ls="--", lw=1.0,
                   label="nominal level 0.90"),
            Line2D([], [], color="dimgray", ls=":", lw=1.2,
                   label="noise scale s(x), right axis")]
ax.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.95)

# (b) set sizes per DGP/arm
ax = axes[1]
positions, ticklabs = [], []
pos = 0
for dgp, tag in [("gamma_ls", "skewed Gamma\n(homosced.)"),
                 ("hetero_gauss", "heterosced.\nGaussian")]:
    for arm in ["ET", "pcp", "pcp_scaled"]:
        v = df[(df.dgp == dgp) & (df.arm == arm)]["size"].values
        ax.errorbar(pos, v.mean(), yerr=v.std() / np.sqrt(len(v)), fmt="D",
                    ms=7, color=COLOR[arm], capsize=4, lw=1.5)
        positions.append(pos)
        ticklabs.append(arm.replace("pcp_scaled", "pcp\nscaled"))
        pos += 1
    pos += 0.8
ax.set_xticks(positions)
ax.set_xticklabels(ticklabs, fontsize=8)
ax.text(1.0, ax.get_ylim()[0], "", fontsize=8)
ax.set_xlabel("skewed Gamma (homosced.)        heterosced. Gaussian")
ax.set_ylabel("mean set size (Lebesgue measure, y units)")
ax.set_title("(b) Efficiency: set size (mean $\\pm$ 1 SE, 50 macroreps)")
handles = [Line2D([], [], marker="D", ls="none", ms=7, color=COLOR[a],
                  label=LABEL[a]) for a in ["ET", "pcp", "pcp_scaled"]]
ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.95)

fig.tight_layout()
fig.savefig(BASE / "pilot_pcp_fig.png", dpi=150)
print("saved")
