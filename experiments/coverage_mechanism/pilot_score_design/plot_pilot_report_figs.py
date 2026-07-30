"""Paper-style figures for the score-design pilot report (overlap-free:
all line/marker explanations live in legends, no in-axes text)."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
a = pd.read_csv(BASE / "pilot_n50_b600.csv");  a["budget"] = 600
b = pd.read_csv(BASE / "pilot_n50_b2000.csv"); b["budget"] = 2000
df = pd.concat([a, b])
ARMS = ["ET", "opt", "bopt"]
LABEL = {"ET": "equal-tailed (ET)", "opt": "width-optimized (opt)",
         "bopt": "bias-constrained opt (b-opt)"}
COLOR = {"ET": "#4878d0", "opt": "#d65f5f", "bopt": "#6acc64"}
ORACLE_ET, ORACLE_SHORT = 0.931, 0.816

# ---------------- Figure 1: validity + width by arm and budget -----------
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

ax = axes[0]
for j, b in enumerate([600, 2000]):
    for i, a in enumerate(ARMS):
        v = df[(df.budget == b) & (df.arm == a)].cov_score.values
        xpos = j * 4 + i
        ax.scatter(np.full(len(v), xpos) + rng.uniform(-0.12, 0.12, len(v)),
                   v, s=14, color=COLOR[a], alpha=0.65)
        ax.errorbar(xpos, v.mean(), yerr=v.std() / np.sqrt(len(v)),
                    fmt="_", ms=22, color="black", capsize=4, lw=1.6, zorder=5)
ax.axhline(0.9, color="gray", ls="--", lw=1.0)
ax.set_xticks([0, 1, 2, 4, 5, 6])
ax.set_xticklabels(["ET", "opt", "b-opt", "ET", "opt", "b-opt"])
ax.set_xlim(-0.7, 6.7)
ax.set_ylim(0.828, 0.952)
ax.set_xlabel("total budget B = 600                total budget B = 2000")
ax.set_ylabel("marginal coverage of the conformal set")
ax.set_title("(a) Validity")
handles_a = [
    Line2D([], [], color="gray", ls="--", lw=1.0, label="nominal level 0.90"),
    Line2D([], [], color="black", marker="_", ls="none", ms=14, mew=1.6,
           label="mean $\\pm$ 1 SE"),
]
ax.legend(handles=handles_a, fontsize=8, loc="lower right", framealpha=0.9)

ax = axes[1]
for j, b in enumerate([600, 2000]):
    for i, a in enumerate(ARMS):
        v = df[(df.budget == b) & (df.arm == a)].width.values
        xpos = j * 4 + i
        ax.scatter(np.full(len(v), xpos) + rng.uniform(-0.12, 0.12, len(v)),
                   v, s=14, color=COLOR[a], alpha=0.65)
        ax.errorbar(xpos, v.mean(), yerr=v.std() / np.sqrt(len(v)),
                    fmt="_", ms=22, color="black", capsize=4, lw=1.6, zorder=5)
ax.axhline(ORACLE_ET, color="#4878d0", ls=":", lw=1.3)
ax.axhline(ORACLE_SHORT, color="#d65f5f", ls=":", lw=1.3)
ax.set_xticks([0, 1, 2, 4, 5, 6])
ax.set_xticklabels(["ET", "opt", "b-opt", "ET", "opt", "b-opt"])
ax.set_xlim(-0.7, 6.7)
ax.set_ylim(0.55, 3.25)
ax.set_xlabel("total budget B = 600                total budget B = 2000")
ax.set_ylabel("mean interval width (y units)")
ax.set_title("(b) Efficiency")
handles_b = [
    Line2D([], [], marker="o", ls="none", color=COLOR[a], label=LABEL[a])
    for a in ARMS
] + [
    Line2D([], [], color="#4878d0", ls=":", lw=1.3,
           label="oracle equal-tailed width (0.931)"),
    Line2D([], [], color="#d65f5f", ls=":", lw=1.3,
           label="oracle shortest width (0.816)"),
    Line2D([], [], color="black", marker="_", ls="none", ms=14, mew=1.6,
           label="mean $\\pm$ 1 SE"),
]
ax.legend(handles=handles_b, fontsize=8, loc="upper right", framealpha=0.9)

fig.tight_layout()
fig.savefig(BASE / "pilot_fig1_validity_width.png", dpi=150)

# ---------------- Figure 2: paired width difference vs ET ----------------
fig, ax = plt.subplots(figsize=(7.6, 4.6))
positions, xticklab = [], []
pos = 0
for b in [600, 2000]:
    p = df[df.budget == b].pivot(index="seed", columns="arm", values="width")
    for a in ["opt", "bopt"]:
        d = 100 * (p[a] - p["ET"]) / p["ET"]
        ax.scatter(np.full(len(d), pos) + rng.uniform(-0.10, 0.10, len(d)),
                   d, s=16, color=COLOR[a], alpha=0.7)
        ax.errorbar(pos, d.mean(), yerr=d.std() / np.sqrt(len(d)), fmt="D",
                    ms=7, color="black", capsize=5, lw=1.6, zorder=5)
        positions.append(pos)
        xticklab.append(f"{'opt' if a == 'opt' else 'b-opt'}\nB={b}")
        pos += 1
    pos += 0.8
ax.axhline(0, color="gray", lw=1.0)
ax.axhline(-12.3, color="#d65f5f", ls=":", lw=1.3)
ax.set_xticks(positions)
ax.set_xticklabels(xticklab)
ax.set_ylim(-30, 240)
ax.set_ylabel("paired width difference vs ET (%)")
ax.set_title("Paired width differences against the equal-tailed baseline\n"
             "(negative = narrower than ET)")
handles = [
    Line2D([], [], marker="o", ls="none", color=COLOR["opt"],
           label="width-optimized (opt)"),
    Line2D([], [], marker="o", ls="none", color=COLOR["bopt"],
           label="bias-constrained opt (b-opt)"),
    Line2D([], [], color="black", marker="D", ls="none", ms=6,
           label="mean $\\pm$ 1 SE"),
    Line2D([], [], color="gray", lw=1.0, label="parity with ET (0%)"),
    Line2D([], [], color="#d65f5f", ls=":", lw=1.3,
           label="oracle gain of shortest band ($-12.3\\%$)"),
]
ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig(BASE / "pilot_fig2_paired_deltas.png", dpi=150)
print("figures saved")
