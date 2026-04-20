"""
plot_branin_surface.py

Visualize Branin-Hoo true surface, noise std, and noisy samples.
"""
from __future__ import annotations
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from Two_stage.sim_functions.exp3 import exp3_true_function, EXP3_X_BOUNDS
from Two_stage.sim_functions.sim_branin_gauss import _sigma

out_dir = Path(__file__).parent / "output"
out_dir.mkdir(exist_ok=True)

# Grid
nx = 100
x1 = np.linspace(-5, 10, nx)
x2 = np.linspace(0, 15, nx)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.column_stack([X1.ravel(), X2.ravel()])

Z_true = exp3_true_function(X_grid).reshape(nx, nx)
Z_sigma = _sigma(X_grid).reshape(nx, nx)

# Noisy sample
rng = np.random.default_rng(42)
Z_noisy = Z_true + rng.normal(0, Z_sigma)

fig = plt.figure(figsize=(18, 5))

# Panel 1: True surface
ax1 = fig.add_subplot(131, projection="3d")
ax1.plot_surface(X1, X2, Z_true, cmap=cm.viridis, alpha=0.9, linewidth=0, antialiased=True)
ax1.set_xlabel("$x_1$")
ax1.set_ylabel("$x_2$")
ax1.set_zlabel("$f(x)$")
ax1.set_title("True Branin-Hoo $f(x)$", fontsize=12)
ax1.view_init(elev=25, azim=-60)

# Panel 2: Noise std sigma(x)
ax2 = fig.add_subplot(132)
c2 = ax2.contourf(X1, X2, Z_sigma, levels=20, cmap="OrRd")
plt.colorbar(c2, ax=ax2, label=r"$\sigma(x)$")
ax2.set_xlabel("$x_1$")
ax2.set_ylabel("$x_2$")
ax2.set_title(r"Noise std $\sigma(x) = 0.4(4\,\tilde{x}_1 + 1)$", fontsize=12)

# Panel 3: True + noise (one realization)
ax3 = fig.add_subplot(133, projection="3d")
ax3.plot_surface(X1, X2, Z_noisy, cmap=cm.plasma, alpha=0.9, linewidth=0, antialiased=True)
ax3.set_xlabel("$x_1$")
ax3.set_ylabel("$x_2$")
ax3.set_zlabel("$Y$")
ax3.set_title("One noisy realization $Y = f(x) + \\varepsilon$", fontsize=12)
ax3.view_init(elev=25, azim=-60)

fig.suptitle("Branin-Hoo (2D): domain $x_1 \\in [-5,10],\\; x_2 \\in [0,15]$\n"
             r"$f$ range $\approx [0, 310]$, $\sigma$ range $\approx [0.4, 2.0]$",
             fontsize=13, y=1.02)
fig.tight_layout()

# --- New figure: true surface + noisy scatter overlay ---
n_pts = 500
X_samp = np.column_stack([
    rng.uniform(-5, 10, n_pts),
    rng.uniform(0, 15, n_pts),
])
Y_true_samp = exp3_true_function(X_samp)
Y_noisy_samp = Y_true_samp + rng.normal(0, _sigma(X_samp))

fig3 = plt.figure(figsize=(14, 6))

# Left: true surface + scatter from one angle
ax3a = fig3.add_subplot(121, projection="3d")
ax3a.plot_surface(X1, X2, Z_true, cmap=cm.viridis, alpha=0.35, linewidth=0, antialiased=True)
ax3a.scatter(X_samp[:, 0], X_samp[:, 1], Y_noisy_samp,
             c="red", s=6, alpha=0.7, label="Noisy $Y$", zorder=5)
ax3a.set_xlabel("$x_1$")
ax3a.set_ylabel("$x_2$")
ax3a.set_zlabel("$Y$")
ax3a.set_title("True surface + noisy samples (view 1)", fontsize=11)
ax3a.view_init(elev=25, azim=-60)
ax3a.legend(fontsize=9, loc="upper left")

# Right: different angle
ax3b = fig3.add_subplot(122, projection="3d")
ax3b.plot_surface(X1, X2, Z_true, cmap=cm.viridis, alpha=0.35, linewidth=0, antialiased=True)
ax3b.scatter(X_samp[:, 0], X_samp[:, 1], Y_noisy_samp,
             c="red", s=6, alpha=0.7, label="Noisy $Y$", zorder=5)
ax3b.set_xlabel("$x_1$")
ax3b.set_ylabel("$x_2$")
ax3b.set_zlabel("$Y$")
ax3b.set_title("True surface + noisy samples (view 2)", fontsize=11)
ax3b.view_init(elev=35, azim=30)
ax3b.legend(fontsize=9, loc="upper left")

fig3.suptitle(f"Branin-Hoo: true surface (translucent) + {n_pts} noisy samples (red)\n"
              r"$\sigma \in [0.4, 2.0]$ — noise barely visible at this scale",
              fontsize=12, y=1.02)
fig3.tight_layout()
fig3.savefig(out_dir / "branin_surface_with_scatter.png", dpi=150, bbox_inches="tight")
print(f"Scatter fig saved")

# Also make a 2D contour of true function for clearer view
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

c_true = axes2[0].contourf(X1, X2, Z_true, levels=30, cmap="viridis")
plt.colorbar(c_true, ax=axes2[0], label="$f(x)$")
axes2[0].set_xlabel("$x_1$")
axes2[0].set_ylabel("$x_2$")
axes2[0].set_title("True $f(x)$ contour", fontsize=12)
# Mark 3 global minima
minima_x1 = [-np.pi, np.pi, 9.42478]
minima_x2 = [12.275, 2.275, 2.475]
axes2[0].scatter(minima_x1, minima_x2, c="red", s=80, marker="*", zorder=5, label="Global minima")
axes2[0].legend()

c_noisy = axes2[1].contourf(X1, X2, Z_noisy, levels=30, cmap="plasma")
plt.colorbar(c_noisy, ax=axes2[1], label="$Y$")
axes2[1].set_xlabel("$x_1$")
axes2[1].set_ylabel("$x_2$")
axes2[1].set_title("Noisy $Y = f(x) + \\varepsilon$ contour", fontsize=12)

fig2.suptitle("Branin-Hoo contour view", fontsize=13)
fig2.tight_layout()

fig.savefig(out_dir / "branin_surface_3d.png", dpi=150, bbox_inches="tight")
fig2.savefig(out_dir / "branin_surface_contour.png", dpi=150, bbox_inches="tight")
plt.close("all")

print(f"f(x) range: [{Z_true.min():.1f}, {Z_true.max():.1f}]")
print(f"sigma(x) range: [{Z_sigma.min():.2f}, {Z_sigma.max():.2f}]")
print(f"SNR (f_range / sigma_max): {(Z_true.max() - Z_true.min()) / Z_sigma.max():.1f}")
print(f"Saved to {out_dir}")
