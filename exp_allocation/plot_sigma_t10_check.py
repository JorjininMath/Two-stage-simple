"""Quick visual check for sigma_t10 simulator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from Two_stage.sim_functions.sim_sigma_t10 import (
    sigma_t10_simulator, sigma_t10_true_function, _sigma, SIGMA_T10_X_BOUNDS
)

x_lo, x_hi = 0.0, 2 * np.pi
x_dense = np.linspace(x_lo, x_hi, 500)

# --- generate samples ---
rng = np.random.default_rng(0)
x_samp = rng.uniform(x_lo, x_hi, size=800)
y_samp = sigma_t10_simulator(x_samp, random_state=1)

f_dense  = sigma_t10_true_function(x_dense)
sig_dense = _sigma(x_dense)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

# --- panel 1: true function + scatter ---
ax = axes[0]
ax.scatter(x_samp, y_samp, s=6, alpha=0.3, color="steelblue", label="Y samples")
ax.plot(x_dense, f_dense, "r-", lw=2, label="f(x) = exp(x/10)sin(x)")
ax.plot(x_dense, f_dense + 2*sig_dense, "k--", lw=1, label="±2σ(x)")
ax.plot(x_dense, f_dense - 2*sig_dense, "k--", lw=1)
ax.set_xlabel("x"); ax.set_title("True function + samples")
ax.legend(fontsize=7)

# --- panel 2: noise scale sigma(x) ---
ax = axes[1]
ax.plot(x_dense, sig_dense, "darkorange", lw=2)
ax.set_xlabel("x"); ax.set_ylabel("σ(x)")
ax.set_title("Noise scale σ(x) = 0.01 + 0.2(x−π)²")

# --- panel 3: noise histogram at 3 x locations ---
ax = axes[2]
x_pts = [np.pi/2, np.pi, 3*np.pi/2]
colors = ["steelblue", "darkorange", "green"]
for xp, col in zip(x_pts, colors):
    y_local = sigma_t10_simulator(
        np.full(3000, xp), random_state=int(xp * 100)
    )
    noise = y_local - sigma_t10_true_function(np.array([xp]))[0]
    ax.hist(noise, bins=60, density=True, alpha=0.5, color=col,
            label=f"x={xp:.2f}, σ={_sigma(np.array([xp]))[0]:.2f}")
ax.set_xlabel("noise ε"); ax.set_title("Noise dist at 3 x locations (Student-t ν=10)")
ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "sigma_t10_check.png", dpi=120)
print("Saved: exp_allocation/sigma_t10_check.png")
plt.show()
