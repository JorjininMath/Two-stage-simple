from __future__ import annotations

"""
Plot the current skew / heavy-tail noise used in exp2_test.

Generates samples of the exp2_test noise at a few x values
and compares them against Gaussian noise with the same local variance.

exp2_test noise (per coordinate x):
    Y = ζ(x) + ε(x),
    ε(x) ~ (1 - π(x)) N(-δ(x), σ²(x)) + π(x) N(δ(x), σ²(x)),
where
    σ(x) = 0.3 * (1 + |x - π|),
    δ(x) = c * σ(x), c = 3,
    π(x) = logit^{-1}(a sin x), a = 2.
"""

import sys
from pathlib import Path
from math import pi

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def exp2_sigma(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.3 * (1.0 + np.abs(x - pi))


def sample_mixture_noise(x: float, n_samples: int = 100_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    sigma = float(exp2_sigma(x))
    c = 3.0
    delta = c * sigma
    a = 2.0
    pi_x = 1.0 / (1.0 + np.exp(-a * np.sin(x)))
    U = rng.uniform(0.0, 1.0, size=n_samples)
    Z = rng.normal(0.0, 1.0, size=n_samples)
    means = np.where(U < pi_x, delta, -delta)
    eps = means + sigma * Z
    return eps, sigma


def main():
    # Two representative x values
    x_vals = [pi, 0.5 * pi]
    n_samples = 100_000

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, x in zip(axes, x_vals):
        eps, sigma = sample_mixture_noise(x, n_samples=n_samples, seed=42)

        rng = np.random.default_rng(123)
        eps_gauss = rng.normal(0.0, sigma, size=n_samples)

        bins = 100
        ax.hist(eps_gauss, bins=bins, density=True, alpha=0.4, label="Gaussian N(0, σ(x)^2)")
        ax.hist(eps, bins=bins, density=True, alpha=0.4, label="Skew heavy-tail noise")

        xs = np.linspace(-6 * sigma, 6 * sigma, 400)
        ax.plot(xs, norm.pdf(xs, loc=0.0, scale=sigma), "k--", lw=1, label="Gaussian pdf")

        ax.set_title(f"x = {x:.2f}, σ(x) ≈ {sigma:.3f}")
        ax.set_xlabel("ε")
        ax.set_ylabel("density")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "exp2_test_noise_skew_heavy_tail.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

