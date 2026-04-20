"""
plot_nongauss_noise.py

Compare the updated non-Gaussian noise distributions (Small / Large variants)
against the shared target variance sigma_tar(x) = 0.1 + 0.1*(x-pi)^2.

Layout: 2 rows x 3 cols
  Row 0 — Small (lighter non-Gaussianity): t(nu=10) | Gamma(k=9) | Mixture(pi=0.02)
  Row 1 — Large (stronger non-Gaussianity): t(nu=3)  | Gamma(k=2) | Mixture(pi=0.10)

Each panel (at fixed x = x_val):
  - histogram of noise samples
  - true noise pdf
  - reference Gaussian N(0, Var_tar(x))

Usage (from project root):
    python exp_nongauss/plot_nongauss_noise.py
    python exp_nongauss/plot_nongauss_noise.py --x_val 3.14 --save exp_nongauss/output/nongauss_noise.png
    python exp_nongauss/plot_nongauss_noise.py --n_samples 500000 --save exp_nongauss/output/nongauss_noise.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from math import pi

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t as scipy_t, gamma as scipy_gamma

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


# ---------------------------------------------------------------------------
# Shared target variance function
# ---------------------------------------------------------------------------
def sigma_tar(x):
    return 0.1 + 0.1 * (x - pi) ** 2

def var_tar(x):
    return sigma_tar(x) ** 2


# ---------------------------------------------------------------------------
# Student-t noise:  Y = f(x) + s_nu(x) * T_nu
# Scale s_nu(x) = sigma_tar(x) * sqrt((nu-2)/nu)  =>  Var = sigma_tar^2
# ---------------------------------------------------------------------------
def t_scale(x, nu):
    return sigma_tar(x) * np.sqrt((nu - 2) / nu)

def t_sample(x, n, seed, nu):
    return np.random.default_rng(seed).standard_t(df=nu, size=n) * t_scale(x, nu)

def t_pdf(eps, x, nu):
    return scipy_t.pdf(eps, df=nu, loc=0.0, scale=t_scale(x, nu))


# ---------------------------------------------------------------------------
# Centered Gamma noise:  Y = f(x) + Gamma(k, theta(x)) - k*theta(x)
# theta(x) = sigma_tar(x) / sqrt(k)  =>  Var = k*theta^2 = sigma_tar^2
# ---------------------------------------------------------------------------
def gamma_theta(x, k):
    return sigma_tar(x) / np.sqrt(k)

def gamma_sample(x, n, seed, k):
    rng = np.random.default_rng(seed)
    theta = gamma_theta(x, k)
    return rng.gamma(shape=k, scale=theta, size=n) - k * theta

def gamma_pdf(eps, x, k):
    theta = gamma_theta(x, k)
    return scipy_gamma.pdf(eps + k * theta, a=k, scale=theta)


# ---------------------------------------------------------------------------
# Gaussian mixture noise (shifted, heteroscedastic):
#   epsilon | x ~ (1-pi)*N(mu1(x),(a*s1*sig)^2) + pi*N(mu2(x),(a*s2*sig)^2)
#   sig     = sigma_tar(x) = 0.1 + 0.1*(x-pi)^2
#   a(pi)   = 1 / sqrt((1-pi)*s1^2 + pi*s2^2 + pi*(1-pi)*delta^2)
#   mu1(x)  = -a*pi*delta*sig     (inlier, left-shifted)
#   mu2(x)  =  a*(1-pi)*delta*sig (outlier, right-shifted)
#   => E[eps|x]=0,  Var[eps|x] = sigma_tar(x)^2  (exact, all x)
# ---------------------------------------------------------------------------
_S1, _S2, _DELTA = 0.35, 1.0, 4.0

def _mixture_a(pi_mix):
    return 1.0 / np.sqrt(
        (1 - pi_mix) * _S1 ** 2 + pi_mix * _S2 ** 2 + pi_mix * (1 - pi_mix) * _DELTA ** 2
    )

def mixture_sample(x, n, seed, pi_mix):
    rng = np.random.default_rng(seed)
    sig = sigma_tar(x)
    a   = _mixture_a(pi_mix)
    is_outlier = rng.uniform(size=n) < pi_mix
    mu  = np.where(is_outlier,  a * (1 - pi_mix) * _DELTA * sig,
                               -a * pi_mix        * _DELTA * sig)
    std = np.where(is_outlier, a * _S2 * sig, a * _S1 * sig)
    return rng.normal(loc=mu, scale=std)

def mixture_pdf(eps, x, pi_mix):
    sig = sigma_tar(x)
    a   = _mixture_a(pi_mix)
    mu1 = -a * pi_mix       * _DELTA * sig
    mu2 =  a * (1 - pi_mix) * _DELTA * sig
    return ((1 - pi_mix) * norm.pdf(eps, mu1, a * _S1 * sig)
            +    pi_mix  * norm.pdf(eps, mu2, a * _S2 * sig))


# ---------------------------------------------------------------------------
# DGP table: 6 panels  (row, col)
# ---------------------------------------------------------------------------
DGPS = [
    # ---------- Row 0: Small non-Gaussianity ----------
    dict(
        row=0, col=0,
        label=r"A1-S: Student-t ($\nu$=10)",
        note=r"$\nu=10$, light tails",
        sample_fn=lambda x, n, s: t_sample(x, n, s, nu=10),
        var_fn=var_tar,
        pdf_fn=lambda eps, x: t_pdf(eps, x, nu=10),
        pdf_label=r"$t_{10}$ pdf",
        color="#2166ac", clip_sigma=4,
    ),
    dict(
        row=0, col=1,
        label=r"B2-S: Centered Gamma ($k$=9)",
        note=r"$k=9$, mild skew",
        sample_fn=lambda x, n, s: gamma_sample(x, n, s, k=9),
        var_fn=var_tar,
        pdf_fn=lambda eps, x: gamma_pdf(eps, x, k=9),
        pdf_label="Gamma pdf (k=9)",
        color="#d6604d", clip_sigma=4,
    ),
    dict(
        row=0, col=2,
        label=r"C1-S: Mixture ($\pi$=0.02)",
        note=r"$\pi=0.02$, light contamination",
        sample_fn=lambda x, n, s: mixture_sample(x, n, s, pi_mix=0.02),
        var_fn=var_tar,
        pdf_fn=lambda eps, x: mixture_pdf(eps, x, 0.02),
        pdf_label=r"Mixture pdf ($\pi$=0.02)",
        color="#4dac26", clip_sigma=15,
    ),
    # ---------- Row 1: Large non-Gaussianity ----------
    dict(
        row=1, col=0,
        label=r"A1-L: Student-t ($\nu$=3)",
        note=r"$\nu=3$, heavy tails",
        sample_fn=lambda x, n, s: t_sample(x, n, s, nu=3),
        var_fn=var_tar,
        pdf_fn=lambda eps, x: t_pdf(eps, x, nu=3),
        pdf_label=r"$t_3$ pdf",
        color="#2166ac", clip_sigma=5,
    ),
    dict(
        row=1, col=1,
        label=r"B2-L: Centered Gamma ($k$=2)",
        note=r"$k=2$, strong skew",
        sample_fn=lambda x, n, s: gamma_sample(x, n, s, k=2),
        var_fn=var_tar,
        pdf_fn=lambda eps, x: gamma_pdf(eps, x, k=2),
        pdf_label="Gamma pdf (k=2)",
        color="#d6604d", clip_sigma=4,
    ),
    dict(
        row=1, col=2,
        label=r"C1-L: Mixture ($\pi$=0.10)",
        note=r"$\pi=0.10$, heavy contamination",
        sample_fn=lambda x, n, s: mixture_sample(x, n, s, pi_mix=0.10),
        var_fn=var_tar,
        pdf_fn=lambda eps, x: mixture_pdf(eps, x, 0.10),
        pdf_label=r"Mixture pdf ($\pi$=0.10)",
        color="#4dac26", clip_sigma=8,
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x_val",    type=float, default=pi,
                        help="Input x value to evaluate noise at (default: pi)")
    parser.add_argument("--n_samples", type=int, default=300_000)
    parser.add_argument("--save",     type=str, default=None)
    args = parser.parse_args()

    x = args.x_val
    x_label = f"x = {x:.3f}"

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"Non-Gaussian noise distributions at {x_label}\n"
        r"Shared target: $\sigma_{\rm tar}(x)=0.1+0.1(x-\pi)^2$, "
        r"${\rm Var}_{\rm tar}(x)=\sigma_{\rm tar}(x)^2$",
        fontsize=11,
    )

    for dgp in DGPS:
        ax = axes[dgp["row"], dgp["col"]]

        eps   = dgp["sample_fn"](x, args.n_samples, 42)
        std   = np.sqrt(dgp["var_fn"](x))
        clip  = dgp["clip_sigma"] * std
        eps_c = eps[np.abs(eps) < clip]

        ax.hist(eps_c, bins=100, density=True, alpha=0.4,
                color=dgp["color"], label="samples")

        xs = np.linspace(eps_c.min(), eps_c.max(), 600)
        ax.plot(xs, dgp["pdf_fn"](xs, x),
                color=dgp["color"], lw=2.2, label=dgp["pdf_label"])
        ax.plot(xs, norm.pdf(xs, 0, std),
                "k--", lw=1.6, label=f"N(0, {std**2:.4f})")

        ax.set_title(dgp["label"], fontsize=10)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel("Density")
        ax.text(0.97, 0.97, dgp["note"], transform=ax.transAxes,
                fontsize=8, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    plt.tight_layout()

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
