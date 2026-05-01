"""
plot_noise_functions.py

Plot conditional noise std sigma(x) for the 4 simulators chosen for exp_adaptive_h:
  1. wsc_gauss     - smooth U-shape (low at x=pi, high at boundary)
  2. gibbs_s1      - periodic zeros (sigma touches 0)
  3. exp1          - boundary explosion (sigma -> infty as x -> 0.9)
  4. nongauss_A1L  - heavy-tail U-shape (Student-t nu=3, scale = sigma_tar)

Output: exp_adaptive_h/noise_functions.png
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Imports from the project simulators
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Two_stage.sim_functions.exp1 import exp1_noise_variance_function, EXP1_X_BOUNDS
from Two_stage.sim_functions.sim_gibbs_s1 import _gibbs_s1_sigma, GIBBS_S1_X_BOUNDS
from Two_stage.sim_functions.sim_exp2_gauss import exp2_gauss_noise_std, EXP2_GAUSS_X_BOUNDS
from Two_stage.sim_functions.sim_nongauss_A1 import _sigma_tar
from Two_stage.sim_functions.exp2 import EXP2_X_BOUNDS


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # 1. wsc_gauss: sigma(x) = 0.01 + 0.20 * (x - pi)^2, x in [0, 2pi]
    ax = axes[0, 0]
    lo, hi = float(EXP2_GAUSS_X_BOUNDS[0][0]), float(EXP2_GAUSS_X_BOUNDS[1][0])
    x = np.linspace(lo, hi, 600)
    sig = exp2_gauss_noise_std(x, sigma_base=0.01, sigma_slope=0.20)
    ax.plot(x, sig, lw=2, color="tab:blue")
    ax.axvline(np.pi, ls="--", color="gray", alpha=0.5)
    ax.text(np.pi, sig.min() + 0.05, " min @ x=pi", fontsize=9, color="gray", va="bottom")
    ax.set_title("wsc_gauss  (smooth U: Gaussian noise)\n"
                 r"$\sigma(x)=0.01+0.20(x-\pi)^{2}$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\sigma(x)$")
    ax.grid(alpha=0.3)

    # 2. gibbs_s1: sigma(x) = |sin(x)|, x in [-3, 3]; single interior zero at x=0
    #    (+-pi ~ +-3.14 are OUTSIDE this domain, so |sin(x)| has only one zero in range)
    ax = axes[0, 1]
    lo, hi = float(GIBBS_S1_X_BOUNDS[0][0]), float(GIBBS_S1_X_BOUNDS[1][0])
    x = np.linspace(lo, hi, 600)
    sig = _gibbs_s1_sigma(x)
    ax.plot(x, sig, lw=2, color="tab:orange")
    ax.axvline(0.0, ls="--", color="gray", alpha=0.5)
    ax.text(0.0, 0.05, " sigma=0 at x=0", fontsize=9, color="gray", va="bottom")
    ax.set_title("gibbs_s1  (interior zero: Gaussian noise)\n"
                 r"$\sigma(x)=|\sin x|$, $x\in[-3,3]$ (zero only at $x=0$)")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\sigma(x)$")
    ax.grid(alpha=0.3)

    # 3. exp1: sigma(x) = sqrt(noise_variance), x in [0.1, 0.9]; explodes at x -> 0.9
    ax = axes[1, 0]
    lo, hi = float(EXP1_X_BOUNDS[0][0]), float(EXP1_X_BOUNDS[1][0])
    x = np.linspace(lo, hi, 600)
    sig = np.sqrt(exp1_noise_variance_function(x))
    ax.plot(x, sig, lw=2, color="tab:green")
    ax.axvline(hi, ls="--", color="gray", alpha=0.5)
    ax.text(hi - 0.02, sig.max() * 0.5, " sigma -> infty\n as x -> 0.9", fontsize=9,
            color="gray", ha="right")
    ax.set_title("exp1  (boundary explosion: MG1 queue, Gaussian noise)\n"
                 r"$\sigma(x)$ from Pollaczek-Khinchine, $x\in[0.1,0.9]$")
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\sigma(x)$")
    ax.grid(alpha=0.3)

    # 4. nongauss_A1L: scale s(x) = 0.01 + 0.2*(x-pi)^2, Student-t nu=3
    #    Oracle h(x) tracks s(x) (solid). Plug-in's sample std converges to
    #    sigma(Y|x) = s(x)*sqrt(nu/(nu-2)) = s(x)*sqrt(3) (dotted reference).
    ax = axes[1, 1]
    lo, hi = float(EXP2_X_BOUNDS[0][0]), float(EXP2_X_BOUNDS[1][0])
    x = np.linspace(lo, hi, 600)
    s = _sigma_tar(x)
    nu = 3.0
    sig = s * np.sqrt(nu / (nu - 2))  # = s * sqrt(3); plug-in asymptote
    ax.plot(x, s, lw=2, color="tab:red",
            label=r"scale $s(x)$  (oracle $h(x)$ tracks this)")
    ax.plot(x, sig, lw=1.5, ls=":", color="tab:red", alpha=0.7,
            label=r"$\sigma(Y|x)=s(x)\sqrt{3}$  (plug-in asymptote)")
    ax.axvline(np.pi, ls="--", color="gray", alpha=0.5)
    ax.set_title(r"nongauss_A1L  (heavy-tail U: Student-t $\nu=3$)" + "\n"
                 r"oracle uses $s(x)=0.01+0.2(x-\pi)^2$;  plug-in $\to s(x)\sqrt{3}$")
    ax.set_xlabel("x")
    ax.set_ylabel("scale / std")
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(alpha=0.3)

    fig.suptitle("Conditional noise std $\\sigma(x)$ for the 4 chosen simulators",
                 fontsize=13, y=1.00)
    fig.tight_layout()

    out_path = Path(__file__).parent / "noise_functions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # also report dynamic range and key stats (oracle bandwidth target)
    print("\n=== oracle bandwidth target s(x) summary ===")
    for name, x_lo, x_hi, fn in [
        ("wsc_gauss",     0.0, 2 * np.pi, lambda x: exp2_gauss_noise_std(x, 0.01, 0.20)),
        ("gibbs_s1",     -3.0, 3.0,       lambda x: _gibbs_s1_sigma(x)),
        ("exp1",          0.1, 0.9,       lambda x: np.sqrt(exp1_noise_variance_function(x))),
        ("nongauss_A1L",  0.0, 2 * np.pi, lambda x: _sigma_tar(x)),
    ]:
        xs = np.linspace(x_lo, x_hi, 2000)
        s = fn(xs)
        rho = s.max() / max(s.min(), 1e-12)
        print(f"  {name:14s}  s(x) in [{s.min():.4f}, {s.max():.4f}]  "
              f"ratio max/min = {rho:.2f}")


if __name__ == "__main__":
    main()
