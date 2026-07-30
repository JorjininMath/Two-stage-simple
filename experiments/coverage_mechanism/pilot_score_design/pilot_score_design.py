"""Pilot E4-score: ET vs naive-opt vs bias-constrained-opt PIT bands.

DGP: Y = m(x) + s*eps, m(x)=exp(x/10)sin(x), s=0.3 const,
eps = (Gamma(2,1)-2)/sqrt(2)  (skew 1.41; oracle ET width penalty +14%).
Same fitted CKME + same calibration set per macrorep; only the score differs.

Arms:
  ET       : theta = alpha/2 fixed (score |u-1/2|-equivalent band)
  opt      : theta*(x) = argmin width, theta in [0, alpha] (naive DCP-opt)
  bopt     : same argmin, but endpoints must satisfy a density floor
             fhat(endpoint|x) >= GAMMA * max_t fhat(t|x)  (T5 1/f proxy)

Outputs: per-arm marginal score coverage, y-interval coverage, width mean/SD,
theta* stats. Appends one CSV row per macrorep so progress is pollable.
"""
import os
import sys
from pathlib import Path

import numpy as np

# This file is two levels below ``experiments``; bootstrap the src layout so
# the pilot remains runnable as a direct script without an editable install.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _PROJECT_ROOT / "src"
for _import_path in (_SRC_DIR, _PROJECT_ROOT):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

from CKME.ckme import CKMEModel
from CKME.parameters import Params
from Two_stage.design import generate_space_filling_design

ALPHA = 0.1
N0, R0 = 60, 10
N_CAL, N_TEST = 300, 1000
S_CONST = 0.3
GAMMA = 0.15
THETAS = np.linspace(1e-4, ALPHA, 41)
# Pilot outputs stay with the pilot rather than depending on a machine-specific
# session mount.  The file is append-only, matching the original behavior.
OUT = Path(__file__).resolve().parent / "pilot_score_design.csv"

m_fun = lambda x: np.exp(x / 10.0) * np.sin(x)
BOUNDS = (0.0, 2.0 * np.pi)


def draw(x, rng):
    eps = (rng.gamma(2.0, 1.0, size=len(x)) - 2.0) / np.sqrt(2.0)
    return m_fun(x) + S_CONST * eps


def run_macrorep(seed):
    rng = np.random.default_rng(seed)
    Xs = generate_space_filling_design(n=N0, d=1, bounds=BOUNDS,
                                       random_state=seed).ravel()
    Xtr = np.repeat(Xs, R0)
    Ytr = draw(Xtr, rng)
    tg = np.linspace(Ytr.min() - 1.0, Ytr.max() + 1.0, 220)

    model = CKMEModel().fit(Xtr.reshape(-1, 1), Ytr,
                            params=Params(ell_x=0.3, lam=1e-3, h=0.05),
                            t_grid=tg, r=1)

    Xcal = rng.uniform(0, 2 * np.pi, N_CAL)
    Ycal = draw(Xcal, rng)
    Xte = rng.uniform(0, 2 * np.pi, N_TEST)
    Yte = draw(Xte, rng)

    def proj_rows(Xq):
        F = model.predict_cdf(Xq.reshape(-1, 1), tg)
        return np.clip(np.maximum.accumulate(F, axis=1), 0.0, 1.0)

    Fc, Ft = proj_rows(Xcal), proj_rows(Xte)
    u_cal = np.array([np.interp(y, tg, Fc[i]) for i, y in enumerate(Ycal)])
    u_te = np.array([np.interp(y, tg, Ft[i]) for i, y in enumerate(Yte)])

    def qtile(Frow, tau):
        idx = np.searchsorted(Frow, tau)
        return tg[min(idx, len(tg) - 1)]

    def theta_star(Frow, constrained):
        dens = np.gradient(Frow, tg)
        dmax = dens.max() + 1e-12
        best, best_w = ALPHA / 2, np.inf
        for th in THETAS:
            lo, hi = qtile(Frow, th), qtile(Frow, th + 1 - ALPHA)
            if constrained:
                d_lo = np.interp(lo, tg, dens)
                d_hi = np.interp(hi, tg, dens)
                if min(d_lo, d_hi) < GAMMA * dmax:
                    continue
            w = hi - lo
            if w < best_w:
                best_w, best = w, th
        return best  # fallback = ET theta if nothing feasible

    rows = {}
    for arm in ["ET", "opt", "bopt"]:
        if arm == "ET":
            th_cal = np.full(N_CAL, ALPHA / 2)
            th_te = np.full(N_TEST, ALPHA / 2)
        else:
            c = (arm == "bopt")
            th_cal = np.array([theta_star(Fc[i], c) for i in range(N_CAL)])
            th_te = np.array([theta_star(Ft[i], c) for i in range(N_TEST)])
        # score: signed PIT distance outside band [theta, theta+1-alpha]
        s_cal = np.maximum(th_cal - u_cal, u_cal - th_cal - (1 - ALPHA))
        k = min(int(np.ceil((1 - ALPHA) * (1 + N_CAL))), N_CAL)
        qh = float(np.sort(s_cal)[k - 1])
        s_te = np.maximum(th_te - u_te, u_te - th_te - (1 - ALPHA))
        cov_s = float(np.mean(s_te <= qh))
        lo_lv = np.clip(th_te - qh, 0.0, 1.0)
        hi_lv = np.clip(th_te + (1 - ALPHA) + qh, 0.0, 1.0)
        L = np.array([qtile(Ft[i], lo_lv[i]) for i in range(N_TEST)])
        U = np.array([qtile(Ft[i], hi_lv[i]) for i in range(N_TEST)])
        cov_i = float(np.mean((Yte >= L) & (Yte <= U)))
        rows[arm] = (cov_s, cov_i, float(np.mean(U - L)), qh,
                     float(np.mean(th_te)))
    return rows


def main():
    global N0
    n_macro = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    if len(sys.argv) > 2:
        N0 = int(sys.argv[2])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a") as f:
        if new:
            f.write("seed,arm,cov_score,cov_int,width,q_hat,mean_theta\n")
        for k in range(n_macro):
            seed = (20260716 if N0 == 60 else 20260800) + k
            rows = run_macrorep(seed)
            for arm, v in rows.items():
                f.write(f"{seed},{arm},{v[0]:.4f},{v[1]:.4f},"
                        f"{v[2]:.4f},{v[3]:.4f},{v[4]:.5f}\n")
            f.flush()
            print("done", seed, flush=True)


if __name__ == "__main__":
    main()
