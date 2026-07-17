"""Pilot E4-PCP: PCP (Wang et al. 2023) on top of CKME, vs equal-tailed DCP.

Arms (same fitted CKME + same calibration set per macrorep):
  ET         : equal-tailed DCP band (paper default)
  pcp        : score E = min_k |y - yhat_k|, yhat_k ~ Ftilde(.|x) by inverse
               transform (K samples); set = union of K intervals radius q_hat
  pcp_scaled : score E / shat_NW(x)  (replication-based scale normalization —
               our estimation-layer principle ported to PCP)

DGPs:
  gamma_ls     : m(x)=exp(x/10)sin(x), s=0.3 const, eps std-Gamma(2)
                 (homoscedastic, skewed — width comparison)
  hetero_gauss : same m, s(x)=0.10+0.20(x-pi)^2, eps N(0,1)
                 (heteroscedastic — homogeneity/tilt prediction)

Outputs per macrorep/arm: score-set coverage, set measure (Lebesgue), mean
number of components, per-bin coverage (10 equal-width x-bins).
"""
import sys, os
import numpy as np

sys.path.insert(0, ".")
from CKME.ckme import CKMEModel
from CKME.parameters import Params
from Two_stage.design import generate_space_filling_design

ALPHA = 0.1
N0, R0 = 60, 10
N_CAL, N_TEST = 300, 2000
K = 40
NBIN = 10
NW_BW = 0.3
BOUNDS = (0.0, 2.0 * np.pi)
m_fun = lambda x: np.exp(x / 10.0) * np.sin(x)


def make_dgp(name):
    if name == "gamma_ls":
        s_fun = lambda x: np.full_like(np.asarray(x, dtype=float), 0.3)
        def draw(x, rng):
            eps = (rng.gamma(2.0, 1.0, size=len(x)) - 2.0) / np.sqrt(2.0)
            return m_fun(x) + s_fun(x) * eps
    elif name == "hetero_gauss":
        s_fun = lambda x: 0.10 + 0.20 * (np.asarray(x, dtype=float) - np.pi) ** 2
        def draw(x, rng):
            return m_fun(x) + s_fun(x) * rng.normal(size=len(x))
    else:
        raise ValueError(name)
    return s_fun, draw


def nw_scale(x_query, sites, s_site, bw=NW_BW):
    d2 = (np.asarray(x_query)[:, None] - sites[None, :]) ** 2
    w = np.exp(-0.5 * d2 / bw ** 2)
    return (w @ s_site) / np.maximum(w.sum(axis=1), 1e-12)


def conformal_q(scores, alpha):
    n = len(scores)
    k = min(int(np.ceil((1 - alpha) * (1 + n))), n)
    return float(np.sort(scores)[k - 1])


def union_measure(centers, radius):
    """Total length + #components of union of [c-r, c+r]."""
    c = np.sort(centers)
    lo, hi = c - radius, c + radius
    total, ncomp = 0.0, 1
    cur_lo, cur_hi = lo[0], hi[0]
    for a, b in zip(lo[1:], hi[1:]):
        if a <= cur_hi:
            cur_hi = max(cur_hi, b)
        else:
            total += cur_hi - cur_lo
            ncomp += 1
            cur_lo, cur_hi = a, b
    total += cur_hi - cur_lo
    return total, ncomp


def run_macrorep(dgp_name, seed, out_f):
    s_fun, draw = make_dgp(dgp_name)
    rng = np.random.default_rng(seed)
    Xs = generate_space_filling_design(n=N0, d=1, bounds=BOUNDS,
                                       random_state=seed).ravel()
    Xtr = np.repeat(Xs, R0)
    Ytr = draw(Xtr, rng)
    tg = np.linspace(Ytr.min() - 1.0, Ytr.max() + 1.0, 220)

    model = CKMEModel().fit(Xtr.reshape(-1, 1), Ytr,
                            params=Params(ell_x=0.3, lam=1e-3, h=0.05),
                            t_grid=tg, r=1)
    s_site = Ytr.reshape(N0, R0).std(axis=1, ddof=1)

    Xcal = rng.uniform(*BOUNDS, N_CAL)
    Ycal = draw(Xcal, rng)
    Xte = rng.uniform(*BOUNDS, N_TEST)
    Yte = draw(Xte, rng)
    bins = np.minimum((Xte / (BOUNDS[1] / NBIN)).astype(int), NBIN - 1)

    def proj_rows(Xq):
        F = model.predict_cdf(Xq.reshape(-1, 1), tg)
        return np.clip(np.maximum.accumulate(F, axis=1), 0.0, 1.0)

    Fc, Ft = proj_rows(Xcal), proj_rows(Xte)

    def sample_rows(F_rows, rng):
        """K inverse-transform samples per row from projected CDF."""
        n = F_rows.shape[0]
        u = rng.uniform(0, 1, size=(n, K))
        out = np.empty((n, K))
        for i in range(n):
            idx = np.minimum(np.searchsorted(F_rows[i], u[i]), len(tg) - 1)
            out[i] = tg[idx]
        return out

    Scal = sample_rows(Fc, rng)
    Ste = sample_rows(Ft, rng)
    E_cal = np.abs(Scal - Ycal[:, None]).min(axis=1)
    E_te = np.abs(Ste - Yte[:, None]).min(axis=1)
    sh_cal = nw_scale(Xcal, Xs, s_site)
    sh_te = nw_scale(Xte, Xs, s_site)

    rows = {}
    # --- ET (DCP equal-tailed) ---
    u_cal = np.array([np.interp(y, tg, Fc[i]) for i, y in enumerate(Ycal)])
    u_te = np.array([np.interp(y, tg, Ft[i]) for i, y in enumerate(Yte)])
    qh = conformal_q(np.abs(u_cal - 0.5), ALPHA)
    member = np.abs(u_te - 0.5) <= qh
    tlo, thi = 0.5 - qh, 0.5 + qh
    size = np.empty(N_TEST)
    for i in range(N_TEST):
        lo = tg[min(np.searchsorted(Ft[i], np.clip(tlo, 0, 1)), len(tg) - 1)]
        hi = tg[min(np.searchsorted(Ft[i], np.clip(thi, 0, 1)), len(tg) - 1)]
        size[i] = hi - lo
    rows["ET"] = (member, size, np.ones(N_TEST))

    # --- pcp ---
    qh = conformal_q(E_cal, ALPHA)
    member = E_te <= qh
    size = np.empty(N_TEST); ncomp = np.empty(N_TEST)
    for i in range(N_TEST):
        size[i], ncomp[i] = union_measure(Ste[i], qh)
    rows["pcp"] = (member, size, ncomp)

    # --- pcp_scaled ---
    qh = conformal_q(E_cal / sh_cal, ALPHA)
    member = (E_te / sh_te) <= qh
    size = np.empty(N_TEST); ncomp = np.empty(N_TEST)
    for i in range(N_TEST):
        size[i], ncomp[i] = union_measure(Ste[i], qh * sh_te[i])
    rows["pcp_scaled"] = (member, size, ncomp)

    for arm, (member, size, ncomp) in rows.items():
        bc = [float(member[bins == b].mean()) for b in range(NBIN)]
        out_f.write(f"{seed},{dgp_name},{arm},{member.mean():.4f},"
                    f"{size.mean():.4f},{ncomp.mean():.3f},"
                    + ",".join(f"{v:.4f}" for v in bc) + "\n")
    out_f.flush()


def main():
    n_macro = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out_path = sys.argv[2]
    new = not os.path.exists(out_path)
    with open(out_path, "a") as f:
        if new:
            f.write("seed,dgp,arm,cov,size,ncomp,"
                    + ",".join(f"b{b}" for b in range(NBIN)) + "\n")
        for dgp, base in [("gamma_ls", 20260900), ("hetero_gauss", 20261000)]:
            for k in range(n_macro):
                run_macrorep(dgp, base + k, f)
                print("done", dgp, base + k, flush=True)


if __name__ == "__main__":
    main()
