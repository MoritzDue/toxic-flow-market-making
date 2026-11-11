"""Bayes-optimal quoting when the regime is hidden: HJB in (t, q, p).

p = P(toxic | everything observed so far).  V = cash + q * mid + theta(t, q, p).

Idealised observation model: the informed jump hits the mid right after the
fill, and the market maker sees a noisy signal m of it at the same moment (same distribution
as a real markout).  Then p is a sufficient statistic and

  0 = d/dt theta - phi q^2 + [nu_ct (1-p) - nu_tc p] d/dp theta
        + sup_d A e^{-k d} ( d - pihat(p) eta + E_m[theta(t, q-1, p')] - theta(t, q, p) )   (ask)
        + sup_d A e^{-k d} ( d - pihat(p) eta + E_m[theta(t, q+1, p')] - theta(t, q, p) )   (bid)

with pihat(p) = p pi_toxic + (1-p) pi_calm and p' the Bayes update of p after seeing m.
* The drift term is how p relaxes towards its long-run mean between fills (no news).
* E_m[theta(., p')] is how p jumps at a fill.  It is what certainty equivalence ignores:
  a fill is also an experiment, and theta is convex in p, so information has value.

Numerics: p on a uniform grid, the markout in bins, so E_m is a matrix M acting on the
p-direction; RK4 backward in time; central differences for the drift (one-sided at the ends,
where the drift points inward).
"""
import numpy as np


def belief_jump_matrix(p_grid, w0, w1, pi_c, pi_t):
    """M[i, j]: E_m[f(p')] = sum_j M[i, j] f(p_j) when the belief before the fill is p_i."""
    n = len(p_grid)
    dp = p_grid[1] - p_grid[0]
    r = w1 / np.maximum(w0, 1e-300)                     # likelihood ratio informed/uninformed
    M = np.zeros((n, n))
    for i, p in enumerate(p_grid):
        pihat = p * pi_t + (1 - p) * pi_c
        prob = pihat * w1 + (1 - pihat) * w0            # probability of each bin
        Lt = pi_t * w1 + (1 - pi_t) * w0                # bin likelihood in the toxic regime
        Lc = pi_c * w1 + (1 - pi_c) * w0
        post = p * Lt / np.maximum(p * Lt + (1 - p) * Lc, 1e-300)
        u = np.clip(post / dp, 0, n - 1 - 1e-9)
        j = u.astype(int)
        w = u - j
        np.add.at(M[i], j, prob * (1 - w))
        np.add.at(M[i], j + 1, prob * w)
    return M


def solve_belief_hjb(p, w0, w1, n_p=129, dt=0.1, dt_tab=1.0, n_store=65,
                     pi_c=None, pi_t=None, nu_ct=None, nu_tc=None, eta=None):
    """Returns theta(0, q, p) [nq, n_p], the p grid, and quote tables [time, q, n_store]."""
    pi_c = p.pi_calm if pi_c is None else pi_c
    pi_t = p.pi_toxic if pi_t is None else pi_t
    nu_ct = p.nu_ct if nu_ct is None else nu_ct
    nu_tc = p.nu_tc if nu_tc is None else nu_tc
    eta = p.eta if eta is None else eta
    A, k, phi, Q = p.A, p.k, p.phi, p.Q
    nq = 2 * Q + 1
    pg = np.linspace(0, 1, n_p)
    dp = pg[1] - pg[0]
    MT = belief_jump_matrix(pg, w0, w1, pi_c, pi_t).T.copy()
    pihat = pg * pi_t + (1 - pg) * pi_c
    drift = nu_ct * (1 - pg) - nu_tc * pg
    q = np.arange(-Q, Q + 1)[:, None].astype(float)
    pen = phi * q ** 2
    cost = pihat[None, :] * eta                     # expected adverse-selection loss per fill

    def rhs(th, want_quotes=False):
        E = th @ MT                                  # E[q, i] = E_m[theta(q, p') | p_i]
        dth = np.empty_like(th)
        dth[:, 1:-1] = (th[:, 2:] - th[:, :-2]) / (2 * dp)
        dth[:, 0] = (th[:, 1] - th[:, 0]) / dp
        dth[:, -1] = (th[:, -1] - th[:, -2]) / dp
        v = -pen + drift[None, :] * dth
        ca = E[:-1] - th[1:] - cost              # ask, defined for q > -Q
        da = np.maximum(0.0, 1 / k - ca)
        v[1:] += A * np.exp(-k * da) * (da + ca)
        cb = E[1:] - th[:-1] - cost             # bid, defined for q < Q
        db = np.maximum(0.0, 1 / k - cb)
        v[:-1] += A * np.exp(-k * db) * (db + cb)
        if want_quotes:
            return da, db
        return v

    nt = int(round(p.T / dt_tab))
    nsub = int(round(dt_tab / dt))
    # explicit time stepping of the drift term needs  |drift| * dt / dp  below about 2.8 (RK4);
    # with very fast regime switching the default step is too long, so shorten it
    nsub = max(nsub, int(np.ceil(dt_tab * np.abs(drift).max() / (1.5 * dp))))
    dt = dt_tab / nsub
    step = (n_p - 1) // (n_store - 1)
    assert step * (n_store - 1) == n_p - 1
    tab_a = np.full((nt, nq, n_store), 1e9, dtype=np.float32)
    tab_b = np.full((nt, nq, n_store), 1e9, dtype=np.float32)
    th = np.tile(-p.alpha * q ** 2, (1, n_p))
    for it in range(nt - 1, -1, -1):
        for s in range(nsub):
            if s == nsub // 2:
                da, db = rhs(th, True)
                tab_a[it, 1:, :] = da[:, ::step]
                tab_b[it, :-1, :] = db[:, ::step]
            k1 = rhs(th)
            k2 = rhs(th + 0.5 * dt * k1)
            k3 = rhs(th + 0.5 * dt * k2)
            k4 = rhs(th + dt * k3)
            th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return th, pg, tab_a, tab_b
