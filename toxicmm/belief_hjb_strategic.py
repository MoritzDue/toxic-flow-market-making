"""EXPLORATORY EXTENSION: Bayes-optimal quoting against a strategic informed trader.

In the strategic model the informed trader fills a quote at distance d only if the coming
jump exceeds d.  Two things change compared with belief_hjb.py:

1. The quote controls what is learned.  A fill at distance d is informed with a probability
   that depends on d, and its markout is shifted by d.
2. NOT being filled is information too.  The fill intensity per side in regime z is
       Lam_z(d) = A [ (1 - pi_z) e^{-k d} + pi_z e^{-d/eta} ],
   so while no fill arrives the belief drifts by  -p(1-p) (Lam_toxic - Lam_calm) dt.

HJB for V = cash + q * mid + theta(t, q, p):

  0 = d/dt theta - phi q^2 + [nu_ct (1-p) - nu_tc p] d/dp theta
      + sum over the two sides of  max_d {  income(d, p)
                                           - p (1-p) (Lam_toxic(d) - Lam_calm(d)) d/dp theta
                                           + E_fill[ theta(t, q -+ 1, p') ] - Lam(d, p) theta(t, q, p) }

  income(d, p) = A (1 - pihat) e^{-k d} d  -  A pihat e^{-d/eta} eta      (an informed fill loses eta)

The max is taken over a grid of quotes that includes "no quote".  For each grid quote the
fill-and-update operator is a matrix in the belief direction.
"""
import numpy as np
from . import likelihood

PULLED = 200.0                                   # stored quote value meaning "no quote"
D_GRID = np.append(np.arange(0.0, 12.0001, 0.25), PULLED)


def _operators(p, pg, lf0, lg1, a):
    """For every grid quote: K[d] (fill + belief update), total intensity, income, drift coefficient."""
    A, k, eta = p.A, p.k, p.eta
    m = likelihood.grid()
    f0 = np.exp(lf0); f0 /= f0.sum()
    g1 = np.exp(lg1); g1 /= g1.sum()
    nb = 8                                       # bin the markout: 8 table steps = 0.4 ticks
    nbin = len(m) // nb
    w0 = f0[:nbin * nb].reshape(nbin, nb).sum(1)
    n = len(pg)
    dp = pg[1] - pg[0]
    pihat = pg * p.pi_toxic + (1 - pg) * p.pi_calm
    nd = len(D_GRID)
    K = np.zeros((nd, n, n)); Lam = np.zeros((nd, n)); inc = np.zeros((nd, n)); dc = np.zeros((nd, n))
    for di, d in enumerate(D_GRID):
        if d >= PULLED:
            continue
        eu, ei = np.exp(-k * d), np.exp(-d / eta)
        shift = int(round(d / likelihood.DM))    # informed markout = d + Exp(eta) + noise
        g1s = np.zeros_like(g1)
        g1s[shift:] = g1[:len(g1) - shift] if shift > 0 else g1
        w1 = ((1 - a) * f0 + a * g1s)[:nbin * nb].reshape(nbin, nb).sum(1)
        Rt = A * ((1 - p.pi_toxic) * eu * w0 + p.pi_toxic * ei * w1)    # rate of (fill, bin) if toxic
        Rc = A * ((1 - p.pi_calm) * eu * w0 + p.pi_calm * ei * w1)
        for i, pb in enumerate(pg):
            R = pb * Rt + (1 - pb) * Rc
            post = pb * Rt / np.maximum(R, 1e-300)
            u = np.clip(post / dp, 0, n - 1 - 1e-9)
            j = u.astype(int); w = u - j
            np.add.at(K[di, i], j, R * (1 - w))
            np.add.at(K[di, i], j + 1, R * w)
        lam_t = A * ((1 - p.pi_toxic) * eu + p.pi_toxic * ei)
        lam_c = A * ((1 - p.pi_calm) * eu + p.pi_calm * ei)
        Lam[di] = K[di].sum(1)
        inc[di] = A * (1 - pihat) * eu * d - A * pihat * ei * eta
        dc[di] = -pg * (1 - pg) * (lam_t - lam_c)
    return K, Lam, inc, dc


def solve(p, lik, n_p=65, dt=0.1, dt_tab=1.0):
    """Returns theta(0, q, p), the belief grid and quote tables [time, q, n_p]."""
    lf0, lg1 = lik
    Q = p.Q
    nq = 2 * Q + 1
    pg = np.linspace(0, 1, n_p)
    dp = pg[1] - pg[0]
    K, Lam, inc, dc = _operators(p, pg, lf0, lg1, p.p_arrived)
    drift = p.nu_ct * (1 - pg) - p.nu_tc * pg
    q = np.arange(-Q, Q + 1)[:, None].astype(float)
    pen = p.phi * q ** 2

    def rhs(th, want_quotes=False):
        E = np.tensordot(K, th, axes=([2], [1])).transpose(0, 2, 1)       # [d, q, i]
        dth = np.empty_like(th)
        dth[:, 1:-1] = (th[:, 2:] - th[:, :-2]) / (2 * dp)
        dth[:, 0] = (th[:, 1] - th[:, 0]) / dp
        dth[:, -1] = (th[:, -1] - th[:, -2]) / dp
        own = inc[:, None, :] - Lam[:, None, :] * th[None] + dc[:, None, :] * dth[None]   # [d, q, i]
        ask = own[:, 1:] + E[:, :-1]             # from q to q-1
        bid = own[:, :-1] + E[:, 1:]             # from q to q+1
        if want_quotes:
            return D_GRID[ask.argmax(0)], D_GRID[bid.argmax(0)]
        v = -pen + drift[None, :] * dth
        v[1:] += ask.max(0)
        v[:-1] += bid.max(0)
        return v

    nt = int(round(p.T / dt_tab))
    nsub = int(round(dt_tab / dt))
    tab_a = np.full((nt, nq, n_p), 1e9, dtype=np.float32)
    tab_b = np.full((nt, nq, n_p), 1e9, dtype=np.float32)
    th = np.tile(-p.alpha * q ** 2, (1, n_p))
    for it in range(nt - 1, -1, -1):
        for s in range(nsub):
            if s == nsub // 2:
                da, db = rhs(th, True)
                tab_a[it, 1:] = da
                tab_b[it, :-1] = db
            k1 = rhs(th)
            k2 = rhs(th + 0.5 * dt * k1)
            k3 = rhs(th + 0.5 * dt * k2)
            k4 = rhs(th + dt * k3)
            th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return th, pg, tab_a, tab_b
