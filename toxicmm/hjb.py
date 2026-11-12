"""HJB solver for a market maker who KNOWS the toxicity regime.

Value function:  V = cash + q * mid + theta_z(t, q).   The mid is a martingale that the
market maker cannot influence, so only the fills matter.  theta solves, backward in time
from theta(T, q) = -alpha q^2,

  0 = d/dt theta_z - phi q^2 + sum_z' G[z, z'] (theta_z' - theta_z)
        + sup_d  [expected gain rate from ask fills at distance d]
        + sup_d  [expected gain rate from bid fills at distance d]

Standard model (a fill is informed with probability pi_z, whatever the quote):
    gain rate = A exp(-k d) * ( d - pi_z eta + theta_z(q -+ 1) - theta_z(q) )
  We earn d on the fill and lose the jump xi (mean eta) on the unit just traded if the
  order was informed.  Optimum: d* = max(0, 1/k + pi_z eta - [theta_z(q -+ 1) - theta_z(q)]),
  i.e. the textbook quote shifted out by the expected adverse-selection cost.

Strategic model: the informed trader knows the jump xi ~ Exp(eta) and trades only
if xi > d.  Uninformed intensity A (1-pi) exp(-k d); informed intensity A pi exp(-d/eta),
and an informed fill loses E[xi - d | xi > d] = eta whatever the quote:
    gain rate = A (1-pi) e^{-k d} (d + dtheta) + A pi e^{-d/eta} (dtheta - eta)
  No closed form; the sup is taken on a grid of d (d = 40 ticks is "quotes pulled").

K regimes with generator G; K = 1 gives the constant-pi model (Baseline / Naive), and
"K regimes with G = 0" solves many constant-pi problems at once (certainty-equivalent table).
"""
import numpy as np
from numba import njit

D_GRID = np.arange(0.0, 40.0001, 0.05)     # quote grid for the strategic model (ticks)


@njit(cache=True)
def _H(c, A, k):
    """sup_{d>=0} A exp(-k d) (d + c), and the maximiser."""
    d = 1.0 / k - c
    if d < 0.0:
        d = 0.0
    return A * np.exp(-k * d) * (d + c), d


@njit(cache=True)
def _H_strategic(dtheta, pi, A, eta, d_grid, eu, ei):
    """sup over the quote grid of  A(1-pi) e^{-k d} (d + dtheta) + A pi e^{-d/eta} (dtheta - eta)."""
    best = -1e300
    best_d = 0.0
    for i in range(d_grid.shape[0]):
        d = d_grid[i]
        v = A * (1 - pi) * eu[i] * (d + dtheta) + A * pi * ei[i] * (dtheta - eta)
        if v > best:
            best = v
            best_d = d
    return best, best_d


@njit(cache=True)
def _rhs(theta, G, pis, A, k, eta, phi, Q, strategic, d_grid, eu, ei, da, db):
    K, nq = theta.shape
    out = np.empty_like(theta)
    for z in range(K):
        for iq in range(nq):
            q = iq - Q
            v = -phi * q * q
            for z2 in range(K):
                if z2 != z:
                    v += G[z, z2] * (theta[z2, iq] - theta[z, iq])
            if iq > 0:                       # ask: we sell one unit, inventory q -> q-1
                dth = theta[z, iq - 1] - theta[z, iq]
                if strategic:
                    hv, d = _H_strategic(dth, pis[z], A, eta, d_grid, eu, ei)
                else:
                    hv, d = _H(dth - pis[z] * eta, A, k)
                v += hv
                da[z, iq] = d
            else:
                da[z, iq] = 1e9
            if iq < nq - 1:                  # bid: we buy one unit, inventory q -> q+1
                dth = theta[z, iq + 1] - theta[z, iq]
                if strategic:
                    hv, d = _H_strategic(dth, pis[z], A, eta, d_grid, eu, ei)
                else:
                    hv, d = _H(dth - pis[z] * eta, A, k)
                v += hv
                db[z, iq] = d
            else:
                db[z, iq] = 1e9
            out[z, iq] = v
    return out


@njit(cache=True)
def _solve(G, pis, A, k, eta, phi, alpha, Q, T, dt_tab, nsub, strategic, d_grid):
    K = pis.shape[0]
    nq = 2 * Q + 1
    nt = int(round(T / dt_tab))
    dt = dt_tab / nsub
    eu = np.exp(-k * d_grid)
    ei = np.exp(-d_grid / eta) if eta > 0 else np.zeros_like(d_grid)
    theta = np.empty((K, nq))
    for z in range(K):
        for iq in range(nq):
            theta[z, iq] = -alpha * (iq - Q) ** 2
    tab_a = np.empty((nt, nq, K), dtype=np.float32)
    tab_b = np.empty((nt, nq, K), dtype=np.float32)
    da = np.empty((K, nq)); db = np.empty((K, nq))
    for it in range(nt - 1, -1, -1):
        for s in range(nsub):
            if s == nsub // 2:               # store the quotes at the middle of the table cell
                _rhs(theta, G, pis, A, k, eta, phi, Q, strategic, d_grid, eu, ei, da, db)
                for z in range(K):
                    for iq in range(nq):
                        tab_a[it, iq, z] = da[z, iq]
                        tab_b[it, iq, z] = db[z, iq]
            k1 = _rhs(theta, G, pis, A, k, eta, phi, Q, strategic, d_grid, eu, ei, da, db)
            k2 = _rhs(theta + 0.5 * dt * k1, G, pis, A, k, eta, phi, Q, strategic, d_grid, eu, ei, da, db)
            k3 = _rhs(theta + 0.5 * dt * k2, G, pis, A, k, eta, phi, Q, strategic, d_grid, eu, ei, da, db)
            k4 = _rhs(theta + dt * k3, G, pis, A, k, eta, phi, Q, strategic, d_grid, eu, ei, da, db)
            theta = theta + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return theta, tab_a, tab_b


def solve_known_regime(p, pis, G, eta=None, strategic=False, dt_tab=1.0, nsub=None):
    """Solve the HJB for regimes with informed shares `pis` and generator `G`.

    Returns theta(0, q) with shape (K, 2Q+1) and quote tables [time, inventory, regime]
    (float32; 1e9 means "no quote on that side").
    """
    pis = np.atleast_1d(np.asarray(pis, dtype=float))
    G = np.asarray(G, dtype=float).reshape(len(pis), len(pis))
    if eta is None:
        eta = p.eta
    if nsub is None:
        nsub = 4 if strategic else 10
    return _solve(G, pis, p.A, p.k, float(eta), p.phi, p.alpha, p.Q, p.T, dt_tab, nsub,
                  strategic, D_GRID)


def two_regime_generator(p, nu_ct=None, nu_tc=None):
    a = p.nu_ct if nu_ct is None else nu_ct
    b = p.nu_tc if nu_tc is None else nu_tc
    return np.array([[-a, a], [b, -b]])
