"""Builds the quoting tables of each market maker and runs the simulator.

Market makers
  Baseline  ignores adverse selection (quotes for pi = 0)
  Naive     quotes for the time-averaged informed share
  Oracle    sees the regime; regime-switching HJB
  Learning  hidden-Markov filter for p = P(toxic); quotes for the filtered informed share
            (certainty equivalence: a heuristic, it treats the estimate as if it were known)
  Bayes     quotes from the belief-state HJB in (t, q, p)
"""
from dataclasses import dataclass, field
import numpy as np

from . import hjb, likelihood, belief_hjb
from . import simulate as sim

N_U = 65            # number of belief nodes stored in quote tables


@dataclass
class MM:
    name: str
    src: int
    tab_a: np.ndarray
    tab_b: np.ndarray
    fpar: np.ndarray = field(default_factory=lambda: np.zeros(10))
    lik: tuple = None           # (logf0, logg1) used by this market maker's filter
    value0: float = np.nan      # HJB value at t=0, q=0 (where available)


def lik_tables(p, eta=None):
    """Markout likelihood tables assumed by a filter: noise = diffusion + the other jumps that
    fall in the window (rate lam_tot, the same in both regimes); informed impact has mean eta."""
    return likelihood.markout_tables(p.sigma, p.h, p.lam_tot, p.eta, p.eta if eta is None else eta)


def constant_pi(p, pi, name, strategic=False):
    th, ta, tb = hjb.solve_known_regime(p, [pi], [[0.0]], strategic=strategic)
    return MM(name, sim.SRC_CONST, ta, tb, value0=th[0, p.Q])


def baseline(p, strategic=False):
    return constant_pi(p, 0.0, "Baseline", strategic)


def naive(p, strategic=False):
    return constant_pi(p, p.pi_bar, "Naive", strategic)


def oracle(p, strategic=False):
    th, ta, tb = hjb.solve_known_regime(p, [p.pi_calm, p.pi_toxic], hjb.two_regime_generator(p),
                                        strategic=strategic)
    w = p.toxic_share
    return MM("Oracle", sim.SRC_REGIME, ta, tb, value0=(1 - w) * th[0, p.Q] + w * th[1, p.Q])


def _filter_par(p, pi_c, pi_t, nu_ct, nu_tc, eta):
    f = np.zeros(10)
    f[sim.F_NUCT], f[sim.F_NUTC] = nu_ct, nu_tc
    f[sim.F_PIC], f[sim.F_PIT] = pi_c, pi_t
    f[sim.F_A] = p.p_arrived
    f[sim.F_ETA] = eta
    f[sim.F_P0] = nu_ct / (nu_ct + nu_tc)
    return f


def learning(p, name="Learning", eta_f=1.0, pit_f=1.0, nu_f=1.0, strategic=False, n_u=N_U):
    """Certainty-equivalent learner. eta_f, pit_f, nu_f multiply the learner's assumed
    eta, pi_toxic and switching rates (1.0 = correctly specified)."""
    eta = p.eta * eta_f
    pi_c, pi_t = p.pi_calm, min(0.99, p.pi_toxic * pit_f)
    nu_ct, nu_tc = p.nu_ct * nu_f, p.nu_tc * nu_f
    pis = pi_c + np.linspace(0, 1, n_u) * (pi_t - pi_c)
    _, ta, tb = hjb.solve_known_regime(p, pis, np.zeros((n_u, n_u)), eta=eta, strategic=strategic)
    return MM(name, sim.SRC_BELIEF, ta, tb, _filter_par(p, pi_c, pi_t, nu_ct, nu_tc, eta),
              lik_tables(p, eta))


def bayes(p, name="Bayes", n_p=129):
    lf0, lg1 = lik_tables(p)
    w0, w1 = likelihood.observation_bins(lf0, lg1, p.p_arrived)
    th, pg, ta, tb = belief_hjb.solve_belief_hjb(p, w0, w1, n_p=n_p, n_store=N_U)
    v0 = np.interp(p.toxic_share, pg, th[p.Q])
    return MM(name, sim.SRC_BELIEF, ta, tb,
              _filter_par(p, p.pi_calm, p.pi_toxic, p.nu_ct, p.nu_tc, p.eta),
              (lf0, lg1), value0=v0)


def learning_timing(p, name="Learning+timing", n_u=17):
    """Extension: the certainty-equivalent learner, but with the full filter of the strategic
    model (it also learns from how often it is filled)."""
    m = learning(p, name=name, strategic=True, n_u=n_u)
    m.fpar[sim.F_TIMING] = 1.0
    return m


def bayes_strategic(p, name="Bayes"):
    """Extension: belief-state HJB for the strategic model (see belief_hjb_strategic.py)."""
    from . import belief_hjb_strategic as bs
    lik = lik_tables(p)
    th, pg, ta, tb = bs.solve(p, lik)
    f = _filter_par(p, p.pi_calm, p.pi_toxic, p.nu_ct, p.nu_tc, p.eta)
    f[sim.F_TIMING] = 1.0
    return MM(name, sim.SRC_BELIEF, ta, tb, f, lik, value0=np.interp(p.toxic_share, pg, th[p.Q]))


def pi_observer(p, pi_max=0.95, n_u=N_U, name="Oracle"):
    """Sees the true informed share pi_t (whatever process drives it) and quotes as if it
    were constant.  Used as the informed benchmark when the market is not a 2-regime chain."""
    pis = np.linspace(0, pi_max, n_u)
    _, ta, tb = hjb.solve_known_regime(p, pis, np.zeros((n_u, n_u)))
    f = np.zeros(10)
    f[sim.F_ULO], f[sim.F_UHI] = 0.0, pi_max
    return MM(name, sim.SRC_TRUEPI, ta, tb, f)


# ---------- alternative toxicity processes for the structural-misspecification tests ----------
def three_regime_market(p):
    """Calm / medium / toxic chain with the same mean informed share as the 2-regime model."""
    pis = np.array([0.05, 0.20, 0.55])
    share = np.array([0.65, 0.25, 0.10])
    r_tm = p.nu_tc                              # toxic -> medium
    r_mt = share[2] / share[1] * r_tm           # detailed balance
    r_mc = p.nu_tc / 2
    r_cm = share[1] / share[0] * r_mc
    G = np.array([[-r_cm, r_cm, 0.0], [r_mc, -(r_mc + r_mt), r_mt], [0.0, r_tm, -r_tm]])
    return dict(mode=0, G=G, pis=pis, p0=share, ou=np.zeros(4))


def continuous_market(p, cap=0.55):
    """Informed share = min(cap, exp(m + x)) with x an OU process: same mean, standard
    deviation and autocorrelation time as the 2-regime chain."""
    from scipy.optimize import fsolve
    w = p.toxic_share
    mean = p.pi_bar
    sd = np.sqrt(w * (1 - w)) * (p.pi_toxic - p.pi_calm)
    z, wz = np.polynomial.hermite_e.hermegauss(80)
    wz = wz / wz.sum()

    def moments(v):
        x = np.minimum(cap, np.exp(v[0] + v[1] * z))
        m1 = (wz * x).sum()
        return [m1 - mean, np.sqrt((wz * x * x).sum() - m1 * m1) - sd]

    m, s = fsolve(moments, [np.log(mean) - 0.4, 0.9])
    return dict(mode=1, G=np.zeros((1, 1)), pis=np.zeros(1), p0=np.ones(1),
                ou=np.array([m, abs(s), p.nu_ct + p.nu_tc, cap]))


@dataclass
class Result:
    names: list
    out: np.ndarray         # [paths, mm, stat]
    obj: np.ndarray         # [paths, mm] objective: pnl - alpha q_T^2 - phi * int q^2 dt
    pnl: np.ndarray         # [paths, mm] pnl after terminal liquidation penalty
    extra: dict

    def col(self, name):
        return self.names.index(name)


def run(p, mms, n_paths, seed, ideal=False, strategic=False, xi_mode=0, tox=None,
        n_rec=0, rec_mm=0, rec_dt=0.5, scramble_t=1e300, n_chunks=64):
    """Simulate all market makers in `mms` on the same paths."""
    nq = 2 * p.Q + 1
    nt = mms[0].tab_a.shape[0]
    nu_max = max(m.tab_a.shape[2] for m in mms)
    tab_a = np.zeros((len(mms), nt, nq, nu_max), dtype=np.float32)
    tab_b = np.zeros_like(tab_a)
    liks, lik_id = [], []
    for i, m in enumerate(mms):
        tab_a[i, :, :, :m.tab_a.shape[2]] = m.tab_a
        tab_b[i, :, :, :m.tab_b.shape[2]] = m.tab_b
        if m.lik is None:
            lik_id.append(0)
        else:
            for j, l in enumerate(liks):
                if l is m.lik:
                    lik_id.append(j); break
            else:
                liks.append(m.lik); lik_id.append(len(liks) - 1)
    if not liks:
        liks = [(np.zeros(4), np.zeros(4))]
    logf0 = np.array([l[0] for l in liks]); logg1 = np.array([l[1] for l in liks])
    if tox is None:
        tox = dict(mode=0, G=hjb.two_regime_generator(p), pis=np.array([p.pi_calm, p.pi_toxic]),
                   p0=np.array([1 - p.toxic_share, p.toxic_share]), ou=np.zeros(4))
    # path i of a run uses RNG seed  (seed * 10^7 + i) mod 2^32  (Numba seeds are 32-bit).
    # The seeds used in experiments/ never share paths; other pairs can (e.g. seeds 859 apart).
    res = sim._simulate(
        n_paths, int(seed) * 10_000_000, n_chunks, ideal, strategic, scramble_t,
        p.A, p.k, p.sigma, p.T, p.Q, p.eta, xi_mode, p.lam_tot, p.tau_D, p.h,
        tox["mode"], np.asarray(tox["G"], float), np.asarray(tox["pis"], float),
        np.asarray(tox["p0"], float), np.asarray(tox["ou"], float),
        np.array([m.src for m in mms]), np.array([m.tab_a.shape[2] for m in mms]),
        tab_a, tab_b, p.T / nt, np.array([m.fpar for m in mms]), np.array(lik_id),
        logf0, logg1, likelihood.M_LO, likelihood.DM,
        n_rec, rec_mm, rec_dt, 2.0, 30)
    out, es_sum, es_time, qhist, rec_p, rec_z, rec_m, rec_i, rec_t, rec_n = res
    pnl = out[:, :, sim.PNL] - p.alpha * out[:, :, sim.QT] ** 2
    obj = pnl - p.phi * out[:, :, sim.PEN]
    extra = dict(es_sum=es_sum.sum(0), es_time=es_time.sum(0), qhist=qhist.sum(0),
                 rec_p=rec_p, rec_z=rec_z, rec_m=rec_m, rec_i=rec_i, rec_t=rec_t, rec_n=rec_n)
    return Result([m.name for m in mms], out, obj, pnl, extra)
