import numpy as np
from toxicmm.params import Params
from toxicmm import likelihood, market_makers as mk, belief_hjb

P = Params()


def test_densities_integrate_to_one():
    lf0, lg1 = mk.lik_tables(P)
    assert abs(np.exp(lf0).sum() * likelihood.DM - 1) < 2e-3
    assert abs(np.exp(lg1).sum() * likelihood.DM - 1) < 2e-3


def test_markout_mean_shift():
    """An informed fill whose jump landed has mean markout eta; an uninformed one has mean 0."""
    lf0, lg1 = mk.lik_tables(P)
    m = likelihood.grid()
    f0 = np.exp(lf0) / np.exp(lf0).sum(); g1 = np.exp(lg1) / np.exp(lg1).sum()
    assert abs((m * f0).sum()) < 0.02
    assert abs((m * g1).sum() - P.eta) < 0.05


def test_single_markout_is_not_proof():
    lf0, lg1 = mk.lik_tables(P)
    acc, _, _ = likelihood.single_fill_classifier(lf0, lg1, P.p_arrived, 0.5)
    assert 0.6 < acc < 0.9


def test_belief_jump_matrix_is_a_martingale():
    """Rows sum to one and E[p'] = p: on average an observation does not move the belief."""
    lf0, lg1 = mk.lik_tables(P)
    w0, w1 = likelihood.observation_bins(lf0, lg1, P.p_arrived)
    pg = np.linspace(0, 1, 129)
    M = belief_hjb.belief_jump_matrix(pg, w0, w1, P.pi_calm, P.pi_toxic)
    assert np.allclose(M.sum(1), 1, atol=1e-9)
    assert np.allclose(M @ pg, pg, atol=1e-9)


def reference_filter(p, lik, fill_t, markouts, t_now):
    """Plain-Python version of the filter: uses only markouts already visible at t_now."""
    lf0, lg1 = lik
    grid = likelihood.grid()
    kap = p.nu_ct + p.nu_tc
    pinf = p.nu_ct / kap
    belief, t_f = pinf, 0.0
    for tau, m in zip(fill_t, markouts):
        if tau + p.h > t_now:
            break
        belief = pinf + (belief - pinf) * np.exp(-kap * (tau - t_f))
        r = (1 - p.p_arrived) + p.p_arrived * np.exp(np.interp(m, grid, lg1) - np.interp(m, grid, lf0))
        lt, lc = 1 + p.pi_toxic * (r - 1), 1 + p.pi_calm * (r - 1)
        belief = belief * lt / (belief * lt + (1 - belief) * lc)
        t_f = tau
    return pinf + (belief - pinf) * np.exp(-kap * (t_now - t_f))


def test_simulator_filter_matches_reference():
    L = mk.learning(P)
    ex = mk.run(P, [L], 2000, 9, n_rec=3, rec_mm=0).extra
    for i in range(3):
        k = ex["rec_n"][i]
        for g in (100, 400, 777, 1100):
            ref = reference_filter(P, L.lik, ex["rec_t"][i, :k], ex["rec_m"][i, :k], g * 0.5)
            assert abs(ref - ex["rec_p"][i, g]) < 1e-4
