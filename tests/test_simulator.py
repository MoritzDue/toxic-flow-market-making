import numpy as np
from toxicmm.params import Params
from toxicmm import market_makers as mk, simulate as sim

P = Params()


def constant_quote_mm(p, d):
    nq = 2 * p.Q + 1
    tab = np.full((600, nq, 1), d, dtype=np.float32)
    return mk.MM("const", sim.SRC_CONST, tab, tab.copy())


def test_fill_rate_and_informed_share():
    d = 3.0
    p = P.with_(Q=150)                    # wide inventory bounds so no fill is ever refused
    r = mk.run(p, [constant_quote_mm(p, d)], 4000, 21)
    fills = r.out[:, 0, sim.NFILL].mean()
    expected = 2 * P.A * np.exp(-P.k * d) * P.T
    assert abs(fills / expected - 1) < 0.01
    share = r.out[:, 0, sim.NINF].sum() / r.out[:, 0, sim.NFILL].sum()
    assert abs(share - P.pi_bar) < 0.005


def test_pnl_is_spread_minus_adverse_selection_on_average():
    """The mid is a martingale the market maker cannot move, so E[pnl] = E[spread - jumps on traded units]."""
    r = mk.run(P, [constant_quote_mm(P, 3.0)], 60000, 22, ideal=True)
    pnl = r.out[:, 0, sim.PNL]
    edge = r.out[:, 0, sim.SPREAD] - r.out[:, 0, sim.ASCOST]
    diff = pnl - edge
    assert abs(diff.mean()) < 3 * diff.std() / np.sqrt(len(diff))


def test_public_price_does_not_reveal_the_regime():
    """Markouts of UNINFORMED fills must look the same in calm and toxic regimes."""
    n = 1500
    ex = mk.run(P, [constant_quote_mm(P, 0.0)], n, 23, n_rec=n, rec_mm=0).extra
    calm, toxic = [], []
    for i in range(n):
        k = ex["rec_n"][i]
        t, m, inf = ex["rec_t"][i, :k], ex["rec_m"][i, :k], ex["rec_i"][i, :k]
        z = ex["rec_z"][i, np.minimum((t / 0.5).astype(int), ex["rec_z"].shape[1] - 1)] > 0.2
        calm.append(np.abs(m[(inf == 0) & ~z])); toxic.append(np.abs(m[(inf == 0) & z]))
    calm, toxic = np.concatenate(calm), np.concatenate(toxic)
    se = np.sqrt(calm.var() / len(calm) + toxic.var() / len(toxic))
    assert abs(calm.mean() - toxic.mean()) < 4 * se


def test_same_seed_same_result():
    a = mk.run(P, [mk.naive(P)], 3000, 5).obj
    b = mk.run(P, [mk.naive(P)], 3000, 5, n_chunks=7).obj
    assert np.array_equal(a, b)
