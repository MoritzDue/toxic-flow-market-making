"""E0: verification gate.
(a) Monte Carlo objective of the known-regime policies equals the HJB value (ideal world).
(b) Same for the Bayes-optimal policy and the belief-state HJB.
(c) A single markout cannot identify an informed fill; accuracy of the best rule.
(d) No look-ahead: changing the future does not change past beliefs or fills.
"""
import numpy as np
from toxicmm.params import Params
from toxicmm import market_makers as mk, likelihood
from toxicmm.stats import mean_ci
from experiments.common import n, save

PARAM_SETS = {
    "central": Params(),
    "severe": Params(pi_toxic=0.55, eta_mult=2.0, toxic_share=0.5, dur_fills=5, phi=0.03),
    "mild": Params(pi_toxic=0.2, eta_mult=0.5, dur_fills=100, phi=0.3),
}


def check_values(p, n_paths, seed, strategic=False):
    """MC objective vs HJB value for Naive, Oracle (and Bayes), in the ideal world."""
    mms = [mk.naive(p, strategic), mk.oracle(p, strategic)]
    if not strategic:
        mms.append(mk.bayes(p))
    r = mk.run(p, mms, n_paths, seed, ideal=True, strategic=strategic)
    out = {}
    for i, m in enumerate(mms):
        mc, ci = mean_ci(r.obj[:, i])
        out[m.name] = dict(mc=mc, ci=ci, hjb=float(m.value0), z=(mc - m.value0) / (ci / 1.96),
                           inside_ci=bool(abs(mc - m.value0) <= ci))
    return out


def classifier(p, n_rec=4000, seed=11):
    """Theory and simulation: how well can one markout tell informed from uninformed?"""
    lf0, lg1 = mk.lik_tables(p)
    a = p.p_arrived
    res = {"auc": likelihood.auc(lf0, lg1, a)}
    for prior in (p.pi_calm, p.pi_toxic, 0.5):
        acc, trivial, thr = likelihood.single_fill_classifier(lf0, lg1, a, prior)
        res[f"prior_{prior}"] = dict(accuracy=acc, always_majority=trivial, threshold=float(thr))
    # simulated real markouts of the Learning market maker
    r = mk.run(p, [mk.learning(p)], n_rec, seed, n_rec=n_rec, rec_mm=0)
    m, inf = [], []
    for i in range(n_rec):
        k = r.extra["rec_n"][i]
        m.append(r.extra["rec_m"][i, :k]); inf.append(r.extra["rec_i"][i, :k])
    m, inf = np.concatenate(m), np.concatenate(inf).astype(bool)
    thr = res["prior_0.5"]["threshold"]
    res["simulated_balanced_accuracy"] = 0.5 * ((m[inf] >= thr).mean() + (m[~inf] < thr).mean())
    res["simulated_fills"] = int(len(m))
    return res


def look_ahead(p, n_rec=200, seed=5, t_star=300.0):
    """Scramble every random input after t_star. Everything recorded before t_star must be
    identical; things after it must change (otherwise the test has no power)."""
    out = {}
    for mm in (mk.learning(p), mk.bayes(p)):
        a = mk.run(p, [mm], n_rec, seed, n_rec=n_rec, rec_mm=0).extra
        b = mk.run(p, [mm], n_rec, seed, n_rec=n_rec, rec_mm=0, scramble_t=t_star).extra
        g = int(t_star / 0.5)
        same_belief = bool(np.array_equal(a["rec_p"][:, :g], b["rec_p"][:, :g]))
        same_fills = True
        for i in range(n_rec):
            ta = a["rec_t"][i][(a["rec_t"][i] > 0) & (a["rec_t"][i] <= t_star)]
            tb = b["rec_t"][i][(b["rec_t"][i] > 0) & (b["rec_t"][i] <= t_star)]
            same_fills &= bool(np.array_equal(ta, tb))
        differs_after = bool(np.mean(a["rec_p"][:, g + 20:] != b["rec_p"][:, g + 20:]) > 0.5)
        out[mm.name] = dict(same_belief_before=same_belief, same_fills_before=same_fills,
                            differs_after=differs_after)
    return out


def main():
    res = {"a_b": {}, "c": {}, "d": None}
    for i, (name, p) in enumerate(PARAM_SETS.items()):
        res["a_b"][name] = check_values(p, n(400_000), 1000 + i)
        print(name, {k: (round(v["mc"], 3), round(v["hjb"], 3), round(v["z"], 2)) for k, v in res["a_b"][name].items()}, flush=True)
    res["a_b"]["strategic_central"] = check_values(Params(), n(200_000), 1010, strategic=True)
    print("strategic", {k: round(v["z"], 2) for k, v in res["a_b"]["strategic_central"].items()}, flush=True)
    for h in (0.25, 0.5, 1.0):
        res["c"][str(h)] = classifier(Params(h=h))
        print("classifier h", h, res["c"][str(h)], flush=True)
    res["d"] = look_ahead(Params())
    print("look-ahead", res["d"], flush=True)
    save("e0", res)


if __name__ == "__main__":
    main()
