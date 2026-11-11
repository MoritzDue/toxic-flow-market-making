"""E7: the market is not what the market makers assume.
Naive, Learning and Bayes keep the 2-regime model with the central parameters."""
from toxicmm.params import Params
from toxicmm import market_makers as mk
from toxicmm.analysis import summarise
from experiments.common import n, save


def main():
    p = Params()
    two_regime = [mk.naive(p), mk.oracle(p), mk.learning(p), mk.bayes(p)]
    observer = [mk.naive(p), mk.pi_observer(p), mk.learning(p), mk.bayes(p)]
    cases = {
        "a_continuous_toxicity": dict(mms=observer, tox=mk.continuous_market(p)),
        "b1_constant_jump_size": dict(mms=two_regime, xi_mode=1),
        "b2_lognormal_jump_size": dict(mms=two_regime, xi_mode=2),
        "c_three_regimes": dict(mms=observer, tox=mk.three_regime_market(p)),
    }
    res = {}
    for i, (name, kw) in enumerate(cases.items()):
        mms = kw.pop("mms")
        r = mk.run(p, mms, n(200_000), 700 + i, **kw)
        s = summarise(p, r)
        d = s["mm"]["Learning"]
        print(f"E7 {name}: gap {s['gap']:.2f}±{s['gap_ci']:.2f} | Learning-Naive {d['obj_vs_naive']:.2f}±{d['obj_vs_naive_ci']:.2f}"
              f" | edge {d['edge_rec']:.3f}±{d['edge_rec_ci']:.3f} | Bayes-Naive {s['mm']['Bayes']['obj_vs_naive']:.2f}", flush=True)
        res[name] = s
    save("e7", res)


if __name__ == "__main__":
    main()
