"""X1 (exploratory extension, not part of the locked plan).

Strategic informed trader.  Compares three learners:
  Learning         certainty-equivalent quotes, filter uses markouts only (as in E8)
  Learning+timing  same quotes, filter also uses fill timing (the full filter)
  Bayes            belief-state HJB for the strategic model: knows that quoting wide stops
                   the flow of information, and that not being filled is itself evidence
"""
from toxicmm.params import Params
from toxicmm import market_makers as mk
from toxicmm.analysis import summarise
from toxicmm.stats import mean_ci
from experiments.common import n, save


def cell(p, n_paths, seed):
    B = mk.bayes_strategic(p)
    mms = [mk.naive(p, True), mk.oracle(p, True), mk.learning(p, strategic=True, n_u=17),
           mk.learning_timing(p), B]
    out = {}
    for world, ideal in (("ideal", True), ("real", False)):
        r = mk.run(p, mms, n_paths, seed + ideal, ideal=ideal, strategic=True)
        s = summarise(p, r)
        if ideal:       # verification: Monte Carlo objective of the Bayes policy vs its HJB value
            mc, ci = mean_ci(r.obj[:, 4])
            s["bayes_check"] = dict(mc=mc, ci=ci, hjb=float(B.value0), z=(mc - B.value0) / (ci / 1.96))
        out[world] = s
    out["quotes"] = {m.name: m.tab_a[300, p.Q].tolist() for m in mms[2:]}
    return out


def main():
    res = {}
    i = 0
    for eta in (1.0, 1.5, 2.0):
        for pt in (0.4, 0.55):
            out = cell(Params(eta_mult=eta, pi_toxic=pt), n(100_000), 900 + 2 * i)
            for w in ("ideal", "real"):
                s = out[w]
                print(f"X1 eta={eta} pi_tox={pt} {w}: gap {100 * s['gap_rel']:.1f}% | " +
                      " | ".join(f"{m} {s['mm'][m]['edge_rec']:.3f}±{s['mm'][m]['edge_rec_ci']:.3f}"
                                 for m in ("Learning", "Learning+timing", "Bayes")) +
                      (f" | Bayes MC-vs-HJB z={s['bayes_check']['z']:.2f}" if w == "ideal" else ""), flush=True)
            res[f"{pt}_{eta}"] = out
            i += 1
    save("x1", res)


def trap():
    """The one E8 cell where Naive and the certainty-equivalent learner both stop quoting
    (pi_toxic 0.55, eta 2 half-spreads, toxic half the time)."""
    out = cell(Params(eta_mult=2.0, pi_toxic=0.55, toxic_share=0.5), n(100_000), 950)
    for w in ("ideal", "real"):
        s = out[w]
        print(f"X1 trap {w}: " + " | ".join(
            f"{m} pnl/s {d['pnl_per_s']:.3f}±{d['pnl_per_s_ci']:.3f} fills {d['fills']:.0f}" for m, d in s["mm"].items()), flush=True)
    save("x1_trap", out)


if __name__ == "__main__":
    main()
    trap()
