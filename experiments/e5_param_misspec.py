"""E5: Learning with wrong parameters (eta, pi_toxic, switching rates off by -50%..+50%).
All learners run on the same paths as Naive and Oracle."""
from toxicmm.params import Params
from toxicmm import market_makers as mk
from toxicmm.analysis import summarise
from experiments.common import n, save

FACTORS = (0.5, 0.75, 1.25, 1.5)


def main():
    p = Params()
    mms = [mk.naive(p), mk.oracle(p), mk.learning(p)]
    for f in FACTORS:
        mms.append(mk.learning(p, name=f"eta x{f}", eta_f=f))
        mms.append(mk.learning(p, name=f"pi_toxic x{f}", pit_f=f))
        mms.append(mk.learning(p, name=f"switch rates x{f}", nu_f=f))
    r = mk.run(p, mms, n(200_000), 500)
    s = summarise(p, r)
    for name, d in s["mm"].items():
        if "edge_rec" in d:
            print(f"E5 {name:20s} edge {d['edge_rec']:.3f}±{d['edge_rec_ci']:.3f}  vs Naive {d['obj_vs_naive']:.2f}±{d['obj_vs_naive_ci']:.2f}", flush=True)
    save("e5", s)


if __name__ == "__main__":
    main()
