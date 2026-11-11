"""E1: main comparison at the central setting (real and ideal world), plus the markout horizon."""
import numpy as np
from toxicmm.params import Params
from experiments.common import n, run_cell, show, save


def main():
    p = Params()
    res = {}
    s, r, mms = run_cell(p, n(400_000), 101)
    show("E1 real", s)
    # extras for figures: inventory distribution, fill edge around regime switches, quotes
    s["q_hist"] = (r.extra["qhist"] / r.extra["qhist"].sum(1, keepdims=True)).tolist()
    es_rate = r.extra["es_sum"] / np.maximum(r.extra["es_time"][None], 1e-9)
    s["event_study"] = {"names": r.names, "bin_s": 2.0, "edge_rate": es_rate.tolist()}
    it = 300
    s["quotes"] = {m.name: {"ask_q0": m.tab_a[it, p.Q].tolist(),
                            "ask_by_q": m.tab_a[it, p.Q - 6:p.Q + 7].tolist()} for m in mms}
    res["real"] = s
    s, _, _ = run_cell(p, n(400_000), 102, ideal=True)
    show("E1 ideal", s)
    res["ideal"] = s
    res["h"] = {}
    for i, h in enumerate((0.25, 0.5, 1.0)):
        s, _, _ = run_cell(p.with_(h=h), n(200_000), 110 + i, with_baseline=False)
        show(f"E1 h={h}", s)
        res["h"][str(h)] = s
    save("e1", res)


if __name__ == "__main__":
    main()
