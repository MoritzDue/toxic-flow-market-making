"""E8: strategic informed trader (trades only if the coming jump exceeds the quote).
Repeats E1, E2 and E3. (E8a, informed intensity A exp(-k d), is the main model itself.)"""
import numpy as np
from toxicmm.params import Params
from experiments.common import n, run_cell, show, save
from experiments import e2_switching_speed, e3_severity


def main():
    p = Params()
    res = {}
    s, r, mms = run_cell(p, n(400_000), 801, strategic=True)
    show("E8 E1 real", s)
    es_rate = r.extra["es_sum"] / np.maximum(r.extra["es_time"][None], 1e-9)
    s["event_study"] = {"names": r.names, "bin_s": 2.0, "edge_rate": es_rate.tolist()}
    s["quotes"] = {m.name: {"ask_q0": m.tab_a[300, p.Q].tolist()} for m in mms}
    res["real"] = s
    s, _, _ = run_cell(p, n(200_000), 802, strategic=True, ideal=True)
    show("E8 E1 ideal", s)
    res["ideal"] = s
    save("e8_e1", res)
    e2_switching_speed.main(strategic=True, name="e8_e2", paths=100_000, seed0=820)
    e3_severity.main(strategic=True, name="e8_e3", paths=100_000, seed0=830)


if __name__ == "__main__":
    main()
