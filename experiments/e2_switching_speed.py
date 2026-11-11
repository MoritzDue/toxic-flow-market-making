"""E2: how long must regimes last (in fills) before learning pays?"""
from toxicmm.params import Params
from experiments.common import n, run_cell, show, save

DURATIONS = (2, 5, 10, 20, 50, 100, 200)


def main(strategic=False, name="e2", paths=200_000, seed0=200):
    res = {}
    for i, d in enumerate(DURATIONS):
        s, _, _ = run_cell(Params(dur_fills=d), n(paths), seed0 + i, strategic=strategic, with_baseline=False)
        show(f"{name} duration={d} fills", s)
        res[str(d)] = s
    save(name, res)


if __name__ == "__main__":
    main()
