"""E6: inventory penalty phi at four levels."""
from toxicmm.params import Params
from experiments.common import n, run_cell, show, save


def main():
    res = {}
    for i, phi in enumerate((0.01, 0.03, 0.1, 0.3)):
        s, r, _ = run_cell(Params(phi=phi), n(200_000), 600 + i, with_baseline=False)
        qh = r.extra["qhist"][0]
        s["time_at_inventory_bound"] = float((qh[0] + qh[-1]) / qh.sum())
        show(f"E6 phi={phi}", s)
        res[str(phi)] = s
    save("e6", res)


if __name__ == "__main__":
    main()
