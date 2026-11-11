"""E3: severity, full factorial in (pi_toxic, eta, toxic time share)."""
from toxicmm.params import Params
from experiments.common import n, run_cell, show, save

PI_TOXIC = (0.2, 0.4, 0.55)
ETA = (0.5, 1.0, 2.0)
SHARE = (0.1, 0.25, 0.5)


def main(strategic=False, name="e3", paths=100_000, seed0=300):
    res = {}
    i = 0
    for pt in PI_TOXIC:
        for eta in ETA:
            for w in SHARE:
                p = Params(pi_toxic=pt, eta_mult=eta, toxic_share=w)
                s, _, _ = run_cell(p, n(paths), seed0 + i, strategic=strategic)
                # second-order approximation of the relative gap: 0.5 (k eta)^2 Var(pi)
                s["theory_gap_rel"] = 0.5 * eta ** 2 * w * (1 - w) * (pt - p.pi_calm) ** 2
                show(f"{name} pi_tox={pt} eta={eta} share={w}", s)
                res[f"{pt}_{eta}_{w}"] = s
                i += 1
    save(name, res)


if __name__ == "__main__":
    main()
