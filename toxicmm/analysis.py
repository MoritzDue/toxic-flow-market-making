"""Turn raw simulation output into the numbers reported in the experiments."""
import json, os
import numpy as np
from . import simulate as sim
from .stats import mean_ci, edge_recovered

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def summarise(p, r, naive="Naive", oracle="Oracle"):
    """Per market maker: objective, PnL, secondary metrics, paired differences, edge recovered."""
    o = r.out
    iN = r.col(naive) if naive in r.names else None
    iO = r.col(oracle) if oracle in r.names else None
    tox_time = p.toxic_share * p.T
    res = {"n_paths": int(o.shape[0]), "mm": {}}
    for i, name in enumerate(r.names):
        d = {}
        d["obj"], d["obj_ci"] = mean_ci(r.obj[:, i])
        d["pnl"], d["pnl_ci"] = mean_ci(r.pnl[:, i])
        d["pnl_per_s"], d["pnl_per_s_ci"] = d["pnl"] / p.T, d["pnl_ci"] / p.T
        d["pnl_std"] = r.pnl[:, i].std()
        nf = o[:, i, sim.NFILL].mean()
        d["fills"] = nf
        d["spread_per_fill"] = o[:, i, sim.SPREAD].mean() / nf
        d["as_cost_per_fill"] = o[:, i, sim.ASCOST].mean() / nf
        d["informed_share"] = o[:, i, sim.NINF].mean() / nf
        d["mean_q2"] = o[:, i, sim.PEN].mean() / p.T
        nft = o[:, i, sim.NFILL_TOX].mean()
        d["fills_toxic"] = nft
        d["edge_per_fill_toxic"] = o[:, i, sim.EDGE_TOX].mean() / max(nft, 1e-12)
        d["edge_per_fill_calm"] = o[:, i, sim.EDGE_CALM].mean() / max(nf - nft, 1e-12)
        # fill edge earned while the regime is toxic / calm, per second of that regime
        d["edge_rate_toxic"], d["edge_rate_toxic_ci"] = [x / tox_time for x in mean_ci(o[:, i, sim.EDGE_TOX])]
        d["edge_rate_calm"], d["edge_rate_calm_ci"] = [x / (p.T - tox_time) for x in mean_ci(o[:, i, sim.EDGE_CALM])]
        if iN is not None:
            d["obj_vs_naive"], d["obj_vs_naive_ci"] = mean_ci(r.obj[:, i] - r.obj[:, iN])
            d["pnl_vs_naive"], d["pnl_vs_naive_ci"] = mean_ci(r.pnl[:, i] - r.pnl[:, iN])
        if iN is not None and iO is not None and i not in (iN, iO):
            d["edge_rec"], d["edge_rec_ci"] = edge_recovered(r.obj[:, i], r.obj[:, iN], r.obj[:, iO])
            d["edge_rec_pnl"], d["edge_rec_pnl_ci"] = edge_recovered(r.pnl[:, i], r.pnl[:, iN], r.pnl[:, iO])
        res["mm"][name] = {k: float(v) for k, v in d.items()}
    if iN is not None and iO is not None:
        g, gci = mean_ci(r.obj[:, iO] - r.obj[:, iN])
        res["gap"], res["gap_ci"] = float(g), float(gci)
        res["gap_rel"] = float(g / r.obj[:, iN].mean())      # gap as a fraction of Naive's objective
    return res


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def load(name):
    with open(os.path.join(RESULTS, name + ".json")) as f:
        return json.load(f)
