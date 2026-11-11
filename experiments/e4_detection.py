"""E4: how well and how fast does the filter detect the regime?"""
import numpy as np
from toxicmm.params import Params
from toxicmm import market_makers as mk
from experiments.common import n, save

DT = 0.5


def detection_stats(p, rec_p, rec_z):
    tox = rec_z > 0.5 * (p.pi_calm + p.pi_toxic)
    alarm = rec_p > 0.5
    lags, detected, episodes = [], 0, 0
    for i in range(tox.shape[0]):
        z = tox[i].astype(int)
        starts = np.where(np.diff(z) == 1)[0] + 1
        ends = np.where(np.diff(z) == -1)[0] + 1
        for s in starts:
            e = ends[ends > s]
            e = e[0] if len(e) else len(z)
            episodes += 1
            hit = np.where(alarm[i, s:e])[0]
            if len(hit):
                detected += 1
                lags.append(hit[0] * DT)
    lags = np.array(lags)
    order = np.argsort(rec_p.ravel())
    ranks = np.empty(order.size); ranks[order] = np.arange(order.size)
    t = tox.ravel()
    auc = (ranks[t].mean() - (t.sum() - 1) / 2) / (~t).sum()
    return dict(
        toxic_episodes=int(episodes),
        detected_share=detected / max(episodes, 1),
        median_lag_s=float(np.median(lags)), median_lag_fills=float(np.median(lags) * p.fill_rate0),
        mean_lag_s=float(lags.mean()),
        false_alarm_share_of_calm_time=float(alarm[~tox].mean()),
        missed_share_of_toxic_time=float((~alarm[tox]).mean()),
        mean_belief_calm=float(rec_p[~tox].mean()), mean_belief_toxic=float(rec_p[tox].mean()),
        brier=float(((rec_p - tox) ** 2).mean()), brier_prior=p.toxic_share * (1 - p.toxic_share),
        auc_belief_vs_regime=float(auc))


def main():
    res = {}
    for i, d in enumerate((5, 20, 100)):
        p = Params(dur_fills=d)
        nr = n(4000)
        r = mk.run(p, [mk.learning(p)], nr, 400 + i, n_rec=nr, rec_mm=0, rec_dt=DT)
        s = detection_stats(p, r.extra["rec_p"], r.extra["rec_z"])
        s["example_belief"] = r.extra["rec_p"][:3].tolist()
        s["example_regime"] = (r.extra["rec_z"][:3] > 0.2).astype(int).tolist()
        print("E4 duration", d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if not k.startswith("example")}, flush=True)
        res[str(d)] = s
    save("e4", res)


if __name__ == "__main__":
    main()
