"""Shared helpers for the experiment scripts.  Run scripts from the repo root, e.g.
    python -m experiments.e1_main
Set the environment variable PATH_SCALE (e.g. 0.05) for a quick low-precision run."""
import os, time
from toxicmm.params import Params
from toxicmm import market_makers as mk
from toxicmm.analysis import summarise, save, load

SCALE = float(os.environ.get("PATH_SCALE", "1"))


def n(paths):
    return max(2000, int(paths * SCALE))


def standard_mms(p, strategic=False, with_bayes=True, with_baseline=True):
    mms = [mk.baseline(p, strategic)] if with_baseline else []
    mms += [mk.naive(p, strategic), mk.oracle(p, strategic),
            mk.learning(p, strategic=strategic, n_u=17 if strategic else mk.N_U)]
    if with_bayes and not strategic:
        mms.append(mk.bayes(p))
    return mms


def run_cell(p, n_paths, seed, strategic=False, ideal=False, **kw):
    """Solve all market makers for parameters p, simulate, return the summary dict."""
    t0 = time.time()
    mms = standard_mms(p, strategic, **kw)
    r = mk.run(p, mms, n_paths, seed, ideal=ideal, strategic=strategic)
    s = summarise(p, r)
    s["seconds"] = time.time() - t0
    s["hjb_value"] = {m.name: float(m.value0) for m in mms}
    return s, r, mms


def show(label, s):
    line = f"{label}: gap {s['gap']:.2f}±{s['gap_ci']:.2f} ({100 * s['gap_rel']:.2f}% of Naive)"
    for name in ("Learning", "Bayes"):
        if name in s["mm"]:
            d = s["mm"][name]
            line += f" | {name} edge {d['edge_rec']:.3f}±{d['edge_rec_ci']:.3f}"
    print(line + f" [{s['seconds']:.0f}s, n={s['n_paths']}]", flush=True)
