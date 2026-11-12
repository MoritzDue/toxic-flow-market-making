"""Draws every figure in figures/ from the JSON files in results/."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from toxicmm.analysis import load
from toxicmm.params import Params

FIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
# one fixed colour per market maker, in every figure
COL = {"Naive": "#2a78d6", "Oracle": "#eb6834", "Learning": "#1baf7a", "Bayes": "#eda100",
       "Baseline": "#e87ba4"}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e6e5df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False, "figure.dpi": 130,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def done(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def have(name):
    try:
        return load(name)
    except FileNotFoundError:
        return None


def fig_e1():
    e1, e8 = load("e1"), have("e8_e1")
    panels = [("Standard model, realistic world", e1["real"]), ("Standard model, ideal world", e1["ideal"])]
    if e8:
        panels.append(("Strategic informed trader, realistic world", e8["real"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    for ax, (title, s) in zip(np.atleast_1d(axes), panels):
        names = [m for m in ("Baseline", "Learning", "Bayes", "Oracle") if m in s["mm"]]
        base = s["mm"]["Naive"]["obj"]
        vals = [100 * s["mm"][m]["obj_vs_naive"] / base for m in names]
        errs = [100 * s["mm"][m]["obj_vs_naive_ci"] / base for m in names]
        ax.bar(names, vals, yerr=errs, color=[COL[m] for m in names], width=0.55, capsize=3,
               error_kw=dict(ecolor=INK, lw=1))
        ax.axhline(0, color=MUTED, lw=1)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:+.2f}%", (i, v), xytext=(0, 8 if v >= 0 else -14), textcoords="offset points", ha="center")
        ax.set_title(title)
        ax.set_ylabel("objective vs Naive (% of Naive)")
        ax.margins(y=0.25)
    done(fig, "e1_main.png")


def fig_e2():
    e2, e8 = load("e2"), have("e8_e2")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    d = sorted(e2, key=float)
    x = [float(k) for k in d]
    ax = axes[0]
    for m in ("Learning", "Bayes"):
        y = [e2[k]["mm"][m]["edge_rec"] for k in d]
        ci = [e2[k]["mm"][m]["edge_rec_ci"] for k in d]
        ax.errorbar(x, y, yerr=ci, color=COL[m], marker="o", capsize=3, label=f"{m} (standard model)")
    if e8:
        y = [e8[k]["mm"]["Learning"]["edge_rec"] for k in d]
        ci = [e8[k]["mm"]["Learning"]["edge_rec_ci"] for k in d]
        ax.errorbar(x, y, yerr=ci, color=COL["Learning"], marker="s", ls="--", capsize=3, label="Learning (strategic)")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("mean toxic-regime duration (fills)"); ax.set_ylabel("edge recovered")
    ax.set_title("Share of the Oracle-Naive gap recovered"); ax.legend()
    ax = axes[1]
    ax.plot(x, [100 * e2[k]["gap_rel"] for k in d], color=COL["Oracle"], marker="o", label="standard model")
    if e8:
        ax.plot(x, [100 * e8[k]["gap_rel"] for k in d], color=COL["Oracle"], marker="s", ls="--", label="strategic")
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("mean toxic-regime duration (fills)"); ax.set_ylabel("Oracle - Naive (% of Naive)")
    ax.set_title("Size of the gap itself"); ax.legend()
    done(fig, "e2_switching_speed.png")


def fig_e3():
    e3, e8 = load("e3"), have("e8_e3")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7))
    ax = axes[0]
    th = np.array([100 * v["theory_gap_rel"] for v in e3.values()])
    g = np.array([100 * v["gap_rel"] for v in e3.values()])
    ax.scatter(th, g, color=COL["Oracle"], s=36, edgecolor="white", linewidth=1, label="standard model", zorder=3)
    if e8:
        g8 = np.array([100 * v["gap_rel"] for v in e8.values()])
        ax.scatter(th, g8, color=COL["Baseline"], marker="s", s=36, edgecolor="white", linewidth=1, label="strategic", zorder=3)
    lim = [th.min() * 0.6, max(th.max(), 100) * 1.5]
    ax.plot(lim, lim, color=MUTED, lw=1, ls=":")
    ax.annotate("gap = formula", (lim[1] * 0.3, lim[1] * 0.4), color=MUTED, rotation=38, fontsize=8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"second-order formula  $\frac{1}{2}(k\eta)^2\,\mathrm{Var}(\pi)$  (%)")
    ax.set_ylabel("measured Oracle - Naive (% of Naive)")
    ax.set_title("Cost of ignoring regimes, 27 severity cells"); ax.legend()
    ax = axes[1]
    for src, mk_, lab in ((e3, "o", "standard model"), (e8, "s", "strategic")):
        if not src:
            continue
        xs = [100 * v["gap_rel"] for v in src.values()]
        ys = [v["mm"]["Learning"]["edge_rec"] for v in src.values()]
        cs = [v["mm"]["Learning"]["edge_rec_ci"] for v in src.values()]
        keep = [i for i in range(len(xs)) if cs[i] < 0.25]      # drop cells where the ratio is noise
        ax.errorbar([xs[i] for i in keep], [ys[i] for i in keep], yerr=[cs[i] for i in keep], fmt=mk_,
                    color=COL["Learning"], mfc=COL["Learning"] if mk_ == "o" else "white", capsize=2, lw=1, label=lab)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Oracle - Naive (% of Naive)"); ax.set_ylabel("edge recovered by Learning")
    ax.set_title("Learning recovers more where the gap is larger?"); ax.legend()
    done(fig, "e3_severity.png")


def fig_e4():
    e4 = load("e4")
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 5.6), sharex=True)
    for ax, d in zip(axes, ("5", "20", "100")):
        p = np.array(e4[d]["example_belief"][0]); z = np.array(e4[d]["example_regime"][0])
        t = np.arange(len(p)) * 0.5
        ax.fill_between(t, 0, 1, where=z > 0, color=COL["Oracle"], alpha=0.18, lw=0, label="toxic regime (true)")
        ax.plot(t, p, color=COL["Learning"], lw=1.5, label="filtered P(toxic)")
        ax.set_ylim(0, 1); ax.set_ylabel("belief")
        s = e4[d]
        ax.set_title(f"Toxic regimes last {d} fills on average: {100 * s['detected_share']:.0f}% detected, "
                     f"median lag {s['median_lag_fills']:.0f} fills, false alarms {100 * s['false_alarm_share_of_calm_time']:.0f}% of calm time")
    axes[0].legend(ncol=2, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    done(fig, "e4_detection.png")


def fig_event_study():
    panels = [("Standard model", load("e1")["real"])]
    if have("e8_e1"):
        panels.append(("Strategic informed trader", load("e8_e1")["real"]))
    fig, axes = plt.subplots(len(panels), 2, figsize=(9.5, 3.3 * len(panels)), squeeze=False)
    for row, (title, s) in enumerate(panels):
        es = s["event_study"]
        rate = np.array(es["edge_rate"])
        t = (np.arange(rate.shape[2]) + 0.5) * es["bin_s"]
        for col, lab in ((0, "after a switch INTO the toxic regime"), (1, "after a switch back to calm")):
            ax = axes[row, col]
            for i, m in enumerate(es["names"]):
                if m == "Baseline":
                    continue
                ax.plot(t[:20], rate[i, col, :20], color=COL[m], label=m, lw=1.8)
            ax.set_title(f"{title}: {lab}")
            ax.set_xlabel("seconds since the switch"); ax.set_ylabel("fill edge (ticks / s)")
        axes[row, 1].legend(ncol=2, loc="lower right")
    done(fig, "event_study_switches.png")


def fig_e5():
    e5 = load("e5")["mm"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    f = [0.5, 0.75, 1.0, 1.25, 1.5]
    for key, lab, c, mk_ in (("eta", r"impact size $\eta$", "#2a78d6", "o"), ("pi_toxic", r"$\pi_{toxic}$", "#eb6834", "s"),
                             ("switch rates", "switching rates", "#6250d6", "^")):
        y = [e5["Learning" if x == 1.0 else f"{key} x{x}"]["edge_rec"] for x in f]
        ci = [e5["Learning" if x == 1.0 else f"{key} x{x}"]["edge_rec_ci"] for x in f]
        ax.errorbar(f, y, yerr=ci, marker=mk_, color=c, capsize=3, label=lab)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(f); ax.set_xticklabels(["-50%", "-25%", "correct", "+25%", "+50%"])
    ax.set_xlabel("error in the learner's assumed parameter"); ax.set_ylabel("edge recovered")
    ax.set_title("Learning with a wrong parameter"); ax.legend(title="wrong parameter")
    done(fig, "e5_param_misspec.png")


def fig_quotes():
    e1 = load("e1")["real"]["quotes"]
    p = np.linspace(0, 1, len(e1["Learning"]["ask_q0"]))
    panels = [("Standard model", e1)]
    if have("e8_e1"):
        panels.append(("Strategic informed trader", load("e8_e1")["real"]["quotes"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.5))
    for ax, (title, q) in zip(np.atleast_1d(axes), panels):
        for m in ("Learning", "Bayes"):
            if m in q:
                y = np.minimum(np.array(q[m]["ask_q0"]), 12)
                ax.plot(np.linspace(0, 1, len(y)), y, color=COL[m], label=m)
        ax.axhline(q["Naive"]["ask_q0"][0], color=COL["Naive"], ls="--", lw=1.5, label="Naive")
        oc, ot = q["Oracle"]["ask_q0"][:2]
        ax.plot([0, 1], [oc, min(ot, 12)], "o", color=COL["Oracle"], label="Oracle (calm, toxic)")
        ax.set_xlabel("belief  P(toxic)"); ax.set_ylabel("ask distance from mid at q = 0 (ticks)")
        ax.set_title(title + (" (capped at 12)" if ot > 12 else "")); ax.legend()
    done(fig, "quotes_vs_belief.png")


def fig_h3():
    """Profit by toxicity level: total PnL per second and fill edge earned while toxic."""
    out = [("Standard model", load("e3"))]
    if have("e8_e3"):
        out.append(("Strategic informed trader", load("e8_e3")))
    fig, axes = plt.subplots(len(out), 2, figsize=(9.5, 3.4 * len(out)), squeeze=False)
    for row, (title, e) in enumerate(out):
        keys = [k for k in e if k.endswith("_0.25")]            # toxic share 25%
        sev = [float(k.split("_")[0]) * float(k.split("_")[1]) for k in keys]   # pi_toxic * eta / (1/k)
        order = np.argsort(sev)
        x = np.arange(len(keys))
        labels = [f"{keys[i].split('_')[0]}\n{keys[i].split('_')[1]}" for i in order]
        for col, (field, ylab) in enumerate((("pnl_per_s", "mean P&L (ticks / s)"),
                                             ("edge_rate_toxic", "fill edge while toxic (ticks / s)"))):
            ax = axes[row, col]
            for m in ("Naive", "Learning", "Oracle"):
                y = [e[keys[i]]["mm"][m][field] for i in order]
                ci = [e[keys[i]]["mm"][m][field + "_ci"] for i in order]
                ax.errorbar(x, y, yerr=ci, color=COL[m], marker="o", capsize=2, label=m, lw=1.5)
            ax.axhline(0, color=INK, lw=1)
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
            ax.set_xlabel(r"$\pi_{toxic}$ (top) and $\eta$ in half-spreads (bottom), sorted by their product")
            ax.set_ylabel(ylab); ax.set_title(f"{title}: {'whole session' if col == 0 else 'inside toxic regimes'}")
        axes[row, 0].legend()
    done(fig, "h3_profitability.png")


def fig_x1():
    x1 = load("x1")
    keys = sorted(x1, key=lambda k: x1[k]["real"]["gap_rel"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    ax = axes[0]
    w = 0.26
    names = [("Learning", COL["Learning"], "Learning (markouts only)"),
             ("Learning+timing", "#6250d6", "Learning + fill timing"), ("Bayes", COL["Bayes"], "Bayes-optimal")]
    for j, (m, c, lab) in enumerate(names):
        y = [x1[k]["real"]["mm"][m]["edge_rec"] for k in keys]
        ci = [x1[k]["real"]["mm"][m]["edge_rec_ci"] for k in keys]
        ax.bar(np.arange(len(keys)) + (j - 1) * w, y, w * 0.92, yerr=ci, color=c, label=lab, capsize=2,
               error_kw=dict(ecolor=INK, lw=1))
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels([f"{k.split('_')[0]}, {k.split('_')[1]}\ngap {100 * x1[k]['real']['gap_rel']:.0f}%" for k in keys], fontsize=8)
    ax.set_xlabel(r"$\pi_{toxic}$, $\eta$ in half-spreads, and the Oracle - Naive gap")
    ax.set_ylabel("edge recovered"); ax.set_title("Exploratory: strategic informed trader"); ax.legend(fontsize=8)
    ax = axes[1]
    q = x1["0.55_2.0"]["quotes"]
    for m, c, lab in (("Learning", COL["Learning"], "certainty equivalent"), ("Bayes", COL["Bayes"], "Bayes-optimal")):
        y = np.array(q[m]); pulled = y > 20
        pp = np.linspace(0, 1, len(y))
        ax.plot(pp[~pulled], y[~pulled], color=c, marker="o", ms=3, label=lab)
        if pulled.any():
            ax.axvspan(pp[pulled][0], 1, ymin=0.92 if m == "Bayes" else 0.84, ymax=1.0 if m == "Bayes" else 0.92, color=c, alpha=0.5, lw=0)
    ax.annotate("coloured bars: quotes pulled", (0.99, 0.80), xycoords="axes fraction", ha="right", color=MUTED, fontsize=8)
    ax.set_xlabel("belief  P(toxic)"); ax.set_ylabel("ask distance at q = 0 (ticks)")
    ax.set_title(r"Quotes at $\pi_{toxic}$ = 0.55, $\eta$ = 2 half-spreads"); ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(3.5, 9.5)
    done(fig, "x1_extension.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    for f in (fig_x1, fig_e1, fig_e2, fig_e3, fig_e4, fig_event_study, fig_e5, fig_quotes, fig_h3):
        try:
            f()
        except FileNotFoundError as err:
            print("skipped", f.__name__, "-", err)


if __name__ == "__main__":
    main()
