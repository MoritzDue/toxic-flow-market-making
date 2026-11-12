# Market making under hidden toxicity regimes

A simulation study of a market maker whose order flow switches between a calm regime (5% of
incoming orders come from informed traders) and a toxic one (40%), without being told which.
An informed order is followed by a price jump in its direction, so filling it loses money on average.

**Question.** How much does the market maker lose by ignoring the switching, how much can it
win back by inferring the regime from the markouts of its own fills, and how close is a simple
"plug in the estimate" rule to the Bayes-optimal strategy?

## The five market makers

All are run on identical simulated paths, so differences between them are measured precisely.

- **Baseline** ignores adverse selection.
- **Naive** knows the average informed share and quotes for it.
- **Oracle** sees the regime.
- **Learning** runs a hidden-Markov filter on its own markouts and quotes as if its current
  estimate were the truth.
- **Bayes** solves the full control problem with the belief as a state variable.

## Two models of the informed trader

All experiments are run under two models that differ only in how informed traders decide
which quotes to hit. The market makers know which model applies; they do not observe the regime.

- **Standard.** Informed orders respond to the quote in the same way as uninformed ones: any
  order fills a quote at distance δ with probability e^{−kδ}, regardless of whether the
  subsequent jump exceeds δ. Widening reduces informed and uninformed fills in the same
  proportion, so the optimal response to toxicity is a moderate widening of the quotes. This
  is the textbook setting (Avellaneda–Stoikov, Cartea–Jaimungal–Penalva).
- **Strategic.** The informed trader knows the size of the subsequent jump and trades only if
  it exceeds the quote. When informed traders are less price-sensitive than uninformed ones,
  widening worsens the composition of the flow, and the optimal response to toxicity can be
  to withdraw quotes altogether.

## Main results

| | standard flow | strategic informed trader |
|---|---|---|
| Value of knowing the regime (Oracle over Naive) | 1.06% ± 0.02% of the objective | 2.59% ± 0.03% |
| Learning's gain over Naive, regimes of 20 fills | 0.15% of the objective | 0.59% |
| Share of the Oracle's value recovered by Learning, regimes of 20 fills | 14% ± 1% | 23% ± 1% |
| Same, regimes of 200 fills | 53% ± 2% | 66% ± 1% |
| Learning's gain as a share of Bayes's gain | 99% | 100% when toxicity is mild, 62% when severe (exploratory) |

Under standard assumptions, knowing the toxicity regime is worth about 1% of profit, because
profit is flat around the optimal quote; a closed-form approximation, ½(kη)²·Var(π), predicts
the simulated gap across 27 parameter sets (median ratio 0.95, range 0.58–1.12). Learning
becomes valuable when informed traders are strategic and less price-sensitive than uninformed
traders, so that the optimal response to toxicity is to withdraw quotes. In the most severe
settings, Naive loses 1.99 ticks/s inside toxic regimes, against 0.74/s for Learning and
0.10/s for the Oracle. When the market is toxic half of the time, the certainty-equivalent
learner withdraws its quotes, receives no further fills and therefore cannot detect a return
to the calm regime; the Bayes-optimal market maker continues to quote at an expected loss in
order to keep learning (objective 118 against −4). See [REPORT.md](REPORT.md).

## Central setting

400,000 simulated sessions of 600 s each. The objective is P&L net of inventory penalties,
in ticks (95% CI approximately ±0.5). Adverse selection is the average loss to informed
traders per fill.

**Standard model**

| | objective | P&L / s | fills | adverse selection / fill |
|---|---|---|---|---|
| Baseline | 968.3 | 1.773 | 367 | 0.46 |
| Naive | 978.3 | 1.778 | 318 | 0.46 |
| Learning | 979.8 | 1.781 | 319 | 0.45 |
| Bayes | 979.8 | 1.781 | 319 | 0.45 |
| Oracle | 988.7 | 1.796 | 322 | 0.39 |

**Strategic model** (the corresponding Bayes market maker is part of the exploratory
extension and is reported separately in REPORT.md)

| | objective | P&L / s | fills | adverse selection / fill |
|---|---|---|---|---|
| Baseline | 807.5 | 1.506 | 368 | 0.92 |
| Naive | 820.9 | 1.504 | 309 | 0.99 |
| Learning | 825.7 | 1.512 | 308 | 0.95 |
| Oracle | 842.1 | 1.538 | 304 | 0.81 |

Adverse selection per fill roughly doubles in the strategic model, since informed traders
trade only when the jump exceeds the quote.

![Share of the Oracle's edge recovered by learning, by regime length](figures/e2_switching_speed.png)

Everything is simulated. No market data, no machine learning. Nothing here is claimed as new:
the building blocks are textbook, and the work is in combining and measuring them.

## What is in the repo

| | |
|---|---|
| [REPORT.md](REPORT.md) | model, verification, results, limitations |
| [THEORY_NOTES.md](THEORY_NOTES.md) | derivations: HJB, adverse selection, the filter |
| `experiments/` | one script per experiment (E0–E8, X1), plus figures |
| `tests/` | unit tests and the verification gate |
| `results/`, `figures/` | outputs of the full run |

The model lives in `toxicmm/`:

```
toxicmm/params.py                 parameters (baseline from Guéant-Lehalle-Fernandez-Tapia 2013)
toxicmm/hjb.py                    optimal quotes when the regime is known
toxicmm/likelihood.py             distribution of a fill's markout
toxicmm/belief_hjb.py             Bayes-optimal quotes when the regime is hidden
toxicmm/belief_hjb_strategic.py   same, strategic informed trader (exploratory extension)
toxicmm/simulate.py               Monte Carlo simulator and the online filter (Numba)
toxicmm/market_makers.py          Baseline, Naive, Oracle, Learning, Bayes
```

## Run it

Tested with Python 3.13.

```
pip install -r requirements.txt
python -m pytest -q                       # about 2 minutes, includes the verification gate
PATH_SCALE=0.05 sh run_all.sh             # quick pass, roughly 45 minutes on 2 cores
sh run_all.sh                             # full run, roughly 2-3 hours on 2 cores
```

A single experiment: `python -m experiments.e1_main`. Results go to `results/*.json`, figures
to `figures/`.

## Main references

Avellaneda & Stoikov (2008); Guéant, Lehalle & Fernandez-Tapia (2013); Cartea, Jaimungal &
Penalva (2015); Glosten & Milgrom (1985). Full list in THEORY_NOTES.md.
