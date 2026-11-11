"""Distribution of a fill's markout, and what a single markout says about the fill.

Markout m = (mid move over the horizon h after the fill), signed so that m > 0 is a move
AGAINST the market maker.  Under the filter's model

    m = 1{informed} * 1{impact landed within h} * xi  +  noise,
    noise = sigma * sqrt(h) * N(0,1) + compound Poisson(rate lam_noise * h, sizes +-Exp(etaJ)),
    (the other price jumps that happen to fall in the window; etaJ = eta in the model),

so the density is f0 for an uninformed fill and  (1-a) f0 + a g1  for an informed one, where
a = P(impact within h) and g1 = f0 convolved with Exp(eta).  Densities are computed by plain
numerical convolution on a fine grid and stored as log-tables for the simulator.
"""
import numpy as np
from scipy.signal import fftconvolve

M_LO, M_HI, DM = -40.0, 60.0, 0.05         # table range for markouts (ticks)
_DX = 0.01                                  # internal convolution grid


def markout_tables(sigma, h, lam_noise, etaJ, eta):
    """Return (logf0, logg1) on the grid np.arange(M_LO, M_HI + DM/2, DM)."""
    L = 200.0
    x = np.arange(-L, L + _DX / 2, _DX)
    mid = len(x) // 2
    s = sigma * np.sqrt(h)
    gauss = np.exp(-0.5 * (x / s) ** 2)
    gauss /= gauss.sum() * _DX
    lap = np.exp(-np.abs(x) / etaJ) / (2 * etaJ)
    lap /= lap.sum() * _DX
    mu = lam_noise * h
    # compound Poisson part: sum_n P(N=n) * Laplace^{*n}; the n=0 term is a point mass at 0
    dens = np.zeros_like(x)
    conv = lap.copy()
    w = np.exp(-mu)
    p0 = w
    for n in range(1, 13):
        w *= mu / n
        dens += w * conv
        if n < 12:
            conv = fftconvolve(conv, lap, mode="same") * _DX
    f0 = p0 * gauss + fftconvolve(dens, gauss, mode="same") * _DX
    expo = np.where(x >= 0, np.exp(-x / eta) / eta, 0.0)
    expo /= expo.sum() * _DX
    g1 = fftconvolve(f0, expo, mode="same") * _DX
    grid = np.arange(M_LO, M_HI + DM / 2, DM)
    f0g = np.interp(grid, x, np.maximum(f0, 1e-300))
    g1g = np.interp(grid, x, np.maximum(g1, 1e-300))
    return np.log(f0g), np.log(g1g)


def grid():
    return np.arange(M_LO, M_HI + DM / 2, DM)


def single_fill_classifier(logf0, logg1, a, prior):
    """Best possible rule for 'was this fill informed?' from its markout alone.

    Returns (accuracy of the Bayes rule, accuracy of always guessing the majority class,
    threshold m* above which the rule says 'informed')."""
    m = grid()
    f0 = np.exp(logf0); f0 /= f0.sum() * DM
    g1 = np.exp(logg1); g1 /= g1.sum() * DM
    f1 = (1 - a) * f0 + a * g1
    acc = np.sum(np.maximum(prior * f1, (1 - prior) * f0)) * DM
    says_informed = prior * f1 > (1 - prior) * f0
    thr = m[says_informed].min() if says_informed.any() else np.inf
    return acc, max(prior, 1 - prior), thr


def auc(logf0, logg1, a):
    """P(markout of a random informed fill > markout of a random uninformed fill)."""
    f0 = np.exp(logf0); f0 /= f0.sum()
    g1 = np.exp(logg1); g1 /= g1.sum()
    f1 = (1 - a) * f0 + a * g1
    return float(np.sum(f1 * (np.cumsum(f0) - 0.5 * f0)))


def observation_bins(logf0, logg1, a, n_bins=160):
    """Discretise the markout into bins for the belief-state PDE.

    Returns (w0, w1): probability of each bin for an uninformed / informed fill.
    Bins have roughly equal probability under the 50/50 mixture."""
    f0 = np.exp(logf0); f0 /= f0.sum()
    g1 = np.exp(logg1); g1 /= g1.sum()
    f1 = (1 - a) * f0 + a * g1
    cdf = np.cumsum(0.5 * f0 + 0.5 * f1)
    idx = np.minimum((cdf * n_bins).astype(int), n_bins - 1)
    w0 = np.bincount(idx, weights=f0, minlength=n_bins)
    w1 = np.bincount(idx, weights=f1, minlength=n_bins)
    keep = (w0 + w1) > 0
    return w0[keep], w1[keep]
