"""Confidence intervals. All comparisons are paired (same paths for every market maker)."""
import numpy as np

Z95 = 1.96


def mean_ci(x):
    """Mean and 95% half-width."""
    x = np.asarray(x, float)
    return x.mean(), Z95 * x.std(ddof=1) / np.sqrt(len(x))


def edge_recovered(x, naive, oracle):
    """(x - naive) / (oracle - naive) on paired samples, with a delta-method 95% half-width."""
    a = np.asarray(x) - np.asarray(naive)
    b = np.asarray(oracle) - np.asarray(naive)
    n = len(a)
    R = a.mean() / b.mean()
    cov = np.cov(a, b)
    var = (cov[0, 0] - 2 * R * cov[0, 1] + R * R * cov[1, 1]) / (n * b.mean() ** 2)
    return R, Z95 * np.sqrt(max(var, 0.0))
