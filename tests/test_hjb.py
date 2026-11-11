import numpy as np
from scipy.linalg import expm
from toxicmm.params import Params
from toxicmm import hjb

P = Params()


def test_constant_pi_matches_linear_ode():
    """For constant pi the substitution omega = exp(k theta) turns the HJB into a linear ODE
    (Cartea-Jaimungal / GLFT trick), solvable with a matrix exponential. The grid solver
    also enforces quotes >= 0, which can only lower the value, and only slightly."""
    th, _, _ = hjb.solve_known_regime(P, [P.pi_bar], [[0.0]])
    q = np.arange(-P.Q, P.Q + 1)
    M = np.diag(-P.k * P.phi * q ** 2.0)
    c = P.A * np.exp(-1 - P.k * P.pi_bar * P.eta)
    for i in range(len(q) - 1):
        M[i, i + 1] = M[i + 1, i] = c
    omega = expm(M * P.T) @ np.exp(-P.k * P.alpha * q ** 2.0)
    exact = np.log(omega[P.Q]) / P.k
    assert th[0, P.Q] <= exact + 1e-6
    assert abs(th[0, P.Q] - exact) < 0.05


def test_quotes_are_symmetric_and_widen_with_toxicity():
    _, ta, tb = hjb.solve_known_regime(P, [P.pi_calm, P.pi_toxic], hjb.two_regime_generator(P))
    it = 300
    for q in range(-5, 6):                                  # bid at q mirrors ask at -q
        assert np.allclose(ta[it, P.Q + q], tb[it, P.Q - q], atol=1e-5)
    assert ta[it, P.Q, 1] > ta[it, P.Q, 0]                  # toxic regime: wider
    assert ta[it, P.Q + 3, 0] < ta[it, P.Q, 0] < ta[it, P.Q - 3, 0]   # long: cheaper ask


def test_no_inventory_risk_gives_textbook_quote():
    """With phi = alpha = 0 the optimal distance is 1/k + pi * eta (far from the inventory bounds)."""
    p = P.with_(phi=0.0, alpha=0.0, Q=150)
    _, ta, _ = hjb.solve_known_regime(p, [0.3], [[0.0]])
    assert np.allclose(ta[300, p.Q - 5:p.Q + 6, 0], 1 / p.k + 0.3 * p.eta, atol=1e-3)


def test_equal_regimes_reduce_to_constant_pi():
    a, _, _ = hjb.solve_known_regime(P, [0.2, 0.2], hjb.two_regime_generator(P))
    b, _, _ = hjb.solve_known_regime(P, [0.2], [[0.0]])
    assert np.allclose(a[0], b[0], atol=1e-8) and np.allclose(a[1], b[0], atol=1e-8)


def test_strategic_equals_standard_without_informed_flow():
    a, _, _ = hjb.solve_known_regime(P, [0.0], [[0.0]], strategic=True)
    b, _, _ = hjb.solve_known_regime(P, [0.0], [[0.0]])
    assert abs(a[0, P.Q] - b[0, P.Q]) < 0.05               # quote grid of 0.05 ticks
