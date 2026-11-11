# Theory notes

## 1. Optimal quotes with a known informed share

The setup follows Avellaneda & Stoikov (2008), Guéant et al. (2013) and Cartea et al. (2015,
ch. 10).
- The mid S is a martingale. Quotes are S + δᵃ and S − δᵇ.
- Fills arrive with intensity A·e^{−kδ} per side, and inventory q ∈ {−Q, …, Q}.
- Objective: E[X_T + q_T S_T − α q_T² − φ ∫ q_t² dt]. The running penalty φ plays the role of
  risk aversion (Guéant, 2017).

With the ansatz V = x + qS + θ(t, q), S drops out and θ solves

    0 = ∂_t θ − φq² + sup_δᵃ A e^{−kδᵃ} [δᵃ − πη + θ(q−1) − θ(q)]
                    + sup_δᵇ A e^{−kδᵇ} [δᵇ − πη + θ(q+1) − θ(q)],      θ(T, q) = −αq².

Here π is the probability that a fill is informed, and η is the mean jump against the market
maker that follows an informed order. Since argmax_δ e^{−kδ}(δ + c) = 1/k − c, the optimal
quote is

    δ* = 1/k + πη − [θ(q∓1) − θ(q)].

The three terms are:
- **1/k:** the base half-spread.
- **πη:** the expected adverse-selection cost per fill, as in Glosten & Milgrom (1985).
- **θ(q∓1) − θ(q):** the inventory skew.

The system is 2Q+1 ODEs in t, integrated backward with RK4 (`toxicmm/hjb.py`). For π = 0,
the substitution ω = e^{kθ} makes the system linear, which `tests/test_hjb.py` uses as an
independent check.

With several regimes and a known regime z, each θ_z gains the coupling term
Σ_z' G_zz' (θ_z' − θ_z), where G is the generator of the regime chain.

## 2. Why knowing the regime is worth little

Take the profit rate per side, f(δ) = A e^{−kδ}(δ − πη). Quoting ε away from the optimum
gives

    f(δ* + ε) / f(δ*) = e^{−kε}(1 + kε) ≈ 1 − (kε)²/2.

The loss is second order because f′(δ*) = 0. Quoting for the average π̄ instead of the current
π_Z means ε = (π̄ − π_Z)η, so the expected relative loss is

    ½ (kη)² · Var(π_Z).

At kη = 1, π ∈ {0.05, 0.40} and a 25% toxic share, this gives 1.15%. The simulation measures
1.06%.

## 3. Strategic informed trader

Here the informed trader knows the jump ξ ~ Exp(η) and trades only if ξ > δ.
- It trades with probability e^{−δ/η}.
- By memorylessness, E[ξ − δ | ξ > δ] = η. A wider quote does not reduce the loss per informed
  fill, only the number of informed fills.

The ask term becomes

    A(1−π) e^{−kδ} (δ + Δθ) + A π e^{−δ/η} (Δθ − η).

If η > 1/k, widening loses uninformed flow faster than informed flow. The optimum in a toxic
regime can then be to stop quoting, so the loss from not knowing the regime becomes first
order. There is no closed form, so the sup is taken on a grid of δ.

## 4. Filtering the regime from markouts

The market maker tracks the belief p = P(toxic). The chain switches calm → toxic at rate
ν_ct and toxic → calm at rate ν_tc. This is a standard hidden-Markov filter with
irregularly timed observations (Elliott et al., 1995), implemented in `_filter_update` in `toxicmm/simulate.py`.

**Prediction between observations:**

    p ← p_∞ + (p − p_∞) e^{−(ν_ct + ν_tc)Δt},     p_∞ = ν_ct / (ν_ct + ν_tc).

**Update on a markout m.** The markout densities are:
- uninformed fill: f₀, the Brownian noise plus unrelated jumps;
- informed fill: (1−a)f₀ + a(f₀ ∗ Exp(η)), where a = P(the impact lands within the window).

With r(m) the likelihood ratio between the two (`toxicmm/likelihood.py`), the update is

    p ← p(1 + π_t(r−1)) / [ p(1 + π_t(r−1)) + (1−p)(1 + π_c(r−1)) ].

One observation moves the odds by at most π_t/π_c = 8 and at least (1−π_t)/(1−π_c) = 0.63.
This is why short regimes cannot be detected.

## 5. Quoting under a belief

**Certainty equivalence (Learning).** Quote with the constant-π solution of §1 at
π̂ = pπ_t + (1−p)π_c.

**Bayes-optimal.** Make the belief a state variable, V = x + qS + θ(t, q, p):

    0 = ∂_t θ − φq² + [ν_ct(1−p) − ν_tc p] ∂_p θ
        + sup_δ A e^{−kδ} [δ − π̂(p)η + E_m θ(t, q∓1, p′(m)) − θ(t, q, p)]   (each side).

- **The ∂_p θ term:** the prediction step of §4.
- **E_m θ(p′):** the expected value after the belief update triggered by a fill.

Because θ is convex in p, E_m θ(p′) ≥ θ(p). A fill is therefore worth slightly more than its
spread, since it is also an observation, so Bayes quotes slightly tighter in order to learn.
`toxicmm/belief_hjb.py` solves this on a grid in p, with markouts binned so that E_m is a
matrix.

**When the two differ.**
- **Standard model:** the value of information is second order, so the two rules are
  indistinguishable.
- **Strategic model:** pulling quotes also stops the observations, so the value of information
  is first order, and certainty equivalence falls behind (`toxicmm/belief_hjb_strategic.py`).

## References

- Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book.
  *Quantitative Finance* 8(3), 217–224.
- Cartea, Á., Jaimungal, S. and Penalva, J. (2015). *Algorithmic and High-Frequency Trading*.
  Cambridge University Press.
- Elliott, R. J., Aggoun, L. and Moore, J. B. (1995). *Hidden Markov Models: Estimation and
  Control*. Springer.
- Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist
  market with heterogeneously informed traders. *Journal of Financial Economics* 14(1), 71–100.
- Guéant, O. (2017). Optimal market making. *Applied Mathematical Finance* 24(2), 112–154.
- Guéant, O., Lehalle, C.-A. and Fernandez-Tapia, J. (2013). Dealing with the inventory risk:
  a solution to the market making problem. *Mathematics and Financial Economics* 7(4), 477–507.
