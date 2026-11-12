"""Model parameters.

Baseline (A, k, sigma, T) follow the numerical example of
Gueant, Lehalle & Fernandez-Tapia (2013), "Dealing with the inventory risk":
A = 0.9 /s, k = 0.3 /tick, sigma = 0.3 tick/sqrt(s), T = 600 s.
Two things differ from that example: risk aversion enters through the running inventory
penalty phi instead of a CARA parameter gamma, and the inventory bound is Q = 20.
Prices are in ticks, time in seconds.
"""
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Params:
    # --- baseline market (A, k, sigma, T from the GLFT 2013 example) ---
    A: float = 0.9          # fill intensity scale per side (1/s)
    k: float = 0.3          # intensity decay per tick of quote distance
    sigma: float = 0.3      # diffusion volatility of the mid (ticks / sqrt(s))
    T: float = 600.0        # horizon (s)
    Q: int = 20             # inventory bound
    # --- objective ---
    phi: float = 0.1        # running inventory penalty
    alpha: float = 0.5      # terminal liquidation penalty alpha * q_T^2
    # --- toxicity ---
    pi_calm: float = 0.05   # informed share of fills in the calm regime
    pi_toxic: float = 0.40  # informed share of fills in the toxic regime
    eta_mult: float = 1.0   # mean informed impact, in units of the half-spread 1/k
    toxic_share: float = 0.25   # long-run fraction of time spent in the toxic regime
    dur_fills: float = 20.0     # mean toxic-regime duration, in baseline fills
    # --- information events (price jumps) ---
    lam_tot: float = 1.0    # total jump rate of the mid (1/s), the same in both regimes
    # --- timing of impact and observation ---
    tau_D: float = 0.25     # mean delay between an informed order and its price jump (s)
    h: float = 0.5          # markout horizon (s)

    # ---------- derived quantities ----------
    @property
    def half_spread(self):          # risk-neutral optimal half-spread without adverse selection
        return 1.0 / self.k

    @property
    def eta(self):
        return self.eta_mult / self.k

    @property
    def fill_rate0(self):           # baseline fills per second, both sides, quoting at 1/k
        return 2.0 * self.A * np.exp(-1.0)

    @property
    def nu_tc(self):                # toxic -> calm switching rate
        return self.fill_rate0 / self.dur_fills

    @property
    def nu_ct(self):                # calm -> toxic switching rate
        w = self.toxic_share
        return self.nu_tc * w / (1.0 - w)

    @property
    def pi_bar(self):               # time-averaged informed share
        w = self.toxic_share
        return w * self.pi_toxic + (1 - w) * self.pi_calm

    @property
    def p_arrived(self):            # P(impact lands inside the markout window)
        return 1.0 - np.exp(-self.h / self.tau_D)

    def with_(self, **kw):
        return replace(self, **kw)
