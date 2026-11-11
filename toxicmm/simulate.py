"""Monte Carlo simulator (Numba).

One path of the market:
  * toxicity path: the informed share pi(t) of the order flow (2-state Markov chain by default)
  * candidate orders arrive at rate A per side.  A market maker quoting at distance d is
    filled by an uninformed candidate with probability exp(-k d) (thinning), so its fill
    intensity is exactly A exp(-k d) with continuously updated quotes.
  * each candidate is informed with probability pi(t).  An informed order is followed by a
    price jump of size xi in its own direction, WHETHER OR NOT the market maker fills it.
  * other jumps arrive so that the total jump rate of the mid is lam_tot in every regime.
    The public price path therefore says nothing about the regime; only the link between
    the market maker's own fills and the jumps that follow them does.
  * mid = sigma * Brownian motion + all jumps.  It does not depend on the market maker.

Every market maker in a call sees the SAME path (common random numbers); they differ only
in which candidates they fill.

Two "worlds":
  ideal = False (main experiments): the jump of an informed order comes after a random delay
      (mean tau_D).  The market maker sees each fill's markout, the actual mid change over
      (fill, fill + h], at time fill + h.
  ideal = True (exact verification): the jump comes right after the order, and a signal
      with the markout's distribution is observed at the fill.  Here the HJBs are exact.

P&L accounting: wealth = cash + q * mid, so
  pnl = sum over fills of the quoted distance d  +  sum over time of q * d(mid).
"""
import numpy as np
from numba import njit, prange

N_STATS = 10
# columns of the per-path output
PNL, PEN, QT, NFILL, SPREAD, ASCOST, NINF, NFILL_TOX, EDGE_TOX, EDGE_CALM = range(N_STATS)

SRC_CONST, SRC_REGIME, SRC_BELIEF, SRC_TRUEPI = 0, 1, 2, 3
# columns of the per-market-maker filter parameters
F_NUCT, F_NUTC, F_PIC, F_PIT, F_A, F_ETA, F_ULO, F_UHI, F_P0, F_TIMING = range(10)

MAXN = 4096      # max candidate orders per path
MAXJ = 8192      # max jumps per path
MAXREC = 1024    # max recorded fills per path


@njit(cache=True)
def _lin(tab, x0, dx, x):
    u = (x - x0) / dx
    n = tab.shape[0]
    if u <= 0.0:
        return tab[0]
    if u >= n - 1:
        return tab[n - 1]
    i = int(u)
    w = u - i
    return tab[i] * (1.0 - w) + tab[i + 1] * w


@njit(cache=True)
def _quote(tab, it, iq, nu, src, z, u):
    if src == 0:
        return tab[it, iq, 0]
    if src == 1:
        return tab[it, iq, z]
    x = u * (nu - 1)
    if x <= 0.0:
        return tab[it, iq, 0]
    if x >= nu - 1:
        return tab[it, iq, nu - 1]
    j = int(x)
    w = x - j
    return tab[it, iq, j] * (1.0 - w) + tab[it, iq, j + 1] * w


@njit(cache=True)
def _jump_size(eta, xi_mode):
    if xi_mode == 0:
        return eta * np.random.exponential(1.0)                    # exponential
    if xi_mode == 1:
        return eta                                                 # constant
    return eta * np.exp(np.random.standard_normal() - 0.5)         # lognormal, mean eta


@njit(cache=True)
def _filter_update(p_f, t_f, tau, mk, d_f, strategic, k, kap, pinf, pi_c, pi_t, a, eta_f,
                   lf0, lg1, m_lo, dm):
    """One step of the hidden-Markov filter for p = P(toxic).

    1. predict: between observations p relaxes towards its long-run value pinf
    2. update:  Bayes' rule with the markout mk of a fill that happened at time tau.
    r = likelihood ratio (informed / uninformed) of the markout."""
    p = pinf + (p_f - pinf) * np.exp(-kap * (tau - t_f))
    shift = d_f if strategic else 0.0
    r = (1.0 - a) + a * np.exp(_lin(lg1, m_lo, dm, mk - shift) - _lin(lf0, m_lo, dm, mk))
    if strategic:            # informed traders only trade if xi > d: the quote changes the mix
        wi = np.exp(-d_f / eta_f)
        wu = np.exp(-k * d_f)
        pi_c = pi_c * wi / (pi_c * wi + (1 - pi_c) * wu)
        pi_t = pi_t * wi / (pi_t * wi + (1 - pi_t) * wu)
    lt = 1.0 + pi_t * (r - 1.0)          # likelihood of the markout in the toxic regime (/ f0)
    lc = 1.0 + pi_c * (r - 1.0)
    return p * lt / (p * lt + (1.0 - p) * lc)


@njit(cache=True)
def _fill_rate(d, A, k, eta, pi):
    """Strategic model: fill intensity on one side at quote distance d if the informed share is pi."""
    return A * ((1.0 - pi) * np.exp(-k * d) + pi * np.exp(-d / eta))


@njit(cache=True)
def _advance_with_timing(p, t0, t1, tab_a, tab_b, dt_tab, nt, q, Q, nu, A, k, eta, pi_c, pi_t, kap, pinf):
    """Extension only. Belief update for 'no fill between t0 and t1' in the strategic model,
    where the fill intensity depends on the regime, followed by the usual prediction step.
    Small sub-steps because the quotes move with the belief."""
    t = t0
    while t < t1 - 1e-12:
        dt = min(0.1, t1 - t)
        it = min(int(t / dt_tab), nt - 1)
        lt = 0.0
        lc = 0.0
        if q > -Q:
            d = _quote(tab_a, it, q + Q, nu, 2, 0, p)
            lt += _fill_rate(d, A, k, eta, pi_t); lc += _fill_rate(d, A, k, eta, pi_c)
        if q < Q:
            d = _quote(tab_b, it, q + Q, nu, 2, 0, p)
            lt += _fill_rate(d, A, k, eta, pi_t); lc += _fill_rate(d, A, k, eta, pi_c)
        wt = p * np.exp(-lt * dt)
        wc = (1.0 - p) * np.exp(-lc * dt)
        p = wt / (wt + wc)
        p = pinf + (p - pinf) * np.exp(-kap * dt)
        t += dt
    return p


@njit(parallel=True, cache=True)
def _simulate(n_paths, seed, n_chunks, ideal, strategic, scramble_t,
              A, k, sigma, T, Q, eta, xi_mode, lam_tot, tauD, h,
              tox_mode, G, pis, p0, ou,
              src, nus, tab_a, tab_b, dt_tab, fpar, lik_id, logf0, logg1, m_lo, dm,
              n_rec, rec_mm, rec_dt, es_dt, es_nb):
    n_mm = src.shape[0]
    nq = 2 * Q + 1
    nt = tab_a.shape[1]
    K = pis.shape[0]
    out = np.zeros((n_paths, n_mm, N_STATS))
    es_sum = np.zeros((n_chunks, n_mm, 2, es_nb))     # fill edge by time since last switch
    es_time = np.zeros((n_chunks, 2, es_nb))          # exposure time in each bin
    qhist = np.zeros((n_chunks, n_mm, nq))            # time spent at each inventory
    ng = int(T / rec_dt)
    rec_p = np.zeros((max(n_rec, 1), ng), dtype=np.float32)
    rec_z = np.zeros((max(n_rec, 1), ng), dtype=np.float32)
    rec_m = np.zeros((max(n_rec, 1), MAXREC), dtype=np.float32)
    rec_i = np.zeros((max(n_rec, 1), MAXREC), dtype=np.int8)
    rec_t = np.zeros((max(n_rec, 1), MAXREC))          # fill times
    rec_n = np.zeros(max(n_rec, 1), dtype=np.int64)
    sqrt_h = np.sqrt(h)

    for c in prange(n_chunks):
        lo = (n_paths * c) // n_chunks
        hi = (n_paths * (c + 1)) // n_chunks
        tc = np.empty(MAXN); side = np.empty(MAXN, dtype=np.int64)
        u_acc = np.empty(MAXN); u_inf = np.empty(MAXN); xi = np.empty(MAXN)
        delay = np.empty(MAXN); noise = np.empty(MAXN); arrived = np.empty(MAXN, dtype=np.bool_)
        Sc = np.empty(MAXN); Sh = np.empty(MAXN)
        pi_n = np.empty(MAXN); z_n = np.zeros(MAXN, dtype=np.int64)
        es_type = np.empty(MAXN, dtype=np.int64); es_bin = np.empty(MAXN, dtype=np.int64)
        seg_t = np.empty(8192); seg_pi = np.empty(8192); seg_z = np.zeros(8192, dtype=np.int64)
        jt = np.empty(MAXJ); jv = np.empty(MAXJ)
        qu_t = np.empty(MAXN); qu_s = np.empty(MAXN); qu_d = np.empty(MAXN)
        qu_n = np.empty(MAXN, dtype=np.int64)

        for path in range(lo, hi):
            np.random.seed(seed + path)
            # ---------- 1. toxicity path (piecewise-constant informed share) ----------
            nseg = 0
            if tox_mode == 0:                      # K-state Markov chain
                u = np.random.random()
                z = 0
                acc = p0[0]
                while u > acc and z < K - 1:
                    z += 1
                    acc += p0[z]
                t = 0.0
                seg_t[0] = 0.0; seg_pi[0] = pis[z]; seg_z[0] = z; nseg = 1
                while True:
                    rate = -G[z, z]
                    if rate <= 0.0:
                        break
                    t += np.random.exponential(1.0 / rate)
                    if t >= T:
                        break
                    u = np.random.random() * rate
                    acc = 0.0
                    z2 = z
                    for zz in range(K):
                        if zz != z:
                            acc += G[z, zz]
                            if u <= acc:
                                z2 = zz
                                break
                    z = z2
                    seg_t[nseg] = t; seg_pi[nseg] = pis[z]; seg_z[nseg] = z; nseg += 1
            else:                                  # exp-OU: pi = min(cap, exp(ou[0] + x)), x ~ OU
                x = ou[1] * np.random.standard_normal()
                rho = np.exp(-ou[2] * 1.0)
                sd = ou[1] * np.sqrt(1.0 - rho * rho)
                t = 0.0
                while t < T:
                    seg_t[nseg] = t
                    seg_pi[nseg] = min(ou[3], np.exp(ou[0] + x))
                    nseg += 1
                    x = rho * x + sd * np.random.standard_normal()
                    t += 1.0
            # ---------- 2. candidate orders ----------
            N = 0
            t = 0.0
            while N < MAXN:
                t += np.random.exponential(1.0 / (2.0 * A))
                if t >= T:
                    break
                tc[N] = t
                side[N] = 1 if np.random.random() < 0.5 else -1     # +1: buy order (hits our ask)
                u_acc[N] = np.random.random()
                u_inf[N] = np.random.random()
                xi[N] = _jump_size(eta, xi_mode)
                d_ = np.random.exponential(tauD)
                delay[N] = 0.0 if ideal else d_
                arrived[N] = d_ <= h                 # would the jump land inside the markout window?
                # signal noise for the ideal world: same law as the noise in a real markout
                nz = sigma * sqrt_h * np.random.standard_normal()
                for _ in range(np.random.poisson(lam_tot * h)):
                    s = 1.0 if np.random.random() < 0.5 else -1.0
                    nz += s * _jump_size(eta, xi_mode)
                noise[N] = nz
                N += 1
            # informed share / regime seen by each candidate
            sp = 0
            last_sw = -1.0
            for n in range(N):
                while sp + 1 < nseg and seg_t[sp + 1] <= tc[n]:
                    sp += 1
                    last_sw = seg_t[sp]
                pi_n[n] = seg_pi[sp]
                z_n[n] = seg_z[sp]
                es_bin[n] = -1
                es_type[n] = 0
                if tox_mode == 0 and K == 2 and last_sw >= 0.0:
                    b = int((tc[n] - last_sw) / es_dt)
                    if b < es_nb:
                        es_bin[n] = b
                        es_type[n] = 0 if seg_z[sp] == 1 else 1     # 0: last switch was into toxic
            # optional: change everything after scramble_t (used by the look-ahead test)
            if scramble_t < T:
                for n in range(N):
                    if tc[n] > scramble_t:
                        u_acc[n] = 1.0 - u_acc[n]
                        u_inf[n] = 1.0 - u_inf[n]
                        xi[n] = 0.37 * xi[n] + 0.2
                        if K == 2:
                            pi_n[n] = pis[0] + pis[1] - pi_n[n]
                            z_n[n] = 1 - z_n[n]
            # ---------- 3. jumps of the mid ----------
            nj = 0
            for n in range(N):                     # jumps that follow informed orders
                if u_inf[n] < pi_n[n]:
                    jt[nj] = tc[n] + delay[n]; jv[nj] = side[n] * xi[n]; nj += 1
            t = 0.0
            sp = 0
            while nj < MAXJ:                       # other jumps: total rate stays lam_tot
                t += np.random.exponential(1.0 / lam_tot)
                if t >= T:
                    break
                while sp + 1 < nseg and seg_t[sp + 1] <= t:
                    sp += 1
                u = np.random.random()
                s = 1.0 if np.random.random() < 0.5 else -1.0
                size = _jump_size(eta, xi_mode)
                if u < 1.0 - 2.0 * A * seg_pi[sp] / lam_tot:
                    if t > scramble_t:
                        size = 1.7 * size
                    jt[nj] = t; jv[nj] = s * size; nj += 1
            order = np.argsort(jt[:nj])
            # ---------- 4. mid just before each candidate (Sc) and h later (Sh) ----------
            i = 0; j = 0; jp = 0
            W = 0.0; J = 0.0; tprev = 0.0
            while i < N or j < N:
                ti = tc[i] if i < N else 1e300
                tj = min(tc[j] + h, T) if j < N else 1e300
                is_c = ti <= tj
                tnow = ti if is_c else tj
                dW = sigma * np.sqrt(tnow - tprev) * np.random.standard_normal()
                if tnow > scramble_t:
                    dW = -dW
                W += dW
                if is_c:                           # value just BEFORE time tc[i]
                    while jp < nj and jt[order[jp]] < tnow:
                        J += jv[order[jp]]; jp += 1
                    Sc[i] = W + J; i += 1
                else:                              # value AT time tc[j] + h
                    while jp < nj and jt[order[jp]] <= tnow:
                        J += jv[order[jp]]; jp += 1
                    Sh[j] = W + J; j += 1
                tprev = tnow
            dW = sigma * np.sqrt(T - tprev) * np.random.standard_normal()
            if T > scramble_t:
                dW = -dW
            W += dW
            while jp < nj:
                if jt[order[jp]] <= T:
                    J += jv[order[jp]]
                jp += 1
            S_T = W + J
            # ---------- bookkeeping shared by all market makers ----------
            if tox_mode == 0 and K == 2:
                for s_ in range(1, nseg):
                    end = seg_t[s_ + 1] if s_ + 1 < nseg else T
                    ty = 0 if seg_z[s_] == 1 else 1
                    for b in range(es_nb):
                        a0 = seg_t[s_] + b * es_dt
                        if a0 >= end:
                            break
                        es_time[c, ty, b] += min(end, a0 + es_dt) - a0
            if path < n_rec:
                sp = 0
                for g in range(ng):
                    while sp + 1 < nseg and seg_t[sp + 1] <= g * rec_dt:
                        sp += 1
                    rec_z[path, g] = seg_pi[sp]

            # ---------- 5. run every market maker over the same path ----------
            for m in range(n_mm):
                sm = src[m]
                nu = nus[m]
                f_pic = fpar[m, 2]; f_pit = fpar[m, 3]
                f_a = fpar[m, 4]; f_eta = fpar[m, 5]
                kap = fpar[m, 0] + fpar[m, 1]
                pinf = fpar[m, 0] / kap if kap > 0 else 0.0
                lf0 = logf0[lik_id[m]]; lg1 = logg1[lik_id[m]]
                p_f = fpar[m, 8]; t_f = 0.0          # filter belief p_f, valid at time t_f
                timing = fpar[m, 9] > 0.5            # extension: also learn from fill timing
                q = 0
                pnl = 0.0; pen = 0.0
                nfill = 0.0; spread = 0.0; ascost = 0.0; ninf = 0.0
                nfill_tox = 0.0; edge_tox = 0.0; edge_calm = 0.0
                qh = 0; qt = 0                       # markouts waiting to be observed (FIFO)
                tprev = 0.0; Sprev = 0.0
                gi = 0
                rec = (path < n_rec) and (m == rec_mm)
                nrec = 0
                for n in range(N + 1):
                    tnow = tc[n] if n < N else T
                    # (a) markouts of earlier fills that become visible before tnow
                    while qh < qt and qu_t[qh] + h <= tnow:
                        nn = qu_n[qh]
                        mk = qu_s[qh] * (Sh[nn] - Sc[nn])
                        if rec:
                            while gi < ng and gi * rec_dt < qu_t[qh] + h:
                                rec_p[path, gi] = pinf + (p_f - pinf) * np.exp(-kap * (gi * rec_dt - t_f))
                                gi += 1
                            if nrec < MAXREC:
                                rec_m[path, nrec] = mk; nrec += 1
                        if sm == 2 and timing:       # belief is kept current: update it in place
                            p_f = _filter_update(p_f, t_f, t_f, mk, qu_d[qh], strategic, k, kap, pinf,
                                                 f_pic, f_pit, f_a, f_eta, lf0, lg1, m_lo, dm)
                        elif sm == 2:
                            p_f = _filter_update(p_f, t_f, qu_t[qh], mk, qu_d[qh], strategic, k, kap, pinf,
                                                 f_pic, f_pit, f_a, f_eta, lf0, lg1, m_lo, dm)
                            t_f = qu_t[qh]
                        qh += 1
                    if rec:
                        while gi < ng and gi * rec_dt < tnow:
                            rec_p[path, gi] = pinf + (p_f - pinf) * np.exp(-kap * (gi * rec_dt - t_f))
                            gi += 1
                    # (b) mark the inventory to the mid, accrue the inventory penalty
                    Snow = Sc[n] if n < N else S_T
                    pnl += q * (Snow - Sprev)
                    pen += q * q * (tnow - tprev)
                    qhist[c, m, q + Q] += tnow - tprev
                    Sprev = Snow; tprev = tnow
                    if n == N:
                        break
                    # (c) a candidate order arrives: are we filled?
                    s = side[n]
                    if (s == 1 and q <= -Q) or (s == -1 and q >= Q):
                        continue
                    it = int(tnow / dt_tab)
                    if it >= nt:
                        it = nt - 1
                    if sm == 2 and timing:
                        p_f = _advance_with_timing(p_f, t_f, tnow, tab_a[m], tab_b[m], dt_tab, nt, q, Q, nu,
                                                   A, k, f_eta, f_pic, f_pit, kap, pinf)
                        t_f = tnow
                        u = p_f
                    elif sm == 2:    # current belief = last filter state, predicted forward to now
                        u = pinf + (p_f - pinf) * np.exp(-kap * (tnow - t_f))
                    elif sm == 3:
                        u = (pi_n[n] - fpar[m, 6]) / (fpar[m, 7] - fpar[m, 6])
                    else:
                        u = 0.0
                    if s == 1:
                        d = _quote(tab_a[m], it, q + Q, nu, sm, z_n[n], u)
                    else:
                        d = _quote(tab_b[m], it, q + Q, nu, sm, z_n[n], u)
                    informed = u_inf[n] < pi_n[n]
                    if strategic and informed:
                        filled = xi[n] > d           # the informed trader knows the jump
                    else:
                        filled = u_acc[n] < np.exp(-k * d)
                    if not filled:
                        continue
                    q -= s
                    pnl += d
                    nfill += 1.0; spread += d
                    edge = d
                    if informed:
                        ninf += 1.0; ascost += xi[n]
                        edge -= xi[n]
                    if z_n[n] == 1:
                        nfill_tox += 1.0; edge_tox += edge
                    else:
                        edge_calm += edge
                    if es_bin[n] >= 0:
                        es_sum[c, m, es_type[n], es_bin[n]] += edge
                    if timing:       # the fill itself: odds change by the ratio of fill intensities
                        wt = p_f * _fill_rate(d, A, k, f_eta, f_pit)
                        wc = (1.0 - p_f) * _fill_rate(d, A, k, f_eta, f_pic)
                        p_f = wt / (wt + wc)
                    if rec and nfill <= MAXREC:
                        rec_i[path, int(nfill) - 1] = 1 if informed else 0
                        rec_t[path, int(nfill) - 1] = tnow
                    if ideal:        # signal seen at once
                        mk = noise[n]
                        if informed and arrived[n]:
                            mk += xi[n]
                        if rec and nrec < MAXREC:
                            rec_m[path, nrec] = mk; nrec += 1
                        if sm == 2:
                            p_f = _filter_update(p_f, t_f, tnow, mk, d, strategic, k, kap, pinf,
                                                 f_pic, f_pit, f_a, f_eta, lf0, lg1, m_lo, dm)
                            t_f = tnow
                    else:            # markout becomes visible at tnow + h
                        qu_t[qt] = tnow; qu_s[qt] = s; qu_d[qt] = d; qu_n[qt] = n
                        qt += 1
                o = out[path, m]
                o[0] = pnl; o[1] = pen; o[2] = q; o[3] = nfill; o[4] = spread
                o[5] = ascost; o[6] = ninf; o[7] = nfill_tox; o[8] = edge_tox; o[9] = edge_calm
                if rec:
                    rec_n[path] = nrec
    return out, es_sum, es_time, qhist, rec_p, rec_z, rec_m, rec_i, rec_t, rec_n
