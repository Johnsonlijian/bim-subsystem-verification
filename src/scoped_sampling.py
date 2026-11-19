"""Exact finite-population release verification with subsystem-scoped containment.

Implements:
1. Classical global-lot zero-discrepancy acceptance sampling (R13 baseline).
2. Proposition 1 — Exact Minimax Subsystem-Stratified Audit:
   Constructs a coordinate-minimal feasible initial sample subject to exact worst-case tail risk
       max_{0 <= W_f <= N_f} Pr(sum_{f=1}^K W_f Z_f > tolerance * N) <= alpha,
   using exact reduction to [0, min(N_f, w0)] and exact threshold candidates on
   the largest stratum, with subsystem-scoped escalation.
3. Proposition 2 — Two-Stage Adaptive Containment Audit:
   Stage 1 screens all N issued verdicts with n1 = n*(N, w0, alpha1). If clean,
   releases immediately. If discrepancies appear in subsystem set D != empty,
   escalates only D to full inspection and certifies the pooled un-implicated
   subsystems R with an independent sample n2 = n*(N_R, w0, (alpha-alpha1)/(1-alpha1)),
   strictly bounding Pr(W_escaped >= w0) <= alpha for any partition size K >= 1
   and any fixed error configuration without independence assumptions.
"""
from __future__ import annotations

import itertools
import math
from decimal import Decimal, ROUND_FLOOR
from functools import lru_cache
from numbers import Integral

import numpy as np
from scipy.stats import hypergeom


def unacceptable_count(n_total: int, tolerance: float) -> int:
    if not isinstance(n_total, Integral) or isinstance(n_total, bool) or n_total < 0:
        raise ValueError("Population size must be a nonnegative integer")
    if not math.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError("Tolerance must be finite and in [0, 1)")
    return int((Decimal(str(tolerance)) * int(n_total)).to_integral_value(rounding=ROUND_FLOOR)) + 1


@lru_cache(None)
def miss_probability(n_total: int, n_wrong: int, n_sample: int) -> float:
    if n_wrong <= 0 or n_sample <= 0:
        return 1.0
    if n_sample > n_total - n_wrong:
        return 0.0
    return float(hypergeom.pmf(0, n_total, n_wrong, n_sample))


@lru_cache(None)
def _p0_vec(n_total: int, max_wrong: int, n_sample: int) -> np.ndarray:
    return np.array([miss_probability(n_total, w, n_sample) for w in range(max_wrong + 1)], dtype=np.float64)


@lru_cache(None)
def audit_size_w0(n_total: int, w0: int, alpha: float = 0.05) -> int:
    if not isinstance(n_total, Integral) or isinstance(n_total, bool) or n_total < 0:
        raise ValueError("Population size must be a nonnegative integer")
    if not isinstance(w0, Integral) or isinstance(w0, bool) or w0 < 1:
        raise ValueError("w0 must be a positive integer")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("Miss probability alpha must be finite and in (0, 1)")
    if n_total == 0 or w0 > n_total:
        return 0
    lo, hi = 0, n_total
    while lo < hi:
        mid = (lo + hi) // 2
        if miss_probability(n_total, w0, mid) <= alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo


@lru_cache(None)
def audit_size(n_total: int, tolerance: float = 0.05, alpha: float = 0.05) -> int:
    w0 = unacceptable_count(n_total, tolerance)
    if n_total == 0:
        return 0
    return audit_size_w0(n_total, w0, alpha)


def exact_stratified_bad_prob(ns_pop: tuple[int, ...], ws: tuple[int, ...], ns_samp: tuple[int, ...], w0: int) -> float:
    k = len(ns_pop)
    ps = [miss_probability(ns_pop[i], ws[i], ns_samp[i]) for i in range(k)]
    prob = 0.0
    for mask in range(1, 1 << k):
        if sum(ws[i] for i in range(k) if (mask >> i) & 1) >= w0:
            term = 1.0
            for i in range(k):
                term *= ps[i] if ((mask >> i) & 1) else (1.0 - ps[i])
            prob += term
    return prob


def exact_max_bad_release(ns_pop: tuple[int, ...] | list[int], ns_samp: tuple[int, ...] | list[int], w0: int) -> float:
    """Exact supremum over all 0 <= W_f <= N_f of Pr(sum_f W_f Z_f >= w0) for K <= 4 strata."""
    ns_list = list(ns_pop)
    samp_list = list(ns_samp)
    while len(ns_list) < 4:
        ns_list.append(0)
        samp_list.append(0)
    if len(ns_list) > 4:
        raise ValueError("Exact 4D threshold-simplex solver expects at most 4 active strata")
    if sum(ns_list) < w0:
        return 0.0

    order = sorted(range(4), key=lambda i: min(ns_list[i], w0))
    Ns_s = [ns_list[i] for i in order]
    ns_s = [samp_list[i] for i in order]
    Ms = [min(Nf, w0) for Nf in Ns_s]
    P = [_p0_vec(Ns_s[i], Ms[i], ns_s[i]) for i in range(4)]

    W1 = np.arange(Ms[1] + 1)[:, None]
    W2 = np.arange(Ms[2] + 1)[None, :]
    p1 = P[1][:, None]
    p2 = P[2][None, :]
    P3_full = P[3]
    M3 = Ms[3]

    best = 0.0
    for w_0 in range(Ms[0] + 1):
        p_0 = P[0][w_0]
        q_0 = 1.0 - p_0
        S_list = []
        Pr_list = []
        for b0 in (0, 1):
            pr0 = p_0 if b0 else q_0
            s0 = w_0 if b0 else 0
            for b1 in (0, 1):
                pr01 = pr0 * (p1 if b1 else (1.0 - p1))
                s01 = s0 + (W1 if b1 else 0)
                for b2 in (0, 1):
                    pr012 = pr01 * (p2 if b2 else (1.0 - p2))
                    s012 = s01 + (W2 if b2 else 0)
                    S_list.append(np.broadcast_to(s012, (Ms[1] + 1, Ms[2] + 1)))
                    Pr_list.append(np.broadcast_to(pr012, (Ms[1] + 1, Ms[2] + 1)))
        S = np.stack(S_list, axis=0)
        Pr3 = np.stack(Pr_list, axis=0)

        already = S >= w0
        base_prob = np.sum(Pr3 * already, axis=0)
        cur_max = float(np.max(base_prob))
        if cur_max > best:
            best = cur_max

        need = w0 - S
        for m in range(8):
            cand_w3 = need[m]
            valid = (cand_w3 >= 1) & (cand_w3 <= M3)
            if not np.any(valid):
                continue
            w3_clip = np.where(valid, cand_w3, 0)
            p3_val = P3_full[w3_clip]
            gain_mask = (~already) & (need <= w3_clip[None, :, :])
            tot_prob = base_prob + p3_val * np.sum(Pr3 * gain_mask, axis=0)
            mval = float(np.max(tot_prob[valid]))
            if mval > best:
                best = mval
    return best


@lru_cache(None)
def minimax_stratified_plan_w0(ns_pop: tuple[int, ...], w0: int, alpha: float = 0.05) -> tuple[tuple[int, ...], float]:
    if any(not isinstance(x, Integral) or isinstance(x, bool) or x < 0 for x in ns_pop):
        raise ValueError("Stratum sizes must be nonnegative integers")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("Alpha must be in (0, 1)")
    tot = sum(ns_pop)
    if tot == 0 or tot < w0:
        return tuple(0 for _ in ns_pop), 0.0

    lo, hi = 0.0, 1.0
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        ns = [min(Nf, math.ceil(mid * Nf)) for Nf in ns_pop]
        if exact_max_bad_release(ns_pop, ns, w0) <= alpha:
            hi = mid
        else:
            lo = mid
    ns = [min(Nf, math.ceil(hi * Nf)) for Nf in ns_pop]
    improved = True
    while improved:
        improved = False
        for idx in sorted(range(len(ns_pop)), key=lambda i: (-ns_pop[i], i)):
            if ns[idx] > 0:
                ns[idx] -= 1
                if exact_max_bad_release(ns_pop, ns, w0) <= alpha:
                    improved = True
                else:
                    ns[idx] += 1
    sup_p = exact_max_bad_release(ns_pop, ns, w0)
    return tuple(ns), sup_p


def minimax_stratified_plan(ns_pop: tuple[int, ...], tolerance: float = 0.05, alpha: float = 0.05) -> tuple[tuple[int, ...], float]:
    w0 = unacceptable_count(sum(ns_pop), tolerance)
    return minimax_stratified_plan_w0(tuple(ns_pop), w0, alpha)


def eval_minimax_stratified(
    ns_pop: tuple[int, ...],
    ws: tuple[int, ...],
    ms_missed: tuple[int, ...],
    w0: int,
    alpha: float = 0.05,
    alarms: tuple[int, ...] | None = None,
) -> dict:
    k = len(ns_pop)
    alarms_use = tuple(0 for _ in range(k)) if alarms is None else tuple(int(bool(a)) for a in alarms)
    ns_unalarmed = tuple(0 if alarms_use[i] else ns_pop[i] for i in range(k))
    ns_samp, sup_p = minimax_stratified_plan_w0(ns_unalarmed, w0, alpha)
    ps = [0.0 if alarms_use[i] else miss_probability(ns_pop[i], ws[i], ns_samp[i]) for i in range(k)]

    inspections = 0.0
    collateral = 0.0
    escaped_wrong = 0.0
    escaped_missed = 0.0
    for i in range(k):
        if alarms_use[i]:
            insp_i = float(ns_pop[i])
            if ws[i] == 0:
                collateral += float(ns_pop[i] - audit_size_w0(ns_pop[i], min(w0, max(1, ns_pop[i])), alpha)) if ns_pop[i] >= w0 else float(ns_pop[i])
        else:
            insp_i = ns_samp[i] + (1.0 - ps[i]) * (ns_pop[i] - ns_samp[i])
        inspections += insp_i
        escaped_wrong += ws[i] * ps[i]
        escaped_missed += ms_missed[i] * ps[i]

    bad_prob = 0.0
    for mask in range(1, 1 << k):
        if sum(ws[i] for i in range(k) if (mask >> i) & 1) >= w0:
            term = 1.0
            for i in range(k):
                term *= ps[i] if ((mask >> i) & 1) else (1.0 - ps[i])
            bad_prob += term

    return dict(
        sample_size=sum(ns_samp),
        stratum_samples=ns_samp,
        sup_bad_prob=sup_p,
        bad_release=bad_prob,
        escaped_wrong=escaped_wrong,
        escaped_missed=escaped_missed,
        inspections=inspections,
        clean_stratum_escalation=sum(float(ns_pop[i]) for i in range(k) if alarms_use[i] and ws[i] == 0),
    )


def eval_adaptive_2stage(
    ns_pop: tuple[int, ...],
    ws: tuple[int, ...],
    ms_missed: tuple[int, ...],
    w0: int,
    alpha1: float = 0.035,
    alpha: float = 0.05,
) -> dict:
    """Exact analytical expectations for Two-Stage Adaptive Containment Audit.

    Stage 1 draws S1 uniformly of size n1 = audit_size_w0(N, w0, alpha1).
    If S1 hits 0 errors, release all N verdicts.
    If S1 hits >=1 error in non-empty subsystem subset D, inspect all verdicts in D,
    and draw an independent uniform sample S2 of size n2 = audit_size_w0(N_R, w0, alpha2)
    from R = V \\ D (with alpha2 = (alpha - alpha1) / (1 - alpha1)), counting distinct
    verdicts |(S1 cap R) cup S2| if S2 is clean and N_R if S2 hits an error.
    """
    if not (0.0 < alpha1 < alpha < 1.0):
        raise ValueError("Require 0 < alpha1 < alpha < 1")
    k = len(ns_pop)
    N = sum(ns_pop)
    W = sum(ws)
    M_tot = sum(ms_missed)
    if N == 0:
        return dict(sample_size=0, inspections=0.0, escaped_wrong=0.0, escaped_missed=0.0, bad_release=0.0)
    alpha2 = (alpha - alpha1) / (1.0 - alpha1)
    n1 = audit_size_w0(N, w0, alpha1) if N >= w0 else 0
    if W == 0 or n1 == 0:
        bad = 1.0 if (W >= w0 and n1 == 0) else 0.0
        return dict(sample_size=n1, inspections=float(n1), escaped_wrong=float(W if n1 == 0 else 0.0),
                    escaped_missed=float(M_tot if n1 == 0 else 0.0), bad_release=bad)

    num_masks = 1 << k
    g = [0.0] * num_masks
    gx = [0.0] * num_masks
    for u_mask in range(num_masks):
        w_out = sum(ws[i] for i in range(k) if not ((u_mask >> i) & 1))
        g[u_mask] = miss_probability(N, w_out, n1)
        gx[u_mask] = 0.0 if w_out > N - 1 else (n1 / N) * miss_probability(N - 1, w_out, n1 - 1)

    prob_D = [0.0] * num_masks
    x_D = [0.0] * num_masks
    for m_mask in range(num_masks):
        val_p = 0.0
        val_x = 0.0
        sub = m_mask
        while True:
            sign = -1.0 if (bin(m_mask ^ sub).count("1") & 1) else 1.0
            val_p += sign * g[sub]
            val_x += sign * gx[sub]
            if sub == 0:
                break
            sub = (sub - 1) & m_mask
        prob_D[m_mask] = max(0.0, val_p)
        x_D[m_mask] = max(0.0, val_x)

    exp_insp = prob_D[0] * n1
    exp_esc_w = prob_D[0] * W
    exp_esc_m = prob_D[0] * M_tot
    bad_prob = prob_D[0] if W >= w0 else 0.0

    for m_mask in range(1, num_masks):
        pr = prob_D[m_mask]
        if pr <= 1e-15:
            continue
        N_D = sum(ns_pop[i] for i in range(k) if (m_mask >> i) & 1)
        N_R = N - N_D
        W_R = sum(ws[i] for i in range(k) if not ((m_mask >> i) & 1))
        M_R = sum(ms_missed[i] for i in range(k) if not ((m_mask >> i) & 1))
        if N_R == 0:
            exp_insp += pr * N
            continue
        n2 = audit_size_w0(N_R, w0, alpha2) if N_R >= w0 else 0
        p2 = miss_probability(N_R, W_R, n2)
        clean_R = N_R - W_R
        e_m_R = clean_R * (x_D[m_mask] / pr) if clean_R > 0 else 0.0
        if p2 > 0.0:
            overlap_frac = (n2 / clean_R) if clean_R > 0 else 0.0
            e_union_clean = n2 + e_m_R * (1.0 - overlap_frac)
        else:
            e_union_clean = float(N_R)
        insp_R = p2 * e_union_clean + (1.0 - p2) * N_R
        exp_insp += pr * (N_D + insp_R)
        exp_esc_w += pr * (W_R * p2)
        exp_esc_m += pr * (M_R * p2)
        if W_R >= w0:
            bad_prob += pr * p2

    return dict(
        sample_size=n1,
        inspections=exp_insp,
        escaped_wrong=exp_esc_w,
        escaped_missed=exp_esc_m,
        bad_release=bad_prob,
    )


def verifier_sensitivity_miss_prob(n_total: int, n_wrong: int, n_sample: int, sensitivity: float) -> float:
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError("Sensitivity must be in [0, 1]")
    x = np.arange(min(n_sample, n_wrong) + 1)
    return float(np.sum(hypergeom.pmf(x, n_total, n_wrong, n_sample) * ((1.0 - sensitivity) ** x)))
