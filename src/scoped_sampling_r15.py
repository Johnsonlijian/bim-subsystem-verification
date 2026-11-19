"""R15 extensions: exact spillover for the two-stage gate, ablation plans, and fallible verifiers.

Spillover is defined uniformly for every policy as the expected number of verdicts in
error-free subsystems (W_f = 0) that are pushed into full inspection by escalation:
    S = E[ sum_{f : W_f = 0} N_f * 1(subsystem f fully inspected) ].
"""
from __future__ import annotations

import math
from functools import lru_cache

from scoped_sampling import (
    audit_size_w0,
    exact_max_bad_release,
    miss_probability,
    minimax_stratified_plan_w0,
)


def _bad_prob_independent(ws, ps, w0):
    k = len(ws)
    prob = 0.0
    for mask in range(1, 1 << k):
        if sum(ws[i] for i in range(k) if (mask >> i) & 1) >= w0:
            term = 1.0
            for i in range(k):
                term *= ps[i] if ((mask >> i) & 1) else (1.0 - ps[i])
            prob += term
    return prob


def eval_adaptive_2stage_r15(ns_pop, ws, ms, w0, alpha1=0.045, alpha=0.05):
    """Exact two-stage containment expectations, including Stage-2 escalation spillover."""
    k = len(ns_pop)
    N = sum(ns_pop)
    W = sum(ws)
    M_tot = sum(ms)
    N_clean = sum(n for n, w in zip(ns_pop, ws) if w == 0)
    alpha2 = (alpha - alpha1) / (1.0 - alpha1)
    if N == 0:
        return dict(sample_size=0, inspections=0.0, escaped_wrong=0.0, escaped_missed=0.0,
                    bad_release=0.0, spillover=0.0, p_stage2_escalation=0.0)
    n1 = audit_size_w0(N, w0, alpha1) if N >= w0 else 0
    if W == 0 or n1 == 0:
        bad = 1.0 if (W >= w0 and n1 == 0) else 0.0
        return dict(sample_size=n1, inspections=float(n1), escaped_wrong=float(W if n1 == 0 else 0.0),
                    escaped_missed=float(M_tot if n1 == 0 else 0.0), bad_release=bad,
                    spillover=0.0, p_stage2_escalation=0.0)
    masks = 1 << k
    g = [0.0] * masks
    gx = [0.0] * masks
    for u in range(masks):
        w_out = sum(ws[i] for i in range(k) if not ((u >> i) & 1))
        g[u] = miss_probability(N, w_out, n1)
        gx[u] = 0.0 if w_out > N - 1 else (n1 / N) * miss_probability(N - 1, w_out, n1 - 1)
    prob_D = [0.0] * masks
    x_D = [0.0] * masks
    for m in range(masks):
        vp = vx = 0.0
        sub = m
        while True:
            sign = -1.0 if (bin(m ^ sub).count("1") & 1) else 1.0
            vp += sign * g[sub]
            vx += sign * gx[sub]
            if sub == 0:
                break
            sub = (sub - 1) & m
        prob_D[m] = max(0.0, vp)
        x_D[m] = max(0.0, vx)
    insp = prob_D[0] * n1
    esc_w = prob_D[0] * W
    esc_m = prob_D[0] * M_tot
    bad = prob_D[0] if W >= w0 else 0.0
    p_esc2 = 0.0
    for m in range(1, masks):
        pr = prob_D[m]
        if pr <= 1e-15:
            continue
        N_D = sum(ns_pop[i] for i in range(k) if (m >> i) & 1)
        N_R = N - N_D
        W_R = sum(ws[i] for i in range(k) if not ((m >> i) & 1))
        M_R = sum(ms[i] for i in range(k) if not ((m >> i) & 1))
        if N_R == 0:
            insp += pr * N
            continue
        n2 = audit_size_w0(N_R, w0, alpha2) if N_R >= w0 else 0
        p2 = miss_probability(N_R, W_R, n2)
        clean_R = N_R - W_R
        e_m = clean_R * (x_D[m] / pr) if clean_R > 0 else 0.0
        e_union = (n2 + e_m * (1.0 - (n2 / clean_R if clean_R > 0 else 0.0))) if p2 > 0.0 else float(N_R)
        insp += pr * (N_D + p2 * e_union + (1.0 - p2) * N_R)
        esc_w += pr * W_R * p2
        esc_m += pr * M_R * p2
        if W_R >= w0:
            bad += pr * p2
        p_esc2 += pr * (1.0 - p2)
    return dict(sample_size=n1, inspections=insp, escaped_wrong=esc_w, escaped_missed=esc_m,
                bad_release=bad, spillover=N_clean * p_esc2, p_stage2_escalation=p_esc2)


@lru_cache(None)
def bonferroni_plan_w0(ns_pop, w0, alpha=0.05):
    """Valid but conservative stratified plan: per stratum t = ceil(w0/K), level alpha/K."""
    active = [i for i, n in enumerate(ns_pop) if n > 0]
    k_eff = max(1, len(active))
    t = math.ceil(w0 / k_eff)
    ns = []
    for n in ns_pop:
        ns.append(audit_size_w0(n, t, alpha / k_eff) if n >= t else 0)
    return tuple(ns)


def proportional_split_plan(ns_pop, w0, alpha=0.05):
    """Heuristic: split the global zero-acceptance sample proportionally (NOT a valid design)."""
    N = sum(ns_pop)
    n = audit_size_w0(N, w0, alpha) if N >= w0 else 0
    return tuple(min(nf, max(0, round(n * nf / N))) if N else 0 for nf in ns_pop)


def eval_scoped_plan(ns_pop, ws, ms, ns_samp, w0):
    """Expectations for any per-subsystem zero-acceptance plan with scoped escalation."""
    ps = [miss_probability(nf, wf, sf) for nf, wf, sf in zip(ns_pop, ws, ns_samp)]
    insp = sum(sf + (1.0 - p) * (nf - sf) for nf, sf, p in zip(ns_pop, ns_samp, ps))
    return dict(sample_size=sum(ns_samp), inspections=insp,
                escaped_wrong=sum(w * p for w, p in zip(ws, ps)),
                escaped_missed=sum(m * p for m, p in zip(ms, ps)),
                bad_release=_bad_prob_independent(ws, ps, w0), spillover=0.0,
                design_sup_risk=exact_max_bad_release(tuple(ns_pop), tuple(ns_samp), w0))


def eval_targeted(ns_pop, ws, ms, target_idx, w0, alpha=0.05):
    """Prior-targeted heuristic: inspect subsystem target_idx fully, global audit of the remainder."""
    N_t = ns_pop[target_idx]
    rest = [i for i in range(len(ns_pop)) if i != target_idx]
    N_R = sum(ns_pop[i] for i in rest)
    W_R = sum(ws[i] for i in rest)
    M_R = sum(ms[i] for i in rest)
    n_R = audit_size_w0(N_R, w0, alpha) if N_R >= w0 else 0
    p = miss_probability(N_R, W_R, n_R)
    clean_rest = sum(ns_pop[i] for i in rest if ws[i] == 0)
    spill = (N_t if ws[target_idx] == 0 else 0.0) + clean_rest * (1.0 - p)
    return dict(sample_size=N_t + n_R, inspections=N_t + n_R + (1.0 - p) * (N_R - n_R),
                escaped_wrong=W_R * p, escaped_missed=M_R * p,
                bad_release=p if W_R >= w0 else 0.0, spillover=spill)


def fallible_minimax_plan_w0(ns_pop, w0, alpha, sensitivity):
    """Minimax-stratified plan compensated for a verifier detecting each sampled error w.p. s.

    p_s(N,W,n) = E[(1-s)^X] is non-increasing in W, so the threshold reduction of
    Proposition 1 applies unchanged; we search by increasing a proportional rate.
    """
    from scoped_sampling import verifier_sensitivity_miss_prob
    from itertools import product
    ns = list(minimax_stratified_plan_w0(tuple(ns_pop), w0, alpha)[0])

    def sup_risk(samples):
        Ms = [min(n, w0) for n in ns_pop]
        best = 0.0
        grids = [range(m + 1) for m in Ms]
        for W in product(*grids):
            if sum(W) < w0:
                continue
            ps = [verifier_sensitivity_miss_prob(n, w, s) if n else 1.0
                  for n, w, s in zip(ns_pop, W, samples)]
            best = max(best, _bad_prob_independent(W, ps, w0))
        return best

    while sup_risk(ns) > alpha and any(s < n for s, n in zip(ns, ns_pop)):
        i = max(range(len(ns)), key=lambda j: (ns_pop[j] - ns[j]) / max(1, ns_pop[j]))
        ns[i] = min(ns_pop[i], ns[i] + max(1, ns_pop[i] // 100))
    return tuple(ns), sup_risk(ns)
