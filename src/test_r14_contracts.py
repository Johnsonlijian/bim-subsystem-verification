"""Exhaustive combinatorial verification of R14 mathematical guarantees."""
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scoped_sampling import (  # noqa: E402
    audit_size,
    audit_size_w0,
    exact_max_bad_release,
    exact_stratified_bad_prob,
    eval_adaptive_2stage,
    eval_minimax_stratified,
    minimax_stratified_plan_w0,
    miss_probability,
    unacceptable_count,
)


def check_hypergeometric_and_decimal_boundaries() -> int:
    cases = 0
    for total in range(1, 26):
        for wrong in range(total + 1):
            for sample in range(total + 1):
                exact = (math.comb(total - wrong, sample) / math.comb(total, sample)
                         if sample <= total - wrong else 0.0)
                assert abs(miss_probability(total, wrong, sample) - exact) < 1e-12
                cases += 1
        for tol in (0.0, 0.01, 0.05, 0.10):
            n = audit_size(total, tol, 0.05)
            w0 = unacceptable_count(total, tol)
            if w0 <= total:
                assert miss_probability(total, w0, n) <= 0.05 + 1e-12
                assert n == 0 or miss_probability(total, w0, n - 1) > 0.05
    # Decimal precision boundary: tolerance = 0.049999999999 on N=100 has floor(4.9999999999)+1 = 5, NOT 6!
    assert unacceptable_count(100, 0.049999999999) == 5
    assert unacceptable_count(100, 0.05) == 6
    return cases


def check_proposition1_against_4d_grid() -> int:
    configs = [
        ((12, 6, 8, 10), 3, 0.05),
        ((15, 0, 9, 11), 2, 0.05),
        ((10, 5, 7, 4), 4, 0.10),
    ]
    checked = 0
    for ns_pop, w0, alpha in configs:
        ns_samp, sup_fast = minimax_stratified_plan_w0(ns_pop, w0, alpha)
        assert sup_fast <= alpha + 1e-12
        sup_brute = 0.0
        for w0_i in range(ns_pop[0] + 1):
            for w1_i in range(ns_pop[1] + 1):
                for w2_i in range(ns_pop[2] + 1):
                    for w3_i in range(ns_pop[3] + 1):
                        ws = (w0_i, w1_i, w2_i, w3_i)
                        pr = exact_stratified_bad_prob(ns_pop, ws, ns_samp, w0)
                        sup_brute = max(sup_brute, pr)
                        checked += 1
        assert abs(sup_fast - sup_brute) < 1e-12, (sup_fast, sup_brute)
        res = eval_minimax_stratified(ns_pop, (w0, 0, 0, 0), (w0, 0, 0, 0), w0, alpha)
        assert res["bad_release"] <= alpha + 1e-12
    return checked


def check_proposition2_against_exact_sample_enumeration() -> int:
    """Enumerate ALL (S1, S2) sample pairs on multi-subsystem lots and verify exact agreement!"""
    scenarios = [
        # (ns_pop, ws, ms, w0, alpha1, alpha)
        ((4, 3, 3), (2, 0, 0), (2, 0, 0), 2, 0.20, 0.35),
        ((4, 3, 2), (2, 1, 0), (1, 1, 0), 2, 0.20, 0.35),
        ((5, 4), (2, 2), (2, 1), 2, 0.15, 0.30),
        ((5, 4), (0, 0), (0, 0), 2, 0.15, 0.30),
    ]
    enumerated = 0
    for ns_pop, ws, ms, w0, alpha1, alpha in scenarios:
        analytic = eval_adaptive_2stage(ns_pop, ws, ms, w0, alpha1, alpha)
        N = sum(ns_pop)
        n1 = audit_size_w0(N, w0, alpha1)
        alpha2 = (alpha - alpha1) / (1.0 - alpha1)
        # Build explicit item list: (item_id, subsystem, is_wrong, is_missed)
        items = []
        idx = 0
        for f, (nf, wf, mf) in enumerate(zip(ns_pop, ws, ms)):
            for j in range(nf):
                items.append((idx, f, j < wf, j < mf))
                idx += 1

        exp_insp = 0.0
        exp_esc_w = 0.0
        exp_esc_m = 0.0
        bad_prob = 0.0
        s1_list = list(itertools.combinations(range(N), n1))
        p_s1 = 1.0 / len(s1_list)
        for s1 in s1_list:
            s1_set = set(s1)
            hit_fams = {items[i][1] for i in s1 if items[i][2]}
            if not hit_fams:
                w_all = sum(1 for it in items if it[2])
                m_all = sum(1 for it in items if it[3])
                exp_insp += p_s1 * n1
                exp_esc_w += p_s1 * w_all
                exp_esc_m += p_s1 * m_all
                if w_all >= w0:
                    bad_prob += p_s1
                enumerated += 1
            else:
                d_items = {it[0] for it in items if it[1] in hit_fams}
                r_items = [it[0] for it in items if it[1] not in hit_fams]
                N_R = len(r_items)
                if N_R == 0:
                    exp_insp += p_s1 * N
                    enumerated += 1
                    continue
                n2 = audit_size_w0(N_R, w0, alpha2) if N_R >= w0 else 0
                s2_list = list(itertools.combinations(r_items, n2))
                p_s2 = 1.0 / len(s2_list)
                for s2 in s2_list:
                    enumerated += 1
                    s2_err = any(items[i][2] for i in s2)
                    if not s2_err:
                        insp = len(d_items) + len((s1_set & set(r_items)) | set(s2))
                        w_rem = sum(1 for i in r_items if items[i][2])
                        m_rem = sum(1 for i in r_items if items[i][3])
                        exp_insp += p_s1 * p_s2 * insp
                        exp_esc_w += p_s1 * p_s2 * w_rem
                        exp_esc_m += p_s1 * p_s2 * m_rem
                        if w_rem >= w0:
                            bad_prob += p_s1 * p_s2
                    else:
                        exp_insp += p_s1 * p_s2 * N

        assert abs(analytic["inspections"] - exp_insp) < 1e-12, (analytic["inspections"], exp_insp)
        assert abs(analytic["escaped_wrong"] - exp_esc_w) < 1e-12, (analytic["escaped_wrong"], exp_esc_w)
        assert abs(analytic["escaped_missed"] - exp_esc_m) < 1e-12, (analytic["escaped_missed"], exp_esc_m)
        assert abs(analytic["bad_release"] - bad_prob) < 1e-12, (analytic["bad_release"], bad_prob)
        if sum(ws) >= w0:
            assert bad_prob <= alpha + 1e-12
    return enumerated


def check_cluster_sampling_counterexample() -> dict:
    # 100 verdicts in 10 groups of 10; 10 errors concentrated in 1 group.
    # Sampling 4 whole groups gives marginal inclusion 0.4 per verdict, miss prob = 6/10 = 0.6.
    # Uniform 40-verdict sampling gives comb(90, 40)/comb(100, 40) = 0.00435544...
    cluster_miss = 6 / 10
    uniform_miss = math.comb(90, 40) / math.comb(100, 40)
    assert abs(cluster_miss - 0.6) < 1e-15
    assert abs(uniform_miss - miss_probability(100, 10, 40)) < 1e-15
    assert uniform_miss < 0.005
    return dict(cluster_miss=cluster_miss, uniform_miss=uniform_miss)


def main():
    c_hg = check_hypergeometric_and_decimal_boundaries()
    c_p1 = check_proposition1_against_4d_grid()
    c_p2 = check_proposition2_against_exact_sample_enumeration()
    ce = check_cluster_sampling_counterexample()
    out = dict(
        status="PASS",
        hypergeometric_and_boundary_cases=c_hg,
        proposition1_4d_grid_states_checked=c_p1,
        proposition2_sample_pairs_enumerated=c_p2,
        cluster_counterexample=ce,
    )
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/sampling_contract_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
