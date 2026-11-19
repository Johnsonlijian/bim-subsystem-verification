"""Subsystem-scoped and global cryptographic commitment release gate (bim-release-gate/2).

Supports three exact finite-population audit modes:
- `global`: classical single-lot zero-discrepancy acceptance plan.
- `minimax_stratified`: exact minimax subsystem-stratified plan (Proposition 1);
  escalates only subsystems with discrepancies or unverifiable references.
- `adaptive_2stage`: two-stage containment plan (Proposition 2); Stage 1 screens
  the pooled lot at alpha1, and if discrepancies appear in subsystem set D,
  commits Stage 2 on the remaining subsystems R = V \\ D at alpha2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
from pathlib import Path

from scoped_sampling import (
    audit_size,
    audit_size_w0,
    minimax_stratified_plan_w0,
    unacceptable_count,
)


def freeze(population: list[dict]) -> tuple[list[dict], str]:
    if not isinstance(population, list) or not population:
        raise ValueError("A nonempty issued-verdict list is required")
    required = {"check_id", "verdict", "rule_id", "source_sha256"}
    for r in population:
        if not required <= set(r):
            raise ValueError("Missing record field")
        if r["verdict"] not in ("PASS", "FAIL"):
            raise ValueError("Unissued verdict")
        if not isinstance(r["check_id"], str) or not r["check_id"]:
            raise ValueError("Invalid check ID")
        if not isinstance(r["rule_id"], str) or not r["rule_id"]:
            raise ValueError("Invalid rule ID")
        if not isinstance(r["source_sha256"], str) or not re.fullmatch("[0-9a-fA-F]{64}", r["source_sha256"]):
            raise ValueError("Invalid source hash")
        if "subsystem" in r and (not isinstance(r["subsystem"], str) or not r["subsystem"]):
            raise ValueError("Invalid subsystem label")
    ids = [r["check_id"] for r in population]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate check ID")
    normalized = [
        {
            "check_id": r["check_id"],
            "verdict": r["verdict"],
            "rule_id": r["rule_id"],
            "subsystem": r.get("subsystem", "default"),
            "source_sha256": r["source_sha256"].lower(),
        }
        for r in population
    ]
    records = sorted(normalized, key=lambda x: x["check_id"])
    digest = hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return records, digest


def make_plan(
    population: list[dict],
    tolerance: float,
    alpha: float,
    mode: str = "minimax_stratified",
    alpha1: float = 0.045,
    rng=None,
) -> dict:
    if not 0.0 <= tolerance < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("Invalid operating requirement")
    if mode not in ("global", "minimax_stratified", "adaptive_2stage"):
        raise ValueError(f"Unsupported mode: {mode}")
    rows, digest = freeze(population)
    N = len(rows)
    w0 = unacceptable_count(N, tolerance)
    sampler = rng if rng is not None else secrets.SystemRandom()

    subsystems = sorted({r["subsystem"] for r in rows})
    by_sub = {s: [r["check_id"] for r in rows if r["subsystem"] == s] for s in subsystems}

    if mode == "global":
        n = audit_size_w0(N, w0, alpha)
        selected = sorted(sampler.sample([r["check_id"] for r in rows], n))
        return dict(
            schema="bim-release-gate/2",
            mode=mode,
            stage=1,
            population_sha256=digest,
            N=N,
            w0=w0,
            tolerance=tolerance,
            alpha=alpha,
            subsystems=subsystems,
            n=n,
            selected_check_ids=selected,
            status="AWAITING_VERIFICATION",
        )

    if mode == "minimax_stratified":
        if len(subsystems) > 4:
            raise ValueError("minimax_stratified supports at most 4 subsystems; use adaptive_2stage for >4")
        ns_pop = tuple(len(by_sub[s]) for s in subsystems)
        ns_samp, sup_prob = minimax_stratified_plan_w0(ns_pop, w0, alpha)
        stratum_plan = {}
        selected = []
        for s, nf_samp in zip(subsystems, ns_samp):
            chosen = sorted(sampler.sample(by_sub[s], nf_samp))
            stratum_plan[s] = dict(N=len(by_sub[s]), n=nf_samp)
            selected.extend(chosen)
        selected.sort()
        return dict(
            schema="bim-release-gate/2",
            mode=mode,
            stage=1,
            population_sha256=digest,
            N=N,
            w0=w0,
            tolerance=tolerance,
            alpha=alpha,
            sup_bad_release_bound=round(sup_prob, 8),
            subsystems=subsystems,
            stratum_plan=stratum_plan,
            n=len(selected),
            selected_check_ids=selected,
            status="AWAITING_VERIFICATION",
        )

    # mode == "adaptive_2stage"
    if not (0.0 < alpha1 < alpha < 1.0):
        raise ValueError("adaptive_2stage requires 0 < alpha1 < alpha < 1")
    n1 = audit_size_w0(N, w0, alpha1) if N >= w0 else 0
    selected = sorted(sampler.sample([r["check_id"] for r in rows], n1))
    return dict(
        schema="bim-release-gate/2",
        mode=mode,
        stage=1,
        population_sha256=digest,
        N=N,
        w0=w0,
        tolerance=tolerance,
        alpha=alpha,
        alpha1=alpha1,
        alpha2=round((alpha - alpha1) / (1.0 - alpha1), 8),
        subsystems=subsystems,
        n=n1,
        selected_check_ids=selected,
        status="AWAITING_STAGE1_VERIFICATION",
    )


def _validate_answers(rows: list[dict], selected: list[str], answers: list[dict]) -> tuple[list[str], list[str]]:
    lookup = {r["check_id"]: r for r in rows}
    if len(selected) != len(set(selected)) or not set(selected) <= set(lookup):
        raise ValueError("Invalid sample identifiers")
    if len({a["check_id"] for a in answers}) != len(answers):
        raise ValueError("Duplicate reference answer")
    if {a["check_id"] for a in answers} != set(selected):
        raise ValueError("Reference answers do not match committed sample")
    discrepancies = []
    unverifiable = []
    for a in answers:
        if a.get("reference_verdict") not in ("PASS", "FAIL", "UNVERIFIABLE"):
            raise ValueError("Invalid reference state")
        if not a.get("reference_source") or not a.get("reference_sha256"):
            raise ValueError("Reference provenance required")
        if not re.fullmatch("[0-9a-fA-F]{64}", str(a["reference_sha256"])):
            raise ValueError("Invalid reference hash")
        if a["reference_verdict"] == "UNVERIFIABLE":
            unverifiable.append(a["check_id"])
        elif a["reference_verdict"] != lookup[a["check_id"]]["verdict"]:
            discrepancies.append(a["check_id"])
    return sorted(discrepancies), sorted(unverifiable)


def evaluate(population: list[dict], plan: dict, answers: list[dict], rng=None) -> dict:
    rows, digest = freeze(population)
    if digest != plan.get("population_sha256") or len(rows) != plan.get("N"):
        raise ValueError("Population changed after commitment")
    w0_expected = unacceptable_count(len(rows), plan["tolerance"])
    if plan.get("w0") != w0_expected:
        raise ValueError("Invalid w0 in plan")

    mode = plan.get("mode")
    selected = plan.get("selected_check_ids", [])
    if len(selected) != plan.get("n"):
        raise ValueError("Sample count mismatch")

    lookup = {r["check_id"]: r for r in rows}
    subsystems = sorted({r["subsystem"] for r in rows})

    if mode == "global":
        if plan["n"] != audit_size_w0(len(rows), w0_expected, plan["alpha"]):
            raise ValueError("Invalid global sample size")
        discrepancies, unverifiable = _validate_answers(rows, selected, answers)
        held = bool(discrepancies or unverifiable)
        return dict(
            schema="bim-release-gate/2",
            mode=mode,
            population_sha256=digest,
            plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest(),
            N=len(rows),
            n=plan["n"],
            discrepancies=discrepancies,
            unverifiable=unverifiable,
            escalated_subsystems=subsystems if held else [],
            released_subsystems=[] if held else subsystems,
            total_required_inspections=len(rows) if held else plan["n"],
            status="HOLD_FOR_FULL_VERIFICATION" if held else "SAMPLE_PASSED",
        )

    if mode == "minimax_stratified":
        by_sub = {s: [r["check_id"] for r in rows if r["subsystem"] == s] for s in subsystems}
        ns_pop = tuple(len(by_sub[s]) for s in subsystems)
        ns_samp, _ = minimax_stratified_plan_w0(ns_pop, w0_expected, plan["alpha"])
        for s, nf_exp in zip(subsystems, ns_samp):
            s_sel = [cid for cid in selected if lookup[cid]["subsystem"] == s]
            if len(s_sel) != nf_exp:
                raise ValueError(f"Invalid stratum sample size for {s}")
        discrepancies, unverifiable = _validate_answers(rows, selected, answers)
        flagged_ids = set(discrepancies) | set(unverifiable)
        escalated_subs = sorted({lookup[cid]["subsystem"] for cid in flagged_ids})
        released_subs = [s for s in subsystems if s not in escalated_subs]
        req_insp = sum(
            len(by_sub[s]) if s in escalated_subs else plan["stratum_plan"][s]["n"]
            for s in subsystems
        )
        return dict(
            schema="bim-release-gate/2",
            mode=mode,
            population_sha256=digest,
            plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True).encode("utf-8")).hexdigest(),
            N=len(rows),
            n=plan["n"],
            discrepancies=discrepancies,
            unverifiable=unverifiable,
            escalated_subsystems=escalated_subs,
            released_subsystems=released_subs,
            total_required_inspections=req_insp,
            status="SCOPED_HOLD_ESCALATED_SUBSYSTEMS" if escalated_subs else "SAMPLE_PASSED",
        )

    if mode == "adaptive_2stage":
        stage = plan.get("stage", 1)
        alpha1 = plan["alpha1"]
        alpha2 = (plan["alpha"] - alpha1) / (1.0 - alpha1)
        if stage == 1:
            n1_exp = audit_size_w0(len(rows), w0_expected, alpha1) if len(rows) >= w0_expected else 0
            if plan["n"] != n1_exp:
                raise ValueError("Invalid Stage-1 sample size")
            discrepancies, unverifiable = _validate_answers(rows, selected, answers)
            flagged_ids = set(discrepancies) | set(unverifiable)
            if not flagged_ids:
                return dict(
                    schema="bim-release-gate/2",
                    mode=mode,
                    stage_completed=1,
                    population_sha256=digest,
                    N=len(rows),
                    n=plan["n"],
                    discrepancies=[],
                    unverifiable=[],
                    escalated_subsystems=[],
                    released_subsystems=subsystems,
                    total_required_inspections=plan["n"],
                    status="SAMPLE_PASSED",
                )
            escalated_subs = sorted({lookup[cid]["subsystem"] for cid in flagged_ids})
            rem_subs = [s for s in subsystems if s not in escalated_subs]
            rem_ids = [r["check_id"] for r in rows if r["subsystem"] in rem_subs]
            N_R = len(rem_ids)
            if N_R < w0_expected:
                esc_count = sum(1 for r in rows if r["subsystem"] in escalated_subs)
                s1_in_r = len(set(selected) & set(rem_ids))
                return dict(
                    schema="bim-release-gate/2",
                    mode=mode,
                    stage_completed=2,
                    population_sha256=digest,
                    N=len(rows),
                    discrepancies=discrepancies,
                    unverifiable=unverifiable,
                    escalated_subsystems=escalated_subs,
                    released_subsystems=rem_subs,
                    total_required_inspections=esc_count + s1_in_r,
                    status="SCOPED_HOLD_RESIDUAL_CERTIFIED",
                )
            sampler = rng if rng is not None else secrets.SystemRandom()
            n2 = audit_size_w0(N_R, w0_expected, alpha2)
            s2_ids = sorted(sampler.sample(rem_ids, n2))
            return dict(
                schema="bim-release-gate/2",
                mode=mode,
                stage=2,
                population_sha256=digest,
                N=len(rows),
                w0=w0_expected,
                tolerance=plan["tolerance"],
                alpha=plan["alpha"],
                alpha1=alpha1,
                alpha2=round(alpha2, 8),
                stage1_selected_check_ids=selected,
                stage1_discrepancies=discrepancies,
                stage1_unverifiable=unverifiable,
                escalated_subsystems=escalated_subs,
                residual_subsystems=rem_subs,
                N_R=N_R,
                n=n2,
                selected_check_ids=s2_ids,
                status="AWAITING_STAGE2_RESIDUAL_VERIFICATION",
            )
        elif stage == 2:
            rem_subs = plan["residual_subsystems"]
            rem_ids = [r["check_id"] for r in rows if r["subsystem"] in rem_subs]
            n2_exp = audit_size_w0(len(rem_ids), w0_expected, alpha2)
            if plan["n"] != n2_exp or not set(selected) <= set(rem_ids):
                raise ValueError("Invalid Stage-2 residual plan")
            discrepancies, unverifiable = _validate_answers(rows, selected, answers)
            held_r = bool(discrepancies or unverifiable)
            esc_subs = subsystems if held_r else plan["escalated_subsystems"]
            rel_subs = [] if held_r else rem_subs
            if held_r:
                req_insp = len(rows)
            else:
                esc_count = sum(1 for r in rows if r["subsystem"] in esc_subs)
                union_r = (set(plan["stage1_selected_check_ids"]) & set(rem_ids)) | set(selected)
                req_insp = esc_count + len(union_r)
            return dict(
                schema="bim-release-gate/2",
                mode=mode,
                stage_completed=2,
                population_sha256=digest,
                N=len(rows),
                n=plan["n"],
                stage2_discrepancies=discrepancies,
                stage2_unverifiable=unverifiable,
                escalated_subsystems=esc_subs,
                released_subsystems=rel_subs,
                total_required_inspections=req_insp,
                status="HOLD_FOR_FULL_VERIFICATION" if held_r else "SCOPED_HOLD_RESIDUAL_CERTIFIED",
            )
    raise ValueError(f"Invalid plan mode or stage: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Subsystem-scoped BIM compliance release gate (v2).")
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("plan")
    p.add_argument("population", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--tolerance", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--mode", choices=("global", "minimax_stratified", "adaptive_2stage"), default="minimax_stratified")
    p.add_argument("--alpha1", type=float, default=0.045)

    e = sub.add_parser("evaluate")
    e.add_argument("population", type=Path)
    e.add_argument("plan", type=Path)
    e.add_argument("answers", type=Path)
    e.add_argument("output", type=Path)

    args = parser.parse_args()
    read_json = lambda path: json.loads(path.read_text(encoding="utf-8"))
    if args.action == "plan":
        result = make_plan(read_json(args.population), args.tolerance, args.alpha, mode=args.mode, alpha1=args.alpha1)
    else:
        result = evaluate(read_json(args.population), read_json(args.plan), read_json(args.answers))
    with args.output.open("x", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
