"""Study III: External SHACL + IfcOpenShell reproduction and release_gate_v2 verification."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R13_DIR = ROOT  # portable source registry, no sibling-project dependency
VENDOR = R13_DIR / "validation" / "vendor"
sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ROOT / "src"))

import ifcopenshell  # noqa: E402
import ifcopenshell.util.unit  # noqa: E402
from pyshacl import validate  # noqa: E402
from rdflib import Graph, Literal, Namespace, RDF, URIRef, XSD  # noqa: E402

from release_gate_v2 import evaluate, make_plan  # noqa: E402
from scoped_sampling import audit_size, miss_probability  # noqa: E402

OUT = ROOT / "validation" / "results"


def csvout(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_nuyts_shacl_and_ifc() -> tuple[dict, dict, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((R13_DIR / "validation/source_manifest.json").read_text(encoding="utf-8"))
    files = {r["name"]: R13_DIR / r["path"] for r in manifest}
    for r in manifest:
        actual_sha = hashlib.sha256((R13_DIR / r["path"]).read_bytes()).hexdigest()
        assert actual_sha == r["sha256"]
        if r.get("upstream_hash"):
            assert r["upstream_hash"] == actual_sha

    I = Namespace("https://otl.buildingsmart.org/IFC4_ADD2_TC1/def/")
    S = Namespace("https://w3id.org/sml/def#")
    O = Namespace("http://otl.company-x.com/def/")
    SH = Namespace("http://www.w3.org/ns/shacl#")
    REG = Namespace("https://accessibility-regulations.org/shapes/")

    original = Graph().parse(data=files["shapes.ttl"].read_text(encoding="utf-8"), format="turtle")
    adapted = Graph()
    adapted += original
    adapted.set((REG.Attribute_overallWidth_req2, SH.minInclusive, Literal(900, datatype=XSD.decimal)))
    (OUT / "typed_width_shapes.ttl").write_text(adapted.serialize(format="turtle"), encoding="utf-8")

    def values(g, subject, pred):
        return [float(v) for p in g.objects(subject, pred) for v in g.objects(p, RDF.value)]

    def requirement(shape, rg):
        name = str(shape)
        m = re.search(r"req(\d+)", name)
        if m:
            return int(m.group(1))
        path = rg.value(shape, SH.path) or original.value(shape, SH.path)
        if path == S.hasPart:
            return 3
        for parent in original.subjects(SH.property, shape):
            m = re.search(r"req(\d+)", str(parent))
            if m:
                return int(m.group(1))
        raise ValueError("Unmapped source shape: " + name)

    details = []
    summaries = []
    for name, path in sorted(files.items()):
        if not name.startswith("HS_"):
            continue
        g = Graph().parse(data=path.read_text(encoding="utf-8"), format="turtle")
        outputs = {}
        for mode, shapes in (("original", original), ("typed_width", adapted)):
            conforms, report, report_text = validate(g, shacl_graph=shapes, advanced=True, inference="none", inplace=False)
            fails = set()
            for result in report.subjects(RDF.type, SH.ValidationResult):
                node = report.value(result, SH.focusNode)
                req = requirement(report.value(result, SH.sourceShape), report)
                fails.add((str(node), req))
            outputs[mode] = fails
            summaries.append(dict(
                model=name,
                mode=mode,
                conforms=bool(conforms),
                violations=len(fails),
                violations_by_id=";".join(f"{n.split('/')[-1]}:R{r}" for n, r in sorted(fails)),
            ))
        for cls, sub_name, reqs in ((I.Door, "door", [1, 2]), (I.Stair, "stair", [3, 4]), (I.Ramp, "ramp", [5])):
            for node in sorted(g.subjects(RDF.type, cls), key=str):
                for req in reqs:
                    width = values(g, node, I.attribute_overallWidth)
                    height = values(g, node, I.property_riserHeight)
                    tread = values(g, node, I.property_treadLength)
                    dh = values(g, node, O.property_heightDifference)
                    slope = values(g, node, O.property_slope)
                    if req == 1:
                        direct = len(list(g.objects(node, I.attribute_overallWidth))) == 1
                    elif req == 2:
                        direct = all(x >= 900 for x in width)
                    elif req == 3:
                        direct = len({x for x in g.objects(node, S.hasPart) if (x, RDF.type, I.Railing) in g}) >= 2
                    elif req == 4:
                        direct = all(570 < 2 * h + t < 630 for h, t in itertools.product(height, tread))
                    else:
                        direct = all(250 < h < 500 for h in dh) and all(s <= 0.0625 for s in slope)
                    qty_missing = (req == 2 and not width) or (req == 4 and (not height or not tread)) or (req == 5 and (not dh or not slope))
                    details.append(dict(
                        model=name,
                        subsystem=sub_name,
                        node=str(node),
                        guid=str(g.value(node, I.attribute_ifcGuid)),
                        rule=req,
                        original_pass=(str(node), req) not in outputs["original"],
                        typed_width_pass=(str(node), req) not in outputs["typed_width"],
                        direct_pass=direct,
                        quantity_missing=qty_missing,
                        integrated_reference="UNVERIFIABLE" if qty_missing else ("PASS" if direct else "FAIL"),
                    ))

    csvout(OUT / "benchmark_verdicts.csv", details)
    csvout(OUT / "benchmark_runs.csv", summaries)

    pub = files["validation output.txt"].read_text(encoding="utf-8")
    expected = {("doorA_fail01", 1), ("doorB_fail02", 2), ("stairA_fail03-04-06", 3), ("stairA_fail03-04-06", 4), ("rampA_fail05", 5)}
    assert all(node in pub for node, _ in expected)
    reproduction = {}
    for mode in ("original", "typed_width"):
        rows_m = [x for x in details if x["model"] == "HS_fail_preprocessed.ttl" and not x[mode + "_pass"]]
        obs = {(x["node"].split("/")[-1], x["rule"]) for x in rows_m}
        reproduction[mode] = dict(
            published_failures=len(expected),
            observed_failures=len(obs),
            matching=len(expected & obs),
            extra=sorted(obs - expected),
            missing=sorted(expected - obs),
        )

    # Independent IFC GlobalId interrogation
    ifc_root = R13_DIR / "validation" / "ifc"
    ifc_manifest = list(csv.DictReader((ifc_root / "download_manifest.csv").open(encoding="utf-8")))
    ifc_rows = []
    for r in ifc_manifest:
        if not r["name"].endswith(".ifc"):
            continue
        p = ifc_root / r["local_path"]
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert h == r["sha256"]
        model = ifcopenshell.open(str(p))
        scale = ifcopenshell.util.unit.calculate_unit_scale(model)
        ttl_name = r["name"].replace(".ifc", ".ttl")
        graph = Graph().parse(data=files[ttl_name].read_text(encoding="utf-8"), format="turtle")
        for cls in (I.Door, I.Stair):
            for node in graph.subjects(RDF.type, cls):
                guid = str(graph.value(node, I.attribute_ifcGuid))
                obj = model.by_guid(guid)
                if cls == I.Door:
                    ref = [float(x) for prop in graph.objects(node, I.attribute_overallWidth) for x in graph.objects(prop, RDF.value)]
                    native = obj.OverallWidth
                    val = None if native is None else native * scale * 1000.0
                    ifc_rows.append(dict(
                        model=r["name"],
                        guid=guid,
                        quantity="OverallWidth_mm",
                        ifc_value=val,
                        rdf_value=ref[0] if len(ref) == 1 else None,
                        agreement=(len(ref) == 1 and val is not None and abs(val - ref[0]) < 1e-6),
                        reference_state="PRESENT" if ref else "RDF_MISSING",
                    ))
                else:
                    child = {c.GlobalId for rel in obj.IsDecomposedBy for c in rel.RelatedObjects if c.is_a("IfcRailing")}
                    ref = {str(graph.value(c, I.attribute_ifcGuid)) for c in graph.objects(node, S.hasPart) if (c, RDF.type, I.Railing) in graph}
                    ifc_rows.append(dict(
                        model=r["name"],
                        guid=guid,
                        quantity="railing_children",
                        ifc_value=len(child),
                        rdf_value=len(ref),
                        agreement=(child == ref),
                        reference_state="PRESENT",
                    ))
    csvout(OUT / "ifc_rdf_comparison.csv", ifc_rows)

    # Assemble source-traced engineering reference
    graphs = {r["name"]: Graph().parse(data=(R13_DIR / r["path"]).read_text(encoding="utf-8"), format="turtle") for r in manifest if r["name"].startswith("HS_")}
    for r in details:
        state = r["integrated_reference"]
        prov = "Direct predicate, published numeric convention"
        if r["rule"] == 2:
            pair = next(p for p in ifc_rows if p["model"] == r["model"].replace(".ttl", ".ifc") and p["guid"] == r["guid"] and p["quantity"] == "OverallWidth_mm")
            state = ("PASS" if float(pair["ifc_value"]) >= 900.0 else "FAIL") if pair["ifc_value"] is not None else "UNVERIFIABLE"
            prov = "Original IFC OverallWidth converted to mm by project units; GlobalId match"
        elif r["rule"] == 5:
            g = graphs[r["model"]]
            node = URIRef(r["node"])
            hprop = g.value(node, O.property_heightDifference)
            sprop = g.value(node, O.property_slope)
            hv = g.value(hprop, RDF.value)
            sv = g.value(sprop, RDF.value)
            hu = str(g.value(hprop, S.hasUnit)).split("/")[-1]
            su = str(g.value(sprop, S.hasUnit)).split("/")[-1]
            if hv is None or sv is None or hu != "MilliM" or su not in ("DEG", "PERCENT"):
                state = "UNVERIFIABLE"
            else:
                slp = math.tan(math.radians(float(sv))) if su == "DEG" else float(sv)
                state = "PASS" if (250.0 < float(hv) < 500.0 and slp <= 0.0625) else "FAIL"
                prov = "Source-specific numeric contract; DEG converted by tan; preprocessed PERCENT stored as rise/run"
        r["engineering_reference"] = state
        r["reference_provenance"] = prov
    csvout(OUT / "engineering_reference.csv", details)

    bench_summary = dict(
        versions={x: importlib.metadata.version(x) for x in ("pyshacl", "rdflib", "ifcopenshell")},
        models=4,
        independent_projects=1,
        rule_verdicts=len(details),
        direct_vs_typed_agreement=sum(r["direct_pass"] == r["typed_width_pass"] for r in details),
        direct_vs_original_agreement=sum(r["direct_pass"] == r["original_pass"] for r in details),
        published_report_comparison=reproduction,
        missing_quantity_vacuous_passes=sum(r["quantity_missing"] and r["typed_width_pass"] for r in details),
    )
    eng_summary = dict(
        rule_cases=len(details),
        original_disagreements=sum(bool(r["original_pass"]) != (r["engineering_reference"] == "PASS") for r in details),
        ifc_recovered_missing_widths=sum(r["rule"] == 2 and r["quantity_missing"] and r["engineering_reference"] != "UNVERIFIABLE" for r in details),
        unit_normalized_ramps=sum(r["rule"] == 5 and "preprocessed" not in r["model"] for r in details),
        unverifiable=sum(r["engineering_reference"] == "UNVERIFIABLE" for r in details),
        ifc_present_pairs=sum(r["reference_state"] == "PRESENT" for r in ifc_rows),
        ifc_agreements=sum(bool(r["agreement"]) for r in ifc_rows),
    )
    (OUT / "benchmark_summary.json").write_text(json.dumps(bench_summary, indent=2), encoding="utf-8")
    (OUT / "engineering_summary.json").write_text(json.dumps(eng_summary, indent=2), encoding="utf-8")
    return bench_summary, eng_summary, {"details": details}


def run_release_gate_v2_tests() -> dict:
    rows = list(csv.DictReader((OUT / "engineering_reference.csv").open(encoding="utf-8")))
    digest = hashlib.sha256((OUT / "engineering_reference.csv").read_bytes()).hexdigest()
    nuyts_cases = []
    integrity_rejections = 0

    for model in sorted({r["model"] for r in rows}):
        selected = [r for r in rows if r["model"] == model]
        pop = [
            dict(
                check_id=f"{r['node']}|R{r['rule']}",
                verdict="PASS" if r["original_pass"] == "True" else "FAIL",
                rule_id=f"Nuyts2024-R{r['rule']}",
                subsystem=r["subsystem"],
                source_sha256=digest,
            )
            for r in selected
        ]
        answers = [
            dict(
                check_id=f"{r['node']}|R{r['rule']}",
                reference_verdict=r["engineering_reference"],
                reference_source=r["reference_provenance"],
                reference_sha256=digest,
            )
            for r in selected
        ]
        for mode in ("global", "minimax_stratified"):
            plan = make_plan(pop, 0.05, 0.05, mode=mode)
            assert plan["n"] == len(pop)
            ev = evaluate(pop, plan, answers)
            assert ev["status"] in ("HOLD_FOR_FULL_VERIFICATION", "SCOPED_HOLD_ESCALATED_SUBSYSTEMS")
            if mode == "minimax_stratified":
                nuyts_cases.append(dict(
                    model=model,
                    N=len(pop),
                    n=plan["n"],
                    discrepancies=len(ev["discrepancies"]),
                    escalated_subsystems=ev["escalated_subsystems"],
                    released_subsystems=ev["released_subsystems"],
                    status=ev["status"],
                ))

        # Integrity tamper tests
        plan_g = make_plan(pop, 0.05, 0.05, mode="global")
        for kind in ("changed_verdict", "changed_source", "missing_answer", "duplicate_answer", "changed_subsystem"):
            pp = copy.deepcopy(pop)
            aa = copy.deepcopy(answers)
            if kind == "changed_verdict":
                pp[0]["verdict"] = "PASS" if pp[0]["verdict"] == "FAIL" else "FAIL"
            elif kind == "changed_source":
                pp[0]["source_sha256"] = "0" * 64
            elif kind == "changed_subsystem":
                pp[0]["subsystem"] = "tampered"
            elif kind == "missing_answer":
                aa = aa[:-1]
            elif kind == "duplicate_answer":
                aa.append(aa[0])
            try:
                evaluate(pp, plan_g, aa)
            except ValueError:
                integrity_rejections += 1
            else:
                raise AssertionError(f"Tamper {kind} not rejected")

    # End-to-end execution on the real 709-case DE multi-family empirical IFC batch!
    pop_de = json.loads((OUT / "empirical_de709_population.json").read_text(encoding="utf-8"))
    ref_de = json.loads((OUT / "empirical_de709_reference_map.json").read_text(encoding="utf-8"))
    det_rng = random.Random(20261117)

    # 1) Global mode on DE-709
    plan_de_glob = make_plan(pop_de, 0.05, 0.05, mode="global", rng=det_rng)
    ans_de_glob = [ref_de[cid] for cid in plan_de_glob["selected_check_ids"]]
    ev_de_glob = evaluate(pop_de, plan_de_glob, ans_de_glob)
    assert ev_de_glob["status"] == "HOLD_FOR_FULL_VERIFICATION"
    assert ev_de_glob["total_required_inspections"] == 709

    # 2) Minimax-Stratified mode on DE-709
    det_rng_m = random.Random(20261117)
    plan_de_mm = make_plan(pop_de, 0.05, 0.05, mode="minimax_stratified", rng=det_rng_m)
    ans_de_mm = [ref_de[cid] for cid in plan_de_mm["selected_check_ids"]]
    ev_de_mm = evaluate(pop_de, plan_de_mm, ans_de_mm)
    assert ev_de_mm["status"] == "SCOPED_HOLD_ESCALATED_SUBSYSTEMS"
    assert "door" in ev_de_mm["escalated_subsystems"]
    assert "space" in ev_de_mm["released_subsystems"]
    assert ev_de_mm["total_required_inspections"] < 709

    # 3) Adaptive 2-Stage mode on DE-709 (Stage 1 -> Stage 2 commitment -> Evaluation)
    det_rng_a = random.Random(20261117)
    plan_de_a1 = make_plan(pop_de, 0.05, 0.05, mode="adaptive_2stage", alpha1=0.045, rng=det_rng_a)
    ans_de_a1 = [ref_de[cid] for cid in plan_de_a1["selected_check_ids"]]
    plan_de_a2 = evaluate(pop_de, plan_de_a1, ans_de_a1, rng=det_rng_a)
    assert plan_de_a2["status"] == "AWAITING_STAGE2_RESIDUAL_VERIFICATION"
    ans_de_a2 = [ref_de[cid] for cid in plan_de_a2["selected_check_ids"]]
    ev_de_a2 = evaluate(pop_de, plan_de_a2, ans_de_a2, rng=det_rng_a)
    assert ev_de_a2["status"] in ("SCOPED_HOLD_RESIDUAL_CERTIFIED", "HOLD_FOR_FULL_VERIFICATION")

    gate_report = dict(
        status="PASS",
        nuyts_engineering_cases=nuyts_cases,
        integrity_rejections=integrity_rejections,
        empirical_de709_demo=dict(
            N=len(pop_de),
            global_5pct=dict(
                initial_n=plan_de_glob["n"],
                required_inspections=ev_de_glob["total_required_inspections"],
                escalated=ev_de_glob["escalated_subsystems"],
                released=ev_de_glob["released_subsystems"],
            ),
            minimax_stratified_5pct=dict(
                initial_n=plan_de_mm["n"],
                stratum_plan=plan_de_mm["stratum_plan"],
                required_inspections=ev_de_mm["total_required_inspections"],
                escalated=ev_de_mm["escalated_subsystems"],
                released=ev_de_mm["released_subsystems"],
            ),
            adaptive_2stage_5pct=dict(
                stage1_n=plan_de_a1["n"],
                stage2_n=plan_de_a2["n"],
                required_inspections=ev_de_a2["total_required_inspections"],
                escalated=ev_de_a2["escalated_subsystems"],
                released=ev_de_a2["released_subsystems"],
                final_status=ev_de_a2["status"],
            ),
        ),
    )
    (OUT / "release_gate_v2_tests.json").write_text(json.dumps(gate_report, indent=2), encoding="utf-8")
    return gate_report


def main() -> None:
    bench, eng, _ = run_nuyts_shacl_and_ifc()
    gate = run_release_gate_v2_tests()
    assert bench["published_report_comparison"]["original"]["matching"] == 5
    assert bench["direct_vs_typed_agreement"] == 24
    assert eng["original_disagreements"] == 5
    print(json.dumps(dict(benchmark=bench, engineering=eng, gate_v2=gate["empirical_de709_demo"]), indent=2))


if __name__ == "__main__":
    main()
