"""Deduplicated support and support-proportional change credit.

The operational support unit is one identified hypothesis or source observation,
conditional on its explicit scientific support argument. Counts are not posterior
probabilities, evidence independence estimates, or grades of scientific quality.
"""

from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json
import math

from ..case_reports.research.support_observations import observation_identity


SUPPORT_FIELDS = ("support", "applications", "facts", "argument", "checks")
CHANGE_FIELDS = ("break", "source", "target", "importance", "argument", "assessment")


def _id(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def claim_id(reference):
    return reference["key"] + "@" + reference["version"]


def controller_projection(points, resolution):
    """Do not silently turn exact scientific fractions into zero-yield grades.

    The existing controller uses floats and rounds its total to ten places.
    Scientific calculations remain available when that projection is unsupported.
    """
    try:
        step = float(Fraction(resolution))
        values = {key: float(Fraction(value)) for key, value in points.items()}
        if not math.isfinite(step) or step <= 0 or round(step, 10) == 0 or any(
                not math.isfinite(value) or (Fraction(points[key]) > 0 and round(value, 10) == 0)
                for key, value in values.items()):
            raise ValueError("unrepresentable resolution")
    except (OverflowError, ValueError):
        return {"supported": False, "reason": "Exact scientific fractions exceed the existing float controller's "
                "representable resolution; calculations remain valid but numerical continuation is not bound."}
    return {"supported": True, "values": values, "resolution": step}


def claim_of(sheet, expectation):
    row = sheet.snapshot(expectation)
    key = row["data"]["inference"]
    return {"key": key, "version": row["dependencies"][key]}


def references(kind, data):
    if kind == "support_contribution":
        return [data["support"], data["argument"], *data["applications"],
                *(item["evidence"] for item in data["facts"]), *data.get("checks", [])]
    return [data["break"], data["importance"], data["argument"], *([data["assessment"]] if data.get("assessment") else [])]


def context_pins(sheet, kind, data):
    if kind == "support_contribution":
        support = sheet.rows[data["support"]]["data"]
        refs = [support["inference"], support["expectation"]]
    else:
        refs = [data["source"], data["target"], claim_of(sheet, data["source"]), claim_of(sheet, data["target"])]
    pins = {}
    for ref in refs:
        pins.setdefault(ref["key"], set()).add(ref["version"])
    return pins


def accepted_fields(kind, data):
    """Replay legacy operand records without giving text-only claims credit."""
    required = set(SUPPORT_FIELDS if kind == "support_contribution" else CHANGE_FIELDS)
    modern = "checks" if kind == "support_contribution" else "assessment"
    allowed = [required, required - {modern}]
    return set(data) if isinstance(data, dict) and any(set(data) == fields or set(data) == fields | {"verification_required"}
                                         for fields in allowed) else set()


def validate(sheet, key, kind, data):
    if not accepted_fields(kind, data):
        raise ValueError("Scientific credit record requires its exact executable operands")
    if "verification_required" in data and type(data["verification_required"]) is not bool:
        raise ValueError("Malformed retired audit field")
    sheet._require_kind(data["argument"], "reasoning")
    if kind == "support_contribution":
        sheet._require_kind(data["support"], "expectation_support")
        support = sheet.rows[data["support"]]["data"]
        if not isinstance(data["applications"], list) or not isinstance(data["facts"], list):
            raise ValueError("Select supporting applications and identified observations explicitly")
        if not data["applications"] and not data["facts"]:
            raise ValueError("A support contribution needs a hypothesis or an additional observation")
        for ref in data["applications"]:
            sheet._require_kind(ref, "application")
            row = sheet.rows[ref]
            if (row["data"]["inference"] != support["inference"]["key"] or
                    row["dependencies"][support["inference"]["key"]] != support["inference"]["version"]):
                raise ValueError("Supporting hypothesis must use the selected exact G")
        selected = [*data["applications"]]
        for fact in data["facts"]:
            if not isinstance(fact, dict) or set(fact) != {"evidence", "observation"}:
                raise ValueError("A fact needs its evidence key and stable source observation identifier")
            sheet._require_kind(fact["evidence"], "evidence")
            observation_identity(sheet.rows[fact["evidence"]]["data"], fact["observation"])
            selected.append(fact["evidence"])
        for ref in selected:
            if not any(sheet._depends_on(ground, ref) for ground in support["grounds"]):
                raise ValueError("Selected contribution must belong to the B_G support argument's grounds")
            if not sheet._depends_on(data["argument"], ref):
                raise ValueError("The support derivation must address every counted application/observation")
        previous = sheet.rows.get(key)
        if previous and previous["data"]["support"] != data["support"]:
            raise ValueError("A contribution retains its B_G support identity")
        checks = data.get("checks", [])
        if not isinstance(checks, list) or len(checks) > 16 or len(set(checks)) != len(checks):
            raise ValueError("Select at most sixteen distinct executable support assessments")
        for check in checks:
            sheet._require_kind(check, "support_assessment")
            if sheet.rows[check]["data"]["support"] != data["support"]:
                raise ValueError("Support contribution and subscoring assessments must share one exact B_G support claim")
            formalization = sheet.rows[check]["data"].get("source_formalization", "")
            if formalization and not sheet._depends_on(data["argument"], formalization):
                raise ValueError("The public support derivation must address the formalized source statement")
        return

    sheet._require_kind(data["break"], "expectation_break")
    for name in ("source", "target"):
        if sheet.snapshot(data[name])["kind"] != "expectation":
            raise ValueError("Change endpoints are exact expected-generalizability-bounds snapshots")
    if sheet.rows[data["break"]]["data"]["expectation"] != data["source"]:
        raise ValueError("Change must start from the expectation challenged by its break")
    source_claim, target_claim = claim_of(sheet, data["source"]), claim_of(sheet, data["target"])
    if source_claim["key"] != target_claim["key"]:
        raise ValueError("A change retains G's identity and explicitly distinguishes its versions")
    cursor = data["target"]
    while cursor != data["source"]:
        revision = sheet.snapshot(cursor)["data"]["revision"]
        if revision is None:
            raise ValueError("Changed expectation must descend from the challenged expectation")
        cursor = revision["prior"]
    if data["source"] == data["target"]:
        raise ValueError("A change needs a different target expectation")
    sheet._require_kind(data["importance"], "importance")
    protocol = sheet.rows[data["importance"]]["data"]
    application = sheet.rows[protocol["application"]]
    if (application["data"]["inference"] != source_claim["key"] or
            application["dependencies"][source_claim["key"]] not in {source_claim["version"], target_claim["version"]}):
        raise ValueError("Importance must measure the selected original/changed G in an affected application")
    trigger_keys = {ref["key"] for ref in sheet.rows[data["break"]]["data"]["triggers"]}
    for ref in (data["break"], data["importance"], *trigger_keys):
        if not sheet._depends_on(data["argument"], ref):
            raise ValueError("The necessity/usefulness argument must address the break, alteration and actual triggering grounds")
    previous = sheet.rows.get(key)
    if previous and any(previous["data"][name] != data[name] for name in ("break", "source")):
        raise ValueError("A change case retains its original break and expectation")
    if any(other != key and row["data"]["break"] == data["break"] for other, row in sheet.of_kind("causal_change")):
        raise ValueError("Update the existing change case; alternative repairs/adapters do not create another break reward")
    if data.get("assessment"):
        sheet._require_kind(data["assessment"], "repair_assessment")
        repair = sheet.rows[data["assessment"]]["data"]
        source_test = sheet.rows[repair["expectation_test"]]["data"]
        if (repair["break"] != data["break"] or repair["target"] != data["target"] or
                source_test["expectation"] != data["source"] or source_test["assessment"] != data["importance"]):
            raise ValueError("Change credit must use its exact break, old/new bounds and executed repair comparison")



def _current_operands(sheet, key, pins):
    if not sheet.usable(key, _pins=pins):
        return False, "stale_operands"
    # This establishes version consistency, not scientific correctness.
    return True, "recorded_operands_current"


SUPPORT_POLICY = {
    "name": "executed-support-equal-component-units-v2",
    "hypothesis_unit": "One canonical theory/hypothesis group with executed G dependence and an aligned same-method source formalization inside its expected bounds; contexts and parameterizations do not multiply it",
    "observation_unit": "One identified source observation aligned with a computed prediction in a dependent application inside expected bounds",
    "aggregation": "hypothesis_count + observation_count",
    "change_factor": "qualifying restored original-scope circumstance mass / original-scope mass",
    "origin": "Explicit implementation convention for component aggregation and restored-mass scaling; not user-specified relative scientific weights",
    "meaning": "Counts are support for G as represented, not certainty, empirical probabilities, verified source transcription or statistical independence",
}


def compile_support(sheet):
    """Compile executed subscoring operands; prose-only legacy claims stay pending."""
    from ..case_reports.research.support_grading import grade_support
    from ..case_reports.research.repairs import grade_repair
    from ..case_reports.research.importance import result_key
    claims, expectations, contributions, points, point_dependencies = {}, {}, {}, {}, {}
    check_cache = {}
    hypothesis_units = []
    for key, row in sheet.of_kind("support_contribution"):
        data = row["data"]
        support = sheet.snapshot({"key": data["support"], "version": row["dependencies"][data["support"]]})["data"]
        cid, bid = claim_id(support["inference"]), claim_id(support["expectation"])
        pins = context_pins(sheet, "support_contribution", data)
        current, status = _current_operands(sheet, key, pins)
        entry = claims.setdefault(cid, {"inference": support["inference"], "hypotheses": {}, "facts": {}})
        bound = expectations.setdefault(bid, {"expectation": support["expectation"], "claim": cid, "hypotheses": {}, "facts": {}})
        checks = data.get("checks", [])
        graded = []
        if current:
            for check in checks:
                if check not in check_cache:
                    check_cache[check] = grade_support(sheet, check)
                graded.append(check_cache[check])
        pending = []
        units = []
        for application in dict.fromkeys(data["applications"]):
            candidates = [check for check in graded if check["application"] == application]
            accepted = [check for check in candidates if check["hypothesis"]["status"] == "demonstrated"]
            if not accepted:
                pending.append({"kind": "hypothesis", "application": application,
                    "status": "stale_operands" if not current else
                    candidates[0]["hypothesis"]["status"] if candidates else "awaiting_executable_dependence"})
            for check in accepted:
                identity = check["hypothesis"]["identity"]
                dependencies = {key, data["support"], data["argument"], application,
                    check["assessment"], check["importance"], check["bounds_test"]}
                dependencies.update(result_key(check["importance"], case) for case in check["hypothesis"]["witness_cases"])
                units.append(("hypotheses", identity, {"application": application,
                    "theory": sheet.canonical_theory(sheet.rows[application]["data"]["theory"]),
                    "assessment": check["assessment"], "dependence": check["dependence"]}, dependencies))
        for fact in data["facts"]:
            candidates = [(check, result) for check in graded for result in check["facts"]
                          if (result["evidence"], result["observation"]) == (fact["evidence"], fact["observation"])]
            accepted = [(check, result) for check, result in candidates if result["status"] == "aligned"]
            if not accepted:
                pending.append({"kind": "observation", **fact,
                    "status": "stale_operands" if not current else
                    candidates[0][1]["status"] if candidates else "awaiting_executable_alignment"})
            for check, result in accepted:
                dependencies = {key, data["support"], data["argument"], fact["evidence"],
                    check["assessment"], check["application"], check["importance"], check["bounds_test"],
                    result_key(check["importance"], result["case_id"])}
                dependencies.update(result_key(check["importance"], case) for case in check["hypothesis"]["witness_cases"])
                units.append(("facts", result["identity"], {**fact, "assessment": check["assessment"],
                    "comparison_basis": result["comparison_basis"], "baseline_matches": result["baseline_matches"],
                    "altered_matches": result["altered_matches"], "discriminating": result["discriminating"],
                    "source": result["source"]}, dependencies))
        contributions[key] = {"claim": cid, "expectation": bid,
            "status": "stale_operands" if not current else "partially_resolved" if pending and units else
                      "awaiting_executable_operands" if pending else "executed_operands",
            "checks": list(checks), "pending_operands": pending,
            "pending_hypothesis_identities": sorted({item["application"] for item in pending if item["kind"] == "hypothesis"}),
            "candidate_units": len(data["applications"]) + len(data["facts"]),
            "admitted_units": len(set((kind, identity) for kind, identity, _, _ in units))}
        for kind, identity, detail, dependencies in units:
            if kind == "hypotheses":
                hypothesis_units.append((cid, bid, key, identity, detail, dependencies))
                continue
            entry[kind].setdefault(identity, {**detail, "contributions": []})["contributions"].append(key)
            bound[kind].setdefault(identity, {**detail, "contributions": []})["contributions"].append(key)
            point = "support:" + _id([cid, kind, identity])
            points[point] = "1"
            point_dependencies.setdefault(point, set()).update(dependencies)
    # Same canonical theory across contexts is one hypothesis unit. A repeated
    # formal realization under another name also cannot create another unit.
    # Neither operation establishes general scientific semantic equivalence.
    row_order = {key: index for index, key in enumerate(sheet.rows)}
    parents = {}
    def root(item):
        parents.setdefault(item, item)
        while parents[item] != item:
            parents[item] = parents[parents[item]]
            item = parents[item]
        return item
    def unite(left, right):
        left, right = root(left), root(right)
        if left != right:
            parents[right] = left
    for cid, _, _, formal, detail, _ in hypothesis_units:
        unite((cid, "theory", detail["theory"]), (cid, "formal", formal))
    groups = {}
    for unit in hypothesis_units:
        cid, _, _, formal, _, _ = unit
        groups.setdefault(root((cid, "formal", formal)), []).append(unit)
    local_hypotheses = {}
    for units in groups.values():
        cid = units[0][0]
        witnesses = {dep for *_, deps in units for dep in deps if dep.startswith("importance-result:")}
        first_witness = min(witnesses, key=lambda key: row_order[key])
        identity = _id(["executed-hypothesis-group", first_witness])
        detail = {"theories": sorted({unit[4]["theory"] for unit in units}),
            "formal_realizations": sorted({unit[3] for unit in units}),
            "applications": sorted({unit[4]["application"] for unit in units}),
            "assessments": sorted({unit[4]["assessment"] for unit in units}),
            "contributions": sorted({unit[2] for unit in units}),
            "identity_qualification": "Declared canonical-theory grouping plus structural alias deduplication; not proof of general scientific semantic equivalence"}
        claims[cid]["hypotheses"][identity] = detail
        for bid in {unit[1] for unit in units}:
            expectations[bid]["hypotheses"][identity] = {**detail,
                "contributions": sorted({unit[2] for unit in units if unit[1] == bid})}
        point = "support:" + _id([cid, "hypotheses", identity])
        points[point] = "1"
        point_dependencies[point] = set().union(*(unit[5] for unit in units))
        for unit in units:
            local_hypotheses.setdefault(unit[2], set()).add(identity)
    for key, contribution in contributions.items():
        observation_ids = {identity for entry in claims.values() for identity, fact in entry["facts"].items()
                           if key in fact["contributions"]}
        contribution["admitted_units"] = len(local_hypotheses.get(key, ())) + len(observation_ids)
    for group in (claims, expectations):
        for entry in group.values():
            entry["hypothesis_count"], entry["observation_count"] = len(entry["hypotheses"]), len(entry["facts"])
            entry["support"] = entry["hypothesis_count"] + entry["observation_count"]

    changes, resolutions = {}, [Fraction(1)]
    for key, row in sheet.of_kind("causal_change"):
        data = row["data"]
        cid = claim_id(claim_of(sheet, data["source"]))
        support = claims.get(cid, {}).get("support", 0)
        current, _ = _current_operands(sheet, key, context_pins(sheet, "causal_change", data))
        assessment = data.get("assessment")
        repair = grade_repair(sheet, assessment) if current and assessment else None
        factor = repair["qualifying_restored_fraction"] if repair else {"lower": "0", "upper": "1", "value": None}
        lower = support * Fraction(factor["lower"])
        upper = support * Fraction(factor["upper"])
        status = ("stale_operands" if not current else "awaiting_executable_repair_comparison" if repair is None else
                  "scoped_change_demonstrated" if Fraction(factor["lower"]) > 0 else
                  "scoped_change_unresolved" if factor["value"] is None else "no_qualifying_change_in_declared_comparison")
        changes[key] = {"break": data["break"], "source": data["source"], "target": data["target"],
            "source_claim": cid, "support": support, "importance": data["importance"],
            "assessment": assessment, "status": status, "factor": factor,
            "lower": str(lower), "upper": str(upper),
            "value": str(lower) if factor["value"] is not None else None,
            "rule": "support_for_original_G * qualifying_restored_original_scope_fraction",
            "repair_kind": sheet.snapshot(data["target"])["data"]["revision"]["repair_kind"],
            "current_scope": repair["necessity_scope"] if repair else "No executable repair comparison yet",
            "qualification": repair["qualification"] if repair else "Pending scientific operands are not a negative verdict",
            "repair_summary": {name: repair[name] for name in ("masses", "g_changed", "model_changes", "domain_kind")} if repair else None}
        point = "change:" + key
        points[point] = str(lower)
        dependencies = {key, data["break"], data["argument"], data["importance"]}
        if repair:
            dependencies.add(assessment)
            protocols = [repair["source_protocol"], repair["candidate"], *(item["assessment"] for item in repair["alternatives"])]
            dependencies.update(protocols)
            dependencies.update((repair["source_test"], repair["target_test"]))
            for case in repair["cases"]:
                if case["qualifying_restored"] is True:
                    dependencies.update(result_key(protocol, case["case_id"]) for protocol in protocols)
            denominator = Fraction(repair["masses"]["original_in_scope"])
            if denominator:
                resolutions.append(min(Fraction(case["weight"]) for case in repair["cases"]) / denominator)
        # Later support for old G revalues this same change and receives that gain.
        for kind in ("hypotheses", "facts"):
            for identity in claims.get(cid, {}).get(kind, {}):
                support_point = "support:" + _id([cid, kind, identity])
                dependencies.update(point_dependencies.get(support_point, ()))
        point_dependencies[point] = dependencies
    total = sum((Fraction(value) for value in points.values()), Fraction())
    return {"qualification": "executed_formal_consequences_with_explicit_source_and_model_mapping_limits",
        "policy": SUPPORT_POLICY["name"], "policy_details": deepcopy(SUPPORT_POLICY),
        "claims": claims, "expectations": expectations, "contributions": contributions,
        "changes": changes, "points": points, "total": str(total), "resolution": str(min(resolutions)),
        "point_dependencies": {key: sorted(values) for key, values in point_dependencies.items()},
        "meaning": "Hypothesis support requires executed dependence and an aligned same-method source formalization; fact support requires executed dependence and computed agreement with source observations in expected generalizability bounds. "
            "Change credit uses restored targets not rescued by compared G-preserving alternatives, within the declared comparison only. "
            "Later support revalues original G after a break. Counts and scaling follow the exposed implementation convention, not certainty or statistical independence. "
            "Unexecuted, unrepresentable, stale and text-only operands remain explicit pending entries; model verdicts never supply numerical credit."}


def selected_credit(sheet, key):
    result = compile_support(sheet)
    row = sheet.rows[key]
    if row["kind"] == "causal_change":
        return result["changes"][key]
    if row["kind"] == "expectation":
        entry = result["expectations"].get(claim_id(sheet.anchor(key)), {
            "expectation": sheet.anchor(key), "hypothesis_count": 0, "observation_count": 0, "support": 0})
        return {name: value for name, value in entry.items() if name not in {"hypotheses", "facts"}}
    if row["kind"] in {"support_contribution", "expectation_support"}:
        data = row["data"] if row["kind"] == "expectation_support" else sheet.rows[row["data"]["support"]]["data"]
        entry = result["claims"].get(claim_id(data["inference"]), {})
        summary = {name: value for name, value in entry.items() if name not in {"hypotheses", "facts"}}
        if row["kind"] == "support_contribution":
            summary["contribution"] = result["contributions"][key]
        return summary
    raise ValueError("Select an expectation, support contribution or causal change")


def record_schema(kind, snapshot_schema):
    text = {"type": "string", "minLength": 1}
    if kind == "support_contribution":
        properties = {"support": text, "applications": {"type": "array", "uniqueItems": True, "items": text},
                      "facts": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                          "required": ["evidence", "observation"], "properties": {"evidence": text, "observation": text}}},
                      "argument": text, "checks": {"type": "array", "maxItems": 16, "uniqueItems": True, "items": text}}
    else:
        properties = {"break": text, "source": snapshot_schema, "target": snapshot_schema,
                      "importance": text, "argument": text, "assessment": text}
    return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}
