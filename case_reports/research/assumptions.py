"""Parameter registration, provenance validation, and scientific credit projection."""

from ...task_executors.scenario_metrics.evaluate import CONDITIONS, UNITS, validate_parameter
from ...task_executors.scenario_metrics.catalog import METRICS, PARAMETER_MEANINGS
from .records import Credit, Fact, ResearchResult, identity, assessment_summary, assumption_claim, current_support
from .scenarios import check_case_inputs
from .progress import evidence_key, parameter_support, found_parameter_progress
from .verification import CALIBRATION_PROGRESS, calibration_progress, target_calibration_progress
from .model_defaults import retained_defaults


RUBRIC = {
    "support": "0: no relevant admitted evidence; 1: indirect evidence with an explicit inference; 2: direct applicable evidence resolving the claim, for or against, or an explicit request constraint.",
    "applicability": "0: context unexamined; 1: bounded context comparison with stated limitations; 2: scenario, activity and population match or mismatch established.",
    "physical_consistency": "0: consistency unexamined; 1: plausible mechanism with unresolved checks; 2: relevant checks establish physical consistency or a contradiction.",
    "uncertainty": "0: uncertainty ignored; 1: consequential limitations identified; 2: alternatives or bounds examined and remaining limitations explained.",
}

# One rubric increment for a new usable computation; this is not evidence quality.
PROVISIONAL_OUTPUT_CREDIT = 1 / (2 * len(RUBRIC))

INPUT_ERROR_CODES = {
    "quantity_mismatch": "The recorded canonical field names a different physical quantity from the source statement; matching units alone do not establish matching meaning.",
    "value_mismatch": "The recorded number does not represent the number in the cited source after the stated conversion.",
    "unit_mismatch": "The recorded unit or unit conversion does not match the cited source quantity.",
    "source_attribution_mismatch": "The recorded subject or source attribution is demonstrably different from the cited observation or model parameter.",
}


def input_error_schema(evidence):
    """The existing scientific review can correct exact misrecorded source inputs."""
    return {"type": "array", "uniqueItems": True,
        "description": "Exact source records with demonstrated recording errors. Weak support, an uncertain transfer, a different scenario, or valid contradictory evidence are not recording errors. [] means no demonstrated recording error.",
        "items": {"type": "object", "additionalProperties": False,
            "required": ["evidence_ref", "error"], "properties": {
                "evidence_ref": {"type": "string", "enum": list(evidence)} if evidence else False,
                "error": {"type": "string", "enum": list(INPUT_ERROR_CODES)},
            }}}


def correct_input_records(entries, evidence, ledger):
    """Persist exact input corrections and withdraw dependent mappings immediately.

    The caller supplies the JSON from its existing review. This function performs
    no scientific judgment, model call, value override, or relevance adjustment.
    """
    if not isinstance(entries, list):
        raise ValueError("invalid_evidence must be a list of typed input errors")
    parsed, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"evidence_ref", "error"}:
            raise ValueError("An input error requires evidence_ref and error")
        ref, code = entry["evidence_ref"], entry["error"]
        if (not isinstance(ref, str) or ref not in evidence
                or not isinstance(code, str) or code not in INPUT_ERROR_CODES):
            raise ValueError("Input errors must identify supplied evidence and a declared recording-error code")
        if ref in seen:
            raise ValueError("A source record may have only one primary recording error per review")
        seen.add(ref)
        row = evidence[ref]
        if row.get("kind") not in {"source_observation", "source_model_parameter", "source_evidence"}:
            raise ValueError("Input corrections must identify a source record, not a requester condition or hypothesis")
        parsed.append((row, code))
    if not parsed:
        return ResearchResult()
    selected = found_parameter_progress(ledger.result(), ledger=ledger)
    previous_errors = {item["error_id"]: item for assumption in ledger.assumptions.values()
                       for item in assumption.get("input_errors", ())}
    previous_errors.update((item["input_error"]["error_id"], item["input_error"])
        for item in ledger.diagnostics if item.get("stage") == "input_record_correction"
        and isinstance(item.get("input_error"), dict))
    updates, diagnostics = {}, []
    for row, code in parsed:
        record_key = evidence_key(row)
        fact_keys = sorted(fact.key for fact in ledger.facts.values()
                           if evidence_key(fact.semantic_record()) == record_key)
        affected = []
        for key, assumption in ledger.assumptions.items():
            derivation = assumption.get("derivation", {})
            if (set(fact_keys).intersection(assumption.get("source_fact_keys", ()))
                    or set(fact_keys).intersection(derivation.get("input_fact_keys", ()))
                    or identity(row) in derivation.get("source_evidence_ids", ())):
                affected.append(key)
        selected_before = [{"subject": subject, "group_id": target["group_id"],
                "assumption_key": target["selected_assumption_key"],
                "value": target["value"], "unit": target["unit"]}
            for target in selected if target.get("selected_assumption_key") in affected
            for subject in target["subjects"]]
        error = {"error_id": identity(["input_record_correction", record_key,
                    row.get("source_version", ""), code]),
            "field": row.get("field"), "fields": list(row.get("fields", ())),
            "value": row.get("value"), "unit": row.get("unit"), "error_code": code,
            "evidence_key": record_key, "source_fact_keys": fact_keys,
            "affected_assumption_keys": sorted(affected), "selected_before": selected_before,
            "source": row.get("source", ""), "source_subject": row.get("subject"),
            "source_version": row.get("source_version", "")}
        old_error = previous_errors.get(error["error_id"], {})
        error["selected_before"] = list({identity(item): item for item in (
            *old_error.get("selected_before", ()), *selected_before)}.values())
        for name in ("source_fact_keys", "affected_assumption_keys"):
            error[name] = sorted(set(old_error.get(name, ())) | set(error[name]))
        if error != old_error:
            diagnostics.append({"stage": "input_record_correction", "category": code,
                "status": "applied", "input_error": error, "invalidated_evidence": [row]})
        for key in affected:
            previous = updates.get(key, ledger.assumptions[key])
            invalid = list(previous.get("invalidated_evidence", ()))
            if row not in invalid:
                invalid.append(row)
            errors = {item["error_id"]: item for item in previous.get("input_errors", ())}
            errors[error["error_id"]] = error
            revised = {**previous, "invalidated_evidence": invalid,
                "input_errors": list(errors.values()), "source_candidate_unavailable": True}
            revised.pop("version", None)
            revised["version"] = identity(revised)
            if revised != previous:
                updates[key] = revised
    return ResearchResult(assumptions=tuple(updates.values()), diagnostics=tuple(diagnostics))


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def texts(value, name):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(f"{name} must be a list of nonempty text")
    return value




def assumption_candidate(action, ledger):
    """Validate a hypothesis before acquisition and again before assessment."""
    if not isinstance(action, dict):
        raise ValueError("Candidate assumption must be an object")
    subject, field = text(action.get("subject"), "subject"), action.get("field")
    if subject not in ledger.scenarios:
        scope_instruction = ("Use an exact caller-defined subject; fixed cases cannot be introduced or replaced. "
                             if any(s.get("scope_locked") for s in ledger.scenarios.values()) else
                             "Use an exact existing subject or define the new scope in this proposal's scenarios list. ")
        raise ValueError(f"Assumption subject {subject!r} has no reviewed scenario definition. "
                         f"Existing subjects: {list(ledger.scenarios)!r}. " + scope_instruction +
                         "Source-evidence labels do not automatically define model scenarios.")
    if field not in UNITS and field not in CONDITIONS:
        raise ValueError("Unknown assumption field or model condition")
    definition = ledger.scenarios[subject]
    if field in retained_defaults(definition):
        raise ValueError(f"{field} is a retained model default; the constants judge must promote it before a scenario-specific value is proposed")
    check_case_inputs(definition, {field: action.get("value")} if field in UNITS else {},
                      {field: action.get("statement")} if field in CONDITIONS else {})
    if field in definition.get("fixed_parameters", {}) or field in definition.get("fixed_conditions", {}):
        raise ValueError(f"{field} is already a fixed scenario condition, not an assumption to propose or reassess")
    value = action.get("value")
    if field in UNITS:
        if value is None:
            raise ValueError(f"{field} is a numeric parameter, not a model condition. Supply a finite "
                             "candidate value or keep it as an unresolved information need; omit the "
                             "candidate until a value can be proposed.")
        validate_parameter(field, value, action.get("unit"))
        if any(f.subject == subject and f.field == field and f.kind == "request_constraint" and f.value != value
               for f in ledger.facts.values()):
            raise ValueError("An assumption cannot override an admitted requester constraint")
    elif value is not None:
        raise ValueError("Model conditions use a statement, not a numeric value")
    claim = {"subject": subject, "field": field, "value": value,
             "unit": UNITS.get(field, "condition"),
             "statement": text(action.get("statement"), "statement"),
             "basis": text(action.get("basis"), "basis")}
    if action.get("entry_type") is not None:
        if action["entry_type"] != "best_guess" or field not in UNITS:
            raise ValueError("best_guess entries are inferred numeric parameters")
        claim.update(entry_type="best_guess", derivation=_candidate_derivation(action, claim, ledger))
    elif any(name in action for name in ("source_refs", "input_refs", "added_assumptions", "derivation")):
        raise ValueError("Inference provenance requires entry_type=best_guess")
    return claim


def _candidate_derivation(action, claim, ledger):
    from .records import numeric_references

    evidence = ledger.evidence_options()
    numeric = numeric_references(ledger)
    evidence_ids = {identity(row) for row in evidence.values()}
    numeric_keys = {fact.key: fact for fact in numeric.values()}
    if "derivation" in action:
        derivation = action["derivation"]
        if not isinstance(derivation, dict) or set(derivation) != {
                "source_evidence_ids", "input_fact_keys", "added_assumptions",
                "input_assumption_versions", "input_assumption_claims"}:
            raise ValueError("Invalid retained best-guess derivation")
        sources = texts(derivation["source_evidence_ids"], "source_evidence_ids")
        inputs = texts(derivation["input_fact_keys"], "input_fact_keys")
        added = texts(derivation["added_assumptions"], "added_assumptions")
    else:
        sources = texts(action.get("source_refs"), "source_refs")
        inputs = texts(action.get("input_refs"), "input_refs")
        added = texts(action.get("added_assumptions"), "added_assumptions")
        if any(ref not in evidence for ref in sources) or any(ref not in numeric for ref in inputs):
            raise ValueError("Best-guess references must select this episode's admitted evidence or numeric inputs")
        # Positional option names can move when a child adds source facts.
        sources = [identity(evidence[ref]) for ref in sources]
        inputs = [numeric[ref].key for ref in inputs]
    if any(key not in evidence_ids for key in sources) or any(key not in numeric_keys for key in inputs):
        raise ValueError("Best-guess provenance no longer identifies available evidence or numeric inputs")
    claim_key = identity([claim["subject"], claim["field"], claim["value"], None])
    versions, claims = {}, {}

    def check_lineage(key, path):
        if key == claim_key or key in path:
            raise ValueError("A best guess cannot depend on itself or cyclic input lineage")
        fact = numeric_keys.get(key)
        assumption = ledger.assumptions.get(key)
        if assumption is not None:
            versions[key] = assumption["version"]
            claims[key] = assumption_claim(assumption)
        if fact is not None and fact.kind == "model_assumption" and fact.parents:
            current = ledger.assumptions.get(fact.parents[0])
            if current is not None:
                expected_context = (current["statement"] if fact.subject == current["subject"] else
                                    f"Shared from {current['subject']}: {current['statement']}")
                if (fact.field != current["field"] or fact.value != current["value"]
                        or fact.unit != current["unit"] or fact.context != expected_context):
                    raise ValueError("Best-guess input refers to a changed assumption claim")
        parents = (fact.parents if fact is not None else
                   assumption.get("derivation", {}).get("input_fact_keys", ()) if assumption else ())
        for parent in parents:
            check_lineage(parent, {*path, key})

    for key in inputs:
        check_lineage(key, set())
    if "derivation" in action:
        prior_versions, prior_claims = derivation["input_assumption_versions"], derivation["input_assumption_claims"]
        if (not isinstance(prior_versions, dict) or not isinstance(prior_claims, dict)
                or set(prior_versions) != set(versions) or prior_claims != claims
                or any(not isinstance(value, str) or not value for value in prior_versions.values())):
            raise ValueError("Best-guess input assumption lineage is stale or invalid")
        # Evidence-only reviews may change a version without changing the input claim.
        versions = prior_versions
    return {"source_evidence_ids": list(dict.fromkeys(sources)),
            "input_fact_keys": list(dict.fromkeys(inputs)), "added_assumptions": added,
            "input_assumption_versions": versions, "input_assumption_claims": claims}


def candidate_response_schema(*, scenarios=None, evidence=None, numeric_inputs=None):
    prose = {"type": "string", "minLength": 1}
    properties = {name: prose for name in ("subject", "statement", "basis")}
    properties["subject"] = {**prose, "description": "Exact subject from context.defined_scenarios, "
        "or a new definition in this proposal's scenarios list. A source-evidence subject or "
        "colon-suffixed activity label alone is not a reviewed scenario."}
    if scenarios is not None:
        properties["subject"] = {"type": "string", "enum": list(scenarios),
                                 "description": "Exact reviewed target or calibration context; no new scope or replacement is allowed"}
    def references(options, description):
        item = {"type": "string"} if options is None else (
            {"type": "string", "enum": list(options)} if options else False)
        return {"type": "array", "uniqueItems": True, "items": item, "description": description}
    inferred = {
        "entry_type": {"const": "best_guess", "description": "An LLM-inferred parameter, not a directly reported source fact"},
        "source_refs": references(evidence, "Admitted evidence used in this inference; [] honestly declares no source support yet"),
        "input_refs": references(numeric_inputs, "Existing numeric operands actually used in the recasting or calculation; do not include a rejected value merely to contrast it with the proposed value. Preserve each operand's native meaning"),
        "added_assumptions": {"type": "array", "items": prose,
                              "description": "Explicit approximations or transfer assumptions added by this inference"},
    }
    schema = {"oneOf": [
        {"type": "object", "additionalProperties": False,
         "required": [*properties, "field", "value", "unit", *inferred], "properties": {**properties, **inferred,
             "field": {"type": "string", "enum": sorted(UNITS)},
             "value": {"type": "number", "description": "Finite scalar hypothesis, not an unresolved dependency"},
             "unit": {"type": "string", "enum": sorted(set(UNITS.values())),
                      "description": "The canonical unit for this field in canonical_fields"}}},
        {"type": "object", "additionalProperties": False,
         "required": [*properties, "field", "value", "unit"], "properties": {**properties,
             "field": {"type": "string", "enum": sorted(CONDITIONS)},
             "value": {"type": "null"}, "unit": {"const": "condition"}}},
    ]}
    if scenarios is not None:
        for branch in schema["oneOf"]:
            fields = branch["properties"]["field"]["enum"]
            branch["allOf"] = []
            for subject, definition in scenarios.items():
                choices = [f for f in fields
                    if f not in definition.get("fixed_parameters", {})
                    and f not in definition.get("fixed_conditions", {})
                    and f not in retained_defaults(definition)
                    and f not in definition.get("calculation_variables", ())]
                branch["allOf"].append({"if": {"properties": {"subject": {"const": subject}}},
                    "then": {"properties": {"field": {"enum": choices} if choices else False}}})
    return schema


def unassessed_assumption(candidate, ledger):
    """Retain a validated hypothesis, without inventing an independent judgment."""
    claim = assumption_candidate(candidate, ledger)
    key = identity([claim["subject"], claim["field"], claim["value"],
                    claim["statement"] if claim["field"] in CONDITIONS else None])
    assessment = {**claim, "key": key, "reasonableness": 0.0,
        "role": "inferred_assumption", "evidence": {}, "review": {}}
    assessment["version"] = identity(assessment)
    return assessment


def candidate_review_scope(candidate, ledger):
    """The physical cases and sharing decision to which a claim applies."""
    from .scenario_sharing import group_for

    group = group_for(ledger, candidate["subject"], candidate["field"]) or {}
    subjects = group.get("subjects", (candidate["subject"],))
    return {"scenarios": {name: ledger.scenarios.get(name) for name in subjects}, "group": group}


def register_candidate(candidate, ledger):
    """Retain a validated hypothesis before the common value and review processing.

    A registered statement and its input provenance define the claim. Rewriting
    explanatory prose cannot replace its physical premises without changing that
    statement or the identified provenance.
    """
    claim = assumption_candidate(candidate, ledger)
    key = identity([claim["subject"], claim["field"], claim["value"],
                    claim["statement"] if claim["field"] in CONDITIONS else None])
    current = ledger.assumptions.get(key)
    scope_version = identity(candidate_review_scope(claim, ledger))
    provenance_fields = ("source_evidence_ids", "input_fact_keys", "input_assumption_claims")

    def provenance(row):
        derivation = row.get("derivation", {})
        return {name: sorted(derivation.get(name, ())) if name != "input_assumption_claims"
                else derivation.get(name, {}) for name in provenance_fields}

    same_claim = (current is not None and assumption_claim(current) == assumption_claim(claim)
        and provenance(current) == provenance(claim)
        and current.get("candidate_scope_version", scope_version) == scope_version)
    if same_claim:
        changed_prose = (current["basis"] != claim["basis"] or
            current.get("derivation", {}).get("added_assumptions", ()) !=
            claim.get("derivation", {}).get("added_assumptions", ()))
        diagnostics = ({"stage": "candidate_registration", "category": "registered_claim_retained",
            "subject": claim["subject"], "field": claim["field"], "value": claim["value"],
            "why": "The existing claim, provenance and scope are unchanged. Its registered basis and added assumptions were retained; the proposed rationale rewrite was not applied. A material premise revision must change the claim statement or identified input/source provenance.",
            "proposed_basis": claim["basis"],
            "proposed_added_assumptions": claim.get("derivation", {}).get("added_assumptions", [])},) if changed_prose else ()
        return ResearchResult(assumptions=(current,), diagnostics=diagnostics)
    registered = unassessed_assumption(claim, ledger)
    registered["candidate_scope_version"] = scope_version
    registered.pop("version", None)
    registered["version"] = identity(registered)
    return ResearchResult(assumptions=(registered,))










def frame_scenario(agent, action, ledger, context):
    scenario = {"subject": text(action.get("subject"), "subject"),
                "description": text(action.get("description"), "description"),
                "basis": text(action.get("basis"), "basis"),
                "role": action.get("role", "target")}
    if scenario["role"] not in {"target", "calibration", "validation"}:
        raise ValueError("Scenario role must be target, calibration, or validation")
    if "parameters" in action:
        raise ValueError("Scenario framing cannot inject numeric parameter records")
    existing = ledger.scenarios.get(scenario["subject"])
    if existing:
        changed = [name for name in ("description", "role")
                   if existing.get(name, "target" if name == "role" else None) != scenario[name]]
        if changed:
            raise ValueError(f"Scenario {scenario['subject']!r} is already defined; changed fields: {changed}. "
                             "To reference it, use its existing subject and omit it from scenarios. "
                             "A changed scope needs a new subject; previous assessments cannot silently transfer")
        return ResearchResult(scenarios=(existing,))
    review = agent({"operation": "review_scenario_scope", "proposal": scenario, "context": context,
                    "existing_scenarios": list(ledger.scenarios.values()),
                    "source_evidence": [f.semantic_record() for f in ledger.facts.values()
                                        if f.subject == scenario["subject"] and f.kind.startswith("source_")]
                        + [e for e in ledger.evidence.values() if e["subject"] == scenario["subject"]
                           and e.get("kind") == "source_evidence"],
                    "instruction": (
                        "Check alignment with the original request. A scenario identifies a case or "
                        "conditional experiment, not an equation, model extension or research topic. "
                        "A target receives the requested safety estimates; calibration and validation "
                        "scenarios identify distinct evidential cases. Reject topic-only scopes rather "
                        "than an observed outbreak context: a fixed target scenario set still permits "
                        "separate source-grounded calibration contexts, but no new target cases. Check "
                        "the proposed event identity against source_evidence. Reject invented events rather "
                        "than treating them as additional target cases. Distinguish requester conditions "
                        "from agent-proposed study conditions. When cases are left open, an explicitly "
                        "conditional study case may instantiate that scope without claiming universality. "
                        "Do not replace specified cases or add numerical givens. Framing does not admit "
                        "parameters or evidence; every proposed input still requires acquisition or "
                        "evidence-linked assumption assessment. Explain a rejection so the parent can "
                        "understand which scope requirement was not met."),
                    "response_contract": {"type": "object", "additionalProperties": False,
                        "required": ["admit", "reason"], "properties": {
                            "admit": {"type": "boolean"}, "reason": {"type": "string", "minLength": 1}}}})
    if type(review.get("admit")) is not bool:
        raise ValueError("Scenario review requires a boolean decision")
    text(review.get("reason"), "scenario review reason")
    if not review["admit"]:
        return ResearchResult(findings=(review["reason"],))
    scenario["scope_review"] = review["reason"]
    return ResearchResult(scenarios=(scenario,))


def scientific_credit(result, focus, *, subjects=None, outputs=None):
    """Project independent target discoveries and separately qualified computations."""
    rows = []
    subjects = (set(subjects) if subjects is not None else
                {s["subject"] for s in result.scenarios if s.get("role", "target") == "target"})
    current = {a["key"]: a for a in result.assumptions}
    from .records import Ledger
    ledger = Ledger(result)
    from .scenario_sharing import independent_groups, needed_fields
    from .calculation_steps import calculation_step_status
    from ...task_executors.scenario_metrics.dependencies import CALCULATED_PARAMETER_FIELDS
    scoped_groups = independent_groups(ledger) if any(row.get("kind") == "group"
        for row in result.scenario_parameter_relations) else ()
    found = found_parameter_progress(result, ledger=ledger)
    outputs = set(METRICS) if outputs is None else set(outputs)
    calculation_fields = set(focus).intersection(outputs | CALCULATED_PARAMETER_FIELDS)
    steps = calculation_step_status(result, sorted(outputs), subjects=subjects) if calculation_fields else {"steps": []}
    step_scores = {(row["subject"], row["field"]): row["points"] for row in steps["steps"]}
    calibration = calibration_progress(result) if CALIBRATION_PROGRESS in focus else None
    linked_targets = target_calibration_progress(result, sorted(outputs), calibration=calibration) if calibration else {}
    for field in focus:
        if field == CALIBRATION_PROGRESS:
            score = max((max(PROVISIONAL_OUTPUT_CREDIT, row["support"])
                for row in calibration["calibrations"] if row["outbreak_evidence_acquired"] and row["current"]), default=0)
            rows.append((field, "workflow:outbreak_inversion", score))
            for subject in sorted(subjects):
                alternatives = linked_targets.get(subject, {}).get("alternatives", ())
                score = max((sum(max(PROVISIONAL_OUTPUT_CREDIT, alternative["calibrated_support_by_quantity"].get(name, 0))
                                 if alternative["calibrated_inputs_by_quantity"].get(name) else 0
                                 for name in outputs) / len(outputs)
                             for alternative in alternatives if alternative["coherent"] and outputs), default=0)
                rows.append((field, "target:" + subject, score))
            continue
        if field in calculation_fields:
            rows.extend((field, subject, step_scores.get((subject, field), 0.0)) for subject in sorted(subjects))
            continue
        if field in UNITS:
            groups = [row for row in found if row["parameter"] == field
                      and set(row["subjects"]).intersection(subjects)]
            rows.extend((field, "group:" + row["group_id"], row["points"]) for row in groups)
            if not groups:
                rows.extend((field, subject, 0.0) for subject in sorted(subjects))
            continue
        if field not in outputs and field != "evidence_consistency":
            groups = [group for group in scoped_groups if group["parameter"] == field
                      and set(group["subjects"]).intersection(subjects)]
            if groups:
                for group in groups:
                    learning = parameter_support(result, (field,), hypotheses_only=True,
                                                 subjects=group["subjects"])
                    rows.append((field, "group:" + group["group_id"],
                                 learning[0]["points"] if learning else 0.0))
                continue
        for subject in sorted(subjects):
            if field == "evidence_consistency":
                # Establishing a discrepancy is learning too; a smaller residual is not credit.
                score = max((current_support(c, current, ledger=ledger) for c in result.calculations
                    if c["subject"] == subject and c["tool"] == "compare_predictions_with_observations"
                    and c["output"]["category"] != "calibration_consistency"), default=0)
            else:
                scenario = ledger.scenarios.get(subject, {})
                if (field in retained_defaults(scenario) or field in CALCULATED_PARAMETER_FIELDS
                        or (scoped_groups and field not in needed_fields(scenario))):
                    rows.append((field, subject, 0.0))
                    continue
                learning = parameter_support(result, (field,), hypotheses_only=True, subjects=(subject,))
                score = learning[0]["points"] if learning else 0.0
            rows.append((field, subject, score))
    return Credit(evaluated=result.evaluated, qualities=tuple(rows),
                  quality_resolution=PROVISIONAL_OUTPUT_CREDIT / (max(1, len(outputs)) if calibration else 1))
