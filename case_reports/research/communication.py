"""Processed decision context, with the full evidence and failures left in audit."""

from .records import identity, numeric_references


def input_error_view(ledger, target_values=None, *, include_record_refs=False):
    """Closed error footnotes tied to the current authoritative target selection."""
    import math
    from ...task_executors.scenario_metrics.evaluate import UNITS
    from .progress import found_parameter_progress

    errors = {}
    records = [*ledger.assumptions.values(), *ledger.calculations.values()]
    for record in records:
        for error in record.get("input_errors", ()):
            if isinstance(error, dict) and error.get("error_id"):
                errors[error["error_id"]] = error
    for diagnostic in ledger.diagnostics:
        error = diagnostic.get("input_error")
        if diagnostic.get("stage") == "input_record_correction" and isinstance(error, dict) and error.get("error_id"):
            errors[error["error_id"]] = error
    references = {"facts": {}, "evidence": {}, "assumptions": {}}
    if not errors:
        return {"footnotes": [], "record_refs": references} if include_record_refs else []
    if target_values is None:
        target_values = found_parameter_progress(ledger.result(), ledger=ledger)

    def number(value):
        return value if type(value) in (int, float) and math.isfinite(value) else None

    notes = []
    for error_id, error in errors.items():
        field, code = error.get("field"), error.get("error_code")
        error_fields = ({field} | set(error.get("fields", ()))) & set(UNITS)
        if not error_fields or code not in {
                "quantity_mismatch", "value_mismatch", "unit_mismatch", "source_attribution_mismatch"}:
            continue
        error_ref = "error:" + identity(error_id)[:12]
        for category, keys in (("facts", error.get("source_fact_keys", ())),
                               ("evidence", (error.get("evidence_key"),)),
                               ("assumptions", error.get("affected_assumption_keys", ()))):
            for key in keys:
                if key:
                    references[category].setdefault(key, []).append(error_ref)
        affected_keys = set(error.get("affected_assumption_keys", ()))
        affected_targets = {(row["subject"], row["field"])
                            for key, row in ledger.assumptions.items() if key in affected_keys}
        before = error.get("selected_before", ())
        previous_groups = {row.get("group_id") for row in before}
        effects = []
        for current in target_values:
            if (current.get("group_id") not in previous_groups
                    and not any((subject, current.get("parameter")) in affected_targets
                                for subject in current.get("subjects", ()))):
                continue
            selected_error = current.get("selected_assumption_key") in affected_keys
            was_selected = current.get("group_id") in previous_groups
            resolution = ("error" if selected_error or not current.get("found") else
                          "replaced" if was_selected else "invalid_alternative_excluded")
            effects.append({"group_ref": "target_group:" + identity(current["group_id"])[:12],
                "subjects": list(current.get("subjects", ())), "field": current["parameter"], "resolution": resolution,
                "current_value": number(current.get("value")),
                "current_unit": current.get("unit") if current.get("unit") == UNITS[current["parameter"]] else None,
                "current_value_found": current.get("found") is True})
        source, owner = error.get("source"), error.get("source_subject")
        notes.append({"error_ref": error_ref,
            "field": field if field in UNITS else None, "fields": sorted(error_fields),
            "value": number(error.get("value")),
            "unit": error.get("unit") if error.get("unit") in set(UNITS.values()) else None,
            "error_code": code,
            "source_ref": "source:" + identity(source)[:12] if isinstance(source, str) and source else None,
            "source_subject": "source_subject:" + identity(owner)[:12] if isinstance(owner, str) and owner else None,
            "targets": effects})
    return {"footnotes": notes, "record_refs": references} if include_record_refs else notes


def decision_records(ledger, *, target_subjects=()):
    """Typed decision catalogs and internal resolvers for the same stored records.

    Catalog keys preserve the executor's evidence/input/calculation references.
    Only the catalogs cross the model boundary; the resolver maps stay in code.
    Scientific reviewers still read the original records from the ledger.
    """
    import math
    from ...task_executors.scenario_metrics.catalog import METRICS, metric_catalog_record
    from ...task_executors.scenario_metrics.evaluate import CONDITIONS, UNITS
    from .calculation_steps import calculation_is_up_to_date
    from .records import current_support
    from .progress import evidence_key
    from .tools import TOOLS

    targets = set(target_subjects)
    units = {**{name: metric_catalog_record(name).get("unit") for name in METRICS}, **UNITS}
    fields = set(units) | set(CONDITIONS)
    subjects, parameterizations, conditions, sources = {}, {}, {}, {}
    errors = input_error_view(ledger, target_values=[], include_record_refs=True)["record_refs"]

    def subject(value):
        if not isinstance(value, str):
            return None
        subjects.setdefault(value, value if value in targets else "source_subject:" + identity(value)[:12])
        return subjects[value]

    def number(value):
        return value if type(value) in (int, float) and math.isfinite(value) else None

    def enum(value, options, default="unknown"):
        return value if isinstance(value, str) and value in options else default

    def condition(owner, field, statement):
        ref = "condition:" + identity([owner, field, statement])[:12]
        conditions[ref] = {"subject": owner, "field": field, "statement": statement}
        return ref

    def numeric(row):
        field = row.get("field")
        canonical = units.get(field)
        if field not in units or (row.get("unit") is not None and row["unit"] != canonical):
            raise ValueError("Decision catalog requires a canonical quantity and unit")
        refs = list(errors["facts"].get(identity(row), ()))
        for parent in row.get("parents", ()):
            refs.extend(errors["assumptions"].get(parent, ()))
        refs.extend(errors["assumptions"].get(row.get("key"), ()))
        return {"subject": subject(row.get("subject")), "field": field,
                "value": number(row.get("value")), "unit": canonical,
                "input_status": "error" if refs else "recorded", "error_refs": list(dict.fromkeys(refs)),
                "kind": enum(row.get("kind", row.get("role")), {
                    "user_assumption", "request_constraint", "scenario_constraint", "model_default",
                    "source_observation", "source_model_parameter", "model_assumption", "inferred_assumption", "derived"})}

    def review(row):
        value = row.get("review", {})
        return {"completed": value.get("completed", value.get("verdict") not in (None, "unassessed")) is True,
                "verdict": enum(value.get("verdict"), {
                    "unassessed", "supported", "provisional", "unsupported", "contradicted", "fixed_condition"}),
                "grades": {name: item["grade"] for name, item in value.get("criteria", {}).items()
                    if name in {"support", "applicability", "physical_consistency", "uncertainty"}
                    and isinstance(item, dict) and type(item.get("grade")) is int and item["grade"] in (0, 1, 2)},
                "supporting_evidence_count": len(value.get("supporting_evidence", ())),
                "contradicting_evidence_count": len(value.get("contradicting_evidence", ())),
                "invalid_evidence_count": len(value.get("invalid_evidence", ()))}

    assumptions = {}
    for index, row in enumerate(ledger.assumptions.values(), 1):
        if row.get("field") not in fields:
            continue
        item = numeric(row) if row.get("value") is not None else {
            "subject": subject(row.get("subject")), "field": row["field"], "value": None, "unit": None,
            "condition_ref": condition(row["subject"], row["field"], row["statement"])}
        applicability = row.get("source_applicability", {})
        assumptions[f"assumption:{index}"] = {**item, "review": review(row),
            "reasonableness": number(row.get("reasonableness")),
            "available": row.get("source_candidate_unavailable") is not True,
            "relevance_score": number(applicability.get("score")),
            "entry_type": enum(row.get("entry_type"), {"extracted", "calculated", "best_guess", "source_candidate"})}

    evidence = {}
    for ref, row in ledger.evidence_options().items():
        item = numeric(row) if row.get("field") in units and number(row.get("value")) is not None else {
            "subject": subject(row.get("subject")),
            "fields": sorted(field for field in row.get("fields", ()) if field in fields),
            "kind": enum(row.get("kind"), {"source_evidence", "request_constraint"})}
        if row.get("source"):
            sources[row["source"]] = "source:" + identity(row["source"])[:12]
        constraints = []
        for bound in row.get("numeric_constraints", ()):
            field, relation = bound.get("field"), bound.get("relation")
            if field not in units or relation not in {"interval", "lower_bound", "upper_bound"}:
                continue
            if bound.get("unit") != units[field]:
                raise ValueError("Decision bounds require canonical units")
            constraints.append({"field": field, "relation": relation, "unit": units[field],
                **{name: number(bound[name]) for name in ("minimum", "maximum") if name in bound},
                **{name: bound[name] for name in ("minimum_inclusive", "maximum_inclusive")
                   if type(bound.get(name)) is bool}})
        evidence[ref] = {**item,
            "input_status": "error" if evidence_key(row) in errors["evidence"] else item.get("input_status", "recorded"),
            "error_refs": list(dict.fromkeys([*item.get("error_refs", ()),
                                              *errors["evidence"].get(evidence_key(row), ())])),
            "source_ref": sources.get(row.get("source")),
            "evidence_role": enum(row.get("evidence_role", {"source_observation": "observation",
                "source_model_parameter": "model_statement", "request_constraint": "request_constraint"}.get(row.get("kind"))),
                {"observation", "model_statement", "context", "request_constraint"}),
            "observed_fields": ([row["field"]] if row.get("kind") == "source_observation" and row.get("field") in fields else
                                sorted(field for field in row.get("observed_fields", ()) if field in fields)),
            "numeric_constraints": constraints,
            "relevance": [{"target": relation["target_subject"], "parameter": relation["parameter"],
                           "score": number(relation.get("score"))}
                          for relation in ledger.source_target_relationships.values()
                          if relation.get("source_subject") == row.get("subject")
                          and relation.get("source") == row.get("source")
                          and relation.get("target_subject") in targets and relation.get("parameter") in fields],
            "has_quote": bool(row.get("quote")), "has_context": bool(row.get("context"))}

    numeric_options = {ref: numeric(fact.as_record()) for ref, fact in numeric_references(ledger).items()}
    facts = {ref: numeric(fact.as_record()) for ref, fact in ledger.options().items()}
    calculations = {}
    for ref, row in ledger.calculation_options().items():
        label = row.get("parameterization")
        if isinstance(label, str):
            parameterizations.setdefault(label, "parameterization:" + identity(label)[:12])
        outputs = {}
        for field, output in {**row.get("output", {}), **row.get("intermediate_outputs", {}),
                              **row.get("derived_parameters", {})}.items():
            if field not in units or not isinstance(output, dict):
                continue
            value = number(output.get("value"))
            outputs[field] = {"value": value, "unit": units[field],
                "status": enum(output.get("status", "calculated" if value is not None else "not_calculated"),
                    {"calculated", "not_calculated", "missing_inputs", "inapplicable", "unsupported", "unsupported_variable", "undefined"}),
                "limit": enum(output.get("limit"), {"finite", "unbounded", "empty", "function"}, None),
                "missing_inputs": [field for field in output.get("missing", ()) if field in fields]}
        model_review = row.get("model_review", {})
        calculations[ref] = {"subject": subject(row["subject"]), "tool": enum(row.get("tool"), TOOLS),
            "parameterization": parameterizations.get(label), "outputs": outputs,
            "review_completed": model_review.get("completed", "admit" in model_review) is True,
            "review_admitted": model_review.get("admit") if type(model_review.get("admit")) is bool else None,
            "support_complete": row.get("support_complete", True) is True,
            "current_support": current_support(row, ledger.assumptions, ledger=ledger),
            "current_computed_metrics": [field for field in row.get("output", {})
                                         if field in units and calculation_is_up_to_date(row, field, ledger)],
            "current_intermediate_metrics": [field for field in row.get("intermediate_outputs", {})
                                             if field in units and calculation_is_up_to_date(row, field, ledger)],
            "metric_support": {field: current_support(row, ledger.assumptions, metric=field, ledger=ledger)
                               for field in row.get("metric_support", {}) if field in units},
            "conditions": {field: condition(row["subject"], field, statement)
                           for field, statement in row.get("conditions", {}).items() if field in CONDITIONS}}

    scenarios = {}
    for row in ledger.estimates.values():
        for name in ("parameterization", "label"):
            label = row.get(name)
            if isinstance(label, str):
                parameterizations.setdefault(label, "parameterization:" + identity(label)[:12])
    for name, row in ledger.scenarios.items():
        ref = subject(name)
        fixed_conditions = {field: condition(name, field, statement)
                            for field, statement in row.get("fixed_conditions", {}).items() if field in CONDITIONS}
        if name in targets:
            scenarios[ref] = {**{key: row[key] for key in (
                "subject", "description", "role", "basis", "scope_locked", "fixed_parameters", "output_contracts",
                "assumption_needs", "calculation_variables") if key in row}, "fixed_conditions": fixed_conditions,
                "model_defaults": {field: {"value": number(item.get("value")), "unit": units[field],
                    "decision": enum(item.get("decision"), {"retain", "revisit"})}
                    for field, item in row.get("model_defaults", {}).items() if field in units}}
            continue
        scenarios[ref] = {"subject": ref, "role": enum(row.get("role"), {"target", "calibration", "validation"}),
            "fixed_parameters": {field: {"value": number(item.get("value")), "unit": units[field]}
                                 for field, item in row.get("fixed_parameters", {}).items() if field in units},
            "fixed_conditions": fixed_conditions,
            "assumption_needs": [field for field in row.get("assumption_needs", ()) if field in fields],
            "calculation_variables": [field for field in row.get("calculation_variables", ()) if field in units]}
    return {"assumptions": assumptions, "evidence": evidence, "numeric": numeric_options,
            "facts": facts, "calculations": calculations, "scenarios": scenarios,
            "subjects": subjects, "parameterizations": parameterizations, "conditions": conditions, "sources": sources}


def calculation_deferral_contract(table):
    """Describe and validate decision metadata against the displayed table only."""
    from collections.abc import Mapping
    from ...task_executors.scenario_metrics.evaluate import CONDITIONS, UNITS

    reasons = {
        "missing_value": "A numeric input needs a value before the intended calculation.",
        "input_error": "A selected input may record the wrong quantity, value, unit or source; a recorded error reference is not required for a newly suspected error.",
        "unresolved_condition": "A physical model condition needs to be stated or resolved for the intended calculation.",
        "model_constraint": "The intended calculation has a model or parameter constraint that the chosen work addresses.",
        "improving_evidence_support": "The chosen work aims to strengthen evidence or model support instead of performing this calculation now.",
    }
    related_fields = set(UNITS) | set(CONDITIONS)
    needs_fields = {"missing_value", "input_error", "unresolved_condition"}
    rows = {}
    for values in table.get("rows", ()):
        row = dict(zip(table["columns"], values, strict=True))
        if row["kind"] != "requested_output":
            continue
        key = (row["subject"], row["field"])
        if any(not isinstance(value, str) or not value for value in key) or key in rows:
            raise ValueError("Calculation deferral targets require unique displayed subject/field pairs")
        rows[key] = {
            "routes": frozenset(route["route_ref"] for route in row.get("routes", ())),
            "errors": frozenset(row.get("error_refs", ())),
        }

    def references(options, description):
        return {"type": "array", "uniqueItems": True,
            "items": {"type": "string", "enum": sorted(options)} if options else False,
            "description": description}

    item_fields = {"subject", "field", "reason", "related_fields", "route_ref", "error_refs"}
    route_refs = set().union(*(row["routes"] for row in rows.values()))
    error_refs = set().union(*(row["errors"] for row in rows.values()))
    item_schema = {"type": "object", "additionalProperties": False,
        "required": sorted(item_fields), "properties": {
            "subject": {"type": "string", "enum": sorted({subject for subject, _ in rows}),
                "description": "Target subject of the existing requested-output row being deferred."},
            "field": {"type": "string", "enum": sorted({field for _, field in rows}),
                "description": "Requested output in that subject's displayed calculation row."},
            "reason": {"type": "string", "enum": list(reasons),
                "description": " ".join(f"{name}: {meaning}" for name, meaning in reasons.items())},
            "related_fields": references(related_fields,
                "Canonical numeric-input or model-condition names relevant to this decision. Required for missing_value, input_error and unresolved_condition; may be empty for model_constraint or improving_evidence_support."),
            "route_ref": {"type": ["string", "null"], "enum": [None, *sorted(route_refs)],
                "description": "A route_ref from this exact subject/field row, or null when no particular route is cited."},
            "error_refs": references(error_refs,
                "Only error_refs displayed on this exact subject/field row. Use [] if none is cited, including a newly suspected input error."),
        }, "allOf": [{"if": {"required": ["reason"],
            "properties": {"reason": {"enum": sorted(needs_fields)}}},
            "then": {"properties": {"related_fields": {"minItems": 1}}}}]}
    schema = {"type": "array", "minItems": 1 if rows else 0, "uniqueItems": True,
        "items": item_schema if rows else False,
        "description": "For investigate or evidence_strategy, identify one or more relevant requested-output rows whose calculation you are deferring and state your decision reasons. Do not enumerate the whole table. Return [] only when there are no requested-output rows. These are decision records, not permission checks or instructions to execute a calculation."}

    def validate(proposal):
        action = proposal.get("action")
        if not isinstance(action, str) or action not in {"investigate", "evidence_strategy"}:
            if "calculation_deferrals" in proposal:
                raise ValueError("Calculation and candidates-only proposals must omit calculation_deferrals")
            return
        deferred = proposal.get("calculation_deferrals")
        if not isinstance(deferred, list) or (rows and not deferred):
            raise ValueError("Investigate and evidence_strategy require a calculation_deferrals array, nonempty when requested-output rows are displayed")
        seen = set()
        for item in deferred:
            if not isinstance(item, Mapping) or set(item) != item_fields:
                raise ValueError("Each calculation deferral must contain exactly subject, field, reason, related_fields, route_ref and error_refs")
            subject, field = item["subject"], item["field"]
            if not isinstance(subject, str) or not isinstance(field, str) or (subject, field) not in rows:
                raise ValueError("A calculation deferral must identify a displayed requested-output subject/field pair")
            reason = item["reason"]
            if not isinstance(reason, str) or reason not in reasons:
                raise ValueError("A calculation deferral requires a declared reason code")
            related = item["related_fields"]
            if (not isinstance(related, list)
                    or any(not isinstance(ref, str) or ref not in related_fields for ref in related)
                    or len(related) != len(set(related))
                    or (reason in needs_fields and not related)):
                raise ValueError("Calculation deferral related_fields must be unique canonical input/condition names, nonempty for missing_value, input_error or unresolved_condition")
            row = rows[(subject, field)]
            route = item["route_ref"]
            if route is not None and (not isinstance(route, str) or route not in row["routes"]):
                raise ValueError("Calculation deferral route_ref must belong to its displayed subject/field row")
            errors = item["error_refs"]
            if (not isinstance(errors, list)
                    or any(not isinstance(ref, str) or ref not in row["errors"] for ref in errors)
                    or len(errors) != len(set(errors))):
                raise ValueError("Calculation deferral error_refs must be unique references on its displayed subject/field row")
            signature = (subject, field, reason, tuple(sorted(related)), route, tuple(sorted(errors)))
            if signature in seen:
                raise ValueError("Calculation deferrals must not repeat the same references and reason")
            seen.add(signature)

    return schema, validate


def merged_ranges(ranges):
    result = []
    for start, end in sorted(ranges):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def reading_coverage(attempts):
    """Execution coverage, not evidence of absence or a scientific judgment."""
    groups = {}
    for attempt in attempts:
        key = (attempt["source"], attempt["source_version"])
        group = groups.setdefault(key, {
            "source": key[0], "source_version": key[1], "document_characters": attempt["document_characters"],
            "attempts": 0, "outcomes": {}, "examined_ranges": [], "reviewed_ranges": [], "questions": []})
        group["attempts"] += 1
        status = attempt["status"]
        group["outcomes"][status] = group["outcomes"].get(status, 0) + 1
        if status != "not_started":
            group["examined_ranges"].extend(attempt["ranges"])
        if status == "completed":
            group["reviewed_ranges"].extend(attempt["ranges"])
        if attempt["question"] not in group["questions"]:
            group["questions"].append(attempt["question"])
    for group in groups.values():
        for name in ("examined_ranges", "reviewed_ranges"):
            group[name] = merged_ranges(group[name])
        unread, position = [], 0
        for start, end in group["examined_ranges"]:
            if position < start:
                unread.append([position, start])
            position = end
        if position < group["document_characters"]:
            unread.append([position, group["document_characters"]])
        group["unexamined_ranges"] = unread
        group["interpretation"] = (
            "Half-open character offsets in this source version. Examined means supplied to an extraction attempt; "
            "reviewed means extraction/admission completed, not that every claim was accepted or the question resolved. "
            "Different questions may need rereading. Unexamined text cannot support an absence claim.")
    return list(groups.values())


def decision_diagnostics(diagnostics):
    # A completed scientific judgment, including unsupported or contradicted,
    # is reported on its claim. It is not an unsuccessful operation.
    return (d for d in diagnostics if d.get("stage") != "assumption_assessment")


def compact_assumption(assessment):
    return {k: assessment[k] for k in ("subject", "field", "value", "statement", "reasonableness")}
