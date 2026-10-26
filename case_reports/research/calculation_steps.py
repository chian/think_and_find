"""Versioned calculation progress, separate from finding scientific inputs."""

import math
import operator
from collections.abc import Mapping

from ...task_executors.scenario_metrics.dependencies import (
    input_requirements,
)
from ...task_executors.scenario_metrics.evaluate import UNITS, output_is_resolved, output_meets_contract
from ...task_executors.scenario_metrics.catalog import METRICS
from .model_defaults import retained_defaults
from .records import (
    Ledger, admitted_calculation, assumption_matches,
    current_premises, current_support, identity,
)
from .progress import evidence_key


COMPLETION_CREDIT = 12.5
REUSABLE_METHOD_CREDIT = 2.5
CURRENT_RESULT_CREDIT = COMPLETION_CREDIT - REUSABLE_METHOD_CREDIT
REQUESTED_OUTPUT_MULTIPLIER = 2.0
SUPPORT_CREDIT_BASELINE = 0.125


def _step_fields(scenario, outputs):
    """All reachable executable nodes; alternative recipes remain optional."""
    fixed = {**scenario.get("fixed_parameters", {}), **retained_defaults(scenario)}
    conditions = set(scenario.get("fixed_conditions", {}))
    requirements = input_requirements(outputs, fixed, conditions,
        scenario.get("calculation_variables", ()))["unresolved"]
    fields, visited = set(outputs), set()

    def visit(field):
        if field in visited or field in fixed or field in conditions:
            return
        visited.add(field)
        for route in requirements.get(field, {}).get("routes", ()):
            route_conditions = set(route["conditions"])
            if (("negligible_near_field" in conditions and "continuous_unmasked_jet" in route_conditions)
                    or ("continuous_unmasked_jet" in conditions and "negligible_near_field" in route_conditions)):
                continue
            compatible = True
            for constraint in route.get("parameter_constraints", ()):
                accepts = {"equal": operator.eq, "minimum": operator.ge}[constraint["operator"]]
                if constraint["field"] in fixed and not accepts(fixed[constraint["field"]]["value"], constraint["value"]):
                    compatible = False
            if not compatible:
                continue
            fields.add(field)
            for name in route["inputs"]:
                visit(name)

    for field in outputs:
        visit(field)
    return fields


def _including_intermediates(record):
    metrics = record.get("intermediate_outputs", {})
    if not metrics:
        return record
    key = "metrics" if "metrics" in record else "output"
    return {**record, key: {**record.get(key, {}), **metrics},
            "metric_support": {**record.get("metric_support", {}), **record.get("intermediate_support", {})}}


def _requirements(field, numeric, conditions, options, variables=()):
    """Show executable alternatives without selecting among candidate values."""
    requirements = input_requirements((field,), conditions=conditions, variables=variables)["unresolved"]

    def resolve(name, visiting):
        if name in numeric or name in conditions or name in variables:
            return set(), set(), []
        if name in visiting:
            return {name}, set(), []
        routes = requirements.get(name, {}).get("routes", ())
        if not routes:
            return ({name}, set(), []) if name in UNITS else (set(), {name}, [])
        alternatives = [missing(route, {*visiting, name}) for route in routes]
        def rank(row):
            blocked_inputs = row[0] | {item["field"] for item in row[2]}
            return len(blocked_inputs) + len(row[1]), sorted(blocked_inputs), sorted(row[1])
        return min(alternatives, key=rank)

    def missing(route, visiting):
        absent, absent_conditions, failures = set(), set(), []
        for name in route["inputs"]:
            fields, needed, constraints = resolve(name, visiting)
            absent.update(fields)
            absent_conditions.update(needed)
            failures.extend(constraints)
        for constraint in route.get("parameter_constraints", ()):
            accepts = {"equal": operator.eq, "minimum": operator.ge}[constraint["operator"]]
            values = options.get(constraint["field"], ())
            if not values:
                absent.add(constraint["field"])
            elif not any(accepts(value, constraint["value"]) for value in values):
                failures.append({"field": constraint["field"], "operator": constraint["operator"],
                    "required_value": constraint["value"], "actual_values": sorted(values)})
        return absent, absent_conditions, list({identity(item): item for item in failures}.values())

    rows = []
    for route in requirements.get(field, {}).get("routes", ()):
        absent, absent_conditions, failures = missing(route, {field})
        rows.append({**route, "missing_inputs": sorted(absent),
                     "missing_conditions": sorted(absent_conditions),
                     "constraint_failures": failures,
                     "ready": not absent and not absent_conditions and not failures})
    return rows


def _numerically_completed(record, field, *, derived):
    if derived:
        if field not in record.get("derived_support", {}):
            return False
        row = record.get("derived_parameters", {}).get(field, {})
        value = row.get("value")
        return (bool(row.get("inputs")) and bool(row.get("formula"))
                and type(value) in (int, float) and math.isfinite(value) and value >= 0
                and field not in record.get("input_values", record.get("parameters", {})))
    row = record.get("output", record.get("metrics", {})).get(field, {})
    return isinstance(row, Mapping) and output_is_resolved(row)


def selected_target_snapshot(ledger):
    """One numerical main-selection snapshot, shared by all outputs of execution."""
    from .progress import found_parameter_progress
    return {(subject, row["parameter"]): row
            for row in found_parameter_progress(ledger.result(), ledger=ledger)
            for subject in row["subjects"]}


def selected_target_bindings(facts, selection):
    """Bind direct operands; derived operands retain their producer's snapshot."""
    bindings = []
    for fact in facts:
        if fact.kind == "derived":
            continue
        main = selection.get((fact.subject, fact.field))
        if main is None:
            continue
        bindings.append({"subject": fact.subject, "field": fact.field,
            "group_id": main["group_id"], "input_fact_key": fact.key,
            "assumption_key": fact.parents[0] if fact.kind == "model_assumption" and fact.parents else None,
            "used_value": fact.value, "used_unit": fact.unit,
            "selected_value": main.get("value"), "selected_unit": main.get("unit"),
            "selected_assumption_key": main.get("selected_assumption_key"),
            "follows_main_selection": bool(main.get("found"))
                and (fact.value, fact.unit) == (main.get("value"), main.get("unit"))})
    return bindings


def _main_input_differences(record, dependency, ledger, seen=frozenset()):
    token = identity(dependency)
    if token in seen:
        return []
    differences = []
    if "selected_target_bindings" not in dependency:
        for change in _selected_source_changes(dependency, ledger):
            assumption = ledger.assumptions.get(change.get("key"), {})
            differences.append({"subject": record.get("subject"), "field": change["field"],
                "used_value": change.get("previous_value"), "used_unit": assumption.get("unit"),
                "current_selected_value": change.get("selected_value"),
                "current_selected_unit": assumption.get("unit") if change.get("selected_value") is not None else None})
    selection = selected_target_snapshot(ledger)
    for binding in dependency.get("selected_target_bindings", ()):
        main = selection.get((binding["subject"], binding["field"]), {})
        if (not main.get("found") or (binding["used_value"], binding["used_unit"])
                != (main.get("value"), main.get("unit"))):
            differences.append({"subject": binding["subject"], "field": binding["field"],
                "used_value": binding["used_value"], "used_unit": binding["used_unit"],
                "current_selected_value": main.get("value"), "current_selected_unit": main.get("unit")})
    for key in dependency.get("input_fact_keys", ()):
        fact = ledger.facts.get(key)
        if fact is None or fact.kind != "derived":
            continue
        producers = [row for row in ledger.calculations.values() if key in row.get("output_fact_keys", ())]
        if not producers:
            continue
        producer = producers[-1]
        upstream = producer.get("derived_support", {}).get(fact.field, producer)
        differences.extend(_main_input_differences(producer, upstream, ledger, {*seen, token}))
    return list({identity(row): row for row in differences}.values())


def _selected_source_changes(dependency, ledger):
    """An executed target input must still equal its current main-table value."""
    if "selected_target_bindings" in dependency:
        selection = selected_target_snapshot(ledger)
        changes = []
        for binding in dependency["selected_target_bindings"]:
            current = selection.get((binding["subject"], binding["field"]), {})
            if (current.get("found") and (current.get("value"), current.get("unit"))
                    == (binding["used_value"], binding["used_unit"])):
                continue
            changes.append({"kind": "selected_target_value", "key": binding.get("assumption_key"),
                "field": binding["field"], "previous_value": binding["used_value"],
                "selected_value": current.get("value"),
                "selected_assumption_key": current.get("selected_assumption_key"),
                "reason": "The used numerical input differs from the current main-table value or is no longer available."})
        return changes
    # Older records lack execution-time selection context. Preserve their
    # existing invalidation behavior without inventing a historical snapshot.
    keys = dependency.get("assumption_keys", dependency.get("assumption_versions", {}))
    premises = [ledger.assumptions[key] for key in keys if key in ledger.assumptions]
    if not premises:
        return []
    from .progress import found_parameter_progress
    selected = {row["group_id"]: row for row in found_parameter_progress(ledger.result(), ledger=ledger)}
    by_target = {(subject, row["parameter"]): row for row in selected.values() for subject in row["subjects"]}
    changes = []
    for previous in premises:
        if previous.get("entry_type") == "source_candidate":
            current = selected.get(previous.get("source_candidate_group"), {})
        else:
            current = by_target.get((previous["subject"], previous["field"]))
            if current is None:
                continue
        if current.get("found") and (current["value"], current["unit"]) == (previous["value"], previous["unit"]):
            continue
        changes.append({"kind": "selected_target_value", "key": previous["key"],
            "field": previous["field"], "previous_value": previous["value"],
            "selected_value": current.get("value"),
            "selected_assumption_key": current.get("selected_assumption_key"),
            "reason": "The automatically selected source value changed or is no longer available."})
    return changes


def _changed_dependencies(dependency, ledger, seen=frozenset()):
    token = identity(dependency)
    if token in seen:
        return []
    seen = {*seen, token}
    changes = _selected_source_changes(dependency, ledger)
    for key in dependency.get("assumption_keys", dependency.get("assumption_versions", {})):
        current = ledger.assumptions.get(key)
        if not assumption_matches(dependency, key, current):
            changes.append({"kind": "assumption", "key": key,
                            "field": current.get("field") if current else None,
                            "reason": "The selected claim changed or is no longer available."})
        elif current.get("review", {}).get("verdict") == "contradicted":
            changes.append({"kind": "assumption", "key": key, "field": current["field"],
                            "reason": "The selected claim is now contradicted."})
    for binding in dependency.get("scenario_input_bindings", ()):
        scenario = ledger.scenarios.get(binding["subject"], {})
        current = (retained_defaults(scenario) if binding["kind"] == "model_default"
                   else scenario.get("fixed_parameters", {})).get(binding["field"])
        if current is None or any(current.get(key) != binding.get(key) for key in ("value", "unit")):
            changes.append({"kind": binding["kind"], "field": binding["field"],
                            "reason": "The bound scenario input changed or its default was withdrawn."})
    for binding in dependency.get("scenario_condition_bindings", ()):
        scenario = ledger.scenarios.get(binding["subject"], {})
        if scenario.get("fixed_conditions", {}).get(binding["field"]) != binding["statement"]:
            changes.append({"kind": "condition", "field": binding["field"],
                            "reason": "The bound scenario condition changed or was withdrawn."})
    for key in dependency.get("invalidated_dependencies", ()):
        changes.append({"kind": "input", "key": key, "reason": "Recorded input provenance is invalidated."})
    for key in dependency.get("input_fact_keys", ()):
        fact = ledger.facts.get(key)
        if fact is None or fact.kind != "derived":
            continue
        for producer in ledger.calculations.values():
            if key in producer.get("output_fact_keys", ()):
                upstream = producer.get("derived_support", {}).get(fact.field, producer)
                changes.extend(_changed_dependencies(upstream, ledger, seen))
    return list({identity(change): change for change in changes}.values())


def _fresh_dependencies(record, dependency, ledger, invalid, seen=frozenset()):
    """Numerical freshness does not depend on the calculation's review verdict."""
    token = identity(dependency)
    if (token in seen or not current_premises(dependency, ledger.assumptions, ledger=ledger)
            or _selected_source_changes(dependency, ledger)):
        return False
    inputs = {identity(row): row for row in record.get("input_records", ())}
    for version in record.get("calculation_versions", ()):
        calculation = ledger.calculations.get(version, {})
        inputs.update((identity(row), row) for row in calculation.get("input_records", ()))
    inputs.update((fact.key, fact.as_record()) for fact in ledger.facts.values())
    for key in dependency.get("input_fact_keys", ()):
        fact = inputs.get(key)
        if fact is None or evidence_key(fact) in invalid:
            return False
        if fact.get("kind") != "derived":
            continue
        producers = [row for row in ledger.calculations.values() if key in row.get("output_fact_keys", ())]
        if not producers:
            return False
        producer = producers[-1]
        if not _numerically_completed(producer, fact["field"], derived=True):
            return False
        upstream = producer.get("derived_support", {}).get(fact["field"], producer)
        if not _fresh_dependencies(producer, upstream, ledger, invalid, {*seen, token}):
            return False
    return True


def calculation_is_up_to_date(record, field, ledger, *, derived=False, invalid=None):
    """Check executed arithmetic and current dependencies independently of credit."""
    record = _including_intermediates(record)
    if not _numerically_completed(record, field, derived=derived):
        return False
    if not derived and field not in METRICS and field in record.get("input_values", record.get("parameters", {})):
        return False
    output = record.get("output", record.get("metrics", {})).get(field, {})
    scenario = ledger.scenarios.get(record.get("subject"), {})
    contract = scenario.get("output_contracts", {}).get(field, record.get("output_contracts", {}).get(field))
    if not derived and not output_meets_contract(output, contract):
        return False
    if invalid is None:
        invalid = {evidence_key(evidence) for assumption in ledger.assumptions.values()
                   for evidence in assumption.get("invalidated_evidence", ())}
    dependency = record.get("derived_support" if derived else "metric_support", {}).get(field, record)
    return _fresh_dependencies(record, dependency, ledger, invalid)


def _step_alternatives(records, field, subject, scenario, ledger, invalid, *, requested):
    """The common result projection used by the table and automatic binding."""
    alternatives = {}
    multiplier = REQUESTED_OUTPUT_MULTIPLIER if requested else 1.0
    for record in records:
        if record.get("subject") != subject:
            continue
        for derived in (False, True):
            if derived and requested:
                continue
            if not _numerically_completed(record, field, derived=derived):
                continue
            if not derived and field not in METRICS and field in record.get("input_values", record.get("parameters", {})):
                continue
            dependency = record.get("derived_support" if derived else "metric_support", {}).get(field, record)
            output = record.get("output", record.get("metrics", {})).get(field, {})
            contract = scenario.get("output_contracts", {}).get(field,
                record.get("output_contracts", {}).get(field))
            contract_met = derived or output_meets_contract(output, contract)
            fresh = calculation_is_up_to_date(record, field, ledger, derived=derived, invalid=invalid)
            support_qualified = fresh and admitted_calculation(record, ledger.assumptions, ledger=ledger,
                **({"field": field} if derived else {"metric": field}))
            support = current_support(record, ledger.assumptions, metric=field, ledger=ledger) if support_qualified and requested else 0.0
            method_credit = multiplier * REUSABLE_METHOD_CREDIT
            current_result_credit = multiplier * CURRENT_RESULT_CREDIT if fresh else 0.0
            support_credit = multiplier * max(0.0, support - SUPPORT_CREDIT_BASELINE) if fresh else 0.0
            points = method_credit + current_result_credit + support_credit
            versions = record.get("calculation_versions", ())
            version = record.get("version") or (versions[-1] if versions else None)
            changes = _changed_dependencies(dependency, ledger) if not fresh else []
            if not contract_met:
                changes.append({"kind": "output_contract", "field": field,
                    "reason": "Calculated result does not meet the current requested output contract."})
            if not fresh and not changes:
                changes = [{"kind": "provenance", "reason": "Selected source, shared scope, or upstream calculation is no longer current."}]
            review = record.get("model_review", {})
            review_status = ("incomplete" if not review.get("completed", True) else
                             "accepted" if review.get("admit") is True else "rejected")
            value_row = record.get("derived_parameters", {}).get(field, {}) if derived else output
            limit = "finite" if derived else value_row.get("limit")
            computed_value = (value_row.get("function") if limit == "function" else
                              "unbounded" if limit == "unbounded" else value_row.get("value"))
            main_differences = _main_input_differences(record, dependency, ledger)
            alternative = {
                "parameterization": record.get("parameterization", record.get("label", record.get("basis", ""))),
                "calculation_version": version,
                "status": "calculated_up_to_date" if fresh else "calculated_needs_updating",
                "accepted_current": support_qualified,
                "matches_main_selection": not main_differences,
                "main_input_differences": main_differences,
                "computed_value": computed_value, "unit": value_row.get("unit", UNITS.get(field)),
                "computed_limit": limit,
                "computed_output": dict(value_row),
                "review_status": review_status,
                "points": points, "method_credit": method_credit,
                "current_result_credit": current_result_credit,
                "support_credit": support_credit,
                "input_fields": dependency.get("input_fields", sorted(record.get("input_values", record.get("parameters", {})))),
                "condition_fields": dependency.get("condition_fields", sorted(record.get("conditions", {}))),
                "changed_dependencies": changes,
            }
            key = (alternative["parameterization"], version)
            if key not in alternatives or points > alternatives[key]["points"]:
                alternatives[key] = alternative
    return list(alternatives.values())


def _preferred_result(candidates):
    return max(candidates, key=lambda row: (row["status"] == "calculated_up_to_date", row["points"])) if candidates else None


def current_calculated_inputs(ledger, subject):
    """Use the same current result shown in the calculation table for each field."""
    scenario = ledger.scenarios[subject]
    invalid = {evidence_key(evidence) for assumption in ledger.assumptions.values()
               for evidence in assumption.get("invalidated_evidence", ())}
    records = [_including_intermediates(record) for record in
               (*ledger.calculations.values(), *ledger.estimates.values())]
    facts = [fact for fact in ledger.facts.values() if fact.subject == subject and fact.kind == "derived"]
    selected = []
    for field in dict.fromkeys(fact.field for fact in facts):
        candidates = _step_alternatives(records, field, subject, scenario, ledger, invalid,
            requested=field in scenario.get("output_contracts", {}))
        best = _preferred_result(candidates)
        if best is None or best["status"] != "calculated_up_to_date":
            continue
        producer = ledger.calculations.get(best["calculation_version"], {})
        match = next((fact for fact in facts if fact.field == field
                      and fact.key in producer.get("output_fact_keys", ())
                      and (fact.value, fact.unit) == (best["computed_value"], best["unit"])), None)
        if match is not None:
            selected.append(match)
    return tuple(selected)


def calculation_step_status(result, outputs, *, subjects=None):
    """Enumerate useful steps and project current and reusable-completion credit."""
    ledger = Ledger(result)
    outputs = tuple(dict.fromkeys(outputs))
    targets = {subject: scenario for subject, scenario in ledger.scenarios.items()
               if scenario.get("role", "target") == "target" and (subjects is None or subject in subjects)}
    from .tools import main_calculation_inputs
    invalid = {evidence_key(evidence) for assumption in ledger.assumptions.values()
               for evidence in assumption.get("invalidated_evidence", ())}
    steps = []
    records = [_including_intermediates(record) for record in
               (*ledger.calculations.values(), *ledger.estimates.values())]
    for subject, scenario in targets.items():
        fields = tuple(sorted(_step_fields(scenario, outputs)))
        numeric_options = {}
        for fact in main_calculation_inputs(ledger, subject):
            numeric_options.setdefault(fact.field, set()).add(fact.value)
        conditions = set(scenario.get("fixed_conditions", {})) | {
            a["field"] for a in ledger.assumptions.values() if a["subject"] == subject
            and a.get("value") is None and a.get("review", {}).get("verdict") != "contradicted"}
        for field in fields:
            requested = field in outputs
            candidates = _step_alternatives(records, field, subject, scenario, ledger, invalid, requested=requested)
            best = _preferred_result(candidates)
            routes = _requirements(field, set(numeric_options), conditions, numeric_options,
                scenario.get("calculation_variables", ()))
            status = best["status"] if best else "not_calculated"
            ready = any(route["ready"] for route in routes)
            steps.append({"subject": subject, "field": field,
                "kind": "requested_output" if requested else "intermediate", "status": status,
                "computed_value": best["computed_value"] if best else None,
                "unit": best["unit"] if best else UNITS.get(field),
                "computed_limit": best["computed_limit"] if best else None,
                "computed_output": best["computed_output"] if best else None,
                "parameterization": best["parameterization"] if best else None,
                "calculation_version": best["calculation_version"] if best else None,
                "points": best["points"] if best else 0.0,
                "method_credit": best["method_credit"] if best else 0.0,
                "current_result_credit": best["current_result_credit"] if best else 0.0,
                "support_credit": best["support_credit"] if best else 0.0,
                "accepted_current": best["accepted_current"] if best else False,
                "matches_main_selection": best["matches_main_selection"] if best else None,
                "main_input_differences": best["main_input_differences"] if best else [],
                "credit_calculation_version": best["calculation_version"] if best else None,
                "review_status": best["review_status"] if best else None,
                "available_as_input": field in numeric_options, "routes": routes,
                "ready_to_calculate": ready, "ready_to_recalculate": status == "calculated_needs_updating" and ready,
                "result_alternatives": candidates})
    return {"steps": steps,
        **{status: sum(row["status"] == status for row in steps)
           for status in ("not_calculated", "calculated_up_to_date", "calculated_needs_updating")},
        "points": round(sum(row["points"] for row in steps), 10),
        "credit_policy": {"reusable_completion": REUSABLE_METHOD_CREDIT,
            "current_result": CURRENT_RESULT_CREDIT,
            "completion_floor": COMPLETION_CREDIT,
            "requested_output_multiplier": REQUESTED_OUTPUT_MULTIPLIER,
            "support_bonus_threshold": SUPPORT_CREDIT_BASELINE,
            "requested_output_rates": {
                "reusable_completion": REQUESTED_OUTPUT_MULTIPLIER * REUSABLE_METHOD_CREDIT,
                "current_result": REQUESTED_OUTPUT_MULTIPLIER * CURRENT_RESULT_CREDIT,
                "completion_floor": REQUESTED_OUTPUT_MULTIPLIER * COMPLETION_CREDIT,
                "fully_supported": REQUESTED_OUTPUT_MULTIPLIER * (COMPLETION_CREDIT + 1 - SUPPORT_CREDIT_BASELINE)},
            "requested_output": (
                f"{REQUESTED_OUTPUT_MULTIPLIER:g} * ({COMPLETION_CREDIT:g} + max(0, current support - {SUPPORT_CREDIT_BASELINE:g})) for current completed arithmetic; "
                "current-result credit and support bonus are withdrawn when stale"),
            "meaning": "Calculation credit follows the three numerical statuses: not_calculated has no method or current-result credit; calculated_up_to_date has both; calculated_needs_updating retains method credit and withdraws current-result and support credit until recalculation. Base rates apply to intermediate steps and the requested-output multiplier applies to every endpoint credit component. Scientific support review only qualifies the support bonus and never vetoes method or current-result credit. Changed inputs, provenance, or output contracts affect freshness without erasing completed method work. accepted_current reports scientific support qualification; status reports numerical freshness. Alternative recipes are not a mandatory sequence."}}
