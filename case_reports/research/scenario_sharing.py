"""Define independent target values from scenario features before research starts."""

from __future__ import annotations

from dataclasses import replace

from .records import identity


def needed_fields(scenario):
    from ...task_executors.scenario_metrics.dependencies import (
        CALCULATED_PARAMETER_FIELDS, fixed_input_fields, input_fields,
    )
    from ...task_executors.scenario_metrics.evaluate import DEFAULT_METRICS, UNITS
    from .model_defaults import retained_defaults

    declared = set(scenario.get("assumption_needs", ()))
    fixed = {**retained_defaults(scenario), **scenario.get("fixed_parameters", {})}
    conditions = scenario.get("fixed_conditions", {})
    outputs = tuple(scenario.get("output_contracts", {})) or DEFAULT_METRICS
    dependencies = input_fields(outputs, parameters=UNITS,
        fixed_values={field: row["value"] for field, row in fixed.items()},
        fixed_conditions=conditions)
    return ((declared | (dependencies & UNITS.keys())) - fixed_input_fields(fixed, conditions) - conditions.keys()
            - set(outputs) - set(scenario.get("calculation_variables", ()))
            - CALCULATED_PARAMETER_FIELDS)


def scenario_version(scenario):
    return identity({name: scenario.get(name) for name in
        ("subject", "description", "fixed_parameters", "fixed_conditions", "assumption_needs",
          "calculation_variables", "output_contracts", "model_defaults")})


def _fixed_features(scenarios):
    fields = sorted(set().union(*(set(row.get("fixed_parameters", {})) |
                                  set(row.get("fixed_conditions", {})) for row in scenarios.values())))
    rows = []
    for field in fields:
        values = {subject: scenario.get("fixed_parameters", {}).get(field,
                  scenario.get("fixed_conditions", {}).get(field))
                  for subject, scenario in scenarios.items()}
        rows.append({"kind": "feature", "name": "fixed:" + field,
                     "source": "declared_fixed_input", "values": values,
                     "basis": "Caller-declared numeric input or model condition."})
    return rows


def _unknown(value):
    return value is None or (isinstance(value, str) and value.strip().lower() in
        {"unknown", "unspecified", "not specified", "not stated", "missing", "n/a"})


def initialize_parameter_scope(initial, assessor=None, *, disease=None):
    """Compare multiple targets; one target has one group per required value."""
    scenarios = {row["subject"]: row for row in initial.scenarios
                 if row.get("role", "target") == "target"}
    single_target = len(scenarios) == 1
    scope_assessor = None if single_target else assessor
    by_field = {}
    for subject, scenario in scenarios.items():
        for field in needed_fields(scenario):
            if field not in scenario.get("fixed_parameters", {}) and field not in scenario.get("fixed_conditions", {}):
                by_field.setdefault(field, []).append(subject)
    fixed = _fixed_features(scenarios)
    analysis = (scope_assessor({"disease_of_interest": disease, "scenarios": scenarios,
                          "parameters": sorted(by_field), "fixed_features": fixed})
                if scope_assessor and by_field else {"features": [],
                    "dependencies": [{"parameter": field, "factors": [],
                                      "reason": ("One target needs no cross-scenario equality assessment."
                                                 if single_target else "No scenario scope assessor configured.")}
                                     for field in by_field]})
    if not isinstance(analysis, dict) or not isinstance(analysis.get("features"), list) or \
            not isinstance(analysis.get("dependencies"), list):
        raise ValueError("Scenario scope assessment needs features and parameter dependencies")
    features = {row["name"]: row for row in fixed}
    for row in analysis["features"]:
        if (not isinstance(row, dict) or not isinstance(row.get("name"), str)
                or not row["name"].strip() or row["name"] in features
                or not isinstance(row.get("basis"), str) or not row["basis"].strip()
                or not isinstance(row.get("values"), dict)
                or set(row["values"]) != set(scenarios)
                or any(not isinstance(value, str) or not value.strip()
                       for value in row["values"].values())):
            raise ValueError("Scenario description feature must name every selected target exactly once")
        features[row["name"]] = {"kind": "feature", "name": row["name"],
            "source": "scenario_description", "values": row["values"], "basis": row["basis"]}
    dependencies = {}
    for row in analysis["dependencies"]:
        if (not isinstance(row, dict) or row.get("parameter") not in by_field
                or row["parameter"] in dependencies or not isinstance(row.get("factors"), list)
                or any(name not in features for name in row["factors"])
                or len(set(row["factors"])) != len(row["factors"])
                or not isinstance(row.get("reason"), str) or not row["reason"].strip()):
            raise ValueError("Scenario parameter dependency must use declared factors and one parameter")
        dependencies[row["parameter"]] = {"kind": "dependency", "parameter": row["parameter"],
            "factors": sorted(row["factors"]), "reason": row["reason"],
            "method": "single_target_scope" if single_target else
                      "reasoning_model" if scope_assessor else "unassessed"}
    if set(dependencies) != set(by_field):
        raise ValueError("Scenario scope assessment must cover every needed parameter")
    groups = []
    for field, subjects in sorted(by_field.items()):
        relevant = [name for name in dependencies[field]["factors"]
                    if len({identity(features[name]["values"][subject]) for subject in subjects}) > 1]
        buckets = {}
        for subject in sorted(subjects):
            signature = tuple((name, identity((subject, features[name]["values"][subject]))
                               if _unknown(features[name]["values"][subject]) else
                               identity(features[name]["values"][subject])) for name in relevant)
            buckets.setdefault(signature, []).append(subject)
        for signature, members in buckets.items():
            versions = {subject: scenario_version(scenarios[subject]) for subject in members}
            groups.append({"kind": "group", "group_id": identity((field, members, versions, signature)),
                "parameter": field, "subjects": members, "scenario_versions": versions,
                "relevant_features": relevant, "feature_signature": list(signature),
                "method": "parameter_dependency_signature" if scope_assessor else "singletons_without_assessor"})
    # An absent assessor cannot justify cross-case equality, even with no named factors.
    if scope_assessor is None:
        groups = []
        for field, subjects in sorted(by_field.items()):
            for subject in sorted(subjects):
                version = scenario_version(scenarios[subject])
                groups.append({"kind": "group", "group_id": identity((field, subject, version)),
                    "parameter": field, "subjects": [subject],
                    "scenario_versions": {subject: version}, "relevant_features": [],
                    "feature_signature": [], "method": "single_target_scope" if single_target else "uncompared_singleton"})
    return replace(initial, scenario_parameter_relations=tuple((*features.values(), *dependencies.values(), *groups)))


def independent_groups(ledger):
    """Current groups only; changed scenarios lose old scope decisions."""
    groups = []
    needs = {subject: needed_fields(scenario) for subject, scenario in ledger.scenarios.items()
             if scenario.get("role", "target") == "target"}
    for row in ledger.scenario_parameter_relations.values():
        if row.get("kind") != "group":
            continue
        if all(subject in needs and row["parameter"] in needs[subject]
               and row["scenario_versions"].get(subject) ==
               scenario_version(ledger.scenarios[subject]) for subject in row["subjects"]):
            groups.append(row)
    covered = {(subject, row["parameter"]) for row in groups for subject in row["subjects"]}
    for subject, scenario in ledger.scenarios.items():
        if scenario.get("role", "target") != "target":
            continue
        for field in needed_fields(scenario):
            if field in scenario.get("fixed_parameters", {}) or field in scenario.get("fixed_conditions", {}):
                continue
            if (subject, field) not in covered:
                version = scenario_version(scenario)
                groups.append({"kind": "group", "group_id": identity((field, subject, version)),
                    "parameter": field, "subjects": [subject], "scenario_versions": {subject: version},
                    "relevant_features": [], "feature_signature": [], "method": "uncompared_singleton"})
    return tuple(groups)


def group_for(ledger, subject, field):
    return next((row for row in independent_groups(ledger)
                 if row["parameter"] == field and subject in row["subjects"]), None)


def shared_assumptions(ledger, subject):
    """Resolve a value in the target's group to its one original record."""
    target = ledger.scenarios.get(subject)
    if not target:
        return ()
    resolved = []
    for group in independent_groups(ledger):
        if subject not in group["subjects"]:
            continue
        for assumption in ledger.assumptions.values():
            if (assumption["subject"] == subject or assumption["subject"] not in group["subjects"]
                    or assumption["field"] != group["parameter"] or assumption["value"] is None
                    or assumption.get("source_candidate_unavailable")
                    or assumption.get("review", {}).get("verdict") == "fixed_condition"):
                continue
            origin = ledger.scenarios.get(assumption["subject"])
            if not origin:
                continue
            relation = {"assumption_key": assumption["key"],
                "assumption_version": assumption["version"], "origin_subject": assumption["subject"],
                "target_subject": subject, "parameter": group["parameter"],
                "group_id": group["group_id"],
                "origin_scenario_version": scenario_version(origin),
                "target_scenario_version": scenario_version(target)}
            resolved.append((assumption, relation))
    return tuple(resolved)


def applicable_assumptions(ledger, subject):
    from .model_defaults import retained_defaults
    retained = retained_defaults(ledger.scenarios.get(subject, {}))
    direct = []
    for assumption in ledger.assumptions.values():
        if assumption["subject"] != subject:
            continue
        if assumption.get("role") == "model_default":
            current = retained.get(assumption["field"])
            if (current is None or assumption.get("value") != current["value"]
                    or assumption.get("unit") != current["unit"]
                    or assumption.get("default_decision") != {
                        name: current[name] for name in ("decision", "reason", "scenario_evidence")}):
                continue
        direct.append((assumption, None))
    return (*direct, *shared_assumptions(ledger, subject))
