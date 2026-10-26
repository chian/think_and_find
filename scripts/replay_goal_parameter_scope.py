"""Re-score saved admitted records without providers, extraction, or a workflow run."""

import argparse
from collections import Counter, defaultdict
from dataclasses import fields, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def load_package():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "episodic_workflow", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)


def by_parameter(components):
    points = defaultdict(float)
    for row in components:
        points[row["field"]] += row["points"]
    return {field: round(value, 10) for field, value in sorted(points.items()) if value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    load_package()
    from episodic_workflow.case_reports.research.assumptions import scientific_credit
    from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
    from episodic_workflow.case_reports.research.progress import parameter_support
    from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult
    from episodic_workflow.case_reports.research.scenario_sharing import (
        independent_groups, needed_fields, scenario_version)
    from episodic_workflow.case_reports.research.scenarios import Scenario
    from episodic_workflow.task_executors.scenario_metrics.dependencies import fixed_input_fields
    from episodic_workflow.task_executors.scenario_metrics.catalog import METRICS
    from episodic_workflow.task_executors.scenario_metrics.evaluate import UNITS
    from episodic_workflow.task_executors.scenario_metrics.exposure import (
        BACKGROUND_INPUTS, LOCAL_INPUTS)

    with args.report.open() as stream:
        report = json.load(stream)
    metrics = tuple(report["requested_quantities"])
    saved_status = report["global_scientific_goal"]
    old_focus = tuple(sorted({row["parameter"] for row in saved_status["independent_target_values"]}
                             | set(metrics)))
    result = ResearchResult(
        facts=tuple(Fact(**{**row, "parents": tuple(row.get("parents", ()))})
                    for row in report["evidence_and_assumptions"]),
        assumptions=tuple(report["assumption_assessments"]),
        scenarios=tuple(row["definition"] for row in report["scenarios"].values()),
        estimates=tuple(estimate for row in report["scenarios"].values()
                        for estimate in row["alternatives"]),
        evidence=tuple(report["qualitative_evidence"]),
        calculations=tuple(report["calculations"]),
        reading_attempts=tuple(report["reading_attempts"]),
        source_event_types=tuple(report["source_event_types"]),
        source_target_relationships=tuple(report["source_target_relationships"]),
        scenario_parameter_relations=tuple(report["scenario_parameter_relations"]))
    del report
    before = scientific_credit(result, old_focus, outputs=metrics)
    before_components = [{"field": field, "element": element, "points": points}
                         for field, element, points in before.qualities]
    before_score = round(sum(row["points"] for row in before_components), 10)
    assert before_score == saved_status["weighted_score"], "Old field eligibility no longer reproduces the saved credit"
    after = scientific_goal_status(result, metrics)
    checks = ["old field eligibility reproduces saved credit"]

    # Duplicate actual records, including actual alternatives; do not invent evidence.
    duplicated = replace(result, **{field.name: getattr(result, field.name) * 2
                         for field in fields(result) if isinstance(getattr(result, field.name), tuple)})
    repeated = scientific_goal_status(duplicated, metrics)
    assert repeated["credit_components"] == after["credit_components"]
    assert repeated["weighted_score"] == after["weighted_score"]
    checks.append("duplicated saved records and alternative assumptions add no credit")

    constraints = tuple(row for row in result.assumptions
                        if row.get("role") in {"scenario_constraint", "request_constraint"})
    givens = replace(result, assumptions=constraints, estimates=(), calculations=())
    assert scientific_goal_status(givens, metrics)["weighted_score"] == 0
    checks.append("caller-fixed givens earn no evidence credit")

    ledger = Ledger(result)
    groups = independent_groups(ledger)
    pairs = [(subject, group["parameter"]) for group in groups for subject in group["subjects"]]
    required_pairs = {(scenario["subject"], field) for scenario in result.scenarios
                      if scenario.get("role", "target") == "target" for field in needed_fields(scenario)}
    assert len(pairs) == len(set(pairs)), "A scenario parameter appears in multiple credit groups"
    assert set(pairs) == required_pairs, "Credit groups omit or add a scenario parameter"
    checks.append("every needed scenario parameter occurs in exactly one credit group")

    scenario_fields = {}

    def assignable_inputs(names):
        inputs = set()
        for name in names:
            if name in UNITS:
                inputs.add(name)
            else:
                inputs.update(assignable_inputs(METRICS[name].inputs))
        return inputs

    for scenario in result.scenarios:
        needed = needed_fields(scenario)
        fixed = scenario.get("fixed_parameters", {})
        conditions = scenario.get("fixed_conditions", {})
        supplied = fixed_input_fields(fixed, conditions)
        assert not needed.intersection(supplied)
        assert not needed.intersection(metrics)
        direct = set(BACKGROUND_INPUTS)
        near_field_possible = ("negligible_near_field" not in conditions
                               and fixed.get("mask_penetration", {"value": 1})["value"] == 1)
        if near_field_possible:
            direct.update(LOCAL_INPUTS)
        assignable = assignable_inputs(direct)
        assert (assignable - supplied) <= needed, (scenario["subject"], sorted(assignable - supplied - needed))
        rebuilt = Scenario(scenario["subject"], scenario["description"],
                           tuple((name, row["value"]) for name, row in fixed.items()),
                           tuple(conditions.items()), tuple(scenario.get("assumption_needs", ())),
                           tuple(scenario.get("calculation_variables", ())), scenario["basis"]).as_record()
        assert set(rebuilt["assumption_needs"]) == needed
        scenario_fields[scenario["subject"]] = {
            "count": len(needed), "fields": sorted(needed),
            "caller_fixed_or_fixed_derived_fields": sorted(supplied)}
    checks.append("initial Scenario record contains all applicable direct exposure inputs")
    checks.append("fixed-derived inputs and requested outputs are not evidence needs")

    # These are configuration-only probes, not new cases or source evidence.
    example = result.scenarios[0]
    base = {**example, "assumption_needs": []}
    uniform = {**base, "fixed_conditions": {**base["fixed_conditions"],
                "negligible_near_field": "Configuration-only dependency check."}}
    uniform["fixed_conditions"].pop("continuous_unmasked_jet", None)
    masked = {**base, "fixed_parameters": {**base["fixed_parameters"],
               "mask_penetration": {"value": 0.5, "unit": "fraction"}}}
    jet_only = set(LOCAL_INPUTS) - {"quanta_concentration_exhaled_m3"}
    assert not needed_fields(uniform).intersection(jet_only)
    assert not needed_fields(masked).intersection(jet_only)
    custom = {**base, "fixed_parameters": {}, "fixed_conditions": {}}
    assert assignable_inputs(BACKGROUND_INPUTS + LOCAL_INPUTS) <= needed_fields(custom)
    assert {"room_volume_m3", "floor_area_m2", "room_height_m", "ventilation_flow_m3_s"} <= needed_fields(custom)
    checks.append("explicit uniform and fixed masked configurations exclude incompatible jet inputs")
    checks.append("custom unfixed configuration retains all kernel inputs and geometry alternatives")

    old_parameters = by_parameter(before_components)
    new_parameters = by_parameter(after["credit_components"])
    changes = [{"parameter": field, "before": old_parameters.get(field, 0),
                "after": new_parameters.get(field, 0),
                "difference": round(new_parameters.get(field, 0) - old_parameters.get(field, 0), 10)}
               for field in sorted(old_parameters.keys() | new_parameters.keys())]
    old_groups = [row for row in result.scenario_parameter_relations if row.get("kind") == "group"]
    current_saved_groups = sum(all(row["scenario_versions"].get(subject) == scenario_version(ledger.scenarios[subject])
                                  for subject in row["subjects"]) for row in old_groups)
    learned = parameter_support(result, {row["parameter"] for row in groups}, hypotheses_only=True,
                                subjects=tuple(ledger.scenarios))
    with args.report.open("rb") as stream:
        source_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    output = {
        "source_report": str(args.report.resolve()), "source_report_sha256": source_digest,
        "mode": "Saved-record credit replay only; no providers or new scientific judgments.",
        "before": {"label": "Historical field eligibility applied to the unchanged saved evidence",
                   "saved_credit": saved_status["weighted_score"], "replayed_credit": before_score,
                   "historical_parameter_groups": len(saved_status["independent_target_values"])},
        "after": {"label": "Current dependency eligibility applied to the same saved evidence",
                  "credit": after["weighted_score"], "parameter_credit": changes,
                  "nonzero_components": [row for row in after["credit_components"] if row["points"]],
                  "recorded_target_inputs": after["recorded_target_inputs"],
                  "computed_outputs": after["computed_outputs"], "requested_outputs": after["requested_outputs"]},
        "scope_replay": {"saved_groups": len(old_groups), "saved_groups_with_current_versions": current_saved_groups,
                         "group_methods": dict(Counter(row["method"] for row in groups)),
                         "meaning": "Invalid historical scope versions use unshared singleton groups. No sharing judge was rerun. "
                                    "Dependency alternatives are eligible evidence targets, not a requirement to acquire every alternative."},
        "reconstructed_records": {"facts": len(result.facts), "assessments": len(result.assumptions),
                                  "scenarios": len(result.scenarios), "attempted_estimates": len(result.estimates)},
        "learning_by_parameter": learned, "initial_parameter_lists": scenario_fields,
        "verified_invariants": checks}
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
