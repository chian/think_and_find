"""Replay saved deterministic calculations with their recorded model reviews.

No providers, new evidence, or new scientific judgments are constructed. The
saved report is read-only; corrected outputs are printed, not admitted to a run.
"""

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "episodic_workflow", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult
    from episodic_workflow.case_reports.research.tools import (
        TOOL_METRICS, bind_reference_inputs, execute_tool, input_options)
    from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS
    from episodic_workflow.task_executors.scenario_metrics.exposure import EXPOSURE_OUTPUTS, FIXED_FIELDS

    report = json.loads(args.report.read_text())
    rows = []

    def fact(record):
        return Fact(**{**record, "parents": tuple(record.get("parents", ()))})

    for saved in report["calculations"]:
        if saved["tool"] not in {"calculate_geometry_and_removal", "calculate_transmission_metrics"}:
            continue
        subject = saved["subject"]
        definition = report["scenarios"][subject]["definition"]
        alternatives = report["scenarios"][subject]["alternatives"]
        alternative = next(row for row in alternatives
                           if saved["version"] in row.get("calculation_versions", ()))
        givens = [row for row in report["assumption_assessments"]
                  if row["subject"] == subject and row["role"] == "scenario_constraint"]
        fixed_facts = [fact(row) for row in report["evidence_and_assumptions"]
                       if row["subject"] == subject and row["kind"] == "scenario_constraint"]
        selected = tuple(fact(row) for row in saved["input_records"])
        ledger = Ledger(ResearchResult(facts=(*fixed_facts, *selected),
            assumptions=(*givens, *alternative["assumption_support"]), scenarios=(definition,)))
        before = copy.deepcopy(ledger.result().as_record())
        options = input_options(ledger)
        references = {value.key: key for key, value in options.items()}
        action = {"action": saved["tool"], "subject": subject,
            "facts": [references[row.key] for row in selected], "conditions": saved["conditions"],
            "basis": saved["basis"], "parameterization": saved["parameterization"],
            "metrics": list(saved["output"])}
        bound = bind_reference_inputs(action, ledger, DEFAULT_METRICS)
        assert bind_reference_inputs(bound, ledger, DEFAULT_METRICS) == bound
        assert all(bound["conditions"][name] == value
                   for name, value in definition["fixed_conditions"].items())
        assert all(bound["conditions"][name] == value for name, value in saved["conditions"].items())
        result = execute_tool(bound, ledger, DEFAULT_METRICS, review=saved["model_review"])
        assert ledger.result().as_record() == before
        calculation = result.calculations[0]
        assert calculation["model_review"] == saved["model_review"]
        assert all(calculation["input_values"][name] == value
                   for name, value in saved["input_values"].items())
        for name, row in calculation["output"].items():
            assert row["status"] == "calculated", (name, row)
            if name == "concentration_relaxation_time_s":
                assert math.isclose(row["value"], 1 / saved["input_values"]["removal_rate_s"])
            elif name == "jet_dilution_ratio":
                assert calculation["input_values"]["mask_penetration"] == 1
                assert "Condition continuous_unmasked_jet has no completed reasonableness assessment" in calculation["support_issues"]
                assert "uniform_breathing_and_masks" not in calculation["metric_support"][name]["condition_fields"]
                expected = math.sqrt(saved["input_values"]["mouth_area_m2"]) / (
                    saved["input_values"]["entrainment_coefficient"] * saved["input_values"]["neighbor_distance_m"])
                assert math.isclose(row["value"], expected)
            rows.append({"metric": name, "before_status": saved["output"][name]["status"],
                "before_missing": saved["output"][name].get("missing", []),
                "after_status": row["status"], "value": row["value"], "unit": row["unit"],
                "support_issues": calculation["metric_support"][name]["support_issues"]})

        condition = next(iter(definition["fixed_conditions"]))
        conflicting = {**action, "conditions": {**action["conditions"], condition: "Conflicting scenario condition"}}
        try:
            bind_reference_inputs(conflicting, ledger, DEFAULT_METRICS)
        except ValueError as exc:
            assert "conflicts with the fixed scenario definition" in str(exc)
        else:
            raise AssertionError("A caller-fixed condition was silently overwritten")

    assert rows, "No replayable calculations found"
    bindings = 0
    for scenario in report["scenarios"].values():
        definition = scenario["definition"]
        if definition.get("role") != "target":
            continue
        subject = definition["subject"]
        givens = [row for row in report["assumption_assessments"]
                  if row["subject"] == subject and row["role"] == "scenario_constraint"]
        fixed_facts = [fact(row) for row in report["evidence_and_assumptions"]
                       if row["subject"] == subject and row["kind"] == "scenario_constraint"]
        ledger = Ledger(ResearchResult(facts=tuple(fixed_facts), assumptions=tuple(givens),
                                      scenarios=(definition,)))
        options = input_options(ledger)
        for metric, (operation, _) in EXPOSURE_OUTPUTS.items():
            tool = next(name for name, metrics in TOOL_METRICS.items() if metric in metrics)
            action = {"action": tool, "subject": subject, "facts": [next(iter(options))],
                      "metrics": [metric], "basis": "Saved scenario binding inspection"}
            bound = bind_reference_inputs(action, ledger, DEFAULT_METRICS)
            supplied = {options[ref].field: options[ref] for ref in bound["facts"]}
            for field in FIXED_FIELDS[operation]:
                if field in definition["fixed_parameters"]:
                    assert supplied[field].value == definition["fixed_parameters"][field]["value"]
                    assert supplied[field].kind == "scenario_constraint"
            assert bound["conditions"] == definition["fixed_conditions"]
            assert all(value.kind == "scenario_constraint" for value in supplied.values())
            bindings += 1

    print(json.dumps({"report": str(args.report), "replayed": rows,
        "requested_output_bindings_checked": bindings,
        "checks": ["Caller-fixed conditions and required numeric guards are bound before review/execution",
                   "Recorded input values, model reviews, and source records are unchanged",
                   "Unassessed nonfixed jet condition lowers support but does not block calculation",
                   "Binding is idempotent and preserves rejection of conflicting caller conditions",
                   "Each saved target binds its supplied time/distance reference coordinates for all three requested operations without inventing nonfixed assumptions"],
        "scope": "Offline deterministic replay only; these diagnostic quantities are not final time, distance, or R_in estimates."}, indent=2))


if __name__ == "__main__":
    main()
