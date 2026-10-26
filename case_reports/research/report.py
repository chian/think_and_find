"""Reports and episode progress projected from the current calculation table."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ...method_loop import ClosedRecord
from ...task_executors.scenario_metrics.catalog import METRICS, PAPER, metric_catalog_record
from ...task_executors.scenario_metrics.dependencies import input_requirements
from .calculation_steps import calculation_step_status
from .communication import reading_coverage
from .records import Ledger


def scenario_results(result, metrics, *, calculation_status=None):
    """Use each output's already selected value and numerical freshness status.

    This projection does not choose another parameterization, require a review,
    or require separately executed outputs to share one estimate record.
    """
    metrics = tuple(metrics)
    status = calculation_status if calculation_status is not None else calculation_step_status(result, metrics)
    steps = {(row["subject"], row["field"]): row for row in status["steps"]}
    scenarios = {}
    for definition in result.scenarios:
        if definition.get("role", "target") != "target":
            continue
        subject = definition["subject"]
        coverage = {}
        for name in metrics:
            step = steps[(subject, name)]
            output = step.get("computed_output")
            figures = []
            if output is not None:
                figures.append({**output, "parameterization": step["parameterization"]})
            coverage[name] = {
                "computed": step["status"] == "calculated_up_to_date",
                "recorded_computation": step["status"] != "not_calculated",
                "calculation_status": step["status"],
                "status": step["status"],
                "output_contract": definition.get("output_contracts", {}).get(name),
                "computed_value": step["computed_value"],
                "unit": step["unit"],
                "computed_limit": step["computed_limit"],
                "figures": figures,
                "ready_to_calculate": step["ready_to_calculate"],
                "ready_to_recalculate": step["ready_to_recalculate"],
                "main_input_differences": step["main_input_differences"],
                "points": step["points"],
            }
        scenarios[subject] = {"definition": definition, "coverage": coverage,
            "requested_outputs_complete": all(row["computed"] for row in coverage.values())}
    complete = bool(scenarios) and all(row["requested_outputs_complete"] for row in scenarios.values())
    return scenarios, complete


def scientific_progress(result, metrics, *, decision_view=False):
    """Expose numerical coverage and actionable dependencies to the parent.

    The existing typed calculation view supplies decision fields; raw result
    metadata and child-authored prose do not cross that boundary.
    """
    from .tools import main_calculation_inputs

    ledger = Ledger(result)
    status = calculation_step_status(result, metrics)
    scenarios, _ = scenario_results(result, metrics, calculation_status=status)
    if decision_view:
        from .engine import calculation_decision_view

        view = calculation_decision_view(status, detailed=True, target_subjects=scenarios)
        steps = {(row[0], row[1]): dict(zip(view["columns"], row)) for row in view["rows"]}
    else:
        steps = {(row["subject"], row["field"]): row for row in status["steps"]}
    targets = {}
    for subject, scenario in scenarios.items():
        definition = scenario["definition"]
        inputs = main_calculation_inputs(ledger, subject)
        numeric = {fact.field for fact in inputs}
        conditions = set(definition.get("fixed_conditions", {})) | {
            assumption["field"] for assumption in ledger.assumptions.values()
            if assumption["subject"] == subject and assumption.get("value") is None}
        coverage = scenario["coverage"]
        if decision_view:
            coverage = {name: {
                **{key: row[key] for key in ("computed", "recorded_computation", "calculation_status",
                    "status", "output_contract", "ready_to_calculate", "ready_to_recalculate", "points")},
                **{key: steps[(subject, name)][key] for key in
                   ("computed_value", "unit", "computed_limit", "changed_inputs", "note_ids")},
            } for name, row in coverage.items()}
        targets[subject] = {
            "input_requirements": input_requirements(metrics, numeric, conditions,
                definition.get("calculation_variables", ())),
            "coverage": coverage,
            "requested_outputs_complete": scenario["requested_outputs_complete"],
            "recorded_numeric_fields": sorted(numeric),
            "recalculation_needs": [steps[(subject, name)] for name in metrics
                if coverage[name]["calculation_status"] == "calculated_needs_updating"],
        }
    return {
        "targets": targets,
        "requested_quantities_per_target": list(metrics),
        "quantity_coverage": {name: {
            "requested_cases": len(scenarios),
            "computed_cases": sum(s["coverage"][name]["computed"] for s in scenarios.values()),
            "uncomputed_subjects": [subject for subject, s in scenarios.items()
                if not s["coverage"][name]["computed"]],
        } for name in metrics},
        "recorded_calculated_quantities": sum(row["recorded_computation"] for s in scenarios.values()
            for row in s["coverage"].values()),
        "calculated_quantities": sum(row["computed"] for s in scenarios.values()
            for row in s["coverage"].values()),
        **({"calculation_notes": view["notes"]} if decision_view else {}),
        "interpretation": "Coverage, values and status come from the shared calculation table. "
            "Each requested output counts separately, including outputs executed in separate actions. "
            "Use the ready rows and changed-input notes when choosing work. No additional support "
            "review, calibration lineage or combined-parameterization approval is required for completion.",
    }


@dataclass(frozen=True)
class ResearchReport(ClosedRecord):
    passed: bool
    data: dict

    @classmethod
    def from_result(cls, result, record, run):
        failures = Counter((a["operation"], a["diagnostic"]["category"])
                           for a in run.model_attempts if a["status"] == "response_failed")
        global_goal = run.global_goal.preview()
        scenarios, complete = scenario_results(run.global_goal.ledger.result(), run.metrics,
            calculation_status=global_goal["calculation_steps"])
        computed = sum(row["computed"] for s in scenarios.values() for row in s["coverage"].values())
        data = {
            "objective": record.goal.as_record()["objective"], "reference": PAPER,
            "requested_quantities": list(run.metrics),
            "status": "calculations_complete" if complete else "calculations_incomplete",
            "computational_progress": {
                "calculated_quantities": computed,
                "recorded_calculated_quantities": sum(row["recorded_computation"] for s in scenarios.values()
                    for row in s["coverage"].values()),
                "requested_quantities": len(scenarios) * len(run.metrics),
                "requested_outputs_complete": complete,
            },
            "scenarios": scenarios,
            "evidence_and_assumptions": [f.as_record() for f in result.facts],
            "assumption_assessments": list(result.assumptions),
            "qualitative_evidence": list(result.evidence), "attempt_diagnostics": list(result.diagnostics),
            "source_event_types": [row for row in result.source_event_types if row.get("members")],
            "source_target_relationships": list(run.global_goal.ledger.source_target_relationships.values()),
            "scenario_parameter_relations": list(run.global_goal.ledger.scenario_parameter_relations.values()),
            "global_scientific_goal": global_goal,
            "root_credit_projection": run.root_credit_projection(global_goal),
            "calculations": list(result.calculations),
            "observation_comparisons": [{**c["output"], "subject": c["subject"]}
                for c in result.calculations if c["tool"] == "compare_predictions_with_observations"],
            "research_request": run.research_request,
            "model_attempts": run.model_attempts,
            "model_response_failures": [{"operation": operation, "category": category, "count": count}
                for (operation, category), count in sorted(failures.items())],
            "implementation_sha256_at_start": dict(run.implementation),
            "source_documents": dict(run.documents),
            "provider_events": list(run.provider_events),
            "reading_attempts": list(result.reading_attempts),
            "source_reading_coverage": reading_coverage(result.reading_attempts),
            "episode_tree": {parent.name: [child.name for child in children]
                for parent, children in run.tree.children.items()},
            "findings": list(result.findings),
            "control": {
                "ended_by": record.ended_by, "reason": result.stop_reason,
                "credit_history": list(record.controller_state.history),
                "rarefaction": record.controller_state.rarefaction,
                "model_calls": run.model_calls, "total_units": run.total_units,
                "limits": vars(run.limits),
                "method": "Each requested output is complete when its selected calculation-table row is calculated_up_to_date. "
                    "Numeric completeness is separate from the episode's continuation or termination. "
                    "Found-parameter and calculation-step credit retain the policy recorded in global_scientific_goal. "
                    "Reporting does not execute work, change credit or impose acceptance checks.",
                "input_boundary": ("Runtime disease input and caller-selected fixed scenario conditions."
                    if run.fixed_scenarios else "Original request and recorded scenario definitions."),
            },
            "paper_coverage": {
                "implemented_metrics": [metric_catalog_record(name) for name in METRICS],
                "scope": "Time, distance and R_in use the scenario's explicit model conditions and selected inputs.",
                "not_implemented": [
                    "Size-resolved aerosol distributions and their weighted effective radius (SI Eq. S7).",
                    "Direct integration of activity-dependent aerosol size spectra (scalar volume-fraction rescaling is available).",
                    "Distance-resolved masked leakage or thermal-plume exposure when near-field enhancement is not negligible.",
                ],
            },
            "interpretation": "Calculated estimates under the recorded scenario conditions and selected inputs. "
                "Time fixes separation; distance fixes duration; R_in fixes both. "
                "Each output retains its fixed inputs and risk-model identity.",
        }
        return cls(complete, data)

    def as_record(self):
        return {"passed": self.passed, **self.data}

    def markdown(self):
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")

        def value_text(value):
            if isinstance(value, dict):
                return cell(value.get("expression", value))
            if type(value) in (int, float):
                return f"{value:.8g}"
            return "—" if value is None else cell(value)

        def quantity(row):
            if row.get("limit") == "function":
                return value_text(row.get("function"))
            if row.get("limit") == "unbounded":
                if row.get("operation") == "solve_distance":
                    return "no finite distance satisfies the equation"
                return "unbounded"
            return value_text(row.get("value"))

        def fixed_inputs(row):
            labels = {"neighbor_distance_m": "distance", "exposure_time_s": "time"}
            return "; ".join(f"{labels.get(field, field)} = {value_text(value['value'])} {value['unit']}"
                for field, value in row.get("fixed_inputs", {}).items())

        status_labels = {
            "not_calculated": "not calculated",
            "calculated_up_to_date": "calculated & up to date",
            "calculated_needs_updating": "calculated needs updating",
        }
        progress = self.data["computational_progress"]
        lines = ["# Indoor Transmission Calculation Report", "", self.data["research_request"] or self.data["objective"], "",
            f"**{progress['calculated_quantities']}/{progress['requested_quantities']} requested outputs calculated and up to date.**", "",
            self.data["interpretation"], ""]
        snapshot = self.data.get("saved_snapshot")
        if snapshot:
            lines.extend([f"Saved-state report: {cell(snapshot.get('run_state', 'saved snapshot'))}.", ""])
        goal = self.data["global_scientific_goal"]
        found = goal.get("recorded_target_inputs")
        if found:
            lines.extend([f"{found['recorded']}/{found['required']} values found.", ""])
        scores = goal.get("scores", {})
        lines.extend([f"Found-parameter credit: {scores.get('found_parameters', 0):g}. "
            f"Calculation-step credit: {scores.get('calculation_steps', 0):g}.", ""])
        for subject, scenario in self.data["scenarios"].items():
            definition = scenario["definition"]
            lines.extend([f"## {cell(subject)}", "", definition.get("description", ""), "",
                "| Requested output | Value | Units | Fixed inputs | Status |",
                "| --- | ---: | --- | --- | --- |"])
            for name, row in scenario["coverage"].items():
                figure = row["figures"][0] if row["figures"] else None
                value = quantity(figure) if figure else value_text(row["computed_value"])
                lines.append(f"| {cell(name)} | {value} | {cell(row.get('unit') or '—')} | "
                    f"{cell(fixed_inputs(figure)) if figure else '—'} | {status_labels[row['calculation_status']]} |")
            lines.append("")
            notes = list(dict.fromkeys(figure[key] for row in scenario["coverage"].values()
                for figure in row["figures"] for key in ("interpretation",) if figure.get(key)))
            if notes:
                lines.extend([*notes, ""])
            fixed = definition.get("fixed_parameters", {})
            conditions = definition.get("fixed_conditions", {})
            if fixed or conditions:
                lines.extend(["Scenario inputs:", ""])
                lines.extend(f"- {cell(field)} = {value_text(row['value'])} {cell(row['unit'])}"
                    for field, row in fixed.items())
                lines.extend(f"- {cell(field)}: {cell(statement)}" for field, statement in conditions.items())
                lines.append("")
        lines.extend(["## Selected Found Values", "",
            "Each row is one independent value; shared target scenarios use the same selected value.", "",
            "| Parameter | Value | Units | Target scenarios | Value record subject | Source scenario / origin | Source-to-target relevance | Status |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- |"])
        for row in goal.get("independent_target_values", ()):
            relationships = (row.get("source_applicability") or {}).get("relationships", ())
            sources = list(dict.fromkeys(
                f"{', '.join(relation.get('source_role', ()))}: {relation['source_subject']}"
                for relation in relationships))
            relevance = list(dict.fromkeys(
                f"{relation['target_subject']}: {value_text(relation['score'])}"
                for relation in relationships))
            lines.append(f"| {cell(row['parameter'])} | {value_text(row.get('value'))} | {cell(row.get('unit') or '—')} | "
                f"{cell(', '.join(row['subjects']))} | {cell(row.get('origin_subject') or '—')} | "
                f"{cell('; '.join(sources)) or 'not recorded'} | {cell('; '.join(relevance)) or 'not recorded'} | "
                f"{'found' if row['found'] else 'missing'} |")
        lines.extend(["", "## Calculation Status", "",
            "| Target scenario | Calculation | Kind | Value | Units | Status | Credit | Changed inputs |",
            "| --- | --- | --- | ---: | --- | --- | ---: | --- |"])
        for step in goal["calculation_steps"]["steps"]:
            changes = []
            if step["status"] == "calculated_needs_updating":
                changes = list(dict.fromkeys(change.get("field") or change["reason"]
                    for alternative in step.get("result_alternatives", ())
                    for change in alternative.get("changed_dependencies", ())))
            lines.append(f"| {cell(step['subject'])} | {cell(step['field'])} | {cell(step['kind'])} | "
                f"{value_text(step.get('computed_value'))} | {cell(step.get('unit') or '—')} | "
                f"{status_labels[step['status']]} | {step['points']:g} | {cell('; '.join(changes))} |")
        lines.extend(["", "## Run State", "",
            f"Episode end condition: {cell(self.data['control']['ended_by'])}. "
            f"Recorded reason: {cell(self.data['control']['reason'])}.", "",
            self.data["control"]["method"], ""])
        return "\n".join(lines)
