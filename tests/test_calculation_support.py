"""Synthetic checks: scientific support changes credit, not tool execution."""

import json
import unittest

from episodic_workflow.case_reports.research.assumptions import unassessed_assumption
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, current_support, numeric_references
from episodic_workflow.case_reports.research.report import scenario_results
from episodic_workflow.case_reports.research.tools import execute_tool, main_calculation_inputs, support_for
from episodic_workflow.task_executors._http import ProviderError
from episodic_workflow.task_executors.scenario_metrics.evaluate import UNITS
from episodic_workflow.task_executors.scenario_metrics.dependencies import CALCULATED_PARAMETER_FIELDS
from episodic_workflow.tests.test_research import AdaptiveAgent, METRICS, run_synthetic, supported_review


def guessed_ledger(values):
    ledger = Ledger(ResearchResult(scenarios=({"subject": "target", "description": "Synthetic hypothesis test",
        "assumption_needs": [field for field in values if field not in CALCULATED_PARAMETER_FIELDS]},)))
    for field, value in values.items():
        row = unassessed_assumption({"subject": "target", "field": field, "value": value,
            "unit": UNITS[field], "statement": "Explicit synthetic guess",
            "basis": "No supporting evidence yet"}, ledger)
        if field in CALCULATED_PARAMETER_FIELDS:
            # These tests supply a boundary condition, not a discovered target
            # intermediate that the calculator is allowed to choose ad hoc.
            ledger.scenarios["target"].setdefault("fixed_parameters", {})[field] = {
                "value": value, "unit": UNITS[field]}
            row.update(role="scenario_constraint", reasonableness=1.,
                       review={"verdict": "fixed_condition", "completed": True})
        ledger.merge(ResearchResult(assumptions=(row,)))
    return ledger


def calculate(ledger, *, review=None, **changes):
    action = {"action": "calculate_geometry_and_removal", "subject": "target",
              "metrics": [], "basis": "Synthetic conditional calculation",
              **changes}
    if ledger.scenarios[action["subject"]].get("role", "target") != "target":
        action["source_inputs"] = [key for key, fact in numeric_references(ledger).items()
                                  if fact.subject == action["subject"]]
    return execute_tool(action, ledger, tuple(action["metrics"]), review=review or {
        "admit": True, "reason": "Synthetic conditional test", "limitations": []})


class CalculationSupportTests(unittest.TestCase):
    def test_zero_evidence_guess_is_selectable_without_becoming_source_evidence(self):
        ledger = guessed_ledger({"floor_area_m2": 40, "room_height_m": 2.5})
        self.assertFalse(ledger.facts)
        self.assertFalse(ledger.evidence_options())
        self.assertEqual({fact.field for fact in main_calculation_inputs(ledger, "target")},
                         {"floor_area_m2", "room_height_m"})
        result = calculate(ledger)
        self.assertEqual(result.facts[0].field, "room_volume_m3")
        self.assertEqual(result.facts[0].value, 100)
        calculation = result.calculations[0]
        self.assertEqual(calculation["input_values"], {"floor_area_m2": 40, "room_height_m": 2.5})
        self.assertTrue(all(row["kind"] == "model_assumption" for row in calculation["input_records"]))
        self.assertEqual(current_support(calculation, ledger.assumptions), 0)
        ledger.merge(result)
        self.assertFalse(ledger.evidence_options())
        self.assertFalse(any(score for _, _, score in ledger.result().credit(("room_volume_m3",), scientific=True).qualities))

    def test_negative_scientific_review_does_not_prevent_calibration(self):
        ledger = guessed_ledger({"quanta_emission_rate_s": .02, "breathing_flow_m3_s": .001})
        ledger.scenarios["target"]["role"] = "calibration"
        review = {"admit": False, "reason": "No activity-specific evidence yet", "limitations": []}
        result = calculate(ledger, action="calibrate_infectiousness", operator="cq_from_emission", review=review)
        self.assertEqual(result.facts[0].value, 20)
        self.assertEqual(result.calculations[0]["model_review"], review)
        self.assertEqual(current_support(result.calculations[0], ledger.assumptions), 0)

    def test_derived_guesses_remain_usable_and_keep_transitive_support(self):
        ledger = guessed_ledger({"floor_area_m2": 40, "room_height_m": 2.5, "removal_rate_s": .002})
        first = calculate(ledger)
        ledger.merge(first)
        result = calculate(ledger, metrics=["concentration_relaxation_time_s"],
                           conditions={"well_mixed": "Unreviewed synthetic condition"})
        self.assertEqual(result.estimates[0]["metrics"]["concentration_relaxation_time_s"]["value"], 500)
        self.assertEqual(set(result.calculations[0]["assumption_versions"]), set(ledger.assumptions))
        self.assertEqual(result.estimates[0]["reasonableness"], 0)
        self.assertIn("Condition well_mixed", result.calculations[0]["support_issues"][0])
        ledger.merge(result)
        scenarios, complete = scenario_results(ledger.result(), ("concentration_relaxation_time_s",))
        self.assertFalse(complete)
        self.assertEqual(scenarios["target"]["coverage"]["concentration_relaxation_time_s"]["status"], "unsupported_calculation")

    def test_missing_assessment_is_not_hidden_by_other_supported_inputs(self):
        ledger = guessed_ledger({"room_volume_m3": 100, "removal_capacity_m3_s": .2})
        values = list(ledger.assumptions.values())
        ledger.assumptions[values[0]["key"]] = {**values[0], "reasonableness": 1}
        del ledger.assumptions[values[1]["key"]]
        ledger.merge(ResearchResult(facts=(Fact("target", "removal_capacity_m3_s", .2, "m3/s",
            "user_assumption", "Synthetic unassessed value"),)))
        result = calculate(ledger, metrics=["removal_rate_s"])
        self.assertEqual(result.estimates[0]["metrics"]["removal_rate_s"]["value"], .002)
        self.assertEqual(result.estimates[0]["reasonableness"], 0)
        self.assertTrue(result.estimates[0]["support_issues"])

    def test_historical_derived_value_is_not_reused_by_main_table_binding(self):
        ledger = guessed_ledger({"floor_area_m2": 40, "room_height_m": 2.5, "removal_rate_s": .002})
        original = calculate(ledger)
        ledger.merge(original)
        key = next(iter(ledger.assumptions))
        ledger.assumptions[key] = {**ledger.assumptions[key], "version": "new-judgment",
                                   "statement": "Revised synthetic scope"}
        result = calculate(ledger, metrics=["removal_rate_s"])
        self.assertEqual(result.estimates[0]["metrics"]["removal_rate_s"]["value"], .002)
        self.assertFalse({fact.key for fact in original.facts}
                         & set(result.calculations[0]["input_fact_keys"]))
        from episodic_workflow.case_reports.research.calculation_steps import calculation_is_up_to_date
        self.assertFalse(calculation_is_up_to_date(original.calculations[0], "room_volume_m3", ledger))

    def test_no_gate_in_actual_method_loop_and_results_return_to_investigation(self):
        base = AdaptiveAgent()
        investigated_again = False
        def agent(request):
            nonlocal investigated_again
            if request["operation"] == "propose_guideline_action":
                conditions = request["response_contract"]["properties"]["conditions"]["properties"]
                self.assertTrue(all("including unassessed conditions" in row["description"] for row in conditions.values()))
                self.assertIn("not an execution gate", request["instruction"])
            if request["operation"] == "assess_assumption_reasonableness":
                base.requests.append(request)
                return supported_review(request, grade=0, verdict="unsupported")
            if request["operation"] == "propose_guideline_action" and request["assumption_options"]:
                base.requests.append(request)
                calculated = {m for c in request["calculations"].values() for m in c["output"]}
                if set(METRICS) <= calculated:
                    return {"action": "investigate", "objective": "Interpret computed sensitivity and seek better support",
                            "focus": ["mouth_area_m2"], "subjects": ["target"], "child_model_calls": 2}
                return {"action": "calculate_transmission_metrics" if "indoor_reproductive_number" not in calculated else "calculate_conditional_limits",
                    "objective": "Examine explicit guesses", "subject": "target", "basis": "Unjustified synthetic parameterization",
                    "conditions": {a["field"]: a["statement"] for a in request["assumption_options"].values() if a["value"] is None}}
            if request["operation"] == "propose_investigation_strategy" and request["context"]["calculated_model_behavior"]:
                investigated_again = True
                self.assertTrue(all(c["current_support"] == 0 for c in request["context"]["calculated_model_behavior"].values()))
            response = base(request)
            if request["operation"] == "propose_guideline_action" and response["action"] == "investigate":
                response["child_model_calls"] = 60
            elif request["operation"] == "propose_investigation_strategy":
                response["child_model_calls"] = 16
            return response
        _, (record, report) = run_synthetic(agent)
        self.assertFalse(report.passed)
        self.assertTrue(investigated_again)
        self.assertEqual(report.data["status"], "provisional_estimates_available")
        self.assertEqual(report.data["computational_progress"]["calculated_quantities"], len(METRICS))
        self.assertEqual(report.data["evidence_status"], "incomplete_support")
        self.assertIn("Current Conditional Figures", report.markdown())
        self.assertTrue(report.data["calculations"])
        coverage = report.data["scenarios"]["target"]["coverage"]
        self.assertEqual({r["status"] for r in coverage.values()}, {"unsupported_calculation"})
        self.assertGreater(record.controller_state.history[-1], 0)
        self.assertEqual(record.controller_state.history[-1], max(record.controller_state.history))
        self.assertIn("exploratory calculation only", report.markdown())
        json.dumps(report.as_record(), allow_nan=False)

    def test_failed_support_review_does_not_prevent_root_tools(self):
        base = AdaptiveAgent()
        def agent(request):
            if request["operation"] == "review_research_model":
                raise ProviderError("Synthetic unavailable reviewer")
            return base(request)
        _, (_, report) = run_synthetic(agent)
        self.assertFalse(report.passed)
        self.assertTrue(report.data["calculations"])
        self.assertTrue(all(c["model_review"]["admit"] is False for c in report.data["calculations"]))
        self.assertTrue(any(d["stage"] == "model_support_assessment" for d in report.data["attempt_diagnostics"]))

    def test_merge_cannot_launder_a_failed_model_review(self):
        ledger = guessed_ledger({"floor_area_m2": 40, "room_height_m": 2.5})
        first = calculate(ledger, review={"admit": False, "reason": "Unsupported transfer", "limitations": []})
        second = calculate(ledger)
        left, right = first.estimates[0], second.estimates[0]
        from episodic_workflow.case_reports.research.records import compatible_estimates
        self.assertIsNone(compatible_estimates(left, right))
        self.assertEqual(current_support(left, ledger.assumptions), 0)

    def test_failed_model_reassessment_does_not_erase_prior_supported_output(self):
        ledger = guessed_ledger({"room_volume_m3": 100, "removal_capacity_m3_s": .2})
        for key, assumption in tuple(ledger.assumptions.items()):
            ledger.assumptions[key] = {**assumption, "reasonableness": 1}
        first = calculate(ledger, metrics=["removal_rate_s"])
        ledger.merge(first)
        unavailable = {"admit": False, "completed": False, "reason": "Synthetic unavailable reviewer"}
        ledger.merge(calculate(ledger, metrics=["removal_rate_s"], review=unavailable))
        self.assertEqual(len(ledger.estimates), 2)
        self.assertEqual([current_support(e, ledger.assumptions, ledger=ledger) for e in ledger.estimates.values()], [1, 0])
        self.assertEqual(first.estimates[0]["metrics"]["removal_rate_s"]["value"], .002)
        support, issues, invalidated = support_for(ledger, first.facts, {}, "target")
        self.assertEqual(set(support), set(ledger.assumptions))
        self.assertFalse(issues)
        self.assertFalse(invalidated)
