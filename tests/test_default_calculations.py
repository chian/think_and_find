"""Defaults and reusable calculations keep their actual input dependencies."""

import copy
import unittest

from episodic_workflow.case_reports.research.assumptions import unassessed_assumption
from episodic_workflow.case_reports.research.calculation_steps import calculation_step_status
from episodic_workflow.case_reports.research.model_defaults import initialize_model_defaults
from episodic_workflow.case_reports.research.records import Ledger, ResearchResult, admitted_calculation, current_computation
from episodic_workflow.case_reports.research.scenario_sharing import initialize_parameter_scope
from episodic_workflow.case_reports.research.scenarios import Scenario, initial_scenarios
from episodic_workflow.case_reports.research.tools import execute_tool, main_calculation_inputs, tool_options


REVIEW = {"admit": True, "completed": True, "reason": "Synthetic executable model", "limitations": []}


def planned_ledger(values, conditions=(), assessor=None):
    scenario = Scenario("test-target", "Ordinary indoor water-like respiratory particles",
                        tuple(values.items()), tuple((field, f"Assume {field}") for field in conditions),
                        variables=())
    return Ledger(initialize_model_defaults(initial_scenarios((scenario,)), assessor))


def calculate(ledger, tool, metrics):
    return execute_tool({"action": tool, "subject": "test-target",
                         "metrics": list(metrics), "conditions": {}, "basis": "Synthetic calculation"},
                        ledger, tuple(metrics), review=REVIEW)


class DefaultCalculationTests(unittest.TestCase):
    def test_settling_uses_defaults_without_agent_selecting_them(self):
        ledger = planned_ledger({"effective_radius_m": 1e-6}, ("stokes_particles",))
        result = calculate(ledger, "calculate_geometry_and_removal", ("settling_speed_m_s",))
        calculation = result.calculations[0]
        self.assertEqual(calculation["output"]["settling_speed_m_s"]["status"], "calculated")
        defaults = {row["field"] for row in calculation["input_records"] if row["kind"] == "model_default"}
        self.assertEqual(defaults, {"gravity_m_s2", "air_density_kg_m3",
                                    "air_dynamic_viscosity_pa_s", "particle_density_kg_m3"})
        self.assertFalse(ledger.evidence_options())
        ledger.merge(result)
        self.assertTrue(admitted_calculation(calculation, ledger.assumptions,
                                             ledger=ledger, field="settling_speed_m_s"))

    def test_geometry_without_named_metric_also_receives_defaults(self):
        ledger = planned_ledger({"effective_radius_m": 1e-6}, ("stokes_particles",))
        result = calculate(ledger, "calculate_geometry_and_removal", ())
        self.assertIn("settling_speed_m_s", result.calculations[0]["derived_parameters"])

    def test_promoted_constant_is_not_silently_filled(self):
        def review(item):
            return {"decisions": {subject: {field: {
                "decision": "revisit" if field == "air_dynamic_viscosity_pa_s" else "retain",
                "reason": "Synthetic promotion contract check",
                "scenario_evidence": [item["scenarios"][subject]["description"]]
                    if field == "air_dynamic_viscosity_pa_s" else [],
            } for field in fields} for subject, fields in item["default_candidates"].items()}}
        ledger = planned_ledger({"effective_radius_m": 1e-6}, ("stokes_particles",), review)
        result = calculate(ledger, "calculate_geometry_and_removal", ("settling_speed_m_s",))
        output = result.calculations[0]["output"]["settling_speed_m_s"]
        self.assertEqual(output["status"], "missing_inputs")
        self.assertIn("air_dynamic_viscosity_pa_s", output["missing"])
        self.assertNotIn("air_dynamic_viscosity_pa_s", result.calculations[0]["input_values"])

    def test_changed_default_invalidates_only_calculations_using_it(self):
        ledger = planned_ledger({"effective_radius_m": 1e-6, "breathing_flow_m3_s": .001,
                                 "quanta_concentration_exhaled_m3": 20},
                                ("stokes_particles", "well_mixed", "single_infectious_source",
                                 "uniform_breathing_and_masks"))
        settling = calculate(ledger, "calculate_geometry_and_removal", ("settling_speed_m_s",))
        emission = calculate(ledger, "calculate_transmission_metrics", ("quanta_emission_rate_s",))
        ledger.merge(settling)
        ledger.merge(emission)
        scenario = copy.deepcopy(ledger.scenarios["test-target"])
        scenario["model_defaults"]["gravity_m_s2"]["value"] = 1.62
        ledger.scenarios["test-target"] = scenario
        self.assertFalse(current_computation(settling.calculations[0], ledger.assumptions,
                                             "settling_speed_m_s", ledger=ledger))
        self.assertFalse(admitted_calculation(settling.calculations[0], ledger.assumptions,
                                              ledger=ledger, field="settling_speed_m_s"))
        self.assertTrue(current_computation(emission.calculations[0], ledger.assumptions,
                                            "quanta_emission_rate_s", ledger=ledger))
        self.assertTrue(admitted_calculation(emission.calculations[0], ledger.assumptions,
                                             ledger=ledger, metric="quanta_emission_rate_s"))
        gravity = next(fact for fact in main_calculation_inputs(ledger, "test-target")
                       if fact.field == "gravity_m_s2")
        self.assertEqual(gravity.value, 1.62)

    def test_both_risk_and_limit_tools_publish_recursive_emission(self):
        values = {"breathing_flow_m3_s": .001, "quanta_concentration_exhaled_m3": 20,
                  "mask_penetration": 1, "relative_transmissibility": 1, "removal_rate_s": .002,
                  "room_volume_m3": 100, "susceptible_count": 4, "risk_tolerance": .1,
                  "neighbor_distance_m": 2, "exposure_time_s": 900}
        conditions = ("well_mixed", "single_infectious_source", "uniform_breathing_and_masks",
                      "negligible_near_field")
        for tool, metric in (("calculate_transmission_metrics", "scenario_reproductive_number"),
                             ("calculate_conditional_limits", "allowable_room_time_s")):
            with self.subTest(tool=tool):
                ledger = planned_ledger(values, conditions)
                result = calculate(ledger, tool, (metric,))
                calculation = result.calculations[0]
                self.assertEqual(calculation["output"][metric]["status"], "calculated")
                emitted = {fact.field: fact for fact in result.facts}
                self.assertEqual(emitted["quanta_emission_rate_s"].value, .02)
                support = calculation["derived_support"]["quanta_emission_rate_s"]
                self.assertEqual(set(support["input_fields"]),
                                 {"breathing_flow_m3_s", "quanta_concentration_exhaled_m3"})
                ledger.merge(result)
                self.assertTrue(admitted_calculation(calculation, ledger.assumptions,
                                                     ledger=ledger, field="quanta_emission_rate_s"))
                self.assertEqual(calculation["intermediate_outputs"]["emission_rate_quanta_s"]["value"], .02)
                self.assertTrue(current_computation(calculation, ledger.assumptions,
                                                    "emission_rate_quanta_s", ledger=ledger))
                self.assertTrue(admitted_calculation(calculation, ledger.assumptions,
                                                     ledger=ledger, metric="emission_rate_quanta_s"))
                steps = calculation_step_status(ledger.result(), (metric,))["steps"]
                masked = next(row for row in steps if row["field"] == "emission_rate_quanta_s")
                self.assertEqual((masked["status"], masked["points"]), ("calculated_up_to_date", .125))
                self.assertNotIn("emission_rate_quanta_s", emitted)
                self.assertIn("quanta_emission_rate_s", tool_options(ledger)["calculation:1"]["derived_parameters"])
                self.assertFalse(ledger.evidence_options())

    def test_shared_value_review_does_not_require_repeating_unchanged_arithmetic(self):
        conditions = tuple((name, f"Assume {name}") for name in
                           ("well_mixed", "single_infectious_source", "uniform_breathing_and_masks"))
        cases = tuple(Scenario(subject, "Ordinary indoor target", (("quanta_concentration_exhaled_m3", 20),),
                               conditions, variables=()) for subject in ("origin", "recipient"))
        initial = initialize_model_defaults(initial_scenarios(cases))
        initial = initialize_parameter_scope(initial, lambda item: {"features": [], "dependencies": [
            {"parameter": field, "factors": [], "reason": "Same physical contexts in this fixture"}
            for field in item["parameters"]]})
        ledger = Ledger(initial)
        assumption = unassessed_assumption({"subject": "origin", "field": "breathing_flow_m3_s",
            "value": .001, "unit": "m3/s", "statement": "Synthetic breathing hypothesis",
            "basis": "Synthetic fixture"}, ledger)
        ledger.merge(ResearchResult(assumptions=(assumption,)))
        result = execute_tool({"action": "calculate_transmission_metrics", "subject": "recipient",
            "metrics": ["quanta_emission_rate_s"], "basis": "Shared input fixture"},
            ledger, ("quanta_emission_rate_s",), review=REVIEW)
        ledger.merge(result)
        calculation = result.calculations[0]
        ledger.assumptions[assumption["key"]] = {**assumption, "version": "renewed-evidence-review",
            "reasonableness": .5, "review": {**assumption["review"], "verdict": "provisional"}}
        self.assertTrue(current_computation(calculation, ledger.assumptions,
                                            "quanta_emission_rate_s", ledger=ledger))
        self.assertTrue(admitted_calculation(calculation, ledger.assumptions,
                                             ledger=ledger, metric="quanta_emission_rate_s"))
        ledger.assumptions[assumption["key"]]["review"]["verdict"] = "contradicted"
        self.assertFalse(current_computation(calculation, ledger.assumptions,
                                             "quanta_emission_rate_s", ledger=ledger))

    def test_changed_fixed_condition_marks_affected_step_stale(self):
        ledger = planned_ledger({"effective_radius_m": 1e-6}, ("stokes_particles",))
        result = calculate(ledger, "calculate_geometry_and_removal", ("settling_speed_m_s",))
        ledger.merge(result)
        ledger.scenarios["test-target"]["fixed_conditions"]["stokes_particles"] = "Revised particle-model premise"
        self.assertFalse(current_computation(result.calculations[0], ledger.assumptions,
                                             "settling_speed_m_s", ledger=ledger))
        row = next(row for row in calculation_step_status(ledger.result(), ("settling_speed_m_s",))["steps"]
                   if row["field"] == "settling_speed_m_s")
        self.assertEqual((row["status"], row["points"]), ("calculated_needs_updating", .025))
        self.assertTrue(any(change.get("field") == "stokes_particles"
                            for result in row["result_alternatives"] for change in result["changed_dependencies"]))


if __name__ == "__main__":
    unittest.main()
