"""Synthetic evidence-support dependencies, distinct from numerical execution."""

import unittest

from episodic_workflow.case_reports.research.records import current_support
from episodic_workflow.case_reports.research.assumptions import PROVISIONAL_OUTPUT_CREDIT
from episodic_workflow.case_reports.research.report import scenario_results
from episodic_workflow.case_reports.research.tools import tool_options
from episodic_workflow.task_executors.scenario_metrics.evaluate import evaluate_parameters, output_dependencies
from episodic_workflow.tests import test_approved_tree


class MetricSupportTests(unittest.TestCase):
    def setUp(self):
        self.helper = test_approved_tree.CalculationToolTests()
        self.conditions = {f: f"Assume {f}" for f in
                           ("well_mixed", "single_infectious_source", "uniform_breathing_and_masks")}
        self.values = {"quanta_concentration_exhaled_m3": 20, "breathing_flow_m3_s": .001,
            "removal_capacity_m3_s": .2, "room_volume_m3": 100, "mask_penetration": 1,
            "relative_transmissibility": 1, "exposure_time_s": 600, "susceptible_count": 10,
            "initial_concentration_quanta_m3": 0}
        self.ledger = self.helper.ledger(self.values, self.conditions)

    def weaken(self, field):
        for key, a in tuple(self.ledger.assumptions.items()):
            if a["field"] == field:
                self.ledger.assumptions[key] = {**a, "version": "synthetic-review-correction", "reasonableness": 0}

    def calculate(self, metrics):
        result = self.helper.execute(self.ledger, "calculate_transmission_metrics", self.values,
                                     metrics=metrics, conditions=self.conditions)
        self.ledger.merge(result)
        return result.calculations[0]

    def test_initial_concentration_support_affects_transient_not_steady_results(self):
        steady, transient = "indoor_reproductive_number", "transient_indoor_reproductive_number"
        self.weaken("initial_concentration_quanta_m3")
        calculation = self.calculate([steady, transient])
        self.assertEqual(current_support(calculation, self.ledger.assumptions, ledger=self.ledger, metric=steady), 1)
        self.assertEqual(current_support(calculation, self.ledger.assumptions, ledger=self.ledger, metric=transient), 0)
        scores = {f: score for f, _, score in self.ledger.result().credit((steady, transient), scientific=True).qualities}
        self.assertEqual(scores, {steady: 1, transient: PROVISIONAL_OUTPUT_CREDIT})
        scenarios, complete = scenario_results(self.ledger.result(), (steady, transient))
        self.assertFalse(complete)
        self.assertEqual(scenarios["case"]["coverage"][steady]["status"], "calculated")
        self.assertEqual(scenarios["case"]["coverage"][transient]["status"], "unsupported_calculation")
        self.assertTrue(scenario_results(self.ledger.result(), (steady,))[1])
        self.assertEqual(tool_options(self.ledger)["calculation:1"]["metric_support"], {steady: 1, transient: 0})

    def test_missing_assessment_for_unused_selected_input_does_not_remove_credit(self):
        for key, a in tuple(self.ledger.assumptions.items()):
            if a["field"] == "initial_concentration_quanta_m3":
                del self.ledger.assumptions[key]
        metric = "indoor_reproductive_number"
        calculation = self.calculate([metric])
        self.assertNotIn("initial_concentration_quanta_m3", calculation["input_values"])
        self.assertEqual(current_support(calculation, self.ledger.assumptions, ledger=self.ledger, metric=metric), 1)
        self.weaken("quanta_concentration_exhaled_m3")
        self.assertEqual(current_support(calculation, self.ledger.assumptions, ledger=self.ledger, metric=metric), 0)

    def test_derived_geometry_tracks_only_its_own_dependencies(self):
        values = {"floor_area_m2": 40, "room_height_m": 2.5, "removal_rate_s": .002, "clearance_fraction": .99}
        self.ledger = self.helper.ledger(values)
        self.weaken("clearance_fraction")
        result = self.helper.execute(self.ledger, "calculate_geometry_and_removal", values, metrics=[])
        self.ledger.merge(result)
        followup = self.helper.execute(self.ledger, "calculate_geometry_and_removal",
            ("room_volume_m3", "removal_capacity_m3_s"), metrics=["removal_rate_s"])
        self.assertEqual(current_support(followup.calculations[0], self.ledger.assumptions, ledger=self.ledger, metric="removal_rate_s"), 1)
        # The whole main table is bound automatically. Per-output dependencies,
        # rather than a caller-selected subset, determine scientific support.
        for support in (result.calculations[0]["derived_support"]["room_volume_m3"],
                        followup.calculations[0]["metric_support"]["removal_rate_s"]):
            fields = {self.ledger.assumptions[k]["field"] for k in support["assumption_versions"]}
            self.assertNotIn("clearance_fraction", fields)

    def test_exclusion_requires_only_the_evidence_establishing_inapplicability(self):
        self.values["mask_penetration"] = .3
        self.ledger = self.helper.ledger(self.values, self.conditions)
        self.weaken("quanta_concentration_exhaled_m3")
        metric = "critical_distance_m"
        result = self.helper.execute(self.ledger, "calculate_conditional_limits", self.values,
                                     metrics=[metric], conditions=self.conditions)
        self.assertEqual(result.estimates[0]["metrics"][metric]["status"], "inapplicable")
        self.assertEqual(current_support(result.calculations[0], self.ledger.assumptions, ledger=self.ledger, metric=metric), 1)
        self.weaken("mask_penetration")
        self.assertEqual(current_support(result.calculations[0], self.ledger.assumptions, ledger=self.ledger, metric=metric), 0)

    def test_merged_tool_results_preserve_separate_metric_support(self):
        self.weaken("initial_concentration_quanta_m3")
        self.calculate(["indoor_reproductive_number"])
        self.calculate(["transient_indoor_reproductive_number"])
        self.assertEqual(len(self.ledger.estimates), 1)
        estimate = next(iter(self.ledger.estimates.values()))
        self.assertEqual(current_support(estimate, self.ledger.assumptions, ledger=self.ledger, metric="indoor_reproductive_number"), 1)
        self.assertEqual(current_support(estimate, self.ledger.assumptions, ledger=self.ledger, metric="transient_indoor_reproductive_number"), 0)

    def test_trace_uses_executed_operand_not_an_unused_equivalent_input(self):
        values = {"quanta_emission_rate_s": .02, "quanta_concentration_exhaled_m3": 20,
                  "breathing_flow_m3_s": .001, "mask_penetration": 1}
        direct = evaluate_parameters(values, self.conditions, ("emission_rate_quanta_s",), subject="test", label="test")
        derived = evaluate_parameters(values, self.conditions, ("quanta_emission_rate_s", "emission_rate_quanta_s"),
                                      subject="test", label="test")
        self.assertEqual(output_dependencies(direct, "emission_rate_quanta_s")[0],
                         ["mask_penetration", "quanta_emission_rate_s"])
        self.assertEqual(output_dependencies(derived, "emission_rate_quanta_s")[0],
                         ["breathing_flow_m3_s", "mask_penetration", "quanta_concentration_exhaled_m3"])

    def test_derived_settling_retains_physical_model_condition(self):
        values = {"effective_radius_m": .000001, "particle_density_kg_m3": 1000,
                  "air_density_kg_m3": 1, "air_dynamic_viscosity_pa_s": .00002, "gravity_m_s2": 10}
        estimate = evaluate_parameters(values, {"stokes_particles": "Synthetic low-Re assumption"}, (),
                                       subject="test", label="test")
        fields, conditions = output_dependencies(estimate, "settling_speed_m_s", derived=True)
        self.assertEqual(set(fields), set(values))
        self.assertEqual(conditions, ["stokes_particles"])

    def test_observation_comparison_uses_only_selected_prediction_dependencies(self):
        self.values["attack_probability"] = .05
        self.ledger = self.helper.ledger(self.values, self.conditions)
        self.weaken("initial_concentration_quanta_m3")
        metric = "individual_infection_probability"
        prediction = self.calculate([metric, "transient_individual_infection_probability"])
        observation = next(k for k, f in self.ledger.options().items()
                           if f.field == "attack_probability" and f.kind == "source_observation")
        comparison = self.helper.execute(self.ledger, "compare_predictions_with_observations", (),
            prediction="calculation:1", observation=observation, metric=metric).calculations[0]
        self.assertEqual(current_support(prediction, self.ledger.assumptions, ledger=self.ledger), 0)
        self.assertEqual(current_support(comparison, self.ledger.assumptions, ledger=self.ledger), 1)
        self.assertFalse(comparison["output"]["independent_validation"])
        from episodic_workflow.case_reports.research.records import ResearchResult
        self.ledger.merge(ResearchResult(calculations=(comparison,)))
        option = next(o for o in tool_options(self.ledger).values() if o["tool"] == "compare_predictions_with_observations")
        self.assertEqual(option["current_computed_metrics"], [])

    def test_bulk_flow_timescale_does_not_require_infectious_source_or_mask_assumptions(self):
        values = {"room_volume_m3": 100, "ventilation_flow_m3_s": .02, "recirculation_flow_m3_s": .03,
                  "index_case_count": 3, "mask_penetration": .3}
        name = "air_mixing_time_s"
        result = evaluate_parameters(values, {}, (name,), subject="test", label="test")
        self.assertEqual(result["metrics"][name]["value"], 2000)
        fields, conditions = output_dependencies(result, name)
        self.assertEqual(set(fields), {"room_volume_m3", "ventilation_flow_m3_s", "recirculation_flow_m3_s"})
        self.assertEqual(conditions, [])
        self.assertIn("does not establish actual spatial uniformity", result["metrics"][name]["interpretation"])
