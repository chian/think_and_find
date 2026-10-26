"""Synthetic checks: reassessing a claim is not changing a calculation's inputs."""

from copy import deepcopy
import unittest

from episodic_workflow.case_reports.research.assumptions import assess_assumption
from episodic_workflow.case_reports.research.records import current_support
from episodic_workflow.case_reports.research.report import scenario_results
from episodic_workflow.case_reports.research.tools import tool_options
from episodic_workflow.tests import test_approved_tree
from episodic_workflow.tests.test_research import supported_review


METRIC = "removal_rate_s"


class AssessmentCurrencyTests(unittest.TestCase):
    def setUp(self):
        self.helper = test_approved_tree.CalculationToolTests()
        self.values = {"room_volume_m3": 100, "removal_capacity_m3_s": .2}
        self.ledger = self.helper.ledger(self.values)

    def review(self, field="room_volume_m3", *, grade=2, verdict="supported", **changes):
        old = next(a for a in self.ledger.assumptions.values() if a["field"] == field)
        proposal = {k: old[k] for k in ("subject", "field", "value", "unit", "statement", "basis")}
        proposal.update(changes, evidence=list(self.ledger.evidence_options()))
        def reviewer(request):
            response = supported_review(request, grade=grade, verdict=verdict)
            response["explanation"] = "New independent review of synthetic evidence"
            return response
        result = assess_assumption(reviewer, proposal, self.ledger, {})
        self.ledger.merge(result)
        return result.assumptions[0]

    def calculate(self):
        result = self.helper.execute(self.ledger, "calculate_geometry_and_removal", self.values,
                                     metrics=[METRIC])
        self.ledger.merge(result)
        return result.calculations[0]

    def test_unchanged_claim_retains_output_credit_without_rewriting_calculation(self):
        calculation = self.calculate()
        saved = deepcopy(calculation)
        prior_credit = self.ledger.result().credit((METRIC,), scientific=True)
        revised = self.review()
        self.assertNotEqual(calculation["assumption_versions"][revised["key"]], revised["version"])
        self.assertEqual(current_support(calculation, self.ledger.assumptions), 1)
        self.assertEqual(self.ledger.result().credit((METRIC,), scientific=True), prior_credit)
        scenarios, complete = scenario_results(self.ledger.result(), (METRIC,))
        self.assertTrue(complete)
        self.assertEqual(scenarios["case"]["coverage"][METRIC]["status"], "calculated")
        self.assertEqual(tool_options(self.ledger)["calculation:1"]["current_support"], 1)
        self.assertEqual(calculation, saved)
        self.assertEqual(len(self.ledger.calculations), 1)

    def test_stronger_review_improves_support_of_unchanged_computation(self):
        self.review(grade=1, verdict="provisional")
        calculation = self.calculate()
        self.assertEqual(current_support(calculation, self.ledger.assumptions), .5)
        self.review()
        self.assertEqual(current_support(calculation, self.ledger.assumptions), 1)

    def test_weaker_or_contradicted_review_reduces_current_support(self):
        calculation = self.calculate()
        self.review(grade=1, verdict="provisional")
        self.assertEqual(current_support(calculation, self.ledger.assumptions), .5)
        self.review(verdict="contradicted")
        self.assertEqual(current_support(calculation, self.ledger.assumptions), 0)
        self.assertFalse(scenario_results(self.ledger.result(), (METRIC,))[1])

    def test_changed_numeric_interpretation_does_not_authenticate_old_result(self):
        calculation = self.calculate()
        revised = self.review(statement="Same number now refers to only part of the synthetic room")
        self.assertIn(revised["key"], calculation["assumption_versions"])
        self.assertEqual(current_support(calculation, self.ledger.assumptions), 0)

    def test_legacy_record_without_claim_snapshot_remains_version_strict(self):
        calculation = self.calculate()
        legacy = {k: v for k, v in calculation.items() if k != "assumption_claims"}
        self.assertEqual(current_support(legacy, self.ledger.assumptions), 1)
        self.review()
        self.assertEqual(current_support(legacy, self.ledger.assumptions), 0)

    def test_missing_assessment_and_failed_model_review_remain_unsupported(self):
        calculation = self.calculate()
        revised = self.review()
        failed = {**calculation, "model_review": {"admit": True, "completed": False}}
        self.assertEqual(current_support(failed, self.ledger.assumptions), 0)
        del self.ledger.assumptions[revised["key"]]
        self.assertEqual(current_support(calculation, self.ledger.assumptions), 0)

    def test_unchanged_reassessment_preserves_transitive_derived_support(self):
        self.values = {"floor_area_m2": 40, "room_height_m": 2.5, "removal_rate_s": .002}
        self.ledger = self.helper.ledger(self.values)
        self.ledger.merge(self.helper.execute(self.ledger, "calculate_geometry_and_removal", self.values, metrics=[]))
        self.review("floor_area_m2")
        result = self.helper.execute(self.ledger, "calculate_geometry_and_removal",
            ("room_volume_m3", "removal_capacity_m3_s"), metrics=[METRIC])
        self.assertFalse(result.calculations[0]["support_issues"])
        self.assertEqual(current_support(result.calculations[0], self.ledger.assumptions), 1)

    def test_observation_comparison_preserves_prediction_claim_snapshot(self):
        conditions = {f: f"Assume {f}" for f in
                      ("well_mixed", "single_infectious_source", "uniform_breathing_and_masks")}
        values = {"quanta_concentration_exhaled_m3": 20, "breathing_flow_m3_s": .001,
                  "removal_capacity_m3_s": .2, "mask_penetration": 1,
                  "relative_transmissibility": 1, "exposure_time_s": 600, "attack_probability": .05}
        self.ledger = self.helper.ledger(values, conditions)
        metric = "individual_infection_probability"
        prediction = self.helper.execute(self.ledger, "calculate_transmission_metrics", values,
            metrics=[metric], conditions=conditions)
        self.ledger.merge(prediction)
        saved = deepcopy(prediction.calculations[0])
        self.review("well_mixed")
        observation = next(k for k, f in self.ledger.options().items()
                           if f.field == "attack_probability" and f.kind == "source_observation")
        result = self.helper.execute(self.ledger, "compare_predictions_with_observations", (),
            prediction="calculation:1", observation=observation, metric=metric)
        comparison = result.calculations[0]
        self.assertEqual(current_support(comparison, self.ledger.assumptions), 1)
        self.assertEqual(comparison["assumption_versions"], saved["metric_support"][metric]["assumption_versions"])
        self.assertEqual(comparison["assumption_claims"], saved["metric_support"][metric]["assumption_claims"])
        self.assertFalse(comparison["output"]["independent_validation"])
