"""A working conditional solution can improve without pretending it is evidence."""

from copy import deepcopy
from dataclasses import replace
import json
import unittest

from episodic_workflow.case_reports.research.assumptions import assess_assumption, PROVISIONAL_OUTPUT_CREDIT
from episodic_workflow.case_reports.research.records import current_computation, current_support
from episodic_workflow.case_reports.research.report import scenario_results, scientific_progress
from episodic_workflow.credit_assignment.research import ResearchController
from episodic_workflow.tests import test_approved_tree
from episodic_workflow.tests.test_research import supported_review
from episodic_workflow.tests.test_research import AdaptiveAgent, METRICS, run_synthetic
from episodic_workflow.task_executors._http import ProviderError


RISK = "indoor_reproductive_number"
TIME = "exposure_time_bound_s"


class ProvisionalSolutionTests(unittest.TestCase):
    def setUp(self):
        self.helper = test_approved_tree.CalculationToolTests()
        self.conditions = {name: f"Assume {name}" for name in
                           ("well_mixed", "single_infectious_source", "uniform_breathing_and_masks")}
        self.values = {"quanta_concentration_exhaled_m3": 20, "breathing_flow_m3_s": .001,
                       "removal_capacity_m3_s": .2, "mask_penetration": 1,
                       "relative_transmissibility": 1, "exposure_time_s": 600,
                       "susceptible_count": 10, "risk_tolerance": .1}
        self.ledger = self.helper.ledger(self.values, self.conditions)
        self.review_all(grade=0, verdict="unsupported")

    def review_all(self, *, grade=2, verdict="supported", **changes):
        for old in list(self.ledger.assumptions.values()):
            candidate = {k: old[k] for k in ("subject", "field", "value", "unit", "statement", "basis")}
            candidate.update(changes, evidence=list(self.ledger.evidence_options()))
            self.ledger.merge(assess_assumption(lambda r: supported_review(r, grade=grade, verdict=verdict),
                                                candidate, self.ledger, {}))

    def calculate(self, metric=RISK, *, parameterization="one synthetic case"):
        tool = "calculate_transmission_metrics" if metric == RISK else "calculate_conditional_limits"
        result = self.helper.execute(self.ledger, tool, self.values, metrics=[metric],
            conditions=self.conditions, parameterization=parameterization)
        self.ledger.merge(result)
        return result.estimates[0]

    def credit(self, metric=RISK):
        return self.ledger.result().credit((metric,), scientific=True)

    def test_first_numerical_consequence_is_progress_but_repetition_and_labels_are_not(self):
        controller = ResearchController((RISK,), initial=self.credit(), sum_qualities=True)
        estimate = self.calculate()
        step = controller.observe("first calculation", self.credit(), is_root=True)
        self.assertEqual(step.gain, PROVISIONAL_OUTPUT_CREDIT)
        self.assertEqual(current_support(estimate, self.ledger.assumptions, metric=RISK), 0)
        self.calculate()
        self.assertEqual(controller.observe("repeated", self.credit(), is_root=True).gain, 0)
        self.calculate(parameterization="new label for the same numbers")
        self.assertEqual(controller.observe("relabelled", self.credit(), is_root=True).gain, 0)

    def test_evidence_strength_improves_existing_solution_without_recalculation(self):
        self.calculate()
        calculations = deepcopy(self.ledger.calculations)
        controller = ResearchController((RISK,), initial=self.credit(), sum_qualities=True)
        self.review_all(grade=1, verdict="provisional")
        self.assertEqual(controller.observe("bounded evidence", self.credit(), is_root=True).score, .5)
        self.review_all()
        self.assertEqual(controller.observe("stronger evidence", self.credit(), is_root=True).score, 1)
        self.assertEqual(self.ledger.calculations, calculations)

    def test_changed_or_contradicted_premises_withdraw_provisional_progress(self):
        self.calculate()
        controller = ResearchController((RISK,), initial=self.credit(), sum_qualities=True)
        self.review_all(statement="A genuinely different physical claim")
        step = controller.observe("premise correction", self.credit(), is_root=True)
        self.assertEqual(step.gain, -PROVISIONAL_OUTPUT_CREDIT)
        self.assertEqual(step.score, 0)
        progress = scientific_progress(self.ledger.result(), (RISK,))
        self.assertEqual(progress["recorded_calculated_quantities"], 1)
        self.assertEqual(progress["calculated_quantities"], 0)
        changed = progress["targets"]["case"]["recalculation_needs"][0]["changed_premises"]
        self.assertTrue(changed)
        self.assertTrue(all(p["current_statement"] == "A genuinely different physical claim" for p in changed))
        self.calculate()
        self.review_all(verdict="contradicted")
        self.assertEqual(self.credit().qualities[0][2], 0)

    def test_progress_reports_actual_figures_and_assumption_impact_separately(self):
        estimate = self.calculate()
        progress = scientific_progress(self.ledger.result(), (RISK, TIME))
        target = progress["targets"]["case"]
        self.assertEqual(progress["calculated_quantities"], 1)
        self.assertEqual(progress["evidence_supported_quantities"], 0)
        row = target["coverage"][RISK]
        self.assertTrue(row["computed"])
        self.assertEqual(row["figures"][0]["value"], estimate["metrics"][RISK]["value"])
        self.assertEqual(row["figures"][0]["support"], 0)
        self.assertEqual(row["status"], "unsupported_calculation")
        self.assertFalse(target["provisional_parameterization_available"])
        self.assertTrue(target["recorded_numeric_fields"])
        self.assertFalse(target["assessed_numeric_fields"])
        cq = next(n for n in target["assumption_support_needs"] if n["field"] == "quanta_concentration_exhaled_m3")
        self.assertEqual(cq["affected_quantities"], [RISK])
        self.assertIn("reason", cq)

    def test_compatible_tools_and_unchanged_reassessments_form_one_provisional_solution(self):
        self.calculate()
        self.review_all(grade=0, verdict="unsupported", basis="Reviewed again without changing the claim")
        self.calculate(TIME)
        scenarios, supported = scenario_results(self.ledger.result(), (RISK, TIME))
        self.assertEqual(len(self.ledger.estimates), 1)
        self.assertTrue(scenarios["case"]["provisional_parameterization_available"])
        self.assertFalse(supported)

    def test_separate_alternatives_are_not_silently_combined_into_a_solution(self):
        self.calculate(parameterization="first case")
        self.calculate(TIME, parameterization="different case")
        progress = scientific_progress(self.ledger.result(), (RISK, TIME))
        self.assertEqual(progress["calculated_quantities"], 2)
        self.assertFalse(progress["targets"]["case"]["provisional_parameterization_available"])

    def test_no_credit_for_bare_calculation_label_or_unreviewed_inapplicability(self):
        estimate = self.calculate()
        for row in ({"status": "calculated"}, {"status": "inapplicable"},
                    {"status": "calculated", "value": float("nan"), "limit": "finite"}):
            invalid = {**estimate, "metrics": {RISK: row}}
            result = replace(self.ledger.result(), estimates=(invalid,))
            self.assertFalse(current_computation(invalid, self.ledger.assumptions, RISK))
            self.assertEqual(result.credit((RISK,), scientific=True).qualities[0][2], 0)

    def test_unbounded_consequence_is_reported_and_credited_without_nonfinite_json(self):
        self.conditions["continuous_unmasked_jet"] = "Explicit synthetic jet condition"
        extra = {"mouth_area_m2": .0002, "entrainment_coefficient": .1,
                 "plume_encounter_probability": 1}
        self.values.update(extra)
        self.ledger = self.helper.ledger(self.values, self.conditions)
        self.review_all(grade=0, verdict="unsupported")
        metric = "risk_threshold_distance_m"
        estimate = self.calculate(metric)
        self.assertTrue(current_computation(estimate, self.ledger.assumptions, metric))
        self.assertEqual(estimate["metrics"][metric]["limit"], "unbounded")
        self.assertEqual(self.credit(metric).qualities[0][2], PROVISIONAL_OUTPUT_CREDIT)
        json.dumps(scientific_progress(self.ledger.result(), (metric,)), allow_nan=False)

    def test_invalidated_derived_input_does_not_regain_computational_credit(self):
        self.values = {"floor_area_m2": 40, "room_height_m": 2.5, "removal_rate_s": .002}
        self.ledger = self.helper.ledger(self.values, self.conditions)
        self.ledger.merge(self.helper.execute(self.ledger, "calculate_geometry_and_removal", self.values,
                                             metrics=[], conditions=self.conditions))
        self.review_all(statement="Revised physical scope")
        metric = "concentration_relaxation_time_s"
        result = self.helper.execute(self.ledger, "calculate_geometry_and_removal",
            ("room_volume_m3", "removal_capacity_m3_s"), metrics=[metric], conditions=self.conditions)
        self.ledger.merge(result)
        self.assertTrue(result.estimates[0]["metric_support"][metric]["invalidated_dependencies"])
        self.assertFalse(current_computation(result.estimates[0], self.ledger.assumptions, metric))
        self.assertEqual(self.credit(metric).qualities[0][2], 0)

    def test_initial_reviewer_failure_does_not_block_root_calculations(self):
        base = AdaptiveAgent()
        def agent(request):
            if request["operation"] == "assess_assumption_reasonableness":
                raise ProviderError("Synthetic initial reviewer unavailable")
            if request["operation"] == "propose_guideline_action" and request["assumption_options"]:
                assumptions = request["assumption_options"]
                self.assertTrue(all(a["review"]["verdict"] == "unassessed" for a in assumptions.values()))
                computed = {m for c in request["calculations"].values() for m in c["current_computed_metrics"]}
                return {"action": "calculate_transmission_metrics" if RISK not in computed else "calculate_conditional_limits",
                    "objective": "Calculate explicitly unassessed hypotheses", "subject": "target",
                    "basis": "Synthetic unassessed parameterization", "parameterization": "shared synthetic case",
                    "facts": [k for k,a in assumptions.items() if a["value"] is not None],
                    "conditions": {a["field"]: a["statement"] for a in assumptions.values() if a["value"] is None}}
            response = base(request)
            if request["operation"] == "propose_guideline_action" and response["action"] == "investigate":
                response["child_model_calls"] = 30
            return response
        _, (_, report) = run_synthetic(agent)
        self.assertFalse(report.passed)
        self.assertEqual(report.data["status"], "provisional_estimates_available")
        self.assertEqual(report.data["computational_progress"]["calculated_quantities"], len(METRICS))
        self.assertTrue(report.data["calculations"])
        self.assertTrue(all(a["resolution_credit"] == 0 for a in report.data["assumption_assessments"]))
        self.assertTrue(all(a["review"]["completed"] is False for a in report.data["assumption_assessments"]))
        self.assertTrue(report.data["model_response_failures"])
