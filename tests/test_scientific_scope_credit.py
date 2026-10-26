"""Progress remains comparable as the scientific parent defines its target scope."""

from dataclasses import replace
import unittest

from episodic_workflow.case_reports.research.assumptions import assess_assumption
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun
from episodic_workflow.case_reports.research.records import Credit, ResearchResult
from episodic_workflow.credit_assignment.research import ResearchController
from episodic_workflow.tests.test_assumption_support import FIELD, candidate, fixture_ledger
from episodic_workflow.tests.test_research import supported_review


class ScientificScopeCreditTests(unittest.TestCase):
    def setUp(self):
        self.ledger = fixture_ledger()
        self.ledger.merge(assess_assumption(lambda r: supported_review(r, grade=1, verdict="provisional"),
                                           candidate(), self.ledger, {}))
        self.controller = ResearchController((FIELD,), initial=self.credit(), sum_qualities=True, window=3)

    def credit(self):
        return self.ledger.result().credit((FIELD,), scientific=True)

    def add_empty_target(self):
        self.ledger.merge(ResearchResult(scenarios=({"subject": "second", "description": "Distinct synthetic target"},)))

    def test_new_empty_target_neither_dilutes_progress_nor_earns_credit(self):
        self.add_empty_target()
        step = self.controller.observe("new scope", self.credit(), is_root=True)
        self.assertEqual(step.score, .5)
        self.assertEqual(step.gain, 0)
        self.assertEqual(self.controller.state().history, (.5, .5))
        self.assertEqual(self.controller.state().rarefaction["max_score"], 2)
        self.assertFalse(step.stop)

    def test_scope_growth_and_real_improvement_keep_the_real_gain(self):
        self.add_empty_target()
        self.ledger.merge(assess_assumption(supported_review, candidate(), self.ledger, {}))
        step = self.controller.observe("narrowed original uncertainty", self.credit(), is_root=True)
        self.assertEqual(step.gain, .5)
        self.assertEqual(step.score, 1)
        self.assertFalse(step.stop, "An unresolved second target must not be completed by the first")

    def test_each_scenario_can_advance_without_averaging_down_prior_progress(self):
        self.add_empty_target()
        self.ledger.merge(assess_assumption(supported_review, candidate(subject="second"), self.ledger, {}))
        step = self.controller.observe("second target assessed", self.credit(), is_root=True)
        self.assertEqual(step.score, 1.5)
        self.assertEqual(step.gain, 1)
        repeated = self.controller.observe("same accumulated result", self.credit(), is_root=True)
        self.assertEqual(repeated.gain, 0)
        self.assertEqual(repeated.score, 1.5)

    def test_only_current_best_assessment_per_target_and_requirement_is_counted(self):
        self.ledger.merge(assess_assumption(supported_review, candidate(), self.ledger, {}))
        self.ledger.merge(assess_assumption(supported_review, candidate(value=25), self.ledger, {}))
        self.assertEqual(len(self.ledger.assumptions), 2)
        self.assertEqual(len(self.credit().qualities), 1)
        self.assertEqual(self.credit().qualities[0][2], 1)
        self.controller.observe("alternatives", self.credit(), is_root=True)
        self.ledger.merge(assess_assumption(supported_review, candidate(statement="Equivalent synthetic restatement"), self.ledger, {}))
        self.assertEqual(self.controller.observe("restatement", self.credit(), is_root=True).gain, 0)

    def test_withdrawn_learning_still_decreases_credit(self):
        result = self.ledger.result()
        withdrawn = replace(result, assumptions=tuple({**a, "resolution_credit": 0} for a in result.assumptions))
        step = self.controller.observe("corrected assessment", withdrawn.credit((FIELD,), scientific=True), is_root=True)
        self.assertEqual(step.gain, -.5)
        self.assertEqual(step.score, 0)
        self.assertEqual(self.controller.state().history, (.5, 0))
        self.assertFalse(step.stop)

    def test_one_rubric_increment_has_same_resolution_before_and_after_scope_growth(self):
        initial_resolution = self.credit().quality_resolution
        self.add_empty_target()
        self.assertEqual(self.credit().quality_resolution, initial_resolution)
        unchanged = self.credit()
        for n in range(2):
            self.controller.observe(str(n), unchanged, is_root=True)
        def reviewer(request):
            response = supported_review(request, grade=1, verdict="provisional")
            response["criteria"]["uncertainty"]["grade"] = 2
            return response
        self.ledger.merge(assess_assumption(reviewer, candidate(), self.ledger, {}))
        step = self.controller.observe("one dimension bounded", self.credit(), is_root=True)
        self.assertEqual(step.gain, 1 / 8)
        self.assertFalse(step.stop)

    def test_scoped_child_does_not_credit_unrequested_target(self):
        self.add_empty_target()
        self.ledger.merge(assess_assumption(supported_review, candidate(subject="second"), self.ledger, {}))
        credit = self.ledger.result().credit((FIELD,), scientific=True, subjects=("target",))
        self.assertEqual(credit.qualities, ((FIELD, "target", .5),))

    def test_controller_wiring_sums_scientific_not_acquisition_grades(self):
        run = ResearchRun(ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None), ResearchLimits(),
                          "Synthetic scope test", ("indoor_reproductive_number",))
        for scientific in (True, False):
            run.controller_specs["fixture"] = ((FIELD,), Credit(qualities=((FIELD, "a", .5), (FIELD, "b", .5))), scientific)
            controller = run.controller((("fixture", "fixture"),))
            self.assertEqual(controller.state().score, 1 if scientific else .5)
            self.assertEqual(controller.transitioned("new-epoch").sum_qualities, scientific)
