"""Failed operations have explicit outcomes without losing successful values."""

import unittest

from episodic_workflow.case_reports.research.admission import extract_and_admit
from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.records import ResearchResult
from episodic_workflow.case_reports.research.relevance import assess_numeric_target_relationships
from episodic_workflow.task_executors._http import ProviderError
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS
from . import test_target_values


class ResearchOperationOutcomeTests(unittest.TestCase):
    def extraction(self, *, review_fails=False):
        def agent(request):
            if request["operation"] == "extract_research_evidence":
                base = {"subject": "source", "field": "breathing_flow_m3_s",
                    "unit": "m3/s", "kind": "source_model_parameter",
                    "context": "Recorded source context", "basis": "Printed scalar"}
                return {"facts": [
                    {**base, "value": .001, "quote": "flow 0.001 m3/s"},
                    {**base, "value": .002, "quote": "not in the source"},
                    {**base, "value": .003, "quote": "flow 0.003 m3/s"}],
                    "evidence": [], "findings": []}
            if review_fails:
                raise ProviderError("Independent evidence review unavailable")
            return {"decisions": [
                {"option": 0, "admit": True, "reason": "Faithful source value"},
                {"option": 1, "admit": False, "reason": "Incorrect source attribution"}]}

        return extract_and_admit(agent, text="flow 0.001 m3/s; flow 0.003 m3/s",
            source="https://example.test/source", objective="Extract source flow",
            context={}, question="What flow is reported?")

    def test_rejected_candidates_are_not_committed_but_valid_partial_value_is(self):
        result = self.extraction()
        self.assertTrue(result.evaluated)
        self.assertEqual([fact.value for fact in result.facts], [.001])
        self.assertEqual([row["category"] for row in result.diagnostics],
                         ["invalid_candidate", "rejected_evidence"])

    def test_failed_source_verification_raises_without_claiming_admitted_evidence(self):
        with self.assertRaises(ProviderError) as caught:
            self.extraction(review_fails=True)
        state = caught.exception.extraction_state
        self.assertEqual(state["status"], "independent_review_incomplete")
        self.assertEqual(state["candidate_count"], 2)
        self.assertEqual(len(caught.exception.unreviewed_candidates), 2)

    def test_failed_relevance_keeps_found_value_and_other_completed_grade(self):
        fixture = test_target_values.TargetValueTests()
        fixture.setUp()
        breathing = fixture.fact()
        deactivation = fixture.fact("deactivation_rate_s", .001)
        fixture.ledger.merge(ResearchResult(facts=(breathing, deactivation)))

        def grade(request):
            if request["parameter"] == "breathing_flow_m3_s":
                raise ProviderError("Relevance assessment unavailable")
            return {subject: .7 for subject in request["target_scenarios"]}

        delta = assess_numeric_target_relationships(fixture.ledger, grade)
        fixture.ledger.merge(delta)
        fixture.project()
        self.assertEqual([row["category"] for row in delta.diagnostics],
                         ["assessment_incomplete"])
        self.assertEqual(delta.diagnostics[0]["field"], breathing.field)
        candidates = {a["field"]: a for a in fixture.ledger.assumptions.values()
                      if a.get("entry_type") == "source_candidate"}
        self.assertEqual(candidates[breathing.field]["source_applicability"]["status"], "unknown")
        self.assertEqual(candidates[deactivation.field]["source_applicability"]["score"], .7)
        status = scientific_goal_status(fixture.ledger.result(), DEFAULT_METRICS)
        self.assertEqual(status["recorded_target_inputs"]["recorded"], 2)


if __name__ == "__main__":
    unittest.main()
