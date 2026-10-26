"""Calculation-time support review preserves automatic discovery and lineage."""

from types import SimpleNamespace
from episodic_workflow.case_reports.research.records import numeric_references
import unittest

from episodic_workflow.case_reports.research.assumptions import RUBRIC, assess_selected_source_candidates
from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.records import ResearchResult, assumption_claim, current_support, identity
from episodic_workflow.case_reports.research.target_values import project_source_target_values
from episodic_workflow.case_reports.research.workflow import calculation_leaf
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS
from . import test_target_values


class SelectedSourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_target_values.TargetValueTests()
        self.fixture.setUp()
        self.ledger = self.fixture.ledger
        self.native = self.fixture.fact("effective_radius_m", 1e-6)
        self.ledger.merge(ResearchResult(facts=(self.native,)))
        self.original = self.fixture.project().assumptions[0]
        self.calls = []
        self.fail_review = False
        self.verdict = "provisional"

    def complete(self, request):
        self.calls.append(request)
        if request["operation"] == "review_research_model":
            return {"admit": True, "reason": "Consistent conditional geometry."}
        self.assertEqual(request["operation"], "assess_assumption_reasonableness")
        if self.fail_review:
            raise ValueError("Fixture interrupted support review")
        refs = list(request["evidence"])
        return {"verdict": self.verdict,
            "criteria": {name: {"grade": 1, "reason": "Indirect applicable source evidence.",
                                "evidence": refs} for name in RUBRIC},
            "limitations": ["Target transfer remains conditional."], "alternatives": [],
            "missing_evidence": [], "explanation": "A bounded source-to-target transfer.",
            "request_constraint": None, "supporting_evidence": refs if self.verdict == "provisional" else [],
            "contradicting_evidence": refs if self.verdict == "contradicted" else [], "invalid_evidence": []}

    def source(self):
        return SimpleNamespace(run=SimpleNamespace(complete=self.complete, metrics=DEFAULT_METRICS),
            leaf=lambda callback, action: callback(),
            failure_result=lambda exc, stage: ResearchResult(evaluated=False,
                diagnostics=({"stage": stage, "status": "failed", "why": str(exc)},)))

    def calculate(self, subject="target-alpha"):
        statement = "Small spherical particles in the Stokes regime."
        condition = {"key": "condition:" + subject, "subject": subject, "field": "stokes_particles",
            "value": None, "unit": "condition", "statement": statement,
            "basis": "Explicit fixture premise.", "reasonableness": 1,
            "review": {"completed": True, "verdict": "supported"}, "role": "inferred_assumption"}
        condition["version"] = identity(condition)
        self.ledger.merge(ResearchResult(assumptions=(condition,)))
        return calculation_leaf(self.source(), {"action": "calculate_geometry_and_removal",
            "subject": subject, "metrics": [],
            "conditions": {"stokes_particles": statement}, "basis": "Selected automatic radius."},
            self.ledger, {"disease_of_interest": "fixture"})

    def reviews(self):
        return [row for row in self.calls if row["operation"] == "assess_assumption_reasonableness"]

    def test_selected_shared_value_review_keeps_identity_support_and_cache(self):
        result = self.calculate("target-beta")
        reviewed = self.ledger.assumptions[self.original["key"]]
        for field in ("key", "entry_type", "value", "statement", "source_fact_keys",
                      "source_candidate_group", "source_applicability", "derivation"):
            self.assertEqual(reviewed[field], self.original[field])
        self.assertEqual(assumption_claim(reviewed), assumption_claim(self.original))
        self.assertEqual(reviewed["reasonableness"], .5)
        calculation = result.calculations[-1]
        self.assertIn(self.original["key"], calculation["assumption_versions"])
        support = calculation["derived_support"]["settling_speed_m_s"]
        self.assertEqual(current_support(support, self.ledger.assumptions, ledger=self.ledger), .5)
        self.calculate("target-alpha")
        self.assertEqual(len(self.reviews()), 1)

    def test_failed_review_preserves_found_value_and_executes_arithmetic(self):
        before = scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)["recorded_target_inputs"]
        self.fail_review = True
        result = self.calculate("target-beta")
        self.assertTrue(any(f.field == "settling_speed_m_s" for f in result.facts))
        self.assertEqual(self.ledger.assumptions[self.original["key"]]["review"]["verdict"], "unassessed")
        self.assertTrue(any(d["stage"] == "selected_input_reasonableness_assessment" for d in result.diagnostics))
        after = scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)["recorded_target_inputs"]
        self.assertEqual(after, before)

    def test_grade_only_projection_preserves_review_without_another_assessment(self):
        self.calculate()
        reviewed = self.ledger.assumptions[self.original["key"]]["review"]
        relation = self.fixture.relation(self.native, .8)
        self.ledger.merge(ResearchResult(source_target_relationships=(relation,)))
        self.ledger.merge(project_source_target_values(self.ledger))
        candidate = self.ledger.assumptions[self.original["key"]]
        self.assertEqual(candidate["source_applicability"]["score"], .8)
        self.assertEqual(candidate["review"], reviewed)
        self.calculate()
        self.assertEqual(len(self.reviews()), 1)

    def test_derived_input_reviews_its_actual_automatic_ancestor(self):
        self.fail_review = True
        result = self.calculate()
        derived = next(f for f in result.facts if f.field == "settling_speed_m_s")
        self.fail_review = False
        updates = assess_selected_source_candidates(self.complete, self.ledger, (derived,), {},
            "target-alpha", {}, self.source().failure_result)
        self.assertEqual([a["key"] for a in updates.assumptions], [self.original["key"]])
        self.assertEqual(updates.assumptions[0]["reasonableness"], .5)

    def test_new_evidence_renews_support_review_but_own_version_does_not(self):
        self.calculate()
        self.ledger.merge(ResearchResult(facts=(self.fixture.fact("effective_radius_m", 2e-6,
            source="https://example.org/alternative"),)))
        self.calculate()
        self.calculate()
        self.assertEqual(len(self.reviews()), 2)

    def test_negative_review_keeps_shared_value_executable_without_support(self):
        self.verdict = "contradicted"
        before = scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)["recorded_target_inputs"]
        result = self.calculate("target-beta")
        self.assertTrue(any(f.field == "settling_speed_m_s" for f in result.facts))
        support = result.calculations[-1]["derived_support"]["settling_speed_m_s"]
        self.assertEqual(current_support(support, self.ledger.assumptions, ledger=self.ledger), 0)
        self.assertEqual(scientific_goal_status(self.ledger.result(), DEFAULT_METRICS)["recorded_target_inputs"], before)


if __name__ == "__main__":
    unittest.main()
