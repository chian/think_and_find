"""One registered claim and review cache for inferred and automatic values."""

import unittest
from episodic_workflow.case_reports.research.records import numeric_references

from episodic_workflow.case_reports.research.assumptions import (
    RUBRIC, assess_assumption, assess_registered_candidates, assumption_candidate,
    candidate_review_identity, register_candidate,
)
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, identity
from episodic_workflow.case_reports.research.target_values import project_source_target_values
from .test_source_value_handoff import initial_state


class CandidateReviewReuseTests(unittest.TestCase):
    def setUp(self):
        self.ledger = Ledger(initial_state())
        self.subject = next(subject for subject, row in self.ledger.scenarios.items()
                            if "mask_penetration" not in row["fixed_parameters"])
        facts = tuple(Fact("source-event", "mask_penetration", .5 + index / 1000,
            "fraction", "source_model_parameter", "Native fixture value",
            source="https://example.test/masks", quote=f"Mask input {index}",
            context=f"Source measurement context {index}") for index in range(61))
        self.ledger.merge(ResearchResult(facts=facts))
        self.calls = []
        self.candidate = {"subject": self.subject, "field": "mask_penetration", "value": .7,
            "unit": "fraction", "statement": "Use the same whole-mask factor in both directions.",
            "basis": "Conditional 30 percent reduction for each direction.", "entry_type": "best_guess",
            "source_refs": list(self.ledger.evidence_options())[:7], "input_refs": [],
            "added_assumptions": ["Inward and outward whole-mask factors are equal."]}

    def agent(self, request):
        self.calls.append(request)
        refs = list(request["evidence"])
        return {"verdict": "provisional", "request_constraint": None,
            "criteria": {name: {"grade": 1, "reason": "Indirect fixture evidence.",
                                "evidence": refs[:1]} for name in RUBRIC},
            "limitations": ["Directional symmetry remains an assumption."], "alternatives": [],
            "missing_evidence": [], "explanation": "A conditional estimate with indirect evidence.",
            "supporting_evidence": refs[:1], "contradicting_evidence": [], "invalid_evidence": []}

    def failure(self, error, stage):
        return ResearchResult(issues=(str(error),), diagnostics=({"stage": stage, "why": str(error)},))

    def register(self, candidate=None):
        result = register_candidate(candidate or self.candidate, self.ledger)
        self.ledger.merge(result)
        return result

    def review(self, key):
        result = assess_registered_candidates(self.agent, self.ledger, (key,), {}, self.failure)
        self.ledger.merge(result)
        return result

    def test_same_claim_and_61_evidence_records_reuse_review_despite_paraphrase(self):
        first = self.register().assumptions[0]
        self.review(first["key"])
        reviewed = self.ledger.assumptions[first["key"]]
        repeated = self.register({**self.candidate,
            "basis": "Each direction has a hypothetical reduction of thirty percent.",
            "added_assumptions": ["Assume equal effective inward and outward penetration."]})
        self.review(first["key"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(self.calls[0]["evidence"]), 61)
        self.assertEqual(self.ledger.assumptions[first["key"]], reviewed)
        self.assertEqual(repeated.assumptions[0]["basis"], self.candidate["basis"])
        self.assertEqual(repeated.assumptions[0]["derivation"]["added_assumptions"],
                         self.candidate["added_assumptions"])
        self.assertEqual(repeated.diagnostics[0]["category"], "registered_claim_retained")
        self.assertIn("material premise revision", repeated.diagnostics[0]["why"])
        self.assertTrue(reviewed["review"]["completed"])
        self.assertEqual(reviewed["review"]["assessment_basis_version"],
                         candidate_review_identity(reviewed, self.ledger))

    def test_legacy_manual_review_marker_is_migrated_without_another_judge(self):
        claim = assumption_candidate(self.candidate, self.ledger)
        original = assess_assumption(self.agent, {**claim,
            "evidence": list(self.ledger.evidence_options())}, self.ledger, {})
        assessment = original.assumptions[0]
        assessment["assessment_basis_version"] = identity([claim,
            list(self.ledger.evidence_options().values()), self.ledger.scenarios[self.subject]])
        self.assertNotIn("completed", assessment["review"])
        self.ledger.merge(original)
        self.register({**self.candidate, "basis": "A rewritten rationale."})
        self.review(assessment["key"])
        self.assertEqual(len(self.calls), 1)
        updated = self.ledger.assumptions[assessment["key"]]
        self.assertTrue(updated["review"]["completed"])
        self.assertNotIn("assessment_basis_version", updated)
        self.assertEqual(updated["review"]["assessment_basis_version"],
                         candidate_review_identity(updated, self.ledger))

    def test_new_evidence_input_claim_and_scope_each_require_current_review(self):
        key = self.register().assumptions[0]["key"]
        self.review(key)
        added = Fact("another-source", "mask_penetration", .6, "fraction", "source_model_parameter",
            "New source result", source="https://example.test/new", quote="New result",
            context="Another measured source context")
        self.ledger.merge(ResearchResult(facts=(added,)))
        self.review(key)
        self.assertEqual(len(self.calls), 2)
        reference = next(ref for ref, fact in numeric_references(self.ledger).items() if fact.key == added.key)
        changed_input = {**self.candidate, "input_refs": [reference],
                         "basis": "Infer from the newly identified operand."}
        self.register(changed_input)
        self.review(key)
        self.assertEqual(len(self.calls), 3)
        changed_claim = {**changed_input, "statement": "Use the factor only for the specified outward source model."}
        self.register(changed_claim)
        self.review(key)
        self.assertEqual(len(self.calls), 4)
        self.ledger.scenarios[self.subject] = {**self.ledger.scenarios[self.subject],
            "description": "A materially revised target activity."}
        self.register(changed_claim)
        self.review(key)
        self.assertEqual(len(self.calls), 5)
        self.review(key)
        self.assertEqual(len(self.calls), 5)

    def test_investigation_projects_the_same_inferred_entry_before_cached_review(self):
        from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun
        from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
        from episodic_workflow.case_reports.research.relevance import assess_numeric_target_relationships
        from episodic_workflow.case_reports.research.workflow import InvestigationSource
        from episodic_workflow.method_loop import EpisodeGoal
        from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS

        ledger = Ledger(initial_state())
        native = Fact("source-event", "mask_penetration", 1., "fraction", "source_model_parameter",
            "Published reference factor", source="https://example.test/source", quote="Factor is 1",
            context="Source reference mask setting")
        ledger.merge(ResearchResult(facts=(native,)))
        grades = []
        def grade(request):
            grades.append(request)
            value = request["source_profile"]["facts"][0]["value"]
            return {subject: .9 if value == .7 else .12 for subject in request["target_scenarios"]}
        ledger.merge(assess_numeric_target_relationships(ledger, grade))
        ledger.merge(project_source_target_values(ledger))
        proposal = {"candidates": [{**self.candidate, "source_refs": list(ledger.evidence_options())}]}
        def agent(request):
            if request["operation"] == "propose_investigation_strategy":
                return proposal
            self.assertEqual(request["operation"], "assess_assumption_reasonableness")
            return self.agent(request)
        run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None,
            source_target_assessor=grade), ResearchLimits(), "Review a mask alternative", DEFAULT_METRICS,
            initial=ledger.result())
        source = InvestigationSource(run, ledger.result())
        run.episode("investigate", source,
            goal=EpisodeGoal.root(objective="Estimate mask penetration", result_contract={}), focus=(),
            contract=source.contract("Resolve mask penetration", parameter="mask_penetration",
                subjects=(self.subject,), deliverable="assumption_hypotheses"))
        source.inputs = lambda view, **request: request
        source.leaf = lambda callback, label: callback()
        staged = source.propose(None)
        key = source.candidate_key(self.candidate)
        self.assertEqual(next(row for row in staged.assumptions if row["key"] == key)
                         ["review"]["verdict"], "unassessed")
        source.project(staged)
        source.ledger.merge(source.pending)
        run.global_goal.ledger.merge(source.pending_global)
        selected = scientific_goal_status(source.pending, DEFAULT_METRICS)["target_inputs"]            [self.subject]["mask_penetration"]
        self.assertEqual(selected["value"], .7)
        self.assertEqual(source.ledger.assumptions[key]["source_applicability"]["score"], .9)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(sum(row["field"] == "mask_penetration" and row["value"] == .7
                             for row in source.ledger.assumptions.values()), 1)
        grade_count = len(grades)
        proposal = {"candidates": [{**proposal["candidates"][0], "basis": "A paraphrased rationale.",
            "added_assumptions": ["Symmetric inward/outward effective penetration."]}]}
        source.project(source.propose(None))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(len(grades), grade_count)
        self.assertTrue(any(row.get("category") == "registered_claim_retained"
                            for row in source.pending.diagnostics))

    def test_selected_calculation_reuses_the_registered_inferred_review(self):
        from . import test_selected_source_review

        fixture = test_selected_source_review.SelectedSourceReviewTests()
        fixture.setUp()
        candidate = {"subject": "target-alpha", "field": "effective_radius_m", "value": 1e-6,
            "unit": "m", "statement": "A conditional effective aerosol radius.",
            "basis": "Estimate the radius from the native particle evidence.", "entry_type": "best_guess",
            "source_refs": list(fixture.ledger.evidence_options()), "input_refs": [],
            "added_assumptions": ["The source radius transfers to this target."]}
        registered = register_candidate(candidate, fixture.ledger)
        fixture.ledger.merge(registered)
        key = registered.assumptions[0]["key"]
        fixture.ledger.merge(assess_registered_candidates(fixture.complete, fixture.ledger,
            (key,), {}, fixture.source().failure_result))
        fixture.original = fixture.ledger.assumptions[key]
        fixture.original["source_applicability"] = {"score": .9}
        result = fixture.calculate()
        self.assertTrue(any(fact.field == "settling_speed_m_s" for fact in result.facts))
        self.assertIn(key, result.calculations[-1]["assumption_versions"])
        self.assertEqual(len(fixture.reviews()), 1)
        fixture.calculate()
        self.assertEqual(len(fixture.reviews()), 1)

    def test_automatic_and_inferred_entries_share_cache_and_preserve_relevance(self):
        inferred = self.register().assumptions[0]
        self.ledger.merge(project_source_target_values(self.ledger))
        automatic = next(row for row in self.ledger.assumptions.values()
                         if row.get("entry_type") == "source_candidate" and row["field"] == "mask_penetration")
        keys = (inferred["key"], automatic["key"])
        for key in keys:
            entry = {**self.ledger.assumptions[key], "source_applicability": {"score": .4}}
            self.ledger.merge(ResearchResult(assumptions=(entry,)))
            self.review(key)
            self.assertEqual(self.ledger.assumptions[key]["source_applicability"], {"score": .4})
        self.assertEqual(len(self.calls), 2)
        for key in keys:
            entry = {**self.ledger.assumptions[key], "source_applicability": {"score": .9}}
            self.ledger.merge(ResearchResult(assumptions=(entry,)))
            self.review(key)
            self.assertTrue(self.ledger.assumptions[key]["review"]["completed"])
        self.assertEqual(len(self.calls), 2, "Relevance refreshes do not repeat scientific review")


if __name__ == "__main__":
    unittest.main()
