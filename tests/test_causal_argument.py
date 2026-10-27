"""Checks for causal-argument dependencies and incremental scientific credit."""

from __future__ import annotations

import copy
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from episodic_workflow.causal_argument import (
    argument_brief, argument_contract, argument_progress, argument_refs, normalize_argument,
)
from episodic_workflow.reversal_task import (
    COLUMNS, QUALITY_RUBRIC, RUBRIC, Run, material_vector,
)


def example_argument():
    def step(identity, role, claim, premises, refs):
        return {"id": identity, "role": role, "claim": claim,
                "premise_ids": premises, "warrant_refs": refs,
                "reasoning": "The stated evidence or premises support this scoped claim."}

    def frame(question, steps, conclusion):
        return {"question": question, "actors_and_categories": "Question-specific actors",
                "boundary": "Declared biological system", "intervention": "Declared action",
                "outcome": "Declared response", "scope": "Declared conditions",
                "steps": steps, "conclusion_step_id": conclusion,
                "pragmatic_use": "Answers this frame's biological question"}

    return {
        "frame_one": frame("Question A", [
            step("a1", "source_report", "Observed relation A", [], ["evidence:a"]),
            step("a2", "inference", "Scoped causal conclusion A", ["a1"], [])], "a2"),
        "frame_two": frame("Question B", [
            step("b1", "source_report", "Observed relation B", [], ["evidence:b"])], "b1"),
        "translation": {"source_step_id": "a2", "target_step_id": "b1",
                        "actor_map": "Proposed actor correspondence",
                        "intervention_map": "Proposed action correspondence",
                        "outcome_map": "Proposed outcome correspondence",
                        "relation_map": "Proposed causal relation correspondence",
                        "bridge_premise": "The relation survives the change in question",
                        "transported_inference": "Predicted A consequence in B",
                        "target_inference": "B's warranted consequence",
                        "failure_mechanism": "The proposed actor changes meaning",
                        "consequence": "A different intervention would be chosen",
                        "evidence_refs": ["evidence:a", "evidence:b"]},
        "fixed_category_alternative": {
            "fixed_vocabulary": "The same actors and intervention meanings",
            "added_context_or_dependencies": "A context variable C",
            "reconciliation": "Test whether C alone explains both inferences",
            "status": "challenged", "evidence_refs": ["evidence:b"]},
        "generalization": {
            "source_step_id": "a2", "rule": "Within domain D, intervention X changes Y",
            "premises": "X and Y have the stated causal meanings",
            "causal_conclusion": "Intervention X changes Y",
            "variable_meanings": "X is the intervention and Y its outcome",
            "proposed_domain": "Ecological systems with this interaction",
            "domain_warrant": "Source a motivates extending the rule across relevant systems",
            "abstraction_map": "Map the local actors into ecological roles",
            "domain_evidence_refs": ["evidence:a"],
            "supports": [{
                "id": "h1", "question": "How does X affect the system-level outcome?",
                "theory": "Theory H1", "consequence": "Prediction H1",
                "derivation": "G and its premises imply H1",
                "addition": {"kind": "none", "edits": [],
                             "formal_change": "No new variable or edge",
                             "preservation_argument": "The original prediction remains"},
                "distinct_consequence": "This is the first independently testable consequence",
                "relation_to_rule": "application",
                "relation_reasoning": "H1 answers a different question from G's source test",
                "status": "warranted", "status_reasoning": "Source a supports this consequence",
                "evidence_refs": ["evidence:a"]}],
            "target_domain_status": "supported",
            "target_domain_reasoning": "The proposed target has the same ecological roles",
            "target_domain_refs": ["evidence:a", "evidence:b"],
            "target_theory": "Independently supported theory H-star",
            "target_theory_refs": ["evidence:b"],
            "repair_tests": [{
                "id": "r1", "addition": {"kind": "context_variable",
                    "edits": [{"operation": "add_variable", "subject": "C",
                               "object": "Y", "expression": "Y=f(X,C)"}],
                    "formal_change": "Add context C to Y=f(X,C)",
                    "preservation_argument": "Check predictions for H1"},
                "target_prediction": "The repair still misses H-star",
                "prior_uses_after_repair": "The repair loses H1's prediction",
                "tested_support_ids": ["h1"],
                "independent_constraint": "C cannot be fixed without target fitting",
                "status": "fails", "evidence_refs": ["evidence:a", "evidence:b"]}]},
    }


def fit(**overrides):
    values = {name: {"score": 0.9} for name in (
        "first_frame_warrant", "second_frame_warrant", "translation_specificity",
        "inference_reversal", "category_relation_break", "fixed_category_challenge",
        "pragmatic_pluralism", "generalization_warrant", "target_domain_warrant",
        "target_theory_warrant", "repair_audit", "predictive_loss")}
    for name, value in overrides.items():
        values[name]["score"] = value
    return values


def ratings():
    return {"frame_one": {"a1": {"score": 0.9}, "a2": {"score": 0.8}},
            "frame_two": {"b1": {"score": 0.7}}}


def progress(argument, fit_values=None, step_values=None, support_values=None, repair_values=None):
    return argument_progress(argument,
        fit() if fit_values is None else fit_values,
        ratings() if step_values is None else step_values,
        {"h1": {"score": 0.9}} if support_values is None else support_values,
        {"r1": {"score": 0.9}} if repair_values is None else repair_values)


class CausalArgumentTests(unittest.TestCase):
    def test_inferential_bottleneck_and_unrelated_steps(self):
        argument = normalize_argument(example_argument(), {"evidence:a", "evidence:b"})
        baseline = progress(argument)
        self.assertAlmostEqual(baseline["score"],
            0.05 * 0.8 + 0.95 * (1 - math.exp(-0.8)))
        self.assertEqual(baseline["frame_one_strength"], 0.8)
        extended = copy.deepcopy(argument)
        extended["frame_one"]["steps"].append({
            "id": "a3", "role": "inference", "claim": "Unrelated assertion",
            "premise_ids": [], "warrant_refs": [], "reasoning": "Unrelated to the conclusion"})
        extended_ratings = ratings()
        extended_ratings["frame_one"]["a3"] = {"score": 0.01}
        self.assertEqual(progress(extended, step_values=extended_ratings)["score"],
                         baseline["score"])
        weak = ratings()
        weak["frame_one"]["a2"]["score"] = 0.2
        self.assertLess(progress(argument, step_values=weak)["score"], baseline["score"])

    def test_fixed_category_reconciliation_loses_witness_credit(self):
        argument = example_argument()
        complete = progress(argument)["score"]
        argument["fixed_category_alternative"]["status"] = "reconciles"
        near_miss = progress(argument)
        self.assertAlmostEqual(near_miss["score"],
            0.05 * 0.8 + 0.20 * (1 - math.exp(-0.8)))
        self.assertLess(near_miss["score"], complete)
        self.assertGreater(complete - near_miss["score"], near_miss["score"])
        argument["fixed_category_alternative"]["status"] = "unresolved"
        self.assertEqual(progress(argument)["challenge_strength"], 0)

    def test_domain_not_organism_identity_controls_challenge(self):
        argument = example_argument()
        self.assertGreater(progress(argument)["challenge_strength"], 0)
        argument["generalization"]["target_domain_status"] = "unsupported"
        outside = progress(argument)
        self.assertEqual(outside["challenge_strength"], 0)
        self.assertGreater(outside["support_breadth"], 0)
        argument["generalization"]["target_domain_status"] = "unresolved"
        self.assertEqual(progress(argument)["challenge_strength"], 0)
        argument["generalization"]["target_domain_status"] = "supported"
        argument["generalization"]["target_domain_refs"] = []
        self.assertEqual(progress(argument)["challenge_strength"], 0)

    def test_domain_warrant_scales_only_challenge(self):
        argument = example_argument()
        complete = progress(argument)["score"]
        uncertain = progress(argument, fit_values=fit(target_domain_warrant=0.4))
        self.assertGreater(uncertain["score"], 0)
        self.assertLess(uncertain["score"], complete)
        self.assertAlmostEqual(uncertain["challenge_strength"], 0.4)

    def test_distinct_supported_theory_increases_credit_but_duplicate_does_not(self):
        argument = example_argument()
        baseline = progress(argument)["score"]
        second = copy.deepcopy(argument["generalization"]["supports"][0])
        second.update(id="h2", theory="Theory H2", consequence="Prediction H2",
            derivation="G and dependency Z imply H2",
            distinct_consequence="H2 predicts a different intervention response")
        second["addition"] = {"kind": "dependency",
            "edits": [{"operation": "add_edge", "subject": "Z", "object": "Y",
                       "expression": "Z -> Y"}],
            "formal_change": "Add Z -> Y",
            "preservation_argument": "Z does not alter H1's tested prediction"}
        argument["generalization"]["supports"].append(second)
        with_new = progress(argument, support_values={
            "h1": {"score": 0.9}, "h2": {"score": 0.9}})["score"]
        self.assertGreater(with_new, baseline)
        second["consequence"] = "Prediction H1"
        duplicate = progress(argument, support_values={
            "h1": {"score": 0.9}, "h2": {"score": 0.9}})["score"]
        self.assertAlmostEqual(duplicate, baseline)
        second["consequence"] = "Prediction H2"
        second["theory"] = "Theory H1"
        same_theory = progress(argument, support_values={
            "h1": {"score": 0.9}, "h2": {"score": 0.9}})["score"]
        self.assertAlmostEqual(same_theory, baseline)

    def test_provisional_research_direction_is_not_supported_theory_credit(self):
        argument = example_argument()
        argument["generalization"]["supports"][0]["status"] = "provisional"
        result = progress(argument)
        self.assertEqual(result["support_breadth"], 0)
        self.assertEqual(result["challenge_strength"], 0)
        self.assertAlmostEqual(result["score"], 0.05 * result["generalization_strength"])

    def test_component_of_original_rule_is_not_reuse_credit(self):
        argument = example_argument()
        argument["generalization"]["supports"][0]["relation_to_rule"] = "component"
        result = progress(argument)
        self.assertEqual(result["support_breadth"], 0)
        self.assertEqual(result["challenge_strength"], 0)

    def test_viable_repair_or_unwarranted_target_has_no_large_bonus(self):
        argument = example_argument()
        argument["generalization"]["repair_tests"][0]["status"] = "works"
        self.assertEqual(progress(argument)["challenge_strength"], 0)
        argument["generalization"]["repair_tests"][0]["status"] = "fails"
        argument["generalization"]["target_theory_refs"] = []
        self.assertEqual(progress(argument)["challenge_strength"], 0)
        argument["generalization"]["target_theory_refs"] = ["evidence:b"]
        argument["generalization"]["repair_tests"][0]["status"] = "unresolved"
        self.assertEqual(progress(argument)["challenge_strength"], 0)

    def test_open_frame_gets_only_bounded_partial_credit(self):
        argument = example_argument()
        argument["frame_two"]["conclusion_step_id"] = ""
        result = progress(argument)
        self.assertEqual(result["challenge_strength"], 0)
        self.assertGreater(result["support_breadth"], 0)

    def test_forward_premises_and_unknown_citations_are_rejected(self):
        argument = example_argument()
        argument["frame_one"]["steps"][0]["premise_ids"] = ["a2"]
        with self.assertRaisesRegex(ValueError, "earlier steps"):
            normalize_argument(argument, {"evidence:a", "evidence:b"})
        argument = example_argument()
        argument["translation"]["evidence_refs"] = ["evidence:missing"]
        with self.assertRaisesRegex(ValueError, "available source claims"):
            normalize_argument(argument, {"evidence:a", "evidence:b"})

    def test_malformed_reference_types_raise_validation_error(self):
        argument = example_argument()
        argument["frame_one"]["steps"][1]["premise_ids"] = [["a1"]]
        with self.assertRaises(ValueError):
            normalize_argument(argument, {"evidence:a", "evidence:b"})
        argument = example_argument()
        argument["translation"]["evidence_refs"] = [["evidence:a"]]
        with self.assertRaises(ValueError):
            normalize_argument(argument, {"evidence:a", "evidence:b"})

    def test_generalization_is_required_and_source_linked(self):
        contract = argument_contract({"type": "string"})
        self.assertIn("generalization", contract["required"])
        argument = example_argument()
        argument["generalization"]["domain_evidence_refs"] = ["evidence:missing"]
        with self.assertRaisesRegex(ValueError, "generalization domain_evidence_refs"):
            normalize_argument(argument, {"evidence:a", "evidence:b"})

    def test_formal_addition_requires_typed_graph_edit(self):
        argument = example_argument()
        argument["generalization"]["repair_tests"][0]["addition"]["edits"] = []
        with self.assertRaisesRegex(ValueError, "graph edits matching"):
            normalize_argument(argument, {"evidence:a", "evidence:b"})

    def test_repair_must_link_to_existing_supported_theory(self):
        argument = example_argument()
        argument["generalization"]["repair_tests"][0]["tested_support_ids"] = ["missing"]
        with self.assertRaisesRegex(ValueError, "available supported theories"):
            normalize_argument(argument, {"evidence:a", "evidence:b"})

    def test_argument_citations_cover_every_proof_obligation(self):
        argument = example_argument()
        self.assertEqual(argument_refs(argument), ["evidence:a", "evidence:b"])

    def test_theory_handoff_keeps_dependency_path_and_omits_unrelated_step(self):
        argument = example_argument()
        argument["frame_one"]["steps"].append({
            "id": "a3", "role": "inference", "claim": "Unrelated assertion",
            "premise_ids": [], "warrant_refs": [], "reasoning": "Unrelated"})
        brief = argument_brief(argument)
        self.assertEqual([step["id"] for step in brief["frame_one"]["steps"]], ["a1", "a2"])
        self.assertEqual(brief["frame_one"]["omitted_step_ids"], ["a3"])
        self.assertEqual(brief["frame_one"]["steps"][0]["claim"],
                         argument["frame_one"]["steps"][0]["claim"])
        self.assertEqual(brief["translation"]["bridge_premise"],
                         argument["translation"]["bridge_premise"])

    def test_case_credit_uses_derived_argument_progress(self):
        material = {"epistemic_quality": {name: {"score": 1.0} for name in COLUMNS[:-1]},
                    "candidate_fit": fit(), "argument_progress": {"score": 0.3}}
        self.assertEqual(COLUMNS[-1], "argument_progress")
        self.assertEqual(material_vector(material), (1.0, 1.0, 1.0, 0.3))

    def test_review_wires_support_and_repair_warrants_into_credit(self):
        def measured(score=0.9):
            return {"score": score, "evidence_refs": ["evidence:a"],
                    "rationale": "The cited claim bears on this assessment"}
        case = {"facts": [{"id": "evidence:a"}, {"id": "evidence:b"}],
                "evidence_version": "v1", "current_answer": None,
                "materials": {}, "material_id": None,
                "inference_pair": {"generalizable_rule": "X changes Y across D",
                    "proposed_domain": "Domain D",
                    "domain_rationale": "Shared causal roles"}}
        run = object.__new__(Run)
        run.state = SimpleNamespace(case=lambda key: case)
        run.output = Path("/tmp/generalization-review")
        run.sequence = 1
        captured = {}
        def ask(operation, key, **kwargs):
            captured["contract"] = kwargs["response_contract"]
            captured["inquiry_generalization"] = kwargs["inquiry_generalization"]
            return {"equivalent_to": "new", "identity_rationale": "New argument",
                    "candidate_fit": {name: measured() for name in RUBRIC},
                    "epistemic_quality": {name: measured() for name in QUALITY_RUBRIC},
                    "step_warrants": {
                        "frame_one": {"a1": measured(), "a2": measured()},
                        "frame_two": {"b1": measured()}},
                    "support_warrants": {"h1": measured()},
                    "repair_warrants": {"r1": measured()},
                    "previous_measurements": None, "revision_summary": "Initial review"}
        run.ask = ask
        answer = {"conclusion": "Provisional generalization with a tested limit",
                  "comparison_notes": "Compare source and target theories",
                  "explanation": "The target loses the transported prediction",
                  "inference_pair": {
                      "frame_one_causal_inference": "X changes Y",
                      "frame_one_warrant_refs": ["evidence:a"],
                      "frame_two_causal_inference": "Target theory predicts Z",
                      "frame_two_warrant_refs": ["evidence:b"],
                      "proposed_translation": "Carry X to target",
                      "causal_tension": "Target predicts a different consequence",
                      "generalizable_rule": "X changes Y across domain D",
                      "proposed_domain": "Domain D includes both source and target",
                      "domain_rationale": "Shared causal roles warrant the scope"},
                  "argument": example_argument(), "evidence_refs": []}
        event = run.measure_answer("case_1", answer, "construct", "v1")
        contract = captured["contract"]["properties"]
        self.assertIn("h1", contract["support_warrants"]["properties"])
        self.assertIn("r1", contract["repair_warrants"]["properties"])
        self.assertEqual(captured["inquiry_generalization"]["proposed_domain"], "Domain D")
        self.assertGreater(event["review"]["argument_progress"]["challenge_strength"], 0)


if __name__ == "__main__":
    unittest.main()
