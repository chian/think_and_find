"""Checks for causal-argument dependencies and incremental scientific credit."""

from __future__ import annotations

import copy
import unittest

from episodic_workflow.causal_argument import (
    argument_brief, argument_contract, argument_progress, argument_refs, normalize_argument,
)
from episodic_workflow.reversal_task import COLUMNS, material_vector


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
        "shared_system": {
            "focal_system": "One experimentally matched biological system",
            "organisms_frame_one": ["Species alpha"],
            "organisms_frame_two": ["Species alpha"],
            "overlap_reasoning": "Both frames concern the same focal population",
            "status": "same", "evidence_refs": ["evidence:a", "evidence:b"]},
    }


def fit(**overrides):
    values = {name: {"score": 0.9} for name in (
        "first_frame_warrant", "second_frame_warrant", "translation_specificity",
        "inference_reversal", "category_relation_break", "fixed_category_challenge",
        "pragmatic_pluralism", "same_system_warrant")}
    values["same_system_warrant"]["score"] = 1.0
    for name, value in overrides.items():
        values[name]["score"] = value
    return values


def ratings():
    return {"frame_one": {"a1": {"score": 0.9}, "a2": {"score": 0.8}},
            "frame_two": {"b1": {"score": 0.7}}}


class CausalArgumentTests(unittest.TestCase):
    def test_inferential_bottleneck_and_unrelated_steps(self):
        argument = normalize_argument(example_argument(), {"evidence:a", "evidence:b"})
        baseline = argument_progress(argument, fit(), ratings())
        self.assertAlmostEqual(baseline["score"], 0.71)
        self.assertEqual(baseline["frame_one_strength"], 0.8)
        extended = copy.deepcopy(argument)
        extended["frame_one"]["steps"].append({
            "id": "a3", "role": "inference", "claim": "Unrelated assertion",
            "premise_ids": [], "warrant_refs": [], "reasoning": "Unrelated to the conclusion"})
        extended_ratings = ratings()
        extended_ratings["frame_one"]["a3"] = {"score": 0.01}
        self.assertEqual(argument_progress(extended, fit(), extended_ratings)["score"],
                         baseline["score"])
        weak = ratings()
        weak["frame_one"]["a2"]["score"] = 0.2
        self.assertLess(argument_progress(argument, fit(), weak)["score"], baseline["score"])

    def test_fixed_category_reconciliation_loses_witness_credit(self):
        argument = example_argument()
        complete = argument_progress(argument, fit(), ratings())["score"]
        argument["fixed_category_alternative"]["status"] = "reconciles"
        near_miss = argument_progress(argument, fit(), ratings())
        self.assertAlmostEqual(near_miss["score"], 0.15)
        self.assertLess(near_miss["score"], complete)

    def test_organism_switch_gets_no_case_credit(self):
        argument = example_argument()
        argument["shared_system"]["organisms_frame_two"] = ["Species beta"]
        self.assertEqual(argument_progress(argument, fit(), ratings())["score"], 0)
        argument["shared_system"]["organisms_frame_two"] = ["Species alpha"]
        argument["shared_system"]["status"] = "different"
        self.assertEqual(argument_progress(argument, fit(), ratings())["score"], 0)
        argument["shared_system"]["status"] = "unresolved"
        self.assertEqual(argument_progress(argument, fit(), ratings())["score"], 0)
        argument["shared_system"]["status"] = "same"
        argument["shared_system"]["evidence_refs"] = []
        self.assertEqual(argument_progress(argument, fit(), ratings())["score"], 0)

    def test_same_system_warrant_scales_credit(self):
        argument = example_argument()
        complete = argument_progress(argument, fit(), ratings())["score"]
        uncertain = argument_progress(argument, fit(same_system_warrant=0.4), ratings())
        self.assertAlmostEqual(uncertain["score"], 0.4 * complete)

    def test_open_frame_gets_only_bounded_partial_credit(self):
        argument = example_argument()
        argument["frame_two"]["conclusion_step_id"] = ""
        progress = argument_progress(argument, fit(), ratings())
        self.assertAlmostEqual(progress["score"], 0.08)

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

    def test_shared_system_is_required_and_source_linked(self):
        contract = argument_contract({"type": "string"})
        self.assertIn("shared_system", contract["required"])
        argument = example_argument()
        argument["shared_system"]["evidence_refs"] = ["evidence:missing"]
        with self.assertRaisesRegex(ValueError, "shared system references"):
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


if __name__ == "__main__":
    unittest.main()
