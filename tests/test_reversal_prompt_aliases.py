"""A response key can contain a temporary case label inside a candidate label."""

from __future__ import annotations

import json
import unittest

from episodic_workflow.reversal_prompts import _replace, render_reversal_request
from episodic_workflow.reversal_task import RUBRIC


class ReversalPromptAliasTests(unittest.TestCase):
    def test_case_identity_inside_candidate_label_round_trips_in_object_keys(self):
        case = "case_141e2c98159c05716514"
        candidate = "candidate_update_" + case
        aliases = {case: "ref_inquiry_4"}
        encoded = _replace({"identities": {candidate: {"same_case_as": case}}}, aliases)
        self.assertIn("candidate_update_ref_inquiry_4", encoded["identities"])
        decoded = _replace(encoded, {value: key for key, value in aliases.items()})
        self.assertEqual(decoded, {"identities": {candidate: {"same_case_as": case}}})

    def test_long_search_focus_uses_bounded_context(self):
        request = {"operation": "form_search", "research_question": "Biological question",
                   "acquisition_scope": "existing inquiry",
                   "focus": {"active_argument": "x" * 50000},
                   "response_contract": {"type": "object", "properties": {}}}
        rendered = render_reversal_request(request)
        self.assertLess(len(rendered.iteration_message), 48000)
        self.assertIn('context_compaction', rendered.iteration_message)
        self.assertIn('omitted text as uncertainty', rendered.instructions)

    def test_long_theory_review_keeps_case_identity_without_overflow(self):
        request = {"operation": "measure_theory_quality",
                   "research_question": "Biological question",
                   "source_claims": [{"id": "evidence:one", "claim": "x" * 12000}],
                   "reviewed_cases": [{"key": "case_12345678901234567890",
                                       "title": "Biological case",
                                       "current_answer": {"argument": {
                                           "generalization": {"rule": "x" * 30000}}}}],
                   "answer": {"conclusion": "x" * 30000,
                              "cross_case_mapping": "x" * 30000,
                              "case_refs": ["case_12345678901234567890"],
                              "evidence_refs": ["evidence:one"]},
                   "previous_answer": None, "known_materials": {},
                   "rubric": {"causal_nontransportability": "Biological test"},
                   "quality_rubric": {"source_fidelity": "Source fidelity"},
                   "response_contract": {"type": "object", "properties": {}}}
        rendered = render_reversal_request(request)
        self.assertLess(len(rendered.iteration_message), 48000)
        self.assertIn("ref_inquiry_1", rendered.iteration_message)
        self.assertIn("ref_evidence_1", rendered.iteration_message)
        self.assertIn('context_compaction', rendered.iteration_message)

    def test_search_prompt_uses_explicit_generalizability_domain(self):
        request = {"operation": "form_search", "research_question": "Biological question",
                   "acquisition_scope": "general discovery", "response_contract": {"type": "object"}}
        rendered = render_reversal_request(request)
        self.assertIn("nature of the proposed generalizability", rendered.instructions)
        self.assertIn("Search may cross organisms", rendered.instructions)
        self.assertIn("there is no required order", rendered.instructions)
        self.assertIn("natural phenomenon, biological cause or contrast", rendered.instructions)
        self.assertIn("Laboratory and engineered systems remain eligible", rendered.instructions)
        self.assertIn("model or algorithm names may help find evidence", rendered.instructions)

    def test_algorithmic_input_output_rule_has_no_biological_warrant(self):
        self.assertIn("Score 0 for an algorithm-only rule",
                      RUBRIC["generalization_warrant"])
        self.assertIn("supported mapping to the natural phenomenon",
                      RUBRIC["first_frame_warrant"])

    def test_long_review_keeps_supports_and_repairs_within_episode_bound(self):
        long_text = "Causal premise and biological qualification. " * 35
        references = [f"evidence:{index:024x}" for index in range(36)]
        facts = [{"id": ref, "source": "https://example.org/study",
                  "source_kind": "primary", "side": "mapping",
                  "claim": long_text, "quote": long_text}
                 for ref in references]
        argument = {"frame_one": {"question": long_text,
                    "steps": [{"id": "s1", "claim": long_text,
                               "reasoning": long_text,
                               "warrant_refs": [references[0]]}]},
                    "generalization": {
                        "rule": long_text, "proposed_domain": long_text,
                        "supports": [{"id": f"support_{index}", "theory": long_text,
                                      "derivation": long_text, "evidence_refs": [references[index]]}
                                     for index in range(4)],
                        "repair_tests": [{"id": f"repair_{index}",
                                          "formal_change": long_text,
                                          "evidence_refs": [references[index]]}
                                         for index in range(4)]}}
        answer = {"conclusion": long_text, "argument": argument,
                  "evidence_refs": references[:22]}
        request = {"operation": "measure_answer_quality",
                   "research_question": "Scope of a causal inference",
                   "source_claims": facts,
                   "inquiry_generalization": {"generalizable_rule": long_text,
                                              "proposed_domain": long_text},
                   "answer": answer, "previous_answer": None,
                   "known_materials": {},
                   "rubric": {"generalization_warrant": "Source-grounded rule"},
                   "quality_rubric": {"source_fidelity": "Faithful citation"},
                   "response_contract": {"type": "object", "properties": {}}}
        rendered = render_reversal_request(request)
        message = json.loads(rendered.iteration_message)
        self.assertLessEqual(len(rendered.iteration_message), 48000)
        self.assertEqual(len(message["proposed_answer"]["argument"]["generalization"]["supports"]), 4)
        self.assertEqual(len(message["proposed_answer"]["argument"]["generalization"]["repair_tests"]), 4)
        self.assertEqual(len(message["cited_source_claims"]), 22)
        request["previous_answer"] = answer
        request["known_materials"] = {"answer:previous": answer}
        revised = render_reversal_request(request)
        previous = json.loads(revised.iteration_message)["previous_answer"]
        self.assertLessEqual(len(revised.iteration_message), 48000)
        self.assertIn("frame_conclusions", previous)
        self.assertIn("Give generalization_warrant and first_frame_warrant zero",
                      rendered.instructions)
        self.assertIn("Do not penalize genuine scope evidence", rendered.instructions)


if __name__ == "__main__":
    unittest.main()
