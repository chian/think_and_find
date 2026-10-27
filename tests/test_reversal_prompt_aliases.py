"""A response key can contain a temporary case label inside a candidate label."""

from __future__ import annotations

import unittest

from episodic_workflow.reversal_prompts import _replace, render_reversal_request


class ReversalPromptAliasTests(unittest.TestCase):
    def test_case_identity_inside_candidate_label_round_trips_in_object_keys(self):
        case = "case_141e2c98159c05716514"
        candidate = "candidate_update_" + case
        aliases = {case: "ref_inquiry_4"}
        encoded = _replace({"identities": {candidate: {"same_case_as": case}}}, aliases)
        self.assertIn("candidate_update_ref_inquiry_4", encoded["identities"])
        decoded = _replace(encoded, {value: key for key, value in aliases.items()})
        self.assertEqual(decoded, {"identities": {candidate: {"same_case_as": case}}})

    def test_episode_context_overflow_is_explicit(self):
        request = {"operation": "form_search", "research_question": "Biological question",
                   "acquisition_scope": "existing inquiry",
                   "focus": {"active_argument": "x" * 50000},
                   "response_contract": {"type": "object", "properties": {}}}
        with self.assertRaisesRegex(ValueError, "Episode context exceeds"):
            render_reversal_request(request)

    def test_search_prompt_requires_one_focal_biological_system(self):
        request = {"operation": "form_search", "research_question": "Biological question",
                   "acquisition_scope": "general discovery", "response_contract": {"type": "object"}}
        rendered = render_reversal_request(request)
        self.assertIn("same focal organism", rendered.instructions)


if __name__ == "__main__":
    unittest.main()
