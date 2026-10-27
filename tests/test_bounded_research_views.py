"""Whole-record selection keeps long-running research requests bounded."""

from __future__ import annotations

import json
import unittest

from episodic_workflow.reversal_task import select_identity_catalog
from episodic_workflow.reversal_prompts import _select


def anchor(key, topic):
    return {"key": key, "theme_id": "theme_" + key, "title": topic,
            "topic": topic, "causal_structure": topic + " actors and relations",
            "generalization_defect": topic + " intervention translation",
            "frame_one_question": topic + " cell question",
            "frame_two_question": topic + " tissue question",
            "inference_pair": {
                "frame_one_causal_inference": topic + " cell inference",
                "frame_two_causal_inference": topic + " tissue inference",
                "proposed_translation": topic + " translation",
                "causal_tension": topic + " tension"},
            "occurrence_count": 1}


class BoundedViewTests(unittest.TestCase):
    def test_identity_view_prioritizes_relevant_whole_rows_and_direct_reference(self):
        rows = [anchor("case_a", "neutrophil scaffold"),
                anchor("case_b", "epithelial wound migration"),
                anchor("case_c", "biofilm matrix")]
        candidates = [{"title": "biofilm matrix architecture",
                       "same_case_as": "case_a"}]
        budget = len(json.dumps(rows[:2], ensure_ascii=False))
        selected, omitted = select_identity_catalog(rows, candidates, budget)
        self.assertLessEqual(len(json.dumps(selected, ensure_ascii=False)), budget)
        self.assertEqual(selected[0]["key"], "case_a")
        self.assertIn("case_c", [item["key"] for item in selected])
        self.assertEqual(set(omitted), {"case_b"})

    def test_previous_theory_is_compact_for_measurement(self):
        answer = {"conclusion": "Scoped conclusion", "cross_case_mapping": "Exact map",
                  "explanation": "x" * 10000, "case_refs": ["case_a"],
                  "evidence_refs": ["evidence:a"],
                  "uncertainties": [{"question": "Long uncertainty"}],
                  "next_frame_gap": "Gap", "next_evidence_need": "Need",
                  "avoid_pattern": "Pattern"}
        request = {"operation": "measure_theory_quality", "source_claims": [],
                   "reviewed_cases": [], "answer": answer,
                   "previous_answer": answer, "known_materials": {}}
        selected = _select(request)
        self.assertEqual(selected["proposed_account"]["explanation"], "x" * 10000)
        self.assertNotIn("explanation", selected["previous_account"])
        self.assertEqual(selected["previous_account"]["cross_case_mapping"], "Exact map")


if __name__ == "__main__":
    unittest.main()
