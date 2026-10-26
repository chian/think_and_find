"""Source validity does not depend on the current investigation's assignment."""

import unittest
from episodic_workflow.case_reports.research.records import numeric_references

from episodic_workflow.case_reports.research.admission import extract_and_admit
from episodic_workflow.case_reports.research.records import Ledger


class NativeSourceAdmissionTests(unittest.TestCase):
    def extract(self, facts, *, invalid=()):
        requests = []
        context = {"parent_information_contract": {"measurements": {
            "parameter": "quanta_concentration_exhaled_m3",
            "subjects": ["quiet-speech-target"],
            "target_scenarios": {"quiet-speech-target": {
                "assumption_needs": ["quanta_concentration_exhaled_m3",
                                     "breathing_flow_m3_s", "mouth_area_m2"]}}}}}

        def agent(request):
            requests.append(request)
            if request["operation"] == "extract_research_evidence":
                self.assertEqual(request["context"], context)
                return {"facts": facts, "evidence": [], "findings": []}
            # The reviewer receives native source records, not the quiet-speech
            # assignment that caused the v47 activity-based rejections.
            self.assertEqual(request["context"]["review_scope"], "source_faithfulness_only")
            self.assertNotIn("parent_information_contract", request["context"])
            self.assertNotEqual(request["objective"], "Test quiet-speech infectiousness")
            return {"decisions": [{"option": c["option"],
                "admit": c["raw"]["basis"] not in invalid,
                "reason": "Incorrect native quantity" if c["raw"]["basis"] in invalid
                          else "Native value, unit, activity and modeled status match the source"}
                for c in request["candidates"]]}

        result = extract_and_admit(agent, text="\n".join(row["quote"] for row in facts),
            source="https://example.test/source", objective="Test quiet-speech infectiousness",
            question="Can this independently test the current concentration?", context=context)
        self.assertEqual(len(requests), 2)
        return result, requests

    @staticmethod
    def fact(field, value, unit, quote, context):
        return {"subject": "native-source-case", "field": field, "value": value,
            "unit": unit, "quote": quote, "context": context,
            "basis": "Faithful source input; target applicability remains to be assessed",
            "kind": "source_model_parameter"}

    def test_activity_specific_emissions_all_remain_selectable(self):
        quote = "Delta emission rates are 18, 69, 148, and 676 quanta/hour."
        facts = [self.fact("quanta_emission_rate_s", value, "quanta/h", quote, activity)
                 for value, activity in zip((18, 69, 148, 676),
                                           ("quiet", "speaking", "speaking loudly", "singing"))]
        result, _ = self.extract(facts)
        options = list(numeric_references(Ledger(result)).values())
        self.assertEqual(len(options), 4)
        self.assertEqual({f.context for f in options}, {f["context"] for f in facts})
        self.assertEqual({f.subject for f in options}, {"native-source-case"})
        self.assertEqual({f.kind for f in options}, {"source_model_parameter"})
        for value in (18, 69, 148, 676):
            self.assertTrue(any(abs(f.value - value / 3600) < 1e-12 for f in options))
        self.assertFalse(result.assumptions or result.diagnostics)

    def test_recipient_flows_and_other_target_fields_remain_numeric(self):
        facts = [
            self.fact("breathing_flow_m3_s", .0043, "m3/min",
                      "Sedentary recipient inhalation is 0.0043 m3/min.", "Recipient inhalation"),
            self.fact("breathing_flow_m3_s", .013, "m3/min",
                      "Speaking recipient inhalation is 0.013 m3/min.", "Recipient inhalation"),
            self.fact("breathing_flow_m3_s", 7500, "cm3/min",
                      "Sitting respiratory flow is 7500 cm3/min.", "Sitting model input"),
            self.fact("mouth_area_m2", 3, "cm2", "Mouth area is 3 cm2.", "Source geometry"),
        ]
        result, requests = self.extract(facts)
        self.assertEqual(len(result.facts), 4)
        self.assertFalse(result.evidence)
        self.assertEqual([f.field for f in result.facts], [row["field"] for row in facts])
        for fact, expected in zip(result.facts, (.0043 / 60, .013 / 60, .000125, .0003)):
            self.assertAlmostEqual(fact.value, expected)
            self.assertEqual(fact.subject, "native-source-case")
        self.assertIn("cm3/min", requests[0]["accepted_source_units"]["breathing_flow_m3_s"])

    def test_source_errors_still_excluded_without_a_target_relevance_test(self):
        valid = self.fact("quanta_emission_rate_s", 18, "quanta/h",
                          "The loud-speaking model uses 18 quanta/hour.", "Loud speaking")
        wrong_quantity = self.fact("quanta_concentration_exhaled_m3", 50, "quanta/m3",
                                   "RNA concentration is 50 copies/m3.", "Source RNA measurement")
        wrong_quantity["basis"] = "RNA relabeled as quanta"
        wrong_domain = {**valid, "value": -1}
        result, _ = self.extract([valid, wrong_quantity, wrong_domain], invalid=("RNA relabeled as quanta",))
        self.assertEqual(len(result.facts), 1)
        self.assertEqual(result.facts[0].field, "quanta_emission_rate_s")
        self.assertEqual({d["stage"] for d in result.diagnostics}, {"extraction", "applicability_review"})


if __name__ == "__main__":
    unittest.main()
