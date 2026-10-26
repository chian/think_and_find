"""Numerical failures produce explicit outcomes rather than advisory messages."""

import unittest

from episodic_workflow.case_reports.research.records import Ledger, ResearchResult
from episodic_workflow.task_executors.scenario_metrics.evaluate import evaluate_parameters, output_is_resolved
from episodic_workflow.tests.test_bazant_reference import CLASSROOM, CONDITIONS


class CalculationOutcomeTests(unittest.TestCase):
    def test_inconsistent_geometry_raises_before_returning_a_calculation(self):
        with self.assertRaisesRegex(ValueError, "Inconsistent room_volume_m3"):
            evaluate_parameters({"room_volume_m3": 100, "floor_area_m2": 20, "room_height_m": 3},
                                {}, (), subject="fixture", label="Contradictory geometry")

    def test_outside_jet_domain_is_not_a_calculated_distance(self):
        values = {**CLASSROOM, "air_exchange_rate_s": 8 / 3600, "mask_penetration": 1,
                  "quanta_concentration_exhaled_m3": 1e-6, "mouth_area_m2": .0004,
                  "entrainment_coefficient": .1, "plume_encounter_probability": .1}
        result = evaluate_parameters(values,
            {**CONDITIONS, "continuous_unmasked_jet": "Explicit synthetic jet assumption"},
            ("safe_distance_m",), subject="fixture", label="Outside model domain")
        row = result["metrics"]["safe_distance_m"]
        self.assertEqual(row["status"], "inapplicable")
        self.assertIsNone(row["value"])
        self.assertFalse(output_is_resolved(row))
        self.assertIn("0.1 m", row["applicability"]["reason"])
        self.assertNotIn("applicability_warning", row)
        self.assertNotIn("issues", result)

    def test_result_and_ledger_have_no_passive_issues_collection(self):
        result = ResearchResult()
        ledger = Ledger(result)
        self.assertFalse(hasattr(result, "issues"))
        self.assertFalse(hasattr(ledger, "issues"))
        self.assertNotIn("issues", result.as_record())
        self.assertNotIn("unresolved", result.synthesis())


if __name__ == "__main__":
    unittest.main()
