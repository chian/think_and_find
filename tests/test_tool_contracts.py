"""Decision contracts must describe the executor's actual accepted choices."""

import unittest

from episodic_workflow.case_reports.research.admission import extract_and_admit
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult
from episodic_workflow.case_reports.research.tools import GEOMETRY, LIMITS, execute_tool
from episodic_workflow.case_reports.research.workflow import GuidelineSource
from episodic_workflow.task_executors.scenario_metrics.catalog import METRICS


class ToolContractTests(unittest.TestCase):
    def test_metric_choices_are_scoped_to_the_selected_tool(self):
        run = ResearchRun(ResearchProviders(lambda _: None, lambda *_: (), lambda _: None),
                          ResearchLimits(), "Synthetic contract test", ())
        schema = GuidelineSource(run).proposal_schema()
        expected = {"calculate_geometry_and_removal": GEOMETRY,
                    "calculate_conditional_limits": LIMITS,
                    "calculate_transmission_metrics": set(METRICS) - GEOMETRY - LIMITS}
        for tool, choices in expected.items():
            with self.subTest(tool=tool):
                restrictions = [branch["then"]["properties"]["metrics"] for branch in schema["allOf"]
                    if branch["if"]["properties"].get("action", {}).get("const") == tool
                    and "metrics" in branch["then"].get("properties", {})]
                self.assertEqual(len(restrictions), 1)
                self.assertEqual(set(restrictions[0]["items"]["enum"]), choices)
                self.assertEqual(restrictions[0]["minItems"], 0 if tool == "calculate_geometry_and_removal" else 1)
        self.assertIn("Omit", schema["properties"]["metrics"]["description"])

    def test_invalid_metric_feedback_names_the_mismatch_and_actual_choices(self):
        ledger = Ledger(ResearchResult(
            facts=(Fact("fixture", "floor_area_m2", 40, "m2", "scenario_constraint", "Synthetic fixture input"),
                   Fact("fixture", "room_height_m", 2.5, "m", "scenario_constraint", "Synthetic fixture input")),
            scenarios=({"subject": "fixture", "description": "Synthetic test room",
                "fixed_parameters": {"floor_area_m2": {"value": 40, "unit": "m2"},
                                     "room_height_m": {"value": 2.5, "unit": "m"}}},)))
        with self.assertRaises(ValueError) as raised:
            execute_tool({"action": "calculate_geometry_and_removal", "subject": "fixture",
                "basis": "Synthetic mismatched request",
                "metrics": ["critical_distance_m"]}, ledger, (),
                review={"admit": True, "reason": "Fixture", "limitations": []})
        message = str(raised.exception)
        for term in ("calculate_geometry_and_removal", "critical_distance_m", "air_mixing_time_s"):
            self.assertIn(term, message)
        self.assertIn("no calculation was executed", message)

    def test_units_are_separate_from_variable_names_and_population_labels(self):
        examples = (("occupant_count", 17, "person", "Synthetic source: 17 attendees."),
                    ("mask_penetration", .4, "dimensionless", "Synthetic pm = 0.4 (dimensionless)."))
        for field, value, unit, quote in examples:
            with self.subTest(field=field):
                requests = []
                def agent(request):
                    requests.append(request)
                    if request["operation"] == "extract_research_evidence":
                        return {"facts": [{"subject": "fixture", "field": field, "value": value,
                            "unit": unit, "kind": "source_model_parameter", "quote": quote,
                            "context": "Synthetic population or variable label retained here",
                            "basis": "Software fixture, not empirical evidence"}], "evidence": [], "findings": []}
                    return {"decisions": [{"option": 0, "admit": True, "reason": "Synthetic source agrees"}]}
                result = extract_and_admit(agent, text=quote, source="https://example.test/fixture",
                    objective="Synthetic extraction test", context={}, question="Read the quantity")
                unit_contract = requests[0]["response_contract"]["properties"]["facts"]["items"]["properties"]["unit"]
                self.assertIn("accepted_source_units", unit_contract["description"])
                self.assertIn(field, requests[0]["accepted_source_units"])
                self.assertEqual(result.facts[0].value, value)
                self.assertEqual(result.facts[0].quote, quote)
                self.assertEqual(requests[1]["candidates"][0]["raw"]["unit"], unit)
