"""Report arithmetic freshness follows executed calculations and their lineage."""

import unittest

from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.records import Fact, ResearchResult
from episodic_workflow.case_reports.research.report import ResearchReport, scenario_results, scientific_progress
from episodic_workflow.case_reports.research.tools import execute_tool
from episodic_workflow.tests.test_calculation_steps import OUTPUT, add_assumption, calculate, fixture


def report_projection(ledger):
    result = ledger.result()
    goal = scientific_goal_status(result, (OUTPUT,))
    scenarios, _ = scenario_results(result, (OUTPUT,))
    # Rendering-only metadata; all scientific rows come from the real projections.
    report = ResearchReport(False, {
        "objective": "Offline calculation reporting fixture", "status": "partial",
        "interpretation": "Conditional synthetic calculation", "global_scientific_goal": goal,
        "scenarios": scenarios, "assumption_assessments": list(result.assumptions),
        "observation_comparisons": [],
        "validation": {"calibration": "No calibration in this fixture."},
        "paper_coverage": {"not_implemented": []},
        "control": {"reason": "offline test", "method": "deterministic execution"},
    })
    return goal, scenarios["target"]["coverage"][OUTPUT], report


class CalculationReportingTests(unittest.TestCase):
    def assert_projection(self, ledger, expected_status, expected_value):
        goal, coverage, report = report_projection(ledger)
        step = next(row for row in goal["calculation_steps"]["steps"] if row["field"] == OUTPUT)
        self.assertEqual(step["status"], expected_status)
        self.assertEqual(step["computed_value"], expected_value)
        self.assertEqual(coverage["calculation_status"], step["status"])
        self.assertEqual(coverage["computed"], expected_status == "calculated_up_to_date")
        self.assertEqual(coverage["recorded_computation"], expected_status != "not_calculated")
        progress = scientific_progress(ledger.result(), (OUTPUT,))
        self.assertEqual(progress["calculated_quantities"], int(coverage["computed"]))
        self.assertEqual(progress["recorded_calculated_quantities"], int(coverage["recorded_computation"]))
        self.assertNotIn("computed_outputs", goal)
        statuses = {"not_calculated", "calculated_up_to_date", "calculated_needs_updating"}
        self.assertLessEqual({row["status"] for row in goal["calculation_steps"]["steps"]}, statuses)
        self.assertEqual(sum(goal["calculation_steps"][status] for status in statuses),
                         len(goal["calculation_steps"]["steps"]))
        if coverage["computed"]:
            self.assertTrue(any(figure["value"] == expected_value for figure in coverage["figures"]))
        else:
            self.assertEqual(coverage["figures"], [])
        return step, report

    def test_report_and_global_table_agree_through_transitive_default_withdrawal(self):
        ledger = fixture()
        self.assert_projection(ledger, "not_calculated", None)
        old_volume = next(key for key, row in ledger.assumptions.items() if row["field"] == "room_volume_m3")
        del ledger.assumptions[old_volume]
        default = {"decision": "retain", "value": 100., "unit": "m3",
                   "basis": "Synthetic baseline", "provenance": "Offline fixture"}
        ledger.scenarios["target"]["model_defaults"] = {"room_volume_m3": default}
        ledger.merge(ResearchResult(facts=(Fact("target", "room_volume_m3", 100., "m3",
            "model_default", "Synthetic baseline"),)))
        action = {"action": "calculate_geometry_and_removal", "subject": "target",
                  "metrics": [OUTPUT], "basis": "Compute from supplied baseline",
                  "parameterization": "upstream baseline"}
        ledger.merge(execute_tool(action, ledger, (OUTPUT,),
                     review={"admit": True, "reason": "Compatible fixture"}))
        self.assert_projection(ledger, "calculated_up_to_date", 500.)
        ledger.merge(execute_tool({**action, "parameterization": "downstream reuse"},
                     ledger, (OUTPUT,), review={"admit": True, "reason": "Compatible reuse"}))
        self.assert_projection(ledger, "calculated_up_to_date", 500.)
        default["decision"] = "revisit"
        step, _ = self.assert_projection(ledger, "calculated_needs_updating", 500.)
        self.assertEqual(step["points"], .025)
        self.assertFalse(step["ready_to_recalculate"])
        reused = next(row for row in step["result_alternatives"]
                      if row["parameterization"] == "downstream reuse")
        self.assertTrue(any(change.get("field") == "room_volume_m3"
                            for change in reused["changed_dependencies"]))
        add_assumption(ledger, "room_volume_m3", 200.)
        calculate(ledger, label="recalculated alternative")
        step, _ = self.assert_projection(ledger, "calculated_up_to_date", 1000.)
        self.assertEqual(step["points"], .125)

    def test_markdown_has_one_progress_table_with_same_status_values_and_units(self):
        ledger = fixture()
        _, before = self.assert_projection(ledger, "not_calculated", None)
        calculate(ledger)
        _, current = self.assert_projection(ledger, "calculated_up_to_date", 500.)
        selected = next(row for row in ledger.assumptions.values() if row["field"] == "room_volume_m3")
        selected["review"] = {**selected["review"], "verdict": "contradicted"}
        _, stale = self.assert_projection(ledger, "calculated_needs_updating", 500.)
        for report, label, value in ((before, "not calculated", "—"),
                                    (current, "calculated & up to date", "500.0"),
                                    (stale, "calculated needs updating", "500.0")):
            with self.subTest(status=label):
                markdown = report.markdown()
                self.assertEqual(markdown.count("## Calculation Progress"), 1)
                self.assertEqual(markdown.count("| Target scenario | Calculation |"), 1)
                step = next(row for row in report.data["global_scientific_goal"]["calculation_steps"]["steps"]
                            if row["field"] == OUTPUT)
                self.assertIn(f"| {value} | {step['unit'] or '—'} | {label} |", markdown)
                self.assertNotIn("### Current Progress", markdown)

    def test_review_rejection_or_failure_does_not_change_arithmetic_freshness(self):
        for review in ({"admit": False, "reason": "Incompatible method"},
                       {"admit": False, "completed": False, "reason": "Review unavailable"}):
            with self.subTest(review=review):
                ledger = fixture()
                calculate(ledger, review=review)
                step, report = self.assert_projection(ledger, "calculated_up_to_date", 500.)
                self.assertEqual(step["points"], 0)
                self.assertFalse(step["accepted_current"])
                self.assertEqual(step["uncredited_reason"], review["reason"])
                self.assertIn(review["reason"], report.markdown())
                self.assertIn("| 500.0 | s | calculated & up to date | 0.000 |", report.markdown())


if __name__ == "__main__":
    unittest.main()
