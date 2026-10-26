"""Scenario scope is established before research and credits independent values."""

import json
import tempfile
import unittest
from pathlib import Path

from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.records import Ledger, ResearchResult, current_premises, identity
from episodic_workflow.case_reports.research.scenario_sharing import (
    independent_groups, initialize_parameter_scope,
)
from episodic_workflow.case_reports.research.scenarios import initial_scenarios, load_scenarios
from episodic_workflow.case_reports.research.tools import main_calculation_inputs, support_for
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS, UNITS


class ScenarioSharingTests(unittest.TestCase):
    def setUp(self):
        self.field = "breathing_flow_m3_s"
        def scenario(subject, description, ventilation):
            return {"subject": subject, "role": "target", "description": description,
                    "fixed_parameters": {"air_exchange_rate_s": {
                        "value": ventilation, "unit": UNITS["air_exchange_rate_s"]}},
                    "fixed_conditions": {"well_mixed": "Well mixed room"},
                    "assumption_needs": [self.field], "calculation_variables": []}
        self.scenarios = (
            scenario("case-alpha", "Adults resting; slow ventilation", .0001),
            scenario("case-beta", "Adults resting; fast ventilation", .002),
            scenario("case-gamma", "Adults exercising; slow ventilation", .0001),
        )
        self.assumption = {"key": "reviewed-value", "subject": "case-alpha", "field": self.field,
                           "value": .00015, "unit": UNITS[self.field],
                           "statement": "Resting adult breathing flow", "basis": "Reviewed source evidence",
                           "review": {"verdict": "provisional"}, "reasonableness": .5}
        self.assumption["version"] = identity(self.assumption)

    def scope(self):
        seen = []
        def assessor(request):
            seen.append(request)
            return {"features": [{"name": "activity", "basis": "Declared activity",
                "values": {"case-alpha": "rest", "case-beta": "rest", "case-gamma": "exercise"}}],
                "dependencies": [{"parameter": self.field, "factors": ["activity"],
                                  "reason": "Breathing flow changes with activity, not room ventilation."}]}
        initial = initialize_parameter_scope(ResearchResult(scenarios=self.scenarios), assessor,
            disease="synthetic disease")
        return initial, seen

    def test_preflight_groups_needs_before_any_value_and_credit_counts_groups(self):
        initial, seen = self.scope()
        self.assertEqual(len(seen), 1)
        self.assertEqual(len(seen[0]["scenarios"]), 3)
        self.assertIn("fixed:air_exchange_rate_s", [row["name"] for row in seen[0]["fixed_features"]])
        ledger = Ledger(initial)
        groups = independent_groups(ledger)
        self.assertEqual(len(groups), 2)
        self.assertIn(["case-alpha", "case-beta"], [group["subjects"] for group in groups])
        before = scientific_goal_status(ledger.result(), DEFAULT_METRICS)
        self.assertEqual(before["recorded_target_inputs"], {"recorded": 0, "required": 2})
        ledger.merge(ResearchResult(assumptions=(self.assumption,)))
        after = scientific_goal_status(ledger.result(), DEFAULT_METRICS)
        self.assertEqual(after["recorded_target_inputs"], {"recorded": 1, "required": 2})
        self.assertEqual(after["target_assumptions"], {"fulfilled": 1, "required": 2})
        self.assertGreater(after["weighted_score"], before["weighted_score"])
        self.assertEqual(len([a for a in ledger.assumptions.values() if a["field"] == self.field]), 1)
        shared = next(fact for fact in main_calculation_inputs(ledger, "case-beta")
                      if fact.subject == "case-beta" and fact.field == self.field)
        support, issues, invalidated = support_for(ledger, (shared,), {}, "case-beta")
        self.assertEqual((set(support), issues, invalidated), ({"reviewed-value"}, [], []))
        self.assertFalse(any(f.subject == "case-gamma" and f.field == self.field
                             for f in main_calculation_inputs(ledger, "case-gamma")))
        premise = {"assumption_versions": {self.assumption["key"]: self.assumption["version"]},
            "scenario_parameter_links": [{"assumption_key": self.assumption["key"],
                "relation_version": shared.parents[1], "target_subject": "case-beta"}]}
        self.assertTrue(current_premises(premise, ledger.assumptions, ledger=ledger))
        changed = {**self.scenarios[1], "description": "Children exercising"}
        ledger.merge(ResearchResult(scenarios=(changed,)))
        self.assertFalse(current_premises(premise, ledger.assumptions, ledger=ledger))
        self.assertFalse(any(f.subject == "case-beta" and f.field == self.field
                             for f in main_calculation_inputs(ledger, "case-beta")))

    def test_dependency_signature_is_transitive_across_multiple_irrelevant_changes(self):
        def assessor(request):
            return {"features": [{"name": "activity", "basis": "Declared activity",
                "values": {"case-alpha": "rest", "case-beta": "rest", "case-gamma": "exercise"}}],
                "dependencies": [{"parameter": self.field, "factors": [],
                                  "reason": "This hypothetical parameter is independent of declared changes."}]}
        initial = initialize_parameter_scope(ResearchResult(scenarios=self.scenarios), assessor)
        self.assertEqual(len(independent_groups(Ledger(initial))), 1)

    def test_custom_scenario_file_supplies_preflight_cases(self):
        records = [{"subject": name, "description": description,
                    "fixed": {"air_exchange_rate_s": {"value": ventilation,
                              "unit": UNITS["air_exchange_rate_s"]}},
                    "assumption_needs": [self.field], "variables": []}
                   for name, description, ventilation in (
                       ("trial-one", "People seated", .0001),
                       ("trial-two", "People seated with stronger ventilation", .002))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "caller-scenarios.json"
            path.write_text(json.dumps({"scenarios": records}))
            initial = initial_scenarios(load_scenarios(path))
        seen = []
        scoped = initialize_parameter_scope(initial, lambda request: seen.append(request) or
            {"features": [], "dependencies": [{"parameter": self.field,
                "factors": [], "reason": "Ventilation does not change this input."}]})
        self.assertEqual([row["subject"] for row in scoped.scenarios], ["trial-one", "trial-two"])
        self.assertEqual(set(seen[0]["scenarios"]), {"trial-one", "trial-two"})
        self.assertEqual(len(independent_groups(Ledger(scoped))), 1)


if __name__ == "__main__":
    unittest.main()
