"""Completed and failed operations use the real progress/serialization contract."""

from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
import json
import unittest
from unittest.mock import Mock, patch

from episodic_workflow.configured_providers import (
    WorkflowProviderConfig, firecrawl_argo_research_providers,
)
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchRun, ResearchSource
from episodic_workflow.case_reports.research.records import ResearchResult
from episodic_workflow.method_loop import Context, EpisodeGoal
from episodic_workflow.tests.test_source_value_handoff import initial_state
from episodic_workflow.tests.test_calculation_steps import OUTPUT, calculate, fixture
from episodic_workflow.case_reports.research.goal_state import scientific_goal_status


class OperationOutcomeTests(unittest.TestCase):
    def test_real_progress_callback_and_serialization_distinguish_operation_failure(self):
        config = WorkflowProviderConfig(
            firecrawl_api_key="synthetic", llm_base_url="https://example.test/llm",
            llm_model="synthetic", typesafe_api_key="synthetic")
        trace = []
        with patch("episodic_workflow.configured_providers._llm_client", return_value=Mock()), \
             patch("episodic_workflow.configured_providers.JevDecisionClient", return_value=Mock()), \
             patch("episodic_workflow.configured_providers.firecrawl_retrieve_callable", return_value=Mock()):
            providers = firecrawl_argo_research_providers(config, progress=True, trace_event=trace.append)
        initial = initial_state()
        run = ResearchRun(providers, ResearchLimits(), "Operation outcomes", (), initial=initial)
        source = ResearchSource(run, initial)
        episode = run.episode("evaluate_airborne_safety_guideline", source,
            goal=EpisodeGoal.root(objective="Exercise operation outcomes", result_contract={}), focus=())

        def fail():
            raise ValueError("Synthetic invalid operation")

        def propose(view):
            if view.units_consumed == 0:
                return source.leaf(fail, "invalid_operation")
            if view.units_consumed == 1:
                return source.leaf(lambda: ResearchResult(findings=("Completed examination",)), "completed_operation")
            return None

        source.propose = propose
        stderr = StringIO()
        with redirect_stderr(stderr):
            episode.run(Context(tree=run.tree))
        verdicts = [event for event in trace if event["event"] == "credit_verdict"]
        self.assertEqual([row["operation_status"] for row in verdicts], ["failed", "completed"])
        self.assertEqual([row["evaluated"] for row in verdicts], [False, True])
        self.assertEqual(verdicts[0]["diagnostics"][0]["status"], "failed")
        self.assertEqual(verdicts[0]["diagnostics"][0]["stage"], "invalid_operation")
        self.assertEqual(verdicts[0]["diagnostics"][0]["why"], "Synthetic invalid operation")
        self.assertIn("operation_status=failed", stderr.getvalue())
        self.assertIn("operation_status=completed", stderr.getvalue())
        result = source.ledger.result()
        serialized = json.loads(json.dumps(result.as_record()))
        self.assertNotIn("issues", serialized)
        self.assertFalse(hasattr(result, "issues"))
        self.assertFalse(hasattr(source.ledger, "issues"))
        self.assertTrue(serialized["diagnostics"])
        self.assertEqual(serialized["findings"], ["Completed examination"])
        self.assertTrue(all("issues" not in row for row in verdicts))
        ledger = fixture()
        calculate(ledger)
        status = scientific_goal_status(ledger.result(), (OUTPUT,))
        with redirect_stderr(stderr):
            providers.event({"unit": "calculation", "gain": .25, "stop": False,
                "evaluated": True, "model_calls": 0, "operation_status": "completed",
                "calculation_steps": status["calculation_steps"],
                "requested_outputs": status["requested_outputs"]})
        self.assertIn("outputs=1/1", stderr.getvalue())

    def test_source_division_by_zero_is_failed_operation_without_calculated_result(self):
        from episodic_workflow.case_reports.research.engine import ResearchProviders
        from episodic_workflow.case_reports.research.records import numeric_references
        from episodic_workflow.case_reports.research.workflow import calculation_leaf
        from episodic_workflow.tests.test_derived_target_values import fixture as source_fixture, SOURCE, FIELD
        from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS

        ledger, facts = source_fixture()
        ledger.facts.pop(facts[0].key)
        ledger.merge(ResearchResult(facts=(replace(facts[0], value=0),)))
        trace = []
        run = ResearchRun(ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None,
            trace_event=trace.append), ResearchLimits(), "Numerical failure", DEFAULT_METRICS,
            initial=ledger.result())
        source = ResearchSource(run, ledger.result())
        episode = run.episode("evaluate_airborne_safety_guideline", source,
            goal=EpisodeGoal.root(objective="Check failed arithmetic", result_contract={}), focus=())
        proposal = {"action": "calibrate_infectiousness", "operator": "cq_from_emission",
            "subject": SOURCE, "basis": "Synthetic zero flow",
            "source_inputs": [key for key, fact in numeric_references(ledger).items()
                              if fact.subject == SOURCE]}
        source.propose = lambda view: (calculation_leaf(source, proposal, ledger, {})
                                      if view.units_consumed == 0 else None)
        with patch("episodic_workflow.case_reports.research.workflow.review_model",
                   return_value={"admit": True, "reason": "Synthetic numerical test"}):
            episode.run(Context(tree=run.tree))
        verdict = next(event for event in trace if event["event"] == "credit_verdict")
        self.assertEqual(verdict["operation_status"], "failed")
        self.assertFalse(verdict["evaluated"])
        self.assertEqual(verdict["diagnostics"][0]["stage"], "calibrate_infectiousness")
        self.assertIn("division by zero", verdict["diagnostics"][0]["why"])
        self.assertFalse(source.ledger.calculations)
        self.assertFalse(any(fact.kind == "derived" for fact in source.ledger.facts.values()))
        self.assertFalse(any(row["found"] for row in run.global_goal.preview()["independent_target_values"]
                             if row["parameter"] == FIELD))


if __name__ == "__main__":
    unittest.main()
