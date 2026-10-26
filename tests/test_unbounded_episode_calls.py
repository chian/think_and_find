"""Research episodes stop through yield or source failure, not call quotas."""

from types import SimpleNamespace
import unittest

from episodic_workflow.case_reports.assumption_investigation import AssumptionInvestigation, AssumptionSource
from episodic_workflow.case_reports.research.engine import (
    ResearchLimits, ResearchProviders, ResearchRun, ResearchSource,
)


class UnboundedEpisodeCallsTests(unittest.TestCase):
    def test_research_agent_continues_past_old_call_threshold(self):
        run = ResearchRun(
            ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None),
            ResearchLimits(), "Synthetic call-continuation check", ("mouth_area_m2",),
        )
        for _ in range(300):
            self.assertEqual(run.agent({"operation": "synthetic_probe"}), {})
        self.assertEqual(run.model_calls, 300)

    def test_research_source_does_not_stop_at_old_global_thresholds(self):
        run = ResearchRun(
            ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None),
            ResearchLimits(), "Synthetic unit-continuation check", ("mouth_area_m2",),
        )
        run.model_calls = run.total_units = 300
        source = ResearchSource(run)
        selected = object()
        source.propose = lambda _: selected
        self.assertIs(source.next(SimpleNamespace(units_consumed=300)), selected)
        self.assertEqual(source.failure, "")

    def test_assumption_agent_continues_past_old_call_threshold(self):
        branch = AssumptionInvestigation(
            fields=("mouth_area_m2",), context={}, llm=lambda _: {},
        )
        for _ in range(50):
            self.assertEqual(branch.agent({"operation": "synthetic_probe"}), {})
        self.assertEqual(branch.calls, 50)

        source = AssumptionSource(branch, ())
        selected = object()
        source.propose = lambda _: selected
        self.assertIs(source.next(SimpleNamespace(units_consumed=50)), selected)


if __name__ == "__main__":
    unittest.main()
