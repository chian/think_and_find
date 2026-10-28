"""Transport outages wait outside the scientific Episode controller."""

import unittest
from unittest.mock import patch

from episodic_workflow.reversal_task import Run, provider_retry_delay
from episodic_workflow.task_executors._http import ProviderError
from episodic_workflow.task_executors.evidence_extraction.argo import ModelOutputError


class ReversalProviderBackoffTests(unittest.TestCase):
    def run_with_agent(self, agent):
        run = Run.__new__(Run)
        run.question = "Synthetic biological question"
        run.providers = {"agent": agent}
        run.operation_attempts = {}
        run.operation_successes = {}
        run.failure_streaks = {}
        run.unavailable = None
        events = []
        run.record = events.append
        return run, events

    def test_transient_outage_retries_same_operation_without_episode_failure(self):
        responses = iter((ProviderError("Provider returned HTTP 502"),
                          ProviderError("Provider connection failed: timed out"),
                          ProviderError("Provider returned HTTP 503"),
                          {"accepted": True}))

        def agent(request, *, tier):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

        run, events = self.run_with_agent(agent)
        with patch("episodic_workflow.reversal_task.time.sleep") as sleep:
            answer = run.ask("extract_claims_and_discover", "case_a")
        self.assertEqual(answer, {"accepted": True})
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10, 20])
        self.assertEqual([event["delay_seconds"] for event in events], [5, 10, 20])
        self.assertEqual(run.operation_attempts["extract_claims_and_discover"], 4)
        self.assertEqual(run.operation_successes["extract_claims_and_discover"], 1)
        self.assertEqual(run.failure_streaks["extract_claims_and_discover"], 0)
        self.assertIsNone(run.unavailable)

    def test_nonretryable_errors_surface_without_sleep(self):
        for error in (ProviderError("Provider returned HTTP 400"),
                      ModelOutputError("LLM output was not valid JSON", category="malformed_json",
                                       raw_response="{")):
            with self.subTest(error=str(error)):
                def agent(request, *, tier):
                    raise error

                run, events = self.run_with_agent(agent)
                with patch("episodic_workflow.reversal_task.time.sleep") as sleep:
                    with self.assertRaises(type(error)):
                        run.ask("extract_claims_and_discover", "case_a")
                sleep.assert_not_called()
                self.assertEqual(events, [])
                self.assertEqual(run.operation_attempts["extract_claims_and_discover"], 1)

    def test_backoff_caps_at_five_minutes(self):
        self.assertEqual([provider_retry_delay(n) for n in range(1, 9)],
                         [5, 10, 20, 40, 80, 160, 300, 300])


if __name__ == "__main__":
    unittest.main()
