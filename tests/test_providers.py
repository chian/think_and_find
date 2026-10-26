"""Contract tests for repository-owned provider boundaries."""

from __future__ import annotations

import asyncio
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.error import HTTPError, URLError

from episodic_workflow import (
    ProviderConfigurationError,
    firecrawl_argo_config_from_env,
    run_configured_evaluate_safety_guideline_episode,
)
from episodic_workflow.case_reports.safety_guideline import GUIDELINE_OUTPUTS
from episodic_workflow.cli import main
from episodic_workflow.environment import workflow_environment
from episodic_workflow.method_loop import EpisodeGoal
from episodic_workflow.method_loop.episode_binding.web_search_binding import assess_page_candidate
from episodic_workflow.task_executors._http import ProviderError, post_json
from episodic_workflow.task_executors.evidence_extraction import ArgoClient, argo_extract
from episodic_workflow.task_executors.evidence_extraction.argo import ModelOutputError, parse_json_object
from episodic_workflow.task_executors.outbreak_search import bound_search, firecrawl_search, firecrawl_scrape
from episodic_workflow.task_executors.outbreak_search.jev import JevDecisionClient
from episodic_workflow.tests.fixtures import PROVIDER_ENV, SCENARIO, SOURCE_TEXT, SOURCE_URL, ProviderTransport


def json_response(body):
    return io.BytesIO(json.dumps(body).encode("utf-8"))


async def inline_transport(function, /, *args, **kwargs):
    return function(*args, **kwargs)


class InMemoryTransportTests(unittest.TestCase):
    def setUp(self):
        # The transport is mocked; hermetic tests need no worker wakeup sockets.
        patcher = patch("asyncio.to_thread", side_effect=inline_transport)
        patcher.start()
        self.addCleanup(patcher.stop)


class ProviderHTTPTests(InMemoryTransportTests):
    def test_search_and_scrape_use_local_http_client(self):
        transport = ProviderTransport()
        with patch("episodic_workflow.task_executors._http.urlopen", transport):
            results = firecrawl_search("outbreak", api_key="test-key", limit=2)
            scraped = firecrawl_scrape(SOURCE_URL, api_key="test-key")
        self.assertEqual(results[0]["markdown"], SOURCE_TEXT)
        self.assertEqual(results[0]["retrieval"]["mode"], "search")
        self.assertEqual(scraped["retrieval"]["url"], SOURCE_URL)
        self.assertEqual(transport.calls[0][1]["limit"], 2)
        self.assertEqual(transport.calls[0][2]["Authorization"], "Bearer test-key")

    def test_legacy_search_list_response(self):
        with patch("episodic_workflow.task_executors._http.urlopen", return_value=json_response({"data": [{"url": SOURCE_URL}]})):
            result = firecrawl_search("query", api_key="test-key", limit=1, scrape_results=False)
        self.assertEqual(result[0]["url"], SOURCE_URL)
        self.assertNotIn("retrieval", result[0])

    def test_invalid_search_limits_fail_before_http(self):
        with patch("episodic_workflow.task_executors._http.urlopen") as request:
            for limit in (0, -1, True, 1.5, 101):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    firecrawl_search("query", api_key="test-key", limit=limit)
            request.assert_not_called()

    def test_malformed_search_responses_fail_closed(self):
        for data in (None, {}, {"web": "not a list"}, {"web": [1]}):
            with self.subTest(data=data), patch("episodic_workflow.task_executors._http.urlopen", return_value=json_response({"data": data})), self.assertRaises(ProviderError):
                firecrawl_search("query", api_key="test-key", limit=1)

    def test_scrape_preview_is_not_full_text(self):
        with patch("episodic_workflow.task_executors._http.urlopen", return_value=json_response({"data": {"description": "preview"}})), self.assertRaises(ProviderError):
            firecrawl_scrape(SOURCE_URL, api_key="test-key")

    def test_http_and_network_errors_are_explicit(self):
        for error, message in ((HTTPError("https://api.example", 401, "Unauthorized", {}, None), "HTTP 401"), (URLError("offline"), "connection failed")):
            with self.subTest(error=error), patch("episodic_workflow.task_executors._http.urlopen", side_effect=error), self.assertRaisesRegex(ProviderError, message):
                post_json("https://api.example", {})

    def test_bad_json_and_non_object_bodies_fail(self):
        for raw in (b"not json", b"\xff", b"[]", b"null"):
            with self.subTest(raw=raw), patch("episodic_workflow.task_executors._http.urlopen", return_value=io.BytesIO(raw)), self.assertRaises(ProviderError):
                post_json("https://api.example", {})

    def test_provider_error_redacts_credential(self):
        with patch("episodic_workflow.task_executors._http.urlopen", return_value=json_response({"success": False, "error": "rejected test-secret"})), self.assertRaises(ProviderError) as caught:
            post_json("https://api.example", {}, api_key="test-secret")
        self.assertNotIn("test-secret", str(caught.exception))

    def test_http_error_preserves_reason_without_credentials_or_echoes(self):
        body = {"error": {"message": "Invalid request for test-secret", "type": "invalid_request_error",
                          "code": "invalid_parameter", "param": "temperature",
                          "request": {"Authorization": "Bearer test-secret", "prompt": "private prompt"}}}
        stream = json_response(body)
        error = HTTPError("https://api.example", 400, "Bad Request", {}, stream)
        with patch("episodic_workflow.task_executors._http.urlopen", side_effect=error), self.assertRaises(ProviderError) as caught:
            post_json("https://api.example", {}, api_key="test-secret")
        message = str(caught.exception)
        self.assertIn("HTTP 400", message)
        self.assertIn("invalid_parameter", message)
        self.assertIn("temperature", message)
        self.assertIn("[redacted]", message)
        self.assertNotIn("test-secret", message)
        self.assertNotIn("private prompt", message)
        self.assertNotIn("Authorization", message)
        self.assertTrue(stream.closed)

    def test_unrecognized_or_oversized_http_error_body_keeps_status(self):
        bodies = (b"<html>private proxy error</html>", b"[]", b"\xff",
                  json.dumps({"error": {"message": "x" * 9000}}).encode(),
                  b'{"error":{"request":"private prompt"}}')
        for raw in bodies:
            stream = io.BytesIO(raw)
            error = HTTPError("https://api.example", 400, "Bad Request", {}, stream)
            with self.subTest(raw=raw[:40]), patch("episodic_workflow.task_executors._http.urlopen", side_effect=error), self.assertRaises(ProviderError) as caught:
                post_json("https://api.example", {})
            self.assertEqual(str(caught.exception), "Provider returned HTTP 400")
            self.assertTrue(stream.closed)

    def test_http_error_detail_is_bounded_after_redaction(self):
        body = {"error": "x" * 510 + "test-secret" + "y" * 1000}
        error = HTTPError("https://api.example", 400, "Bad Request", {}, json_response(body))
        with patch("episodic_workflow.task_executors._http.urlopen", side_effect=error), self.assertRaises(ProviderError) as caught:
            post_json("https://api.example", {}, api_key="test-secret")
        self.assertNotIn("test-secret", str(caught.exception))
        self.assertLess(len(str(caught.exception)), 600)

    def test_argo_payload_and_schema(self):
        transport = ProviderTransport()
        client = ArgoClient("https://llm.example/v1", "test-model", user="test-user")
        request = {"source": {"source_ref": SOURCE_URL}, "required_fields": ["attendee_count"], "assumption_admission": {"required": True}}
        with patch("episodic_workflow.task_executors._http.urlopen", transport):
            result = argo_extract(client, request)
        payload = transport.calls[0][1]
        self.assertEqual(payload["user"], "test-user")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn("response_json_schema", payload["messages"][1]["content"])
        self.assertEqual(result["values"][0]["value"], 11)
        self.assertNotIn("Authorization", transport.calls[0][2])

    def test_argo_rejects_incomplete_or_malformed_content(self):
        bodies = ({}, {"choices": []}, {"choices": [{"message": {"content": None}}]}, {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]})
        for body in bodies:
            with self.subTest(body=body), patch("episodic_workflow.task_executors._http.urlopen", return_value=json_response(body)), self.assertRaises(ProviderError):
                ArgoClient("https://llm.example/v1", "test-model").call("prompt")

    def test_malformed_json_preserves_original_and_parser_location(self):
        raw = '{"facts": [{"value": 30}],\n "findings": ["potentially useful"],}'
        with self.assertRaises(ModelOutputError) as caught:
            parse_json_object(raw)
        error = caught.exception
        self.assertEqual(error.raw_response, raw)
        self.assertEqual(error.diagnostic["category"], "malformed_json")
        self.assertEqual(error.diagnostic["line"], 2)
        self.assertIn("property name", error.diagnostic["parser_message"])
        self.assertNotIn("potentially useful", str(error))
        json.dumps(error.diagnostic, allow_nan=False)

    def test_json_rejection_categories_do_not_silently_repair_content(self):
        examples = {
            "[]": "non_object_json",
            '{"value": NaN}': "nonfinite_json_number",
            '{"value": Infinity}': "nonfinite_json_number",
            '{"value": 1e999}': "nonfinite_json_number",
            '{"value": 30, "value": 40}': "duplicate_json_key",
            '```json\n{"value":30}\n```': "malformed_json",
            r'{"findings":["Use \lambda"]}': "malformed_json",
        }
        for raw, category in examples.items():
            with self.subTest(raw=raw), self.assertRaises(ModelOutputError) as caught:
                parse_json_object(raw)
            self.assertEqual(caught.exception.diagnostic["category"], category)
            self.assertEqual(caught.exception.raw_response, raw)
        self.assertEqual(parse_json_object('{"value": 30, "missing": null}'), {"value": 30, "missing": None})
        self.assertEqual(parse_json_object(r'{"findings":["Use \\lambda"]}'), {"findings": [r"Use \lambda"]})

    def test_incomplete_response_retains_content_and_finish_reason(self):
        raw = '{"facts": [{"value": 30}'
        body = {"choices": [{"message": {"content": raw}, "finish_reason": "length"}]}
        with patch("episodic_workflow.task_executors._http.urlopen", return_value=json_response(body)), self.assertRaises(ModelOutputError) as caught:
            ArgoClient("https://llm.example/v1", "test-model").call("prompt")
        self.assertEqual(caught.exception.raw_response, raw)
        self.assertEqual(caught.exception.diagnostic["finish_reason"], "length")

    def test_assumption_operations_keep_their_contract_instead_of_extraction_schema(self):
        request = {"operation": "propose_assumption_investigation",
                   "response_contract": {"action": "gather_evidence or assess_assumption"}}
        with patch.object(ArgoClient, "call", return_value='{"action":"gather_evidence"}') as call:
            response = argo_extract(ArgoClient("https://llm.example/v1", "test-model"), request)
        self.assertEqual(json.loads(call.call_args.args[0]), request)
        self.assertEqual(response["action"], "gather_evidence")

    def test_jev_rejects_missing_wrong_and_nonfinite_probabilities(self):
        answers = ({}, {"type": "text", "noul": 0.9}, {"type": "noul", "noul": True}, {"type": "noul", "noul": float("nan")}, {"type": "noul", "noul": 1.1})
        for answer in answers:
            with self.subTest(answer=answer), patch("episodic_workflow.task_executors._http.urlopen", return_value=json_response({"answers": {"decision": answer}})), self.assertRaises(ProviderError):
                asyncio.run(JevDecisionClient("test-key").noul(state={}, instructions="relevance", criteria={}))

    def test_jev_nouls_keeps_named_target_questions_separate(self):
        body = {"answers": {"caller-a": {"type": "noul", "noul": 0.8},
                            "caller-b": {"type": "noul", "noul": 0.2}}}
        with patch("episodic_workflow.task_executors._http.urlopen",
                   return_value=json_response(body)) as transport:
            values = asyncio.run(JevDecisionClient("test-key").nouls(
                state={"selected_targets": {"caller-a": {}, "caller-b": {}}},
                questions={name: {"instructions": f"Compare {name}", "criteria": {"true": "match"}}
                           for name in ("caller-a", "caller-b")}))
        self.assertEqual(values, {"caller-a": 0.8, "caller-b": 0.2})
        request = json.loads(transport.call_args.args[0].data)
        self.assertEqual(set(request["questions"]), {"caller-a", "caller-b"})
        self.assertTrue(all(row["type"] == "noul" for row in request["questions"].values()))


class SelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_selection_uses_model_choice_without_duplicate_pages(self):
        llm = AsyncMock()
        llm.call_async.side_effect = [json.dumps({"option_id": "page:2"}), json.dumps({"option_id": "page:1"})]
        jev = AsyncMock()
        jev.noul.side_effect = [0.9, 0.2]
        results = await bound_search(provider_search=lambda query, limit: [{"url": "https://one.example", "description": "one"}, {"url": "https://two.example", "description": "two"}], llm=llm, jev=jev, query="query", limit=2)
        self.assertEqual([item["provider_rank"] for item in results], [2, 1])
        self.assertEqual(jev.noul.await_count, 2)
        self.assertEqual(results[0]["page_selection"]["assessment"]["relevance_probability"], 0.2)
        second_prompt = llm.call_async.call_args_list[1].args[0]
        self.assertEqual(len(json.loads(second_prompt.split("\n\n", 1)[1])["candidates"]), 1)

    async def test_reselecting_a_processed_page_fails(self):
        llm = AsyncMock()
        llm.call_async.return_value = json.dumps({"option_id": "page:1"})
        jev = AsyncMock()
        jev.noul.return_value = 0.9
        with self.assertRaisesRegex(ValueError, "undeclared option_id"):
            await bound_search(provider_search=lambda query, limit: [{"url": "https://one.example"}, {"url": "https://two.example"}], llm=llm, jev=jev, query="query", limit=2)

    async def test_empty_search_does_not_call_models(self):
        llm, jev = AsyncMock(), AsyncMock()
        self.assertEqual(await bound_search(provider_search=lambda query, limit: [], llm=llm, jev=jev, query="query", limit=1), ())
        llm.call_async.assert_not_called()
        jev.noul.assert_not_called()

    async def test_long_unicode_text_preserves_all_windows(self):
        text = "\u4e2d\u6587\n" * 20000 + "end"
        jev = AsyncMock()
        jev.noul.return_value = 0.8
        report = await assess_page_candidate(jev, goal=EpisodeGoal.root(objective="query", result_contract={}), query="query", result={"text": text}, provider_rank=1)
        states = [call.kwargs["state"] for call in jev.noul.call_args_list]
        self.assertGreater(len(states), 1)
        self.assertEqual("".join(state["candidate_text_window"] for state in states), text)
        self.assertEqual(report.relevance_probability, 0.8)
        self.assertTrue(all(len(json.dumps(state, ensure_ascii=False).encode()) < 64000 for state in states))


class ConfigurationTests(unittest.TestCase):
    def test_explicit_environment_does_not_read_dotenv_or_process(self):
        with patch.dict(os.environ, {"FIRECRAWL_API_KEY": "process-secret"}), patch("episodic_workflow.environment.ENV_FILE") as path:
            self.assertEqual(workflow_environment({}), {})
            with self.assertRaisesRegex(ProviderConfigurationError, "FIRECRAWL_API_KEY"):
                firecrawl_argo_config_from_env({})
            path.is_file.assert_not_called()

    def test_only_checkout_dotenv_is_loaded_and_shell_wins(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ".env"
            path.write_text("export LOCAL_VALUE='a b' # comment\nOVERRIDE=file\n", encoding="utf-8")
            with patch("episodic_workflow.environment.ENV_FILE", path), patch.dict(os.environ, {"OVERRIDE": "shell"}, clear=True):
                self.assertEqual(workflow_environment(), {"LOCAL_VALUE": "a b", "OVERRIDE": "shell"})

    def test_config_does_not_construct_clients_or_send_requests(self):
        with patch("episodic_workflow.task_executors._http.urlopen") as request:
            config = firecrawl_argo_config_from_env(PROVIDER_ENV)
            self.assertEqual(config.llm_model, "test-model")
            self.assertNotIn("test-firecrawl-key", repr(config))
            self.assertNotIn("test-typesafe-key", repr(config))
            request.assert_not_called()

    def test_invalid_configuration_is_reported_before_http(self):
        for name, value in (("EPISODIC_LLM_BASE_URL", "file:///tmp/model"), ("EPISODIC_REQUEST_TIMEOUT_S", "nan"), ("EPISODIC_REQUEST_TIMEOUT_S", "invalid"), ("FIRECRAWL_SCRAPE_RESULTS", "maybe")):
            with self.subTest(name=name, value=value), self.assertRaises(ProviderConfigurationError):
                firecrawl_argo_config_from_env({**PROVIDER_ENV, name: value})


class ConfiguredWorkflowTests(InMemoryTransportTests):
    def test_all_model_modes_through_configured_http_adapters(self):
        for model in GUIDELINE_OUTPUTS:
            with self.subTest(model=model):
                transport = ProviderTransport()
                with patch("episodic_workflow.task_executors._http.urlopen", transport):
                    _, report = run_configured_evaluate_safety_guideline_episode(disease_name="synthetic-disease", target_scenario=SCENARIO, env=PROVIDER_ENV, transmission_model=model)
                self.assertTrue(report.passed, report.as_record())
                self.assertIn(SOURCE_URL, report.source_refs)
                self.assertIn("compute_safety_bound", report.main_outputs)
                json.dumps(report.as_record(), allow_nan=False)
                self.assertEqual({call[0] for call in transport.calls}, {"/v2/search", "/v2/scrape", "/v1/systemone", "/v1/chat/completions"})

    def test_transport_failure_becomes_structured_workflow_error(self):
        with patch("episodic_workflow.task_executors._http.urlopen", side_effect=URLError("offline")):
            _, report = run_configured_evaluate_safety_guideline_episode(disease_name="synthetic-disease", target_scenario=SCENARIO, env=PROVIDER_ENV)
        self.assertFalse(report.passed)
        self.assertIn("connection failed", json.dumps(report.as_record()))

    def test_cli_emits_one_json_document(self):
        out, err = io.StringIO(), io.StringIO()
        report = SimpleNamespace(passed=True, as_record=lambda: {"passed": True})
        with patch("episodic_workflow.cli.run_configured_safety_guideline_request", return_value=(None, report)) as run:
            code = main(["Investigate synthetic indoor transmission"], stdout=out, stderr=err)
        self.assertEqual(run.call_args.kwargs["research_request"], "Investigate synthetic indoor transmission")
        self.assertEqual(code, 0)
        self.assertTrue(json.loads(out.getvalue())["passed"])
        self.assertEqual(err.getvalue(), "")

    def test_cli_unbounded_result_is_strict_json(self):
        out, err = io.StringIO(), io.StringIO()
        report = SimpleNamespace(passed=True, as_record=lambda: {"distance": {"value": None, "limit": "unbounded"}})
        with patch("episodic_workflow.cli.run_configured_safety_guideline_request", return_value=(None, report)):
            code = main(["Investigate distance"], stdout=out, stderr=err)
        def reject_constant(value):
            raise AssertionError(f"Non-JSON numeric token: {value}")
        result = json.loads(out.getvalue(), parse_constant=reject_constant)
        self.assertEqual(code, 0)
        self.assertEqual(result["distance"], {"value": None, "limit": "unbounded"})
        self.assertEqual(err.getvalue(), "")

    def test_cli_configuration_failure_is_json(self):
        out, err = io.StringIO(), io.StringIO()
        with patch("episodic_workflow.configured_providers.workflow_environment", return_value={}):
            code = main(["Investigate synthetic-disease"], stdout=out, stderr=err)
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(json.loads(err.getvalue())["errors"][0]["code"], "configuration_error")

    def test_cli_rejects_nonfinite_input_before_provider_calls(self):
        for constant in ("Infinity", "-Infinity", "NaN", "1e999"):
            with self.subTest(constant=constant):
                out, err = io.StringIO(), io.StringIO()
                with patch("episodic_workflow.cli.run_configured_safety_guideline_request") as run:
                    code = main(["test", "--target-scenario", '{"risk_tolerance":' + constant + '}'], stdout=out, stderr=err)
                self.assertEqual(code, 2)
                run.assert_not_called()
                self.assertEqual(out.getvalue(), "")
                self.assertEqual(json.loads(err.getvalue())["errors"][0]["code"], "configuration_error")
