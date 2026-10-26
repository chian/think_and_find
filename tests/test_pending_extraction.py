"""Preserve unfinished discovery without treating it as scientific evidence."""

import json
import unittest

from episodic_workflow.case_reports.research.acquisition import PageSource, ReportSource
from episodic_workflow.case_reports.research.documents import Document
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun
from episodic_workflow.case_reports.research.records import condensed_diagnostics
from episodic_workflow.method_loop import Context, EpisodeGoal, EpisodeTree
from episodic_workflow.task_executors._http import ProviderError
from episodic_workflow.task_executors.evidence_extraction.argo import parse_json_object
from episodic_workflow.tests.test_approved_tree import execute_binding
from episodic_workflow.tests.test_research import synthetic_information_review, synthetic_synthesis


FIELD = "mouth_area_m2"
QUOTE = "Synthetic area = 3 cm2."
RANGE = "The synthetic alternative spans 2 to 4 cm2."
TEXT = QUOTE + "\n" + RANGE


def extraction():
    return {"facts": [{"subject": "fixture", "field": FIELD, "value": 3, "unit": "cm2",
        "kind": "source_model_parameter", "basis": "Software fixture, not scientific evidence",
        "quote": QUOTE, "context": "Synthetic source"}],
        "evidence": [{"subject": "fixture", "fields": [FIELD], "claim": "A synthetic range, not a point estimate",
            "quote": RANGE, "context": "Synthetic source", "basis": "Software fixture only"}], "findings": []}


def read_once(agent, *, calls=2, deadline=None, text=TEXT):
    run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None),
        ResearchLimits(text_window_chars=600), "Synthetic incomplete-review test", (FIELD,))
    document = Document("https://example.test/fixture", "Synthetic fixture", text, 600)
    source = ReportSource(run, (), document=document)
    episode = run.episode("report", source,
        goal=EpisodeGoal.root(objective="Resolve a synthetic area", result_contract={}), focus=(FIELD,),
        model_call_deadline=deadline)
    record = episode.run(Context(tree=EpisodeTree(run.grains["report"], {})))
    return run, source, record, source.build_result(record)


class PendingExtractionTests(unittest.TestCase):
    def test_indirect_measurements_remain_evidence_not_inferred_numeric_inputs(self):
        quote = "The synthetic opening diameter is 2 cm."
        def agent(request):
            if request["operation"] == "extract_research_evidence":
                self.assertIn("without a dedicated numeric field", request["instruction"])
                return {"facts": [], "evidence": [{"subject": "fixture", "fields": [FIELD],
                    "claim": "The diameter could constrain area under a separately justified shape assumption",
                    "quote": quote, "context": "Synthetic measurement, not the target population",
                    "basis": "An ingredient for an inference, not an observed area"}], "findings": []}
            if request["operation"] == "review_research_evidence":
                return {"decisions": [{"option": 0, "admit": True, "reason": "Synthetic source reports diameter only"}]}
            return synthetic_information_review(request)
        _, _, _, result = read_once(agent, calls=3, text=quote)
        self.assertFalse(result.facts or result.assumptions)
        self.assertEqual(result.evidence[0]["fields"], [FIELD])
        self.assertEqual(result.evidence[0]["quote"], quote)

    def test_review_failures_retain_checked_candidates_but_no_evidence_or_credit(self):
        for failure in ("malformed_json", "invalid_action_contract", "provider_failure", "operational_limit"):
            with self.subTest(failure=failure):
                def agent(request):
                    if request["operation"] == "extract_research_evidence":
                        return extraction()
                    self.assertEqual(request["operation"], "review_research_evidence")
                    if failure == "malformed_json":
                        return parse_json_object('{"decisions": [')
                    if failure == "invalid_action_contract":
                        return {"decisions": [{"option": 0, "admit": True, "reason": "Incomplete fixture reply"}]}
                    if failure == "provider_failure":
                        raise ProviderError("Synthetic review failure")
                    self.fail("The global ceiling must prevent this provider call")
                run, source, record, result = read_once(agent, calls=1 if failure == "operational_limit" else 2)
                diagnostic = result.diagnostics[0]
                state = diagnostic["extraction_state"]
                self.assertEqual(diagnostic["category"], failure)
                self.assertEqual(state["status"], "independent_review_incomplete")
                self.assertEqual(state["source_version"], source.document.sha256)
                self.assertEqual(state["source"], source.document.source)
                self.assertEqual(state["candidate_count"], 2)
                self.assertAlmostEqual(state["candidates"][0]["value"], .0003)
                self.assertEqual(state["candidates"][0]["unit"], "m2")
                self.assertEqual(state["candidates"][1]["fields"], [FIELD])
                self.assertEqual(diagnostic["unreviewed_candidates"][0]["unit_conversion"]["factor"], .0001)
                location = state["candidates"][0]["location"]
                span = next(s for s in source.document.spans if s.key == location["span"])
                self.assertEqual(span.text[location["start_in_span"]:location["end_in_span"]], QUOTE)
                self.assertNotIn("quote", state["candidates"][0])
                self.assertEqual(diagnostic["unreviewed_candidates"][0]["normalized"]["quote"], QUOTE)
                self.assertNotIn("unreviewed_candidates", condensed_diagnostics([diagnostic])[0])
                self.assertFalse(result.facts or result.evidence or result.assumptions)
                self.assertFalse(result.evaluated)
                self.assertFalse(source.document.processed)
                self.assertEqual(set(record.controller_state.history), {0})
                self.assertGreaterEqual(run.model_calls, 1)
                json.dumps(result.as_record(), allow_nan=False)

    def test_failed_extraction_does_not_claim_candidates_were_found(self):
        def agent(_):
            return parse_json_object('{"facts": [')
        _, _, _, result = read_once(agent, calls=1)
        self.assertNotIn("extraction_state", result.diagnostics[0])

    def test_completed_rejection_is_not_an_unfinished_review(self):
        def agent(request):
            if request["operation"] == "extract_research_evidence":
                return extraction()
            return {"decisions": [{"option": c["option"], "admit": False, "reason": "Synthetic applicability mismatch"}
                                  for c in request["candidates"]]}
        _, source, _, result = read_once(agent)
        self.assertTrue(result.evaluated)
        self.assertTrue(source.document.processed)
        self.assertFalse(result.facts or result.evidence)
        self.assertEqual({d["category"] for d in result.diagnostics}, {"rejected_evidence"})
        self.assertTrue(all("extraction_state" not in d for d in result.diagnostics))

    def test_local_allowance_finishes_review_and_credit_but_cannot_start_another_window(self):
        def agent(request):
            if request["operation"] == "extract_research_evidence":
                return extraction()
            if request["operation"] == "review_research_evidence":
                return {"decisions": [{"option": c["option"], "admit": True, "reason": "Synthetic fixture matches"}
                                      for c in request["candidates"]]}
            response = synthetic_information_review(request)
            for row in response["judgments"]:
                row["status"] = "bounded"
            return response
        run, source, record, result = read_once(agent, calls=10, deadline=1,
            text=TEXT + "\n" + "Additional synthetic context.\n" * 80)
        self.assertEqual([a["operation"] for a in run.model_attempts],
            ["extract_research_evidence", "review_research_evidence", "assess_information_progress"])
        self.assertEqual([a["phase"] for a in run.model_attempts], ["work", "mandatory_completion", "mandatory_completion"])
        self.assertEqual(record.units_consumed, 1)
        self.assertTrue(result.facts and result.evidence)
        self.assertEqual(record.controller_state.history[-1], .5)
        self.assertTrue(source.document.processed)
        self.assertTrue(source.document.remaining())
        self.assertEqual(result.stop_reason, "episode_model_call_budget_exhausted_not_scientific_convergence")

    def test_parent_synthesis_and_next_choice_receive_pending_discovery(self):
        for synthesize in (True, False):
            with self.subTest(synthesize=synthesize):
                requests = []
                def agent(request):
                    requests.append(request)
                    op = request["operation"]
                    if op == "propose_page_child":
                        return {"option": "report" if "report" in request["available_children"] else "lexical_probe",
                                "question": "Find the synthetic quantity", "query": "Synthetic", "child_model_calls": 1}
                    if op == "extract_research_evidence":
                        return extraction()
                    if op == "review_research_evidence":
                        raise ProviderError("Synthetic unavailable review")
                    if op == "synthesize_research_findings":
                        if not synthesize:
                            return {"child_summary": {}}
                        return synthetic_synthesis(request)
                    self.fail(op)
                run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None),
                    ResearchLimits(), "Synthetic parent communication test", (FIELD,))
                source = PageSource(run, (), candidate={"url": "https://example.test/fixture", "text": TEXT})
                record, result = execute_binding(run, source, "page")
                choices = [r for r in requests if r["operation"] == "propose_page_child"]
                self.assertGreaterEqual(len(choices), 2)
                next_choice = choices[1]
                self.assertNotIn("unreviewed_candidates", json.dumps(next_choice))
                diagnostics = ([d for f in next_choice["failure_implications"] for d in f["basis"]] if synthesize else
                               next_choice["attempt_diagnostics"])
                pending = next(d["extraction_state"] for d in diagnostics if "extraction_state" in d)
                self.assertEqual(pending["candidate_count"], 2)
                self.assertEqual(pending["source"], "https://example.test/fixture")
                self.assertEqual(pending["candidates"][0]["field"], FIELD)
                returned = record.unit_records[0].episode_update.report.as_record()
                self.assertIn("independent_review_incomplete", json.dumps(returned["diagnostics"]))
                self.assertFalse(result.facts or result.evidence)
                self.assertEqual(set(record.controller_state.history), {0})
