"""Reading outcomes are steering information, not evidence or completeness."""

import unittest

from episodic_workflow.case_reports.research.acquisition import PageSource, SourceTableSource
from episodic_workflow.case_reports.research.communication import reading_coverage
from episodic_workflow.case_reports.research.documents import Document
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun
from episodic_workflow.case_reports.research.records import Ledger, ResearchResult
from episodic_workflow.task_executors._http import ProviderError
from episodic_workflow.tests.test_approved_tree import execute_binding, local_run
from episodic_workflow.tests.test_pending_extraction import read_once, extraction, TEXT
from episodic_workflow.tests.test_research import run_synthetic, synthetic_synthesis


def attempt(*, version="v1", ranges=((10, 20),), status="completed", question="Synthetic question"):
    return {"source": "https://example.test/source", "source_version": version,
            "document_characters": 100, "ranges": list(ranges), "status": status, "question": question}


class ReadingCoverageTests(unittest.TestCase):
    def test_union_preserves_holes_and_distinguishes_review_failure_from_unexamined(self):
        rows = [attempt(ranges=((10, 30),)), attempt(ranges=((20, 40),)),
                attempt(ranges=((60, 80),), status="failed"),
                attempt(ranges=((80, 100),), status="not_started")]
        coverage = reading_coverage(rows)[0]
        self.assertEqual(coverage["examined_ranges"], [[10, 40], [60, 80]])
        self.assertEqual(coverage["reviewed_ranges"], [[10, 40]])
        self.assertEqual(coverage["unexamined_ranges"], [[0, 10], [40, 60], [80, 100]])
        self.assertEqual(coverage["questions"], ["Synthetic question"])
        self.assertEqual(coverage["outcomes"], {"completed": 2, "failed": 1, "not_started": 1})

    def test_revised_source_versions_never_share_coverage(self):
        rows = reading_coverage([attempt(ranges=((0, 100),)), attempt(version="v2", ranges=((0, 20),))])
        self.assertEqual(rows[0]["unexamined_ranges"], [])
        self.assertEqual(rows[1]["unexamined_ranges"], [[20, 100]])

    def test_ledger_preserves_and_deduplicates_metadata_without_credit(self):
        result = ResearchResult(reading_attempts=(attempt(),))
        ledger = Ledger(result)
        ledger.merge(result)
        self.assertEqual(ledger.result().reading_attempts, result.reading_attempts)
        self.assertFalse(ledger.result().credit(("mouth_area_m2",)).qualities)
        self.assertFalse(ledger.facts or ledger.evidence or ledger.assumptions)

    def test_partial_document_read_records_exact_boundary_without_claiming_absence(self):
        _, source, record, result = read_once(lambda _: {"facts": [], "findings": []},
            calls=10, deadline=1, text="Opening context.\n" * 100 + "Late material.\n")
        self.assertEqual(record.units_consumed, 1)
        coverage = reading_coverage(result.reading_attempts)[0]
        end = max(s.end for s in source.document.spans if s.key in source.document.processed)
        self.assertEqual(coverage["examined_ranges"], [[0, end]])
        self.assertEqual(coverage["unexamined_ranges"], [[end, len(source.document.text)]])
        self.assertEqual(coverage["source_version"], source.document.sha256)
        self.assertEqual(result.reading_attempts[0]["admitted_fields"], [])
        self.assertEqual(set(record.controller_state.history), {0})

    def test_failed_review_records_examined_but_not_reviewed_ranges(self):
        def agent(request):
            if request["operation"] == "extract_research_evidence":
                return extraction()
            raise ProviderError("Synthetic review failure")
        _, _, _, result = read_once(agent)
        coverage = reading_coverage(result.reading_attempts)[0]
        self.assertEqual(coverage["examined_ranges"], [[0, len(TEXT)]])
        self.assertEqual(coverage["reviewed_ranges"], [])
        self.assertEqual(coverage["outcomes"], {"failed": 1})
        self.assertFalse(result.facts or result.evidence)
        self.assertIn("extraction_state", result.diagnostics[0])

    def test_selected_table_rows_are_not_examined_when_extraction_cannot_start(self):
        run = ResearchRun(ResearchProviders(lambda _: {"rows": ["table:1:row:1"], "question": "Synthetic row"},
            lambda *_: (), lambda _: None), ResearchLimits(), "Synthetic task", ("mouth_area_m2",))
        document = Document("https://example.test/table", "Fixture", "| Area | Value |\n| --- | --- |\n| A | 2 |\n", 600)
        _, result = execute_binding(run, SourceTableSource(run, (), document=document, table=document.tables[0]), "source_table")
        coverage = reading_coverage(result.reading_attempts)[0]
        self.assertEqual(coverage["outcomes"], {"not_started": 1})
        self.assertEqual(coverage["examined_ranges"], [])
        self.assertEqual(coverage["unexamined_ranges"], [[0, len(document.text)]])

    def test_revisit_receives_prior_coverage_but_can_choose_a_different_reading_scope(self):
        requests = []
        text = "Opening context.\n" * 100 + "\n## Distinctive synthetic target\nLate evidence location.\n"
        def agent(request):
            requests.append(request)
            if request["operation"] == "propose_page_child":
                return {"option": "lexical_probe", "query": "Distinctive synthetic target",
                        "question": "Examine the late section", "child_model_calls": 1}
            if request["operation"] == "extract_research_evidence":
                return {"facts": [], "findings": []}
            return synthetic_synthesis(request)
        run = local_run(agent)
        document = Document("https://example.test/source", "Fixture", text, 600)
        prior = {**attempt(version=document.sha256, ranges=((0, 500),)), "document_characters": len(text)}
        source = PageSource(run, ResearchResult(reading_attempts=(prior,)),
            candidate={"url": document.source, "text": text})
        run.limits = ResearchLimits(text_window_chars=600)
        _, result = execute_binding(run, source, "page")
        proposal = requests[0]
        self.assertEqual(proposal["source_summary"]["next_document_order_start"], 0)
        self.assertEqual(proposal["source_summary"]["prior_reading"][0]["examined_ranges"], [[0, 500]])
        self.assertIn("cannot jump", proposal["instruction"])
        self.assertIn("does not move the cursor", proposal["available_children"]["report"]["meaning"])
        extract = next(r for r in requests if r["operation"] == "extract_research_evidence")
        self.assertIn("Late evidence location", extract["source_text"])
        self.assertEqual(len(result.reading_attempts), 2)
        synthesis = next(r for r in requests if r["operation"] == "synthesize_research_findings")
        self.assertIn("reading_coverage", synthesis["parent_contract"]["information"])
        self.assertGreater(synthesis["admitted_result"]["reading_coverage"][0]["examined_ranges"][0][0], 500)

    def test_full_tree_carries_reading_coverage_to_root_and_closure(self):
        agent, (_, report) = run_synthetic()
        self.assertTrue(report.data["reading_attempts"])
        self.assertTrue(report.data["source_reading_coverage"])
        root = [r for r in agent.requests if r["operation"] == "propose_guideline_action"]
        self.assertFalse(root[0]["context"]["source_reading_coverage"])
        self.assertTrue(root[1]["context"]["source_reading_coverage"])
        summaries = [r for r in agent.requests if r["operation"] == "synthesize_research_findings"]
        self.assertTrue(all("reading_coverage" in r["admitted_result"] for r in summaries))
        self.assertTrue(all("retrieved-but-unexamined" in r["instruction"] for r in summaries))
