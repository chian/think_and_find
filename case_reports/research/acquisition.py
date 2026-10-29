"""Strategy > search > page > table/report/lexical reading episode bindings."""

from datetime import datetime, timezone
from dataclasses import replace
import asyncio
import json

from ...method_loop.episode_binding.web_search_binding import SearchPageProposer
from ...task_executors._http import ProviderError
from ...task_executors.outbreak_search.queries import (
    source_locator, source_text,
)
from .admission import exact_span
from .assumptions import text as required_text
from .documents import Document, Span, rank_spans
from .communication import merged_ranges, compact_reading_coverage
from .engine import ResearchSource, changed_records, selected_choice
from .model_tiers import ModelTier, TABLE_PAGE_EXTRACTION_TIER
from .records import Ledger, ResearchResult, identity
from .episode_contracts import (SearchStrategyCommission, SearchCommission, PageCommission,
    ReportCommission, LexicalCommission, TableCommission, literature_report)
from . import episode_credit


SOURCE_ASSESSMENT_BATCH_SIZE = 3





class LiteratureSource(ResearchSource):
    """Transport primitives only; concrete episodes own all scientific decisions."""
    def assess(self, result):
        from .observation_admission import assess_returned_observations
        return assess_returned_observations(self, result)

    def inputs(self, view, **kwargs):
        declared = {"task": self.instructions, "objective": str(view.goal.objective),
            "commission": self.commission.as_record(), **self.context(),
            "local_yield": self.credit(self.ledger.result()).as_record(),
            "own_results": self.parent_report(self.ledger.result()), **kwargs}
        return self.bounded_inputs(view, declared)

    def child_contract(self, commission_type, question):
        keys = list(dict.fromkeys([*self.commission.inputs, *self.selected_keys]))
        # Preserve commissioned G/B inputs. Latest evidence is addressable from child returns.
        keys = keys[:8]
        commission = commission_type(question, tuple(keys), tuple(self.scoped_snapshots()))
        commission.request(self.ledger.causal_sheet)
        return commission.report_contract()

    def parent_report(self, result, completion=None, request=None):
        return literature_report(self, result, completion)


class SearchStrategySource(LiteratureSource):
    commission_type = SearchStrategyCommission
    instructions = ("Own repeated choices of search query and search approach for the parent's information need. "
        "Use source findings, coverage, failed searches and open questions to choose the next query. "
        "Return grounded findings and unresolved logical questions that let the scientific parent commission Reasoning. "
        "Local credit follows executed scientific contributions attributable to the evidence returned by these searches.")
    credit_columns = ("strategy_scientific_evidence_gain",)

    def credit(self, result):
        return episode_credit.search_strategy_credit(result, self.initial)

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report["searches"] = {"queries": self.queries[-8:], "omitted": max(0, len(self.queries) - 8)}
        from .episode_contracts import changed_sheet
        delta, sheet = changed_sheet(self, result)
        questions = []
        for key in dict.fromkeys(row["key"] for row in delta.causal_sheet_updates):
            row = sheet.rows[key]
            if row["kind"] == "evidence" and row["data"]["measurements"].get("reasoning_question"):
                questions.append({"evidence": key, "source": row["data"]["source"][:500],
                    "question": row["data"]["measurements"]["reasoning_question"][:800]})
        report["reasoning_questions"] = {"items": questions[:8], "total": len(questions)}
        return report

    def __init__(self, run, inherited, *, approach):
        super().__init__(run, inherited)
        self.approach = approach
        self.queries = []



    def propose(self, view):
        proposal = self.run.agent(self.inputs(view, operation="propose_strategy_query",
            approach=self.approach, previous_queries=self.queries[-8:],
            earlier_query_count=max(0, len(self.queries) - 8),
            search_execution={
                "query": "Passed unchanged to a web search provider as a literal search-engine query.",
                "result": "Source candidates, not execution of a browsing plan. The search episode selects sources; page and reading episodes retrieve and examine their content.",
            },
            instruction=(
                "Choose a literal search-engine query for the delegated scientific question. "
                "Use returned causal_sheet_changes, reading_coverage and operations to revise your approach. "
                "Seek source evidence about theory mechanisms, meaningful uses of the focal inference, "
                "expected generalizability bounds, incompatible causal commitments or proposed repairs, as commissioned. "
                "Query is search text; objective states the specific reading question and necessary context. "
                "Do not copy the whole parent assignment into each query. Repeated search decisions belong "
                "to this Search Strategy episode; there is no fixed number or order. Do not repeat an identical query."),
            response_contract={"type": "object", "additionalProperties": False,
                "required": ["query", "objective"], "properties": {
                    "query": {"type": "string", "minLength": 1, "description": "Literal search-engine query, forwarded unchanged; not browsing or extraction instructions"},
                    "objective": {"type": "string", "minLength": 1, "description": "Information the child should resolve and how it bears on the parent's uncertainty; not part of the search query"}}}))
        self.last_proposal = proposal
        query = required_text(proposal.get("query"), "query")
        if query.casefold().strip() in {q.casefold().strip() for q in self.queries}:
            raise ValueError("This strategy has already searched that query")
        objective = required_text(proposal.get("objective"), "objective")
        child = self.run.child(view, SearchSource(self.run, self.ledger.result(), query=query),
            "search", objective, proposal=proposal,
            contract=self.child_contract(SearchCommission, objective))
        self.queries.append(query)
        return self.edge(child)


class SearchSource(LiteratureSource):
    commission_type = SearchCommission
    instructions = ("Own repeated source selections from ONE literal query's result set. "
        "Compare source descriptions and returned page findings against the assigned information need; "
        "choose the next source to resolve what remains open. Hits are not evidence. "
        "Local credit follows executed scientific contributions attributable to selected sources from this query.")
    credit_columns = ("query_scientific_evidence_gain",)

    def credit(self, result):
        return episode_credit.search_credit(result, self.initial, self.selected_sources)

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report["query"] = self.query
        report["examined_sources"] = [{"source": row["source"], "version": row["source_version"]}
            for row in report["reading_coverage"]]
        return report

    def __init__(self, run, inherited, *, query):
        super().__init__(run, inherited)
        self.query = query
        self.selected_sources = set()
        self.proposer = SearchPageProposer(query=query, limit=run.limits.search_limit,
                                          provider_search=run.search,
                                          jev=run.providers.page_assessor,
                                          select_page=self.select_page,
                                          standing_task="Assess candidate sources for causal-inference generalizability, theories, contradictions and repairs",
                                          information_need=query,
                                          assessment_instructions="Does this source offer substantive mechanisms, evidence, a derivation or a counterexample relevant to the information need?",
                                          assessment_criteria={"true": "Substantive evidence or reasoning relevant to the scoped question", "false": "Only topical mention or unrelated content"})
        self.view = None

    def select_page(self, goal, candidates, previous_pages):
        proposal = self.run.agent(self.inputs(self.view, operation="choose_research_source", query=self.query,
            search_goal=str(self.goal.objective), candidates=candidates, previous_pages=previous_pages,
            response_contract={"type": "object", "additionalProperties": False,
                "required": ["option_id", "objective", "rationale"], "properties": {
                    "option_id": {"type": "string", "enum": [row["option_id"] for row in candidates]},
                    "objective": {"type": "string", "minLength": 1,
                                  "description": "Evidence to seek in the selected source"},
                    "rationale": {"type": "string", "minLength": 1}}},
            instruction="Select for the still-unresolved delegated question using returned causal_sheet_changes, reading_coverage and operations. Give the source child a specific question and necessary context. A missing returned value and a failed reading do not establish absence of evidence. Return an instance of the JSON Schema: the source choice belongs in option_id. Numerical yield controls continuation."))
        self.last_proposal = proposal
        return proposal

    def propose(self, view):
        self.view = view
        candidate = asyncio.run(self.proposer.next(self.goal))
        if candidate is None:
            return None
        proposal = candidate["page_selection"]["proposal"]
        question = proposal["objective"]
        event = {"operation": "select_page", "status": "completed",
            "query": self.query, "source": source_locator(candidate),
            "binding": "SearchPageProposer", "proposal": proposal,
            "jev_assessment": candidate["page_selection"]["assessment"],
            "jev_score_meaning": "Page relevance to the local search question; scientific warrant requires source reading and assessment."}
        self.run.provider_events.append(event)
        self.run.emit_trace("provider_event", provider_event=event)
        self.selected_sources.add(source_locator(candidate))
        source = PageSource(self.run, self.ledger.result(), candidate=candidate)
        child = self.run.child(view, source, "page", question, proposal=proposal,
            contract=self.child_contract(PageCommission, f"Judge what this source establishes about {question}"))
        return self.edge(child)


class PageSource(LiteratureSource):
    commission_type = PageCommission
    instructions = ("Own repeated choices of where and how to examine ONE retrieved source. "
        "Use its outline, tables, remaining text and reader returns to choose ordered reading, "
        "lexical reading or a table query episode. Revisit a passage only for a specific unresolved "
        "question. Local credit follows executed scientific contributions attributable to this source.")
    credit_columns = ("page_scientific_evidence_gain",)

    def credit(self, result):
        return episode_credit.page_credit(result, self.initial, source_locator(self.candidate))

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report["source"] = {"url": source_locator(self.candidate),
                            "version": self.document.sha256 if self.document else None}
        report["remaining_source"] = {"unexamined_characters": sum(len(span.text) for span in self.document.remaining("unexamined")),
                                      "tables": len(self.document.tables)} if self.document else None
        return report

    def __init__(self, run, inherited, *, candidate):
        super().__init__(run, inherited)
        self.candidate, self.document = candidate, None
        self.page_inventory_offset = 0



    def prepare(self):
        locator = source_locator(self.candidate)
        text = self.run.documents.get(locator, {}).get("text", "")
        if not text:
            retrieved = self.run.retrieve(self.candidate)
            text = retrieved if isinstance(retrieved, str) else source_text(retrieved or {})
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Source retrieval returned no document text; this is not negative evidence")
        self.document = Document(locator, self.candidate.get("title", ""), text, self.run.limits.text_window_chars)
        version = {"text": text, "sha256_utf8": self.document.sha256,
            "retrieved_at": (self.run.providers.checkpoint.now()
                if self.run.providers.checkpoint is not None else datetime.now(timezone.utc).isoformat()),
            "spans": [s.as_record() for s in self.document.spans],
            "tables": {t.key: t.summary() for t in self.document.tables}}
        versions = dict(self.run.documents.get(locator, {}).get("versions", {}))
        versions[self.document.sha256] = version
        self.run.documents[locator] = {**version, "versions": versions}

    def propose(self, view):
        if self.document is None:
            try:
                self.prepare()
            except (ProviderError, OSError, ValueError) as exc:
                self.failure = "source_retrieval_failure"
                failed = self.failure_result(exc, "retrieval")
                failed = replace(failed, diagnostics=tuple({**d, "source": source_locator(self.candidate)}
                                                          for d in failed.diagnostics))
                return self.leaf(lambda: failed, "retrieval_failure")
        document = self.document
        page_task = identity([str(self.goal.objective), self.context()])
        attempts = [attempt for attempt in self.ledger.reading_attempts.values()
                    if attempt["source"] == document.source
                    and attempt["source_version"] == document.sha256]
        document.examined_ranges = merged_ranges([*document.examined_ranges,
            *(pair for attempt in attempts
              if attempt["status"] != "not_started" for pair in attempt["ranges"])])
        unread = document.remaining()
        reviewed = merged_ranges(pair for attempt in attempts if attempt["status"] == "completed"
                                 for pair in attempt["ranges"])
        fully_reviewed = all(any(start <= span.start and end >= span.end for start, end in reviewed)
                             for span in document.spans)
        task_attempts = [attempt for attempt in attempts if attempt.get("page_task") == page_task]
        task_reviewed = merged_ranges(pair for attempt in task_attempts if attempt["status"] == "completed"
                                      for pair in attempt["ranges"])
        task_failures = [pair for attempt in task_attempts if attempt["status"] in {"failed", "partial"}
                         for pair in attempt["ranges"] if not any(
                             start <= pair[0] and end >= pair[1] for start, end in task_reviewed)]
        if not unread and fully_reviewed and task_reviewed and not task_failures:
            # A missing answer is not new source material. Return the completed
            # task; a caller may revisit the source with a different task.
            self.cooperative_return_reason = "source_task_examined"
            return None
        reexamination = {span.key: span for span in document.spans
                        if unread or fully_reviewed or not any(
                            start <= span.start and end >= span.end for start, end in reviewed)}
        reading_scopes = ("unexamined", "reexamine") if unread else ("reexamine",)
        tables = {t.key: t for t in document.tables if t.key not in document.attempted_tables}
        offset = self.page_inventory_offset
        table_page = dict(list(tables.items())[offset:offset + 8])
        span_page = dict(list(reexamination.items())[offset:offset + 16])
        headings = [line[:200] for line in document.text.splitlines() if line.startswith("#")]
        options = {key: {"episode": "source_table", "columns": [c[:100] for c in t.columns[:12]],
                         "row_count": len(t.rows), "header": t.header.text[:400]}
                   for key, t in table_page.items()}
        inventory_size = max(len(reexamination), len(tables), len(headings))
        if len(reexamination) > 16 or len(tables) > 8 or len(headings) > 8:
            options["inventory"] = {"operation": "inspect another inventory page", "offset": offset,
                                    "span_count": len(reexamination), "table_count": len(tables),
                                    "heading_count": len(headings)}
        if document.remaining("reexamine"):
            if unread and not document.report_attempted:
                options["report"] = {"episode": "report", "meaning":
                    "Sequential reading with cross-window context; unexamined scope skips previously examined ranges. "
                    "A section name in question does not move the cursor; reaching later sections requires successive windows."}
            options["lexical_probe"] = {"episode": "lexical_probe", "meaning": "Rank source-local chunks by a literal query within the chosen reading scope; unexamined scope excludes previously supplied text from ranking and extraction."}
        if not options:
            return None
        proposal = self.run.agent(self.inputs(view, operation="propose_page_child",
            source_summary={"source": document.source, "title": document.title,
                "characters": len(document.text), "chunks": len(document.spans),
                "opening": document.text[:500], "outline": headings[offset:offset+8],
                "inventory_offset": offset, "table_count": len(tables),
                "prior_reading": compact_reading_coverage(
                    a for a in self.ledger.reading_attempts.values() if a["source"] == document.source)},
            reexamination_spans={key: {"start": span.start, "end": span.end,
                                      "opening": span.text[:240]} for key, span in span_page.items()},
            available_children=options,
            instruction=("Choose among the actual available reading scopes using the delegated information need and prior coverage. "
                "The inventory is paged: choose inventory with inventory_offset to inspect more spans, headings or tables. "
                "A lexical query searches the whole selected reading scope even when its spans are absent from this preview. "
                "report reads sequentially from the next unprocessed span; it cannot jump to a named section in question. "
                "lexical_probe ranks source-local chunks for a targeted query, including material beyond the opening windows. "
                "source_table examines a detected table. For report or lexical_probe, choose reading_scope: "
                "unexamined reads text not previously supplied to extraction in this source version, including "
                "coverage from table rows and earlier visits; reexamine permits deliberate rereading for a new "
                "question, failed extraction, or missing context. For reexamine use lexical_probe and select "
                "explicit reexamine_spans totaling at most one reading window; state the specific changed "
                "question or failed passage in question. It does not start another full-document reader. "
                "Previously examined glossary rows are not unexamined equation discussion merely because "
                "a new query ranks the same terms. Source versions with changed text have separate coverage. "
                "Earlier reading under a different question does not rule out a useful reread. "
                "Distinguish unread material from examined material that did not answer "
                "the question. Give the reader that specific question and necessary context; "
                "its JSON return supplies causal_sheet_changes, reading_coverage, "
                "operations and completion for your next reading choice. No fixed reading order, "
                "no mandatory reading choice. You do not decide continuation."),
            response_contract={"type": "object", "additionalProperties": False,
                "required": ["option", "question"], "properties": {
                    "option": {"type": "string", "enum": list(options)},
                    "question": {"type": "string", "minLength": 1, "description": "Child information need"},
                    "inventory_offset": {"type": "integer", "minimum": 0, "maximum": max(0, inventory_size - 1)},
                    "reading_scope": {"type": "string", "enum": list(reading_scopes),
                                      "description": "For report/lexical_probe: seek unexamined text or deliberately reread; tables retain row-query selection."},
                    "reexamine_spans": {"type": "array", "minItems": 1, "uniqueItems": True,
                        "items": {"type": "string", "enum": list(span_page)},
                        "description": "Exact source chunks for a focused reexamination, within one reading window."},
                    "query": {"type": "string", "minLength": 1, "description": "Literal query for lexical_probe"}},
                "allOf": [{"if": {"properties": {"option": {"const": "lexical_probe"}}},
                           "then": {"required": ["query"]}},
                          {"if": {"properties": {"option": {"enum": ["report", "lexical_probe"]}}},
                           "then": {"required": ["reading_scope"]}},
                          {"if": {"required": ["reading_scope"], "properties": {"reading_scope": {"const": "reexamine"}}},
                           "then": {"required": ["reexamine_spans"], "properties": {"option": {"const": "lexical_probe"}}}},
                          {"if": {"properties": {"option": {"const": "report"}}},
                           "then": {"properties": {"reading_scope": {"const": "unexamined"}}}}]}))
        self.last_proposal = proposal
        option = selected_choice(proposal.get("option"), options, "option")
        question = required_text(proposal.get("question"), "question")
        if option == "inventory":
            requested_offset = proposal.get("inventory_offset")
            if type(requested_offset) is not int or not 0 <= requested_offset < inventory_size:
                raise ValueError("Choose an available inventory offset")
            self.page_inventory_offset = requested_offset
            return self.leaf(lambda: ResearchResult(), "inspect_page_inventory")
        inherited = self.ledger.result()
        if option in tables:
            source = SourceTableSource(self.run, inherited, document=document, table=tables[option])
            grain = "source_table"
        elif option == "report":
            scope = selected_choice(proposal.get("reading_scope", "unexamined"),
                                    ("unexamined",), "reading_scope")
            source = ReportSource(self.run, inherited, document=document, reading_scope=scope)
            grain = "report"
        else:
            scope = selected_choice(proposal.get("reading_scope", "unexamined"),
                                    reading_scopes, "reading_scope")
            selected_spans = None
            if scope == "reexamine":
                selected_spans = proposal.get("reexamine_spans")
                known = reexamination
                if (not isinstance(selected_spans, list) or not selected_spans
                        or any(key not in known for key in selected_spans)
                        or len(set(selected_spans)) != len(selected_spans)
                        or sum(len(known[key].text) for key in selected_spans) > self.run.limits.text_window_chars):
                    raise ValueError("Focused reexamination requires exact source chunks within one reading window")
            source = LexicalProbeSource(self.run, inherited, document=document,
                query=required_text(proposal.get("query"), "query"), reading_scope=scope,
                selected_spans=selected_spans)
            grain = "lexical_probe"
        source.page_task = page_task
        child = self.run.child(view, source, grain, question, proposal=proposal,
            contract=self.child_contract(source.commission_type, f"Establish source-grounded evidence for {question}"))
        if option in tables:
            document.attempted_tables.add(option)
        elif option == "report":
            document.report_attempted = True
        return self.edge(child)


class ReadingSource(LiteratureSource):
    extraction_tier = ModelTier.REASONING

    def __init__(self, run, inherited, *, document):
        super().__init__(run, inherited)
        self.document = document
        self.memory = {"summary": "", "unresolved_links": [], "anchors": []}
        self.seen_spans = {}
        self.pending_spans, self.pending_memory = (), None

    def read(self, spans, question):
        from .causal_reading import read_source_window
        return read_source_window(self, spans, question)


    def validate_memory(self, memory, allowed):
        if not isinstance(memory, dict) or set(memory) != {"summary", "unresolved_links", "anchors"}:
            raise ValueError("Reading memory requires summary, unresolved_links, and source anchors")
        if len(json.dumps(memory)) > self.run.limits.text_window_chars:
            raise ValueError("Reading memory exceeds its declared context capacity")
        if not isinstance(memory["summary"], str) or len(memory["summary"]) > self.run.limits.text_window_chars:
            raise ValueError("Reading summary exceeds its declared context capacity")
        if not isinstance(memory["unresolved_links"], list) or any(not isinstance(v, str) for v in memory["unresolved_links"]):
            raise ValueError("Reading unresolved links must be text")
        if not isinstance(memory["anchors"], list):
            raise ValueError("Reading anchors must be a list")
        for anchor in memory["anchors"]:
            if not isinstance(anchor, dict) or anchor.get("span") not in allowed:
                raise ValueError("Reading anchor must refer to a seen source span")
            exact_span(anchor.get("quote"), allowed[anchor["span"]].text)

    def after_verdict(self, unit, contribution, view):
        if self.pending is not None:
            self.document.examined_ranges = merged_ranges([*self.document.examined_ranges,
                *(pair for attempt in self.pending.reading_attempts
                  if attempt["source"] == self.document.source
                  and attempt["source_version"] == self.document.sha256
                  and attempt["status"] != "not_started" for pair in attempt["ranges"])])
        if self.pending is not None and self.pending.evaluated:
            self.document.processed.update(s.key for s in self.pending_spans)
            self.seen_spans.update((s.key, s) for s in self.pending_spans)
            if self.pending_memory is not None:
                self.memory = self.pending_memory
        super().after_verdict(unit, contribution, view)
        self.pending_spans, self.pending_memory = (), None


class ReportSource(ReadingSource):
    commission_type = ReportCommission
    instructions = ("Read successive document-order windows for the assigned question. Decide which "
        "source claims and premises the new window establishes, which cross-window links remain "
        "unresolved, and which exact anchors must be retained. Preserve document context. "
        "Local credit follows executed scientific contributions attributable to evidence in this reader's document-order scope.")
    credit_columns = ("report_scientific_evidence_gain",)

    def credit(self, result):
        return episode_credit.report_credit(result, self.initial, self.document, self.spans)

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report["document_position"] = {"cursor": self.position, "remaining_spans": len(self.spans) - self.position,
                                       "unresolved_links": self.memory["unresolved_links"][:6]}
        return report

    def __init__(self, *args, reading_scope="unexamined", **kwargs):
        super().__init__(*args, **kwargs)
        self.spans, self.position = self.document.remaining(reading_scope), 0

    def propose(self, view):
        if self.position >= len(self.spans):
            return None
        spans, size = [], 0
        while self.position < len(self.spans):
            span = self.spans[self.position]
            if spans and size + len(span.text) > self.run.limits.text_window_chars:
                break
            spans.append(span)
            size += len(span.text)
            self.position += 1
        return self.leaf(lambda: self.read(tuple(spans), str(view.goal.objective)),
                         f"report_window:{view.units_consumed + 1}")


class LexicalProbeSource(ReadingSource):
    commission_type = LexicalCommission
    instructions = ("Examine successive source-local passages ranked for the assigned literal query. "
        "Decide whether a ranked match actually addresses the scientific question, retain exact "
        "anchors and resolve contextual links before treating keyword matches as evidence. "
        "Local credit follows executed scientific contributions attributable to evidence in the ranked reading scope.")
    credit_columns = ("lexical_scientific_evidence_gain",)

    def credit(self, result):
        return episode_credit.lexical_credit(result, self.initial, self.document, self.parts)

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report["passage_selection"] = {"query": self.query, "considered": self.position,
                                       "remaining": len(self.ranked) - self.position,
                                       "unresolved_links": self.memory["unresolved_links"][:6]}
        return report

    def __init__(self, *args, query, reading_scope="unexamined", selected_spans=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.query, self.position = query, 0
        self.parts = {span.key: (self.document.reading_parts(span, reading_scope)
                                if selected_spans is None or span.key in selected_spans else ())
                      for span in self.document.spans}
        # Empty ranking slots preserve physical neighbors; only original spans are extracted.
        ranking = tuple(Span(span.key, span.start, span.end,
                            "\n".join(part.text for part in self.parts.get(span.key, ())))
                        for span in self.document.spans)
        self.ranked = tuple(span.key for span in rank_spans(ranking, query) if self.parts.get(span.key))

    def propose(self, view):
        while self.position < len(self.ranked):
            key = self.ranked[self.position]
            self.position += 1
            spans = self.parts[key]
            if spans:
                return self.leaf(lambda: self.read(spans, str(view.goal.objective)),
                                 f"chunk:{spans[0].start}:{spans[-1].end}")
        return None


class SourceTableSource(ReadingSource):
    commission_type = TableCommission
    instructions = ("Choose repeated row selections or column/text queries within ONE table. "
        "Use earlier query outcomes to choose rows resolving the remaining question. Interpret "
        "cells with the exact header, conditions and units; preserve missing values. "
        "Local credit follows executed scientific contributions attributable to the selected table rows.")
    credit_columns = ("table_scientific_evidence_gain",)

    def credit(self, result):
        return episode_credit.table_credit(result, self.initial, self.document, self.table)

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report["table_queries"] = {"table": self.table.key,
            "queries": [{"query": q["query"], "column": q["column"], "matches": q["matches"],
                         "selected_rows": q["selected_rows"][:8]} for q in self.queries[-8:]],
            "queried_rows": sorted(self.queried)[:8], "remaining_rows": len(self.table.rows) - len(self.queried)}
        return report

    extraction_tier = TABLE_PAGE_EXTRACTION_TIER

    def __init__(self, *args, table, **kwargs):
        super().__init__(*args, **kwargs)
        self.table, self.queried = table, set()
        self.queries = []

    def propose(self, view):
        remaining = {s.key: (s, values) for s, values in zip(self.table.rows, self.table.cells)
                     if s.key not in self.queried}
        if not remaining:
            return None
        question_schema = {"type": "string", "minLength": 1,
                           "description": "Claim to examine using selected rows and the exact header"}
        response_contract = {"oneOf": [
            {"type": "object", "additionalProperties": False, "required": ["rows", "question"],
             "properties": {
                 "rows": {"type": "array", "minItems": 1, "uniqueItems": True,
                          "items": {"type": "string", "pattern": rf"^{self.table.key}:row:[1-9][0-9]*$"},
                          "description": "Exact unqueried row keys, not row numbers or row contents"},
                 "question": question_schema}},
            {"type": "object", "additionalProperties": False, "required": ["query", "question"],
             "properties": {
                 "query": {"type": "string", "description": "Literal text; empty string scans remaining rows"},
                 "column": {"type": ["string", "null"], "enum": [*self.table.columns, None]},
                 "question": question_schema}},
        ]}
        proposal = self.run.agent(self.inputs(view, operation="propose_table_query",
            table=self.table.summary(),
            available_rows={k: {"cells": list(values), "raw": s.text} for k, (s, values) in list(remaining.items())[:8]},
            remaining_row_count=len(remaining), previous_queries=[
                {"query": row["query"], "column": row["column"], "matches": row["matches"],
                 "selected_row_count": len(row["selected_rows"])} for row in self.queries[-8:]],
            earlier_query_count=max(0, len(self.queries) - 8),
            row_addressing={"format": f"{self.table.key}:row:NUMBER", "first": 1, "last": len(self.table.rows)},
            response_contract=response_contract,
            instruction="Return an instance of one JSON Schema alternative: rows plus question, OR query plus question and an optional column. Omit rows entirely for a text query; do not send rows: []. The preview is not the whole table: a text query searches all remaining rows and returns one bounded execution window. Subsequent queries can examine the remaining matches. Execution returns exact spans, not inferred cells."))
        self.last_proposal = proposal
        if ("rows" in proposal) == ("query" in proposal):
            raise ValueError("Choose exactly one table selection: 'rows' with nonempty exact row keys OR "
                             "'query' with literal text. Omit 'rows' for a text query. "
                             f"Received fields {list(proposal)!r}; no rows were queried.")
        rows = proposal.get("rows")
        question = required_text(proposal.get("question"), "question")
        if "query" in proposal:
            query, column = proposal.get("query"), proposal.get("column")
            if not isinstance(query, str) or (column is not None and column not in self.table.columns):
                raise ValueError("Table query needs literal text and an existing optional column")
            rows, size, matches = [], len(self.table.header.text), 0
            for key, (span, values) in remaining.items():
                searched = values if column is None else (values[self.table.columns.index(column)],)
                if not any(query.casefold() in cell.casefold() for cell in searched):
                    continue
                matches += 1
                if size + len(span.text) <= self.run.limits.text_window_chars:
                    rows.append(key)
                    size += len(span.text)
            self.queries.append({"query": query, "column": column, "matches": matches, "selected_rows": rows})
            if not rows:
                if matches:
                    raise ValueError("A matching table row exceeds the execution window")
                return self.leaf(lambda: self.no_matches(query), f"table_query:{view.units_consumed + 1}")
        if not isinstance(rows, list) or not rows or any(not isinstance(k, str) or k not in remaining for k in rows):
            invalid = [k for k in rows if not isinstance(k, str) or k not in remaining] if isinstance(rows, list) else rows
            raise ValueError("'rows' must be a nonempty list of exact unqueried row keys, not row numbers. "
                             f"Unavailable or invalid selection: {invalid!r}. "
                             f"Available key preview: {list(remaining)[:8]!r}. No rows were queried.")
        if len(set(rows)) != len(rows):
            raise ValueError("'rows' must contain unique row keys; no rows were queried")
        spans = (self.table.header, *(s for s in self.table.rows if s.key in rows))
        if sum(len(s.text) for s in spans) > self.run.limits.text_window_chars:
            raise ValueError("Table query exceeds one execution window; select fewer rows")
        if "rows" in proposal:
            self.queries.append({"query": None, "column": None, "matches": len(rows), "selected_rows": list(rows)})
        return self.leaf(lambda: self.read(spans, question), f"table_query:{view.units_consumed + 1}")

    def after_verdict(self, unit, contribution, view):
        if self.pending is not None and self.pending.evaluated:
            self.queried.update(s.key for s in self.pending_spans if s.key != self.table.header.key)
        super().after_verdict(unit, contribution, view)

    @staticmethod
    def no_matches(query):
        return ResearchResult(findings=(f"No remaining table rows match literal query {query!r}; this is not a scientific refutation.",))
