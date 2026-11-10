"""Discovery > Search > Page > source-reader episode bindings."""

from datetime import datetime, timezone
from dataclasses import replace
import asyncio

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
from .model_tiers import ModelTier
from .records import Ledger, ResearchResult, identity
from .episode_contracts import (DiscoveryCommission, SearchCommission, PageCommission,
    ReportCommission, LexicalCommission, TableCommission, literature_report)
from . import episode_credit
from .discovery_goals import (discovery_context, discovery_credit, source_credit,
    finding_schema, submit_finding)


SOURCE_ASSESSMENT_BATCH_SIZE = 3





class LiteratureSource(ResearchSource):
    """Transport primitives; each acquisition episode owns its next choice."""
    def inputs(self, view, **kwargs):
        declared = {"task": self.instructions, "objective": str(view.goal.objective),
            "commission": self.commission.as_record(), **self.context(),
            "local_yield": self.credit(self.ledger.result()).as_record(),
            **kwargs}
        return self.bounded_inputs(view, declared)

    def context(self):
        return {**super().context(), **discovery_context(self)}

    def child_contract(self, commission_type, question):
        keys = list(dict.fromkeys([*self.commission.inputs, *self.selected_keys]))
        commission = commission_type(question=question, inputs=tuple(keys),
            snapshots=tuple(self.scoped_snapshots()), discovery_goal=self.commission.discovery_goal)
        commission.request(self.ledger.causal_sheet)
        return commission.report_contract()

    def parent_report(self, result, completion=None, request=None):
        return literature_report(self, result, completion)


class DiscoverySource(LiteratureSource):
    commission_type = DiscoveryCommission
    assessment_fields = {
        "commission_answer": "New source-grounded information answering the exact discovery goal, with findings and provenance references.",
        "coverage_and_gaps": "Literal native matches, separately fulfilled scientific requirements, pending questions and what targeted acquisition could resolve them.",
        "qualifications": "Source qualifications, competing accounts, missing measurements and unresolved correspondence to the supplied scientific definitions.",
        "scientific_questions": "Precise unresolved scientific interpretations or theory-use questions for Root to pass to Scientific."}
    instructions = ("Own repeated targeted literature queries for the discovery goal received through Root. "
        "The commission supplies scientific definitions, an acceptable answer schema and individually identified criteria. "
        "Use those criteria, the current fulfillment receipts, source findings and your assessment to choose a query "
        "or submit an answer. Return new information with exact source-claim selectors and qualifications. "
        "Finding a mention, finding a theory's explicit assertion and establishing meaningful dependence on G are "
        "different findings; address the actual commissioned criterion. Preserve native quantities and conditions. "
        "The host validates the answer and checks the declared requirement; never assign a pass verdict or numeric grade. "
        "Native matches certify literal fields only. A scientific candidate filter cannot certify the requested "
        "mechanism or relationship; linked typed scientific structure or the specified formal result must do that. "
        "When a scientific correspondence cannot be verified from the available evidence, report the exact pending "
        "question for Scientific through Root. Do not replace missing measurements with a formula or turn retrieval "
        "failure into evidence of absence. Scientific owns theory reconstruction, repair and comparison. "
        "Discovery's local credit counts native matches and scientific fulfillment at their declared scopes, with repeated criterion/finding identities "
        "counted once. Its numerical yield estimator owns continuation.")
    credit_columns = ("discovery_request_gain",)

    def credit(self, result):
        return discovery_credit(result, self.initial, self.commission.discovery_goal, self.credit_columns[0])

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report["searches"] = {"query_count": len(self.queries), "latest_query": self.queries[-1] if self.queries else None}
        report["scientific_questions"] = self.working_assessment["scientific_questions"]
        report["discovery_goal"] = self.commission.discovery_goal
        report["request_credit"] = self.credit(result).as_record()
        return report

    def __init__(self, run, inherited):
        super().__init__(run, inherited)
        self.queries = []



    def propose(self, view):
        proposal = self.decide(self.inputs(view, operation="propose_discovery_query",
            latest_query=self.queries[-1] if self.queries else None,
            query_count=len(self.queries),
            search_execution={
                "query": "Passed unchanged to a web search provider as a literal search-engine query.",
                "result": "Source candidates, not execution of a browsing plan. The search episode selects sources; page and reading episodes retrieve and examine their content.",
            },
            instruction=(
                self.instructions + " Choose action=search to acquire information addressing a specified criterion, "
                "or action=submit_finding to submit an answer in the goal's exact answer schema. "
                "Use actual child returns and criterion verification receipts to revise your approach. "
                "An answer cites existing exact source-claim selectors and scientific record versions; "
                "the host checks those references and the supplied verifier. An unresolved result is useful "
                "feedback for the next search or a pointed Scientific question. A query should target the "
                "particular missing definition, relationship, observation, condition or source needed by the goal. "
                "Query is search text; objective states the specific reading question and necessary context. "
                "Do not copy the whole parent assignment into each query. Repeated search decisions belong "
                "to Discovery; there is no fixed number or order. Repeating a query remains possible; use prior outcomes to judge its value."),
            response_contract={"oneOf": [{"type": "object", "additionalProperties": False,
                "required": ["action", "query", "objective"], "properties": {
                    "action": {"const": "search"},
                    "query": {"type": "string", "minLength": 1, "description": "Literal search-engine query, forwarded unchanged; not browsing or extraction instructions"},
                    "objective": {"type": "string", "minLength": 1, "description": "Specific criterion or information need this search should address; not part of the literal query"}}},
                finding_schema()]}))
        self.last_proposal = proposal
        if proposal.get("action") == "recall":
            return self.recall(proposal)
        if proposal.get("action") == "submit_finding":
            return self.leaf(lambda: submit_finding(self, proposal), "submit_discovery_finding")
        selected_choice(proposal.get("action"), ("search",), "action")
        query = required_text(proposal.get("query"), "query")
        objective = required_text(proposal.get("objective"), "objective")
        child = self.run.child(view, SearchSource(self.run, self.ledger.result(), query=query),
            "search", objective, proposal=proposal,
            contract=self.child_contract(SearchCommission, objective))
        self.queries.append(query)
        return self.edge(child)


class SearchSource(LiteratureSource):
    commission_type = SearchCommission
    assessment_fields = {
        "query_answer": "What the examined sources establish for this query's commissioned information need, with exact references.",
        "source_comparison": "How the examined sources' definitions, formulations, assumptions and derivations agree or differ.",
        "remaining_information_need": "Which criterion, measurement, source link or scientific correspondence remains unresolved for Discovery."}
    instructions = ("Own repeated source selections from ONE literal query's result set. "
        "Compare source descriptions and returned Page findings against Discovery's precise information need and goal criteria. "
        "Choose the next source to resolve a missing claim, quantity or condition, or submit a grounded finding "
        "using the goal's answer schema and exact source-claim references. Hits and topical mentions are source leads. "
        "Keep source formulations and qualifications explicit, reuse existing identities, and return unsupported "
        "scientific correspondences as unresolved. The host applies the goal's verifier; do not issue a pass verdict. "
        "Theory reconstruction, repair and comparison belong to Scientific. Local credit measures new "
        "source information relevant to the request, independently of "
        "global scientific points and raw extraction volume. Numerical yield controls continuation.")
    credit_columns = ("query_information_gain",)

    def credit(self, result):
        return source_credit(result, self.initial, self.commission.discovery_goal, self.credit_columns[0],
                             selected_sources=self.selected_sources)

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report["query"] = self.query
        report["examined_sources"] = [{"source": url, "version": self.run.documents.get(url, {}).get("sha256_utf8")}
                                      for url in sorted(self.selected_sources)]
        return report

    def __init__(self, run, inherited, *, query):
        super().__init__(run, inherited)
        self.query = query
        self.selected_sources = set()
        self.proposer = SearchPageProposer(query=query, limit=run.limits.search_limit,
                                          provider_search=run.search,
                                          jev=run.providers.page_assessor,
                                          select_page=self.select_page,
                                          standing_task="Find source material addressing the commissioned discovery goal and information need",
                                          assessment_instructions=("Could this source supply a definition, quantity, equation, causal relation, "
                                              "mechanism, assumption, derivation, approximation condition or evidence resolving the commissioned "
                                              "information need? Judge the stated information request, not keyword overlap with the literal query. "
                                              "This is candidate selection evidence, not scientific verification of the source's claim or use of G."),
                                          assessment_criteria={"true": "Source material could supply a component or dependency requested by the commission", "false": "Only topical mention or unrelated content"})
        self.view = None

    def select_page(self, goal, candidates, previous_pages):
        choices = [finding_schema()]
        if candidates:
            choices.insert(0, {"type": "object", "additionalProperties": False,
                "required": ["action", "option_id", "objective", "rationale"], "properties": {
                    "action": {"const": "read_source"},
                    "option_id": {"type": "string", "enum": [row["option_id"] for row in candidates]},
                    "objective": {"type": "string", "minLength": 1,
                                  "description": "The specific source claim, measurement, condition or other criterion this Page should investigate"},
                    "rationale": {"type": "string", "minLength": 1}}})
        proposal = self.decide(self.inputs(self.view, operation="choose_research_source", query=self.query,
            search_goal=str(self.goal.objective), candidates=candidates,
            latest_selection=previous_pages[-1] if previous_pages else None, selection_count=len(previous_pages),
            response_contract={"oneOf": choices},
            instruction=("Choose action=read_source with an exact option_id to investigate an available source, "
                "or action=submit_finding to submit already returned evidence against the goal's typed answer schema. "
                "Use the supplied scientific definitions, fulfillment criteria and unresolved information needs, "
                "together with actual Page returns and the current job assessment. Source candidates are only leads. "
                "A submitted finding uses exact source-claim selectors and scientific record versions; "
                "the host determines criterion fulfillment. Keep alternative formulations "
                "and missing premises explicit. Give each Page a specific reading question and the context needed "
                "to resolve it. A failed reading or absent returned value does not prove absence from the literature. "
                "When no candidates remain, returned evidence can still be submitted or recalled. Numerical yield owns continuation.")))
        self.last_proposal = proposal
        return proposal

    def propose(self, view):
        self.view = view
        self.proposer.information_need = self.commission.question
        candidates = asyncio.run(self.proposer.candidates(self.goal))
        proposal = self.select_page(self.goal.as_record(), candidates, self.proposer.history)
        if proposal.get("action") == "recall":
            return self.recall(proposal)
        if proposal.get("action") == "submit_finding":
            return self.leaf(lambda: submit_finding(self, proposal), "submit_discovery_finding")
        selected_choice(proposal.get("action"), ("read_source",), "action")
        candidate = self.proposer.select(proposal, candidates)
        proposal = candidate["page_selection"]["proposal"]
        question = proposal["objective"]
        event = {"operation": "select_page", "status": "completed",
            "query": self.query, "source": source_locator(candidate),
            "binding": "SearchPageProposer", "proposal": proposal,
            "jev_assessment": candidate["page_selection"]["assessment"],
            "jev_score_meaning": "Possible contribution to the commissioned information need; actual findings and goal verifiers determine Discovery credit."}
        self.run.provider_events.append(event)
        self.run.emit_trace("provider_event", provider_event=event)
        self.selected_sources.add(source_locator(candidate))
        source = PageSource(self.run, self.ledger.result(), candidate=candidate)
        child = self.run.child(view, source, "page", question, proposal=proposal,
            contract=self.child_contract(PageCommission, f"Judge what this source establishes about {question}"))
        return self.edge(child)


class PageSource(LiteratureSource):
    commission_type = PageCommission
    assessment_fields = {
        "source_answer": "What this source establishes about the commissioned question, linked to exact evidence.",
        "assessment_status": "Which source claims, definitions, measurements and conditions have been extracted and what remains incomplete.",
        "qualifications": "Source limitations, alternative formulations, scope conditions and unresolved explanatory dependencies.",
        "remaining_reading_need": "What additional reading or upstream reasoning could resolve."}
    instructions = ("Own repeated choices of where and how to examine ONE retrieved source and assess what it establishes. "
        "Use its outline, tables, remaining text and reader returns to choose ordered reading, "
        "lexical reading or a table query episode. Revisit a passage only for a specific unresolved "
        "question. Readers return extracted source records and qualifications. Aggregate what the source "
        "actually supplies for the delegated information need: definitions and properties, quantities, "
        "relationships, mechanisms, assumptions, derivations, approximation conditions and measurements. "
        "Keep citation records and bibliographic metadata in source leads and provenance; report follow-up "
        "leads in source_answer or remaining_reading_need. Do not present a cited publication or its "
        "bibliographic metadata as an extracted scientific claim or observation. A reference entry alone "
        "does not establish the cited work's scientific definitions, equations or claims. "
        "Return grounded definitions and relationships as they become clear. A qualified partial source "
        "answer is useful while passages, equations or source references remain unresolved. "
        "Use the source's scientific vocabulary and exact evidence versions; preserve supplied quantity meanings. "
        "Preserve the source's native scope and alternative formulations. Distinguish a source claim, an "
        "assumption, and a derived consequence. Record missing explanatory connections for focused follow-up. "
        "Do not force a source into a preselected equation or manufacture an unspecified relationship. "
        "Local credit aggregates newly extracted source information without duplicate credit; the numerical controller owns continuation.")
    credit_columns = ("page_extraction_gain",)

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
        self.failure = ""

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
        reexamination = {span.key: span for span in document.spans}
        reading_scopes = ("unexamined", "reexamine") if unread else ("reexamine",)
        tables = {t.key: t for t in document.tables}
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
            options["report"] = {"episode": "report", "meaning":
                "Sequential reading with cross-window context in the chosen scope. Unexamined skips prior coverage; "
                "reexamine rereads in document order. A section name in question does not move the cursor."}
            options["lexical_probe"] = {"episode": "lexical_probe", "meaning": "Rank source-local chunks by a literal query within the chosen reading scope; unexamined scope excludes previously supplied text from ranking and extraction."}
        proposal = self.decide(self.inputs(view, operation="propose_page_child",
            source_summary={"source": document.source, "title": document.title,
                "characters": len(document.text), "chunks": len(document.spans),
                "opening": document.text[:500], "outline": headings[offset:offset+8],
                "inventory_offset": offset, "table_count": len(tables),
                "prior_reading": compact_reading_coverage(attempts),
                "recent_reading_attempts": [{"question": a["question"][:500], "status": a["status"],
                    "ranges": a["ranges"][:8], "same_page_task": a.get("page_task") == page_task}
                    for a in attempts[-8:]], "earlier_attempt_count": max(0, len(attempts) - 8)},
            reexamination_spans={key: {"start": span.start, "end": span.end,
                                      "opening": span.text[:240]} for key, span in span_page.items()},
            available_children=options,
            instruction=(self.instructions + " Choose action=read with one of the available reading options. "
                "Use extracted_records and table data from reader returns to identify what is established "
                "and what further reading could resolve for the commissioned discovery goal. "
                "Choose among the actual available reading scopes using the delegated information need and prior coverage. "
                "The inventory is paged: choose inventory with inventory_offset to inspect more spans, headings or tables. "
                "A lexical query searches the whole selected reading scope even when its spans are absent from this preview. "
                "report reads sequentially in the chosen scope; it cannot jump to a named section in question. "
                "lexical_probe ranks source-local chunks for a targeted query, including material beyond the opening windows. "
                "source_table examines a detected table. For report or lexical_probe, choose reading_scope: "
                "unexamined reads text not previously supplied to extraction in this source version, including "
                "coverage from table rows and earlier visits; reexamine permits deliberate rereading for a new "
                "question, failed extraction, or missing context. Report can reread in document order; "
                "lexical_probe can search all earlier text or focus on explicit reexamine_spans. "
                "State the changed question or failed passage when rereading. "
                "Previously examined glossary rows are not unexamined equation discussion merely because "
                "a new query ranks the same terms. Source versions with changed text have separate coverage. "
                "Earlier reading under a different question does not rule out a useful reread. "
                "Distinguish unread material from examined material that did not answer "
                "the question. Give the reader that specific question and necessary context; "
                "its JSON return supplies extracted records, local extraction credit, exact references, "
                "qualifications, unresolved extraction errors and completion for your next Page decision. No fixed reading order, "
                "no mandatory reading choice. You do not decide continuation."),
            response_contract={"type": "object", "additionalProperties": False,
                "required": ["action", "option", "question"], "properties": {
                    "action": {"const": "read"},
                    "option": {"type": "string", "enum": list(options)},
                    "question": {"type": "string", "minLength": 1, "description": "Child information need"},
                    "inventory_offset": {"type": "integer", "minimum": 0, "maximum": max(0, inventory_size - 1)},
                    "reading_scope": {"type": "string", "enum": list(reading_scopes),
                                      "description": "For report/lexical_probe: seek unexamined text or deliberately reread; tables retain row-query selection."},
                    "reexamine_spans": {"type": "array", "minItems": 1, "uniqueItems": True,
                        "items": {"type": "string", "enum": list(span_page)},
                        "description": "Optional exact source chunks for a focused lexical reexamination, within one reading window."},
                    "query": {"type": "string", "minLength": 1, "description": "Literal query for lexical_probe"}},
                "allOf": [{"if": {"properties": {"option": {"const": "lexical_probe"}}},
                           "then": {"required": ["query"]}},
                          {"if": {"properties": {"option": {"enum": ["report", "lexical_probe"]}}},
                           "then": {"required": ["reading_scope"]}},
                          {"if": {"required": ["reexamine_spans"]},
                           "then": {"properties": {"option": {"const": "lexical_probe"},
                                                   "reading_scope": {"const": "reexamine"}}}}]}))
        self.last_proposal = proposal
        if proposal.get("action") == "recall":
            return self.recall(proposal)
        selected_choice(proposal.get("action"), ("read",), "action")
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
            scope = selected_choice(proposal.get("reading_scope", reading_scopes[0]),
                                    reading_scopes, "reading_scope")
            source = ReportSource(self.run, inherited, document=document, reading_scope=scope)
            grain = "report"
        else:
            scope = selected_choice(proposal.get("reading_scope", reading_scopes[0]),
                                    reading_scopes, "reading_scope")
            selected_spans = None
            if scope == "reexamine":
                selected_spans = proposal.get("reexamine_spans")
                if selected_spans is not None:
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
            contract=source.commission_type(question, (),
                discovery_goal=self.commission.discovery_goal).report_contract())
        return self.edge(child)


class ReadingSource(LiteratureSource):
    extraction_tier = ModelTier.REASONING

    def context(self):
        return {"question": self.commission.question}

    def __init__(self, run, inherited, *, document):
        super().__init__(run, inherited)
        self.document = document
        self.memory = {"summary": "", "unresolved_links": [], "anchors": [], "retained_evidence": []}
        self.pending_memory_update = False
        self.seen_spans = {}
        self.pending_spans, self.pending_memory = (), None
        self.extraction_repair = None

    def repair_unit(self, view):
        from .extraction_repair import ExtractionRepairSource

        repair = self.extraction_repair
        source = ExtractionRepairSource(self.run, self.ledger.result(),
            document=self.document, repair=repair, memory=self.memory,
            seen_spans={anchor["span"]: self.seen_spans[anchor["span"]]
                        for anchor in self.memory["anchors"]})
        source.page_task = getattr(self, "page_task", None)
        child = self.run.child(view, source, "extraction_repair", repair["question"],
            proposal={"operation": "repair_source_extraction", "errors": repair["errors"]},
            contract=source.commission_type(repair["question"], (),
                discovery_goal=self.commission.discovery_goal).report_contract())
        return self.edge(child)

    def read(self, spans, question):
        from .causal_reading import read_source_window
        return read_source_window(self, spans, question)

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report["extraction_credit"] = self.credit(result).as_record()
        report["extracted_records"] = list(changed_records(result, self.initial).causal_sheet_updates)
        report["extraction_errors"] = self.extraction_repair["errors"] if self.extraction_repair is not None else []
        report["source_anchors"] = {"source": self.document.source, "version": self.document.sha256,
            "spans": [anchor["span"] for anchor in self.memory["anchors"]]}
        return report


    def validate_memory(self, memory, allowed):
        if not isinstance(memory, dict) or set(memory) != {"summary", "unresolved_links", "anchors", "retained_evidence"}:
            raise ValueError("Reading memory requires summary, unresolved_links, anchors and retained_evidence")
        if not isinstance(memory["summary"], str):
            raise ValueError("Reading summary must be text")
        if not isinstance(memory["unresolved_links"], list) or any(not isinstance(v, str) for v in memory["unresolved_links"]):
            raise ValueError("Reading unresolved links must be text")
        if not isinstance(memory["anchors"], list):
            raise ValueError("Reading anchors must be a list")
        for anchor in memory["anchors"]:
            if not isinstance(anchor, dict) or anchor.get("span") not in allowed:
                raise ValueError("Reading anchor must refer to a seen source span")
            exact_span(anchor.get("quote"), allowed[anchor["span"]].text)
        if not isinstance(memory["retained_evidence"], list):
            raise ValueError("Retained evidence must be a list of exact supplied references")
        for ref in memory["retained_evidence"]:
            if (not isinstance(ref, dict) or set(ref) != {"key", "version"}
                    or (ref["key"], ref["version"]) not in self.known_records):
                raise ValueError("Retained evidence must reference an exact record already supplied to this reader")
            self.ledger.causal_sheet.snapshot(ref)

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
            if self.pending_memory_update:
                rows = changed_records(self.pending, self.ledger.result()).causal_sheet_updates
                refs = [*self.memory["retained_evidence"],
                        *({"key": row["key"], "version": row["version"]} for row in rows if row["kind"] == "evidence")]
                self.memory["retained_evidence"] = list({(ref["key"], ref["version"]): ref for ref in refs}.values())
                self.working_assessment.update(reading_answer=self.memory["summary"],
                    open_questions="\n".join(self.memory["unresolved_links"]),
                    record_refs=self.memory["retained_evidence"])
                self.unreviewed_results = []
        super().after_verdict(unit, contribution, view)
        if contribution.episode_update is not None:
            returned = contribution.episode_update.report.as_record()
            repair = returned["repair_status"]
            self.validate_memory(repair["reading_memory"], self.seen_spans)
            self.memory = repair["reading_memory"]
            self.working_assessment.update(reading_answer=self.memory["summary"],
                open_questions="\n".join(self.memory["unresolved_links"]),
                record_refs=self.memory["retained_evidence"])
            if repair["repaired"]:
                self.extraction_repair = None
            else:
                self.extraction_repair = {**self.extraction_repair,
                    "response": repair["rejected_response"], "errors": repair["errors"]}
        self.pending_spans, self.pending_memory = (), None
        self.pending_memory_update = False


class ReportSource(ReadingSource):
    commission_type = ReportCommission
    assessment_fields = {
        "reading_answer": "Source-grounded synthesis across document-order windows; preserve cross-window qualifications and exact evidence references.",
        "open_questions": "Unresolved source links, missing context or extraction questions for Page."}
    instructions = ("Read successive document-order windows for the assigned question. Decide which "
        "source claims and premises the new window establishes, which cross-window links remain "
        "unresolved, and which exact anchors must be retained. Preserve document context. "
        "Local credit counts newly returned source information; repeated information adds no credit.")
    credit_columns = ("report_extraction_gain",)

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
        if self.extraction_repair is not None:
            return self.repair_unit(view)
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
    assessment_fields = {
        "reading_answer": "What ranked passages establish about the targeted question; retain source context and qualifications.",
        "open_questions": "What the selected passages leave unresolved for Page and which source links matter."}
    instructions = ("Examine successive source-local passages ranked for the assigned literal query. "
        "Decide whether a ranked match actually addresses the scientific question, retain exact "
        "anchors and resolve contextual links before treating keyword matches as evidence. "
        "Local credit counts newly returned source information; repeated information adds no credit.")
    credit_columns = ("lexical_extraction_gain",)

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
        if self.extraction_repair is not None:
            return self.repair_unit(view)
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
    assessment_fields = {
        "reading_answer": "What queried rows establish with their headers, conditions and qualifications; distinguish absent cells from contrary evidence.",
        "open_questions": "Unresolved row, header or source-context questions to return to Page."}
    instructions = ("Choose repeated row selections or column/text queries within ONE table. "
        "Use earlier query outcomes to choose rows resolving the remaining question. Interpret "
        "cells with the exact header, conditions and units; preserve missing values. "
        "Return native table cells. Local credit counts newly returned cells; repeated cells add no credit.")
    credit_columns = ("table_extraction_gain",)

    def credit(self, result):
        return episode_credit.table_credit(result, self.initial, self.document, self.table)

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        report["table"] = {"source": self.document.source, "source_version": self.document.sha256,
            "id": self.table.key, "header": self.table.header.text, "columns": list(self.table.columns),
            "rows": [row["data"]["measurements"]["native"]["table_row"]
                for _, row in Ledger(result).causal_sheet.of_kind("evidence")
                if row["data"]["source"] == self.document.source
                and row["data"]["measurements"]["source_version"] == self.document.sha256
                and row["data"]["measurements"]["native"].get("table_row", {}).get("table") == self.table.key]}
        report["table_queries"] = {"table": self.table.key,
            "queries": [{"query": q["query"], "column": q["column"], "matches": q["matches"],
                         "question": q["question"][:500], "selected_rows": q["selected_rows"][:8],
                         "outcome": q.get("outcome")} for q in self.queries[-8:]],
            "queried_rows": sorted(self.queried)[:8], "remaining_rows": len(self.table.rows) - len(self.queried),
            "revisitable_rows": len(self.queried)}
        return report

    def read(self, spans, question):
        """Return parsed native cells; Page owns interpretation."""
        working = Ledger(self.ledger.result())
        rows = {span.key: values for span, values in zip(self.table.rows, self.table.cells)}
        updates = []
        for span in spans:
            if span.key == self.table.header.key:
                continue
            data = {"source": self.document.source, "locator": f"characters {span.start}:{span.end}",
                "study_id": self.document.source, "finding": span.text,
                "measurements": {"source_version": self.document.sha256, "quote": span.text,
                    "question": question, "native": {"observations": {}, "claims": {},
                        "table_row": {"table": self.table.key, "row": span.key,
                            "start": span.start, "end": span.end, "raw": span.text,
                            "columns": list(self.table.columns), "header": self.table.header.text,
                            "cells": list(rows[span.key])}}}}
            key = "evidence:" + identity([self.document.source, self.document.sha256, span.key])[:24]
            updates.append(working.causal_sheet.put(key, "evidence", data))
        self.pending_spans = spans
        return ResearchResult(causal_sheet_updates=tuple(updates), reading_attempts=({
            "source": self.document.source, "source_version": self.document.sha256,
            "document_characters": len(self.document.text), "episode_id": self.goal.goal_id,
            "question": question, "ranges": [[span.start, span.end] for span in spans],
            "status": "completed", "reading_summary": "Returned native table rows.", "unresolved_links": []},))

    def __init__(self, *args, table, **kwargs):
        super().__init__(*args, **kwargs)
        self.table, self.queried = table, set()
        self.queries = []
        self.pending_query = None

    def propose(self, view):
        available = {s.key: (s, values) for s, values in zip(self.table.rows, self.table.cells)}
        if not available:
            return None
        remaining = {key: row for key, row in available.items() if key not in self.queried}
        question_schema = {"type": "string", "minLength": 1,
                           "description": "Claim to examine using selected rows and the exact header"}
        response_contract = {"oneOf": [
            {"type": "object", "additionalProperties": False, "required": ["rows", "question"],
             "properties": {
                 "rows": {"type": "array", "minItems": 1, "uniqueItems": True,
                          "items": {"type": "string", "pattern": rf"^{self.table.key}:row:[1-9][0-9]*$"},
                          "description": "Exact table row keys, including previously read rows when the question changes"},
                 "question": question_schema}},
            {"type": "object", "additionalProperties": False, "required": ["query", "question"],
             "properties": {
                 "query": {"type": "string", "description": "Literal text; empty string scans all unqueried rows"},
                 "column": {"type": ["string", "null"], "enum": [*self.table.columns, None]},
                 "question": question_schema}},
        ]}
        proposal = self.decide(self.inputs(view, operation="propose_table_query",
            table=self.table.summary(),
            available_rows={k: {"cells": list(values), "raw": s.text}
                for k, (s, values) in list(remaining.items())[:8]},
            unqueried_row_count=len(remaining), previous_queries=[
                {"query": row["query"], "column": row["column"], "matches": row["matches"],
                 "question": row["question"][:500], "selected_rows": row["selected_rows"][:8],
                 "selected_row_count": len(row["selected_rows"]), "outcome": row.get("outcome")}
                for row in self.queries[-8:]],
            earlier_query_count=max(0, len(self.queries) - 8),
            row_addressing={"format": f"{self.table.key}:row:NUMBER", "first": 1, "last": len(self.table.rows)},
            response_contract=response_contract,
            instruction="Return an instance of one JSON Schema alternative: rows plus question, OR query plus question and an optional column. Omit rows entirely for a text query; do not send rows: []. The preview shows unqueried rows and is not the whole table. A text query searches all unqueried rows and returns one execution window; repeating it can reach later matching rows. Use prior questions, selected rows and outcomes to choose what information remains useful. Explicit row keys can select previously read rows for a changed question. Execution returns exact spans, not inferred cells."))
        self.last_proposal = proposal
        if proposal.get("action") == "recall":
            return self.recall(proposal)
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
            self.pending_query = {"query": query, "column": column, "matches": matches,
                                  "selected_rows": rows, "question": question}
            self.queries.append(self.pending_query)
            if not rows:
                if matches:
                    raise ValueError("A matching table row exceeds the execution window")
                return self.leaf(lambda: self.no_matches(query), f"table_query:{view.units_consumed + 1}")
        if not isinstance(rows, list) or not rows or any(not isinstance(k, str) or k not in available for k in rows):
            invalid = [k for k in rows if not isinstance(k, str) or k not in available] if isinstance(rows, list) else rows
            raise ValueError("'rows' must be a nonempty list of exact table row keys, not row numbers. "
                             f"Unavailable or invalid selection: {invalid!r}. "
                             f"Available key preview: {list(available)[:8]!r}. No rows were queried.")
        if len(set(rows)) != len(rows):
            raise ValueError("'rows' must contain unique row keys; no rows were queried")
        spans = (self.table.header, *(s for s in self.table.rows if s.key in rows))
        if sum(len(s.text) for s in spans) > self.run.limits.text_window_chars:
            raise ValueError("Table query exceeds one execution window; select fewer rows")
        if "rows" in proposal:
            self.pending_query = {"query": None, "column": None, "matches": len(rows),
                                  "selected_rows": list(rows), "question": question}
            self.queries.append(self.pending_query)
        return self.leaf(lambda: self.read(spans, question), f"table_query:{view.units_consumed + 1}")

    def after_verdict(self, unit, contribution, view):
        if self.pending_query is not None and self.pending is not None:
            delta = changed_records(self.pending, self.ledger.result())
            self.pending_query["outcome"] = {
                "evaluated": self.pending.evaluated,
                "admitted_records": len(delta.causal_sheet_updates),
                "reading_statuses": [attempt["status"] for attempt in delta.reading_attempts],
                "credit_gain": view.controller_step.gain,
                "diagnostics": [{k: diagnostic.get(k) for k in ("stage", "category", "status", "why")}
                                for diagnostic in delta.diagnostics[-3:]],
            }
        if self.pending is not None and self.pending.evaluated:
            self.queried.update(s.key for s in self.pending_spans if s.key != self.table.header.key)
        super().after_verdict(unit, contribution, view)
        self.pending_query = None

    @staticmethod
    def no_matches(query):
        return ResearchResult(findings=(f"No unqueried table rows match literal query {query!r}; this is not a scientific refutation.",))
