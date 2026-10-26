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
from .admission import extract_and_admit, exact_span
from .assumptions import text as required_text
from .documents import Document, Span, rank_spans
from .communication import merged_ranges, reading_coverage
from .engine import ResearchSource, changed_records, selected_choice
from .model_tiers import ModelTier, TABLE_PAGE_EXTRACTION_TIER
from .progress import parameter_requirements
from .records import Ledger, ResearchResult, identity
from .relevance import (
    assess_source_types, assess_source_types_with_jev, source_profiles,
    target_comparison_context,
)


SOURCE_ASSESSMENT_BATCH_SIZE = 3


def _init_relevance_state(source):
    source.event_types = {row["type_id"]: dict(row) for row in source.ledger.source_event_types.values()}
    source.relationships = {(row["source"], row["source_subject"], row["target_subject"], row["parameter"]): dict(row)
                            for row in source.ledger.source_target_relationships.values()}
    source.profile_versions = {}
    source.next_type_number = 1


class StrategySource(ResearchSource):
    def __init__(self, run, inherited, *, approach):
        super().__init__(run, inherited)
        self.approach = approach
        self.queries = []
        _init_relevance_state(self)

    def assess(self, result):
        scoped_targets = self.parent_contract.measurements.get("target_scenarios", {})
        if not scoped_targets or not result.evaluated:
            return result
        shared = getattr(getattr(self.run, "global_goal", None), "ledger", None)
        working = Ledger(shared.result() if shared is not None else self.ledger.result())
        working.merge(changed_records(result, self.ledger.result()))
        self.event_types.update((row["type_id"], dict(row)) for row in working.source_event_types.values())
        self.relationships.update(((row["source"], row["source_subject"], row["target_subject"], row["parameter"]), dict(row))
                                  for row in working.source_target_relationships.values())
        targets = {name: target_comparison_context({**definition, "subject": name},
                    working.assumptions.values()) for name, definition in scoped_targets.items()}
        # Relation fingerprints share the canonical comparison context with the
        # central value grader; local scored fields still exclude defaults.
        profiles = source_profiles(working.result(), parameter_requirements(self.parameter))
        versions = {}
        for source in {locator for locator, _ in profiles}:
            versions[source] = identity({"subjects": {subject: profile for (locator, subject), profile in profiles.items()
                                                        if locator == source}, "targets": targets})
        assigned = {tuple(member) for group in self.event_types.values()
                    for member in group["members"]}
        for source, version in versions.items():
            scoped = {subject: profile for (locator, subject), profile in profiles.items() if locator == source}
            if (all((source, subject) in assigned for subject in scoped)
                    and all((relation := self.relationships.get((source, subject, target, self.parameter), {})).get("target_version")
                            == identity(targets[target]) and relation.get("source_profile_version") == identity(scoped[subject])
                            for subject in scoped for target in targets)):
                self.profile_versions[source] = version
        changed = [source for source, version in sorted(versions.items())
                   if self.profile_versions.get(source) != version]
        if not changed:
            return result
        groups = {key: {**group, "members": list(group["members"])}
                  for key, group in self.event_types.items()}
        relationships = dict(self.relationships)
        next_number = self.next_type_number
        processed = []
        failure = None
        jev_assessor = getattr(getattr(self.run, "providers", None), "source_target_assessor", None)
        for source in changed:
            previous_relationships = tuple(row for key, row in relationships.items() if key[0] == source)
            for group in groups.values():
                group["members"] = [member for member in group["members"] if member[0] != source]
            subjects = sorted(subject for locator, subject in profiles if locator == source)
            batch_size = len(subjects) if jev_assessor is not None else SOURCE_ASSESSMENT_BATCH_SIZE
            for start in range(0, len(subjects), batch_size):
                batch = {(source, subject): profiles[(source, subject)]
                         for subject in subjects[start:start + batch_size]}
                try:
                    if jev_assessor is not None:
                        decisions, assessed_relations = assess_source_types_with_jev(
                            self.run.complete, jev_assessor, source, batch, targets, groups,
                            self.run.context.get("disease_of_interest"), parameter=self.parameter,
                            previous_relationships=previous_relationships)
                    else:
                        decisions, assessed_relations = assess_source_types(
                            self.run.complete, source, batch, targets, groups,
                            self.run.context.get("disease_of_interest"), parameter=self.parameter,
                            previous_relationships=previous_relationships)
                except (ProviderError, ValueError, TypeError, KeyError, OSError) as exc:
                    failure = self.failure_result(exc, "source_event_relevance")
                    break
                for decision in decisions:
                    type_id = decision["type_id"]
                    if type_id == "new":
                        # A source group is scientific state, not an episode
                        # identity. Its key must survive a child-to-parent return.
                        type_id = "source_type:" + identity({
                            "source": source, "subjects": sorted(decision["subjects"]),
                            "description": decision["description"], "basis": decision["basis"],
                            "sequence": next_number,
                        })
                        next_number += 1
                    group = groups.setdefault(type_id, {"members": []})
                    group.update(description=decision["description"], basis=decision["basis"],
                                 target_relevance=dict(decision["target_relevance"]))
                    group["members"].extend([source, subject] for subject in decision["subjects"])
                for row in assessed_relations:
                    key = (row["source"], row["source_subject"], row["target_subject"], row["parameter"])
                    # Event-level relevance serves the existing acquisition
                    # handoff. It must not replace value-specific target grades.
                    if "source_value_scores" not in relationships.get(key, {}):
                        relationships[key] = row
            if failure:
                break
            processed.append(source)
        self.event_types = groups
        self.relationships = relationships
        self.profile_versions.update({source: versions[source] for source in processed})
        self.next_type_number = next_number
        return replace(result, source_event_types=tuple({"type_id": key, **group}
                                                        for key, group in groups.items()),
                       source_target_relationships=tuple(relationships.values()),
                       diagnostics=(*result.diagnostics, *(failure.diagnostics if failure else ())))

    def propose(self, view):
        if self.evidence_handoff_ready():
            return None
        proposal = self.run.agent(self.inputs(view, operation="propose_strategy_query",
            approach=self.approach, previous_queries=self.queries,
            search_execution={
                "query": "Passed unchanged to a web search provider as a literal search-engine query.",
                "result": "Source candidates, not execution of a browsing plan. The search episode selects sources; page and reading episodes retrieve and examine their content.",
            },
            instruction=(
                "Propose a search addressing the delegated uncertainty and unresolved hypotheses in the parent information contract. "
                "Use returned parameter_results, reviews, calculations, reading_coverage and operations to identify remaining inputs and failed attempts. "
                "Your investigation parent owns candidate assumption proposals and their reasonableness "
                "assessment of inferred hypotheses. Every extracted native numeric value is automatically "
                "compared with target parameter needs and added to matching target groups. The highest "
                "value-specific source relevance selects the main-table value and updates shared credit. "
                "The query field is search text, not instructions to an agent. Use searchable terms, identifiers, "
                "phrases or a concise search question. Instructions to open URLs, inspect XML, follow links, "
                "extract equations or stop are not executed by the search provider. Put the information need "
                "and interpretation requirements in objective, separate from the literal query. Later source "
                "selection and reading episodes examine returned sources; merely including a route or action "
                "in query does not establish that it was attempted. A source inventory or another equation "
                "mention is not progress unless the contract needs it. The original research request is context, "
                "not a requirement to explore every topic. Numerical yield controls continuation. "
                "The assigned parameter guides this search. Credit uses the shared target-value and "
                "calculation tables; useful values for other target parameters also count automatically. "
                "A query or objective cannot add scored requirements. "
                "Give the child only its specific information question, including necessary context. "
                "Its return contains defined JSON fields populated from the existing work responses and execution records. "
                "Do not repeat an identical query."),
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
            contract=self.contract(objective, subjects=self.requested_subjects,
                target_scenarios=self.parent_contract.measurements.get("target_scenarios")))
        self.queries.append(query)
        return self.edge(child)


class SearchSource(ResearchSource):
    def __init__(self, run, inherited, *, query):
        super().__init__(run, inherited)
        self.query = query
        self.proposer = SearchPageProposer(query=query, limit=run.limits.search_limit,
                                          provider_search=run.search,
                                          jev=run.providers.page_assessor,
                                          select_page=self.select_page)
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
            instruction="Select for the still-unresolved delegated question using returned parameter_results, reviews, calculations, reading_coverage and operations. Give the source child a specific question and necessary context. A missing returned value and a failed reading do not establish absence of evidence. Return an instance of the JSON Schema: the source choice belongs in option_id. Numerical yield controls continuation."))
        self.last_proposal = proposal
        return proposal

    def propose(self, view):
        if self.evidence_handoff_ready():
            return None
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
            "jev_score_meaning": "Page relevance to the local search question; source-event relevance to a target is not yet assessed.",
            "investigated_target_subjects": list(self.requested_subjects),
            "source_target_relation_status": "unknown_until_source_event_assessed"}
        self.run.provider_events.append(event)
        self.run.emit_trace("provider_event", provider_event=event)
        source = PageSource(self.run, self.ledger.result(), candidate=candidate)
        child = self.run.child(view, source, "page", question, proposal=proposal,
            contract=self.contract(f"Judge what this source establishes about {question}",
                subjects=self.requested_subjects,
                target_scenarios=self.parent_contract.measurements.get("target_scenarios")))
        return self.edge(child)


class PageSource(ResearchSource):
    def __init__(self, run, inherited, *, candidate):
        super().__init__(run, inherited)
        self.candidate, self.document = candidate, None
        _init_relevance_state(self)

    def assess(self, result):
        # Restore event classification/local handoff; numeric target projection
        # additionally happens centrally on every reading commit.
        return StrategySource.assess(self, result)

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
        if self.evidence_handoff_ready():
            return None
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
        task_failures = [pair for attempt in task_attempts if attempt["status"] == "failed"
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
        options = {key: {"episode": "source_table", **t.summary()} for key, t in tables.items()}
        if document.remaining("reexamine"):
            if unread and not document.report_attempted:
                options["report"] = {"episode": "report", "meaning":
                    "Sequential reading with cross-window context; unexamined scope skips previously examined ranges. "
                    "A section name in question does not move the cursor; reaching later sections requires successive windows."}
            options["lexical_probe"] = {"episode": "lexical_probe", "meaning": "Rank source-local chunks by a literal query within the chosen reading scope; unexamined scope excludes previously supplied text from ranking and extraction."}
        if not options:
            return None
        proposal = self.run.agent(self.inputs(view, operation="propose_page_child",
            source_summary={**document.summary(), "prior_reading": reading_coverage(
                a for a in self.ledger.reading_attempts.values() if a["source"] == document.source)},
            reexamination_spans={key: {"start": span.start, "end": span.end,
                                      "opening": span.text[:240]} for key, span in reexamination.items()},
            available_children=options,
            instruction=("Choose among the actual available reading scopes using the delegated information need and prior coverage. "
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
                "its JSON return supplies parameter_results, reviews, calculations, reading_coverage, "
                "operations and completion for your next reading choice. No fixed reading order, "
                "no mandatory reading choice. You do not decide continuation."),
            response_contract={"type": "object", "additionalProperties": False,
                "required": ["option", "question"], "properties": {
                    "option": {"type": "string", "enum": list(options)},
                    "question": {"type": "string", "minLength": 1, "description": "Child information need"},
                    "reading_scope": {"type": "string", "enum": list(reading_scopes),
                                      "description": "For report/lexical_probe: seek unexamined text or deliberately reread; tables retain row-query selection."},
                    "reexamine_spans": {"type": "array", "minItems": 1, "uniqueItems": True,
                        "items": {"type": "string", "enum": list(reexamination)},
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
            contract=self.contract(f"Establish source-grounded evidence for {question}"))
        if option in tables:
            document.attempted_tables.add(option)
        elif option == "report":
            document.report_attempted = True
        return self.edge(child)


class ReadingSource(ResearchSource):
    extraction_tier = ModelTier.REASONING

    def __init__(self, run, inherited, *, document):
        super().__init__(run, inherited)
        self.document = document
        self.memory = {"summary": "", "unresolved_links": [], "anchors": []}
        self.seen_spans = {}
        self.pending_spans, self.pending_memory = (), None

    def read(self, spans, question):
        allowed = {}
        for anchor in self.memory["anchors"]:
            original = self.seen_spans[anchor["span"]]
            quote = exact_span(anchor["quote"], original.text)
            start = original.start + original.text.index(quote)
            key = f"anchor:{start}:{start + len(quote)}"
            allowed[key] = Span(key, start, start + len(quote), quote)
        allowed.update((s.key, s) for s in spans)
        if spans and all(left.end == right.start for left, right in zip(spans, spans[1:])):
            start, end = spans[0].start, spans[-1].end
            window = Span(f"window:{start}:{end}", start, end, self.document.text[start:end])
            allowed[window.key] = window
        proposed_memory = None
        memory_diagnostics = []
        extraction_started = False
        first_call = self.run.model_calls + 1

        def agent(request):
            nonlocal proposed_memory, extraction_started
            if request["operation"] == "extract_research_evidence":
                # Joining nonadjacent rows creates text that cannot be quoted
                # as one original span. Present the actual spans instead.
                request = {k: v for k, v in request.items() if k != "source_text"}
                memory_spans = list(dict.fromkeys([*allowed,
                    *(anchor["span"] for anchor in self.memory["anchors"])]))
                request = {**request, "reading_memory": self.memory,
                    "source_spans": {k: s.as_record() for k, s in allowed.items()},
                    "response_contract": {**request["response_contract"],
                        "properties": {**request["response_contract"]["properties"], "reading_memory": {
                            "type": "object", "additionalProperties": False,
                            "required": ["summary", "unresolved_links", "anchors"], "properties": {
                                "summary": {"type": "string", "description": "Concise working context, not evidence"},
                                "unresolved_links": {"type": "array", "items": {"type": "string"}},
                                "anchors": {"type": "array", "items": {
                                    "type": "object", "additionalProperties": False, "required": ["span", "quote"],
                                    "properties": {"span": {"type": "string", "enum": memory_spans,
                                                            "description": "Available or retained source span key"},
                                                   "quote": {"type": "string", "minLength": 1}}}}}}}},
                    "reading_instruction": (
                        "source_spans contains the original text at each addressable source range. "
                        "These entries are not necessarily adjacent in the document. Every quote must "
                        "be entirely contained in one entry's text: do not concatenate rows or add "
                        "separators, line breaks, headings or other formatting. Table headers and other "
                        "provided spans may inform interpretation without being joined into the quote. "
                        "Use retained context to resolve subjects across windows, never as quoted evidence. "
                        "Choose memory anchors from the declared available or retained span keys. "
                        "Keep memory concise; it earns no credit.")}
            before = self.run.model_calls
            try:
                response = self.run.complete(request, tier=(self.extraction_tier
                    if request["operation"] == "extract_research_evidence" else None))
            finally:
                if request["operation"] == "extract_research_evidence":
                    extraction_started = self.run.model_calls > before
            if request["operation"] == "extract_research_evidence":
                proposed_memory = response.get("reading_memory", self.memory)
                try:
                    self.validate_memory(proposed_memory, {**self.seen_spans, **allowed})
                except ValueError as exc:
                    memory_diagnostics.append({"stage": "reading_memory", "category": "invalid_memory", "status": "not_applied",
                                               "why": str(exc), "source": self.document.source})
                    proposed_memory = self.memory
            return response

        try:
            result = extract_and_admit(agent, text="\n".join(s.text for s in spans),
                source=self.document.source, objective=str(self.goal.objective),
                context=self.context(), question=question,
                feedback=(self.parent_report(self.ledger.result()),),
                source_spans={k: s.text for k, s in allowed.items()}, source_version=self.document.sha256)
        except (ProviderError, ValueError, TypeError, KeyError, OSError) as exc:
            result = self.failure_result(exc, "source_reading")
        result = replace(result, reading_attempts=(*result.reading_attempts, {
            "source": self.document.source, "source_version": self.document.sha256,
            "document_characters": len(self.document.text), "episode_id": self.goal.goal_id,
            "first_model_call": first_call, "question": question,
            "page_task": getattr(self, "page_task", None),
            "ranges": [[s.start, s.end] for s in spans],
            "status": "completed" if result.evaluated else "failed" if extraction_started else "not_started",
            "admitted_fields": sorted({f.field for f in result.facts} | {f for e in result.evidence for f in e.get("fields", ())}),
            "diagnostic_categories": sorted({d.get("category", "unknown") for d in result.diagnostics}),
        }))
        if memory_diagnostics:
            result = replace(result, diagnostics=(*result.diagnostics, *memory_diagnostics))
        if result.evaluated:
            self.pending_spans, self.pending_memory = spans, proposed_memory
            self.seen_spans.update(allowed)
        return result

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
            remaining_row_count=len(remaining), previous_queries=self.queries,
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
        return self.leaf(lambda: self.read(spans, question), f"table_query:{view.units_consumed + 1}")

    def after_verdict(self, unit, contribution, view):
        if self.pending is not None and self.pending.evaluated:
            self.queried.update(s.key for s in self.pending_spans if s.key != self.table.header.key)
        super().after_verdict(unit, contribution, view)

    @staticmethod
    def no_matches(query):
        return ResearchResult(findings=(f"No remaining table rows match literal query {query!r}; this is not a scientific refutation.",))
