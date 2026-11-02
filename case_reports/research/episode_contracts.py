"""Role-specific literature commissions and bounded deterministic returns."""
from dataclasses import dataclass

from ...method_loop.communication import ReportContract
from .communication import compact_reading_coverage
from .records import Ledger

COMMON_RETURN = {
    "completion": "Execution status; yield or exhaustion does not establish scientific completion.",
    "causal_sheet_changes": "At most eight findings with record IDs, source anchors, bearing on the question and open reasoning questions. Further records are addressable through pagination.",
    "reading_coverage": "At most eight coverage records; an examined passage need not supply evidence.",
    "operations": "At most three failures or limitations; failed retrieval is not negative scientific evidence.",
}

GRADING_RETURN = {
    "score_component_changes": "Numerical changes to all five scientific score components since this episode began; repeated calculations earn zero.",
    "source_findings": "Latest reader synthesis per source and reading episode, with unresolved questions; source records retain exact evidence.",
    "grading_needs": "Source-linked assessment questions and qualifications, including needs outside the generic findings preview. Outcomes link actual Page actions; they do not certify resolution.",
    "grading_outcomes": "Latest actual Page sheet/grader outcomes per need and operation, with provenance, computed results, numerical score changes and diagnostics.",
    "grading_action_count": "Number of recorded Page tool actions; execution itself earns zero.",
    "grading_status_meaning": "Interpretation of needs and factual outcomes; no model verdict gates credit.",
}


@dataclass(frozen=True)
class ScientificAttemptCommission:
    """One independent start at the shared scientific objective."""

    question: str
    inputs: tuple[str, ...]
    snapshots: tuple[dict, ...] = ()

    def __post_init__(self):
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("A scientific attempt needs a starting question")
        if not isinstance(self.inputs, (list, tuple)) or len(self.inputs) > 8 or any(
                not isinstance(key, str) for key in self.inputs):
            raise ValueError("Select at most eight scientific starting inputs")
        if not isinstance(self.snapshots, (list, tuple)) or len(self.snapshots) > 8 or any(
                not isinstance(ref, dict) or set(ref) != {"key", "version"} for ref in self.snapshots):
            raise ValueError("Select at most eight exact historical starting records")
        object.__setattr__(self, "inputs", tuple(dict.fromkeys(self.inputs)))
        object.__setattr__(self, "snapshots", tuple(dict(ref) for ref in self.snapshots))

    def as_record(self):
        return {"question": self.question, "inputs": list(self.inputs),
                "snapshots": list(self.snapshots)}

    def request(self, sheet):
        return {"question": self.question, "inputs": sheet.view(self.inputs),
                "historical_inputs": sheet.snapshot_view(self.snapshots)}

    def report_contract(self):
        return ReportContract(decision=self.question,
            measurements={"scope": self.as_record()},
            information={**COMMON_RETURN, **GRADING_RETURN,
                "starting_question": "The starting question of this attempt, retained to avoid repeating the same unproductive start.",
                "scientific_changes": "Kinds and keys of changed scientific records, excluding source evidence; counts remain visible.",
                "score_change": "Numeric sum of the five score-component changes relative to this attempt's starting sheet.",
                "score_component_changes": "Numeric changes for G registration, expected generalizability bounds registration, theory/context application registration, support for G and causal change."})


@dataclass(frozen=True)
class LiteratureCommission:
    question: str
    inputs: tuple[str, ...]
    snapshots: tuple[dict, ...] = ()

    def __post_init__(self):
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("Literature needs a specific information question")
        if not isinstance(self.inputs, (list, tuple)) or len(self.inputs) > 8 or any(not isinstance(k, str) for k in self.inputs):
            raise ValueError("Select at most eight literature inputs")
        if not isinstance(self.snapshots, (list, tuple)) or len(self.snapshots) > 8 or any(
            not isinstance(ref, dict) or set(ref) != {"key", "version"} for ref in self.snapshots):
            raise ValueError("Select at most eight exact historical records")
        object.__setattr__(self, "inputs", tuple(dict.fromkeys(self.inputs)))
        object.__setattr__(self, "snapshots", tuple(dict(ref) for ref in self.snapshots))

    def as_record(self):
        return {"question": self.question, "inputs": list(self.inputs), "snapshots": list(self.snapshots)}

    def request(self, sheet):
        return {"question": self.question, "inputs": sheet.view(self.inputs),
                "historical_inputs": sheet.snapshot_view(self.snapshots)}

    def report_contract(self):
        return ReportContract(decision=self.question, measurements={"scope": self.as_record()},
                              information={**COMMON_RETURN, **GRADING_RETURN, **self.return_fields})


class SearchStrategyCommission(LiteratureCommission):
    return_fields = {"searches": "Last eight literal queries and count omitted; guides changes of search approach.",
                     "reasoning_questions": "At most eight source-linked unresolved questions for the scientific parent to turn into reasoning commissions."}


class SearchCommission(LiteratureCommission):
    return_fields = {"query": "The single literal query whose results this episode selects among.",
                     "examined_sources": "At most eight sources actually read, with versions; not just search hits."}


class PageCommission(LiteratureCommission):
    return_fields = {"source": "The retrieved document identity and version.",
                     "remaining_source": "Unexamined character count and available table count for selecting further reading."}


class ReportCommission(LiteratureCommission):
    return_fields = {"reading_findings": "Reader's concise synthesis and unresolved links, with exact source anchors.",
                     "document_position": "Document-order cursor and remaining windows; cross-window context stays with the reader."}


class LexicalCommission(LiteratureCommission):
    return_fields = {"reading_findings": "Reader's concise synthesis and unresolved links, with exact source anchors.",
                     "passage_selection": "Literal source-local query, number of ranked passages considered and remaining."}


class TableCommission(LiteratureCommission):
    return_fields = {"reading_findings": "Reader's concise synthesis and unresolved links, with exact source anchors.",
                     "table_queries": "Last eight row queries, selected row keys and remaining row count; findings retain headers and exact row anchors."}


def changed_sheet(source, result):
    from .engine import changed_records
    delta = changed_records(result, source.initial)
    sheet = Ledger(source.initial)
    sheet.merge(delta)
    return delta, sheet.causal_sheet


def completion_fields(result, completion):
    return {"ended_by": getattr(completion, "ended_by", result.stop_reason) or "in_progress",
            "evaluated": result.evaluated}


def operation_fields(delta):
    return [{k: row.get(k) for k in ("stage", "category", "status", "why")} for row in delta.diagnostics[-3:]]


def literature_report(source, result, completion):
    from .grading_handoff import grading_report
    from ...credit_assignment.causal_progress import compile_episode_points
    from fractions import Fraction
    delta, sheet = changed_sheet(source, result)
    before = compile_episode_points(Ledger(source.initial).causal_sheet)["components"]
    after = compile_episode_points(sheet)["components"]
    summaries = {}
    for attempt in delta.reading_attempts:
        if attempt.get("reading_summary"):
            summaries[(attempt["source"], attempt["source_version"], attempt["episode_id"])] = {
                name: attempt[name] for name in ("source", "source_version", "question", "reading_summary", "unresolved_links")}
    return {"completion": completion_fields(result, completion),
            **grading_report(delta, sheet),
            "score_component_changes": {name: float(Fraction(value) - Fraction(before[name]))
                                        for name, value in after.items()},
            "source_findings": list(summaries.values()),
            "causal_sheet_changes": sheet.return_page(delta.causal_sheet_updates),
            "reading_coverage": compact_reading_coverage(delta.reading_attempts),
            "operations": operation_fields(delta)}


def reasoning_report(source, result, completion):
    delta, sheet = changed_sheet(source, result)
    keys = list(dict.fromkeys(row["key"] for row in delta.causal_sheet_updates))
    arguments, unresolved = [], []
    for key in keys:
        row = sheet.rows[key]
        data = row["data"]
        if row["kind"] == "reasoning":
            arguments.append({"key": key, "premises": data["premises"][:8],
                "premise_count": len(data["premises"]), "conclusion": data["conclusion"][:800],
                "limitations": data["limitations"][:800], "current": sheet.current(key),
                "derivation": {"key": key, "version": row["version"]}})
            for premise in data["premises"]:
                dependency = sheet.rows[premise]
                linked_ground = any(sheet.rows[ref]["kind"] in {"evidence", "reasoning"} and sheet.current(ref)
                                    for ref in dependency["dependencies"])
                if (dependency["kind"] == "premise" and not linked_ground) or not sheet.current(premise):
                    unresolved.append({"argument": key, "premise": premise,
                        "reason": "Input is stale" if not sheet.current(premise) else "No current evidence or argument is linked to this premise"})
        elif row["kind"] == "verification":
            arguments.append({"key": key, "argument": data["reasoning"], "legacy_model_opinion": data["verdict"], "scoring_effect": "none",
                "checks": [text[:400] for text in data["checks"][:4]], "counterexample": data["counterexample"][:800]})
    return {"completion": completion_fields(result, completion),
            "causal_sheet_changes": sheet.return_page(delta.causal_sheet_updates,
                calculation_context=source.commission.calculation_context),
            "reasoning_results": {"items": arguments[:8], "total": len(arguments)},
            "unresolved_premises": {"items": unresolved[:8], "total": len(unresolved)},
            "work_outcomes": {"items": source.own_action_outcomes,
                              "total": source.own_action_count},
            "operations": operation_fields(delta)}
