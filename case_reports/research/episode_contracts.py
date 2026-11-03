"""Role-specific commissions and episode-owned job returns."""
from dataclasses import dataclass

from ...method_loop.communication import ReportContract

from .episode_memory import RETURN_FIELDS, job_report


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
            information={**RETURN_FIELDS,
                "assessment": "Scientific answer about G and its theoretical role, expected generalizability bounds and their revisions, qualifications and next uncertainties.",
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
                              information={**RETURN_FIELDS, "assessment": self.assessment_meaning, **self.return_fields})


class SearchStrategyCommission(LiteratureCommission):
    assessment_meaning = "Commission answer across searches, coverage and gaps, qualifications and specific remaining reasoning questions, with selected references."
    return_fields = {"searches": "Query count and latest literal query; the strategy assessment explains coverage and gaps.",
                     "reasoning_questions": "The strategy assessment's remaining logical questions for the scientific parent."}


class SearchCommission(LiteratureCommission):
    assessment_meaning = "Answer from this query's examined sources, their comparison and remaining information need, with selected references."
    return_fields = {"query": "The single literal query whose results this episode selects among.",
                     "examined_sources": "Sources selected from this query, with retrieved versions when available; a selection alone is not successful reading."}


class PageCommission(LiteratureCommission):
    assessment_meaning = "Source answer, status of scientific assessment tied to actual grader receipts, qualifications and remaining reading need."
    return_fields = {"source": "The retrieved document identity and version.",
                     "remaining_source": "Unexamined character count and available table count for selecting further reading."}


class ReportCommission(LiteratureCommission):
    assessment_meaning = "Source-grounded answer across document-order windows, cross-window qualifications and unresolved links or grading needs for Page."
    return_fields = {"source_anchors": "Source identity, version and exact spans retained by this reader; findings and open questions belong to its assessment.",
                     "document_position": "Document-order cursor and remaining windows; cross-window context stays with the reader."}


class LexicalCommission(LiteratureCommission):
    assessment_meaning = "Answer from source-local ranked passages, context and qualifications, and unresolved source links or grading needs for Page."
    return_fields = {"source_anchors": "Source identity, version and exact spans retained by this reader; findings and open questions belong to its assessment.",
                     "passage_selection": "Literal source-local query, number of ranked passages considered and remaining."}


class TableCommission(LiteratureCommission):
    assessment_meaning = "Answer from queried table rows with their headers and conditions, qualifications and unresolved comparisons or grading needs for Page."
    return_fields = {"source_anchors": "Source identity, version and exact spans retained by this reader; findings and open questions belong to its assessment.",
                     "table_queries": "Last eight row queries, selected row keys and remaining row count; findings retain headers and exact row anchors."}


def completion_fields(result, completion):
    return {"ended_by": getattr(completion, "ended_by", result.stop_reason) or "in_progress",
            "evaluated": result.evaluated}


def literature_report(source, result, completion):
    return job_report(source, result, completion)


def reasoning_report(source, result, completion):
    return job_report(source, result, completion)
