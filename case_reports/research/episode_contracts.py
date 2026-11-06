"""Role-specific commissions and episode-owned job returns."""
from dataclasses import dataclass, field
from copy import deepcopy

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
                "assessment": "Scientific answer about G in each explanatory task, expected generalizability bounds and their revisions, attempted transfers, evidence of failure and its explanation, tested repairs and their costs, qualifications and next uncertainties.",
                "starting_question": "The starting question of this attempt, retained to avoid repeating the same unproductive start.",
                "scientific_changes": "Kinds and keys of changed scientific records, excluding source evidence; counts remain visible.",
                "score_change": "Numeric sum of the five score-component changes relative to this attempt's starting sheet.",
                "score_component_changes": "Numeric changes for G registration, expected generalizability bounds registration, theory/context application registration, support for G and causal change."})


@dataclass(frozen=True)
class LiteratureCommission:
    question: str
    inputs: tuple[str, ...]
    snapshots: tuple[dict, ...] = ()
    theory_request: dict = field(kw_only=True)

    def __post_init__(self):
        from .theory_runtime import validate_request
        object.__setattr__(self, "theory_request", validate_request(self.theory_request))
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("Literature needs a specific information question")
        if not isinstance(self.inputs, (list, tuple)) or any(not isinstance(k, str) for k in self.inputs):
            raise ValueError("Literature inputs must be record keys")
        if not isinstance(self.snapshots, (list, tuple)) or any(
            not isinstance(ref, dict) or set(ref) != {"key", "version"} for ref in self.snapshots):
            raise ValueError("Literature snapshots require exact keys and versions")
        object.__setattr__(self, "inputs", tuple(dict.fromkeys(self.inputs)))
        object.__setattr__(self, "snapshots", tuple(dict(ref) for ref in self.snapshots))

    def as_record(self):
        return {"question": self.question, "inputs": list(self.inputs), "snapshots": list(self.snapshots),
                "theory_request": deepcopy(self.theory_request)}

    def request(self, sheet):
        from .theory_runtime import validate_request
        validate_request(self.theory_request, sheet)
        return {"question": self.question, "inputs": sheet.view(self.inputs),
                "historical_inputs": sheet.snapshot_view(self.snapshots),
                "theory_request": deepcopy(self.theory_request)}

    def report_contract(self):
        information = {key: value for key, value in RETURN_FIELDS.items()
                       if key != "score_component_changes"}
        information.update(THEORY_RETURN_FIELDS)
        return ReportContract(decision=self.question, measurements={"scope": self.as_record()},
                              information={**information, "assessment": self.assessment_meaning, **self.return_fields})


THEORY_RETURN_FIELDS = {
    "theory_fragment": "The shared typed reconstruction with its exact stored identity and version.",
    "theory_contributions": "Typed additions addressing this request, deduplicated contribution identities, and unresolved validation repair."}


@dataclass(frozen=True)
class TheoryReconstructionCommission(LiteratureCommission):
    seed: dict | None = field(kw_only=True)
    assessment_meaning = "Connected account of the commissioned theory, its mechanisms, assumptions, derivations and substantive gaps."
    return_fields = {}

    def as_record(self):
        return {**super().as_record(), "seed": deepcopy(self.seed)}

    def request(self, sheet):
        from .causal_sheet import CausalSheet
        key = self.theory_request["reconstruction"]
        if key not in sheet.rows:
            if self.seed is None:
                raise ValueError("A new reconstruction needs the parent's explicit declaration seed")
            sheet = CausalSheet(sheet.history)
            sheet.put(key, "theory_fragment", self.seed)
        elif self.seed is not None:
            raise ValueError("Continue an existing reconstruction with seed=null; additions use typed integration")
        return super().request(sheet)


class TheoryGapCommission(LiteratureCommission):
    assessment_meaning = "What focused reading and additional literature establish about the selected reconstruction gaps; unresolved or intentionally unspecified relationships remain explicit."
    return_fields = {}


class TheoryFormatRepairCommission(LiteratureCommission):
    assessment_meaning = "Corrections of rejected typed theory additions using exact validator errors and supplied evidence."
    return_fields = {"repair_status": "Host validation outcome, unresolved errors and rejected additions; no scientific truth verdict."}


class SearchStrategyCommission(LiteratureCommission):
    assessment_meaning = "Commission answer across searches, coverage and gaps, qualifications and specific remaining reasoning questions, with selected references."
    return_fields = {"searches": "Query count and latest literal query; the strategy assessment explains coverage and gaps.",
                     "reasoning_questions": "The strategy assessment's remaining logical questions for the scientific parent."}


class SearchCommission(LiteratureCommission):
    assessment_meaning = "Answer from this query's examined sources, their comparison and remaining information need, with selected references."
    return_fields = {"query": "The single literal query whose results this episode selects among.",
                     "examined_sources": "Sources selected from this query, with retrieved versions when available; a selection alone is not successful reading."}


class PageCommission(LiteratureCommission):
    assessment_meaning = "Source answer, typed reconstruction additions, source qualifications and remaining reading need."
    return_fields = {"source": "The retrieved document identity and version.",
                     "remaining_source": "Unexamined character count and available table count for selecting further reading."}


class ReadingCommission(LiteratureCommission):
    """A processing child returns extraction, without scientific score fields."""

    def report_contract(self):
        information = {name: meaning for name, meaning in RETURN_FIELDS.items()
                       if name != "score_component_changes"}
        information.update(
            assessment=self.assessment_meaning,
            extracted_records="Source records extracted by this reader with exact versions and provenance.",
            extraction_credit="Local credit for distinct returned source information; repeated information adds zero.",
            extraction_errors="Unresolved validator or request errors for Page, including the rejected fields when available.",
            **self.return_fields)
        return ReportContract(decision=self.question, measurements={"scope": self.as_record()},
                              information=information)


class ReportCommission(ReadingCommission):
    assessment_meaning = "Extracted source information, qualifications and unresolved source links for Page."
    return_fields = {"source_anchors": "Source identity, version and exact spans retained by this reader; findings and open questions belong to its assessment.",
                     "document_position": "Document-order cursor and remaining windows; cross-window context stays with the reader."}


class LexicalCommission(ReadingCommission):
    assessment_meaning = "Extracted passage information, qualifications and unresolved source links for Page."
    return_fields = {"source_anchors": "Source identity, version and exact spans retained by this reader; findings and open questions belong to its assessment.",
                     "passage_selection": "Literal source-local query, number of ranked passages considered and remaining."}


class TableCommission(ReadingCommission):
    assessment_meaning = "Native table rows, headers and conditions returned to Page."
    return_fields = {"source_anchors": "Source identity, version and exact spans retained by this reader; findings and open questions belong to its assessment.",
                     "table": "Native table identity, columns, headers and selected rows with source provenance.",
                     "table_queries": "Last eight row queries, selected row keys and remaining row count; findings retain headers and exact row anchors."}


class ExtractionRepairCommission(ReadingCommission):
    assessment_meaning = "Corrected source extraction, retained valid findings, and remaining exact validation errors."
    return_fields = {
        "source_anchors": "Source identity, version and exact spans supplied for correction.",
        "repair_status": "Host extraction validation status, corrected reading memory, rejected output and unresolved errors."}


def completion_fields(result, completion):
    return {"ended_by": getattr(completion, "ended_by", result.stop_reason) or "in_progress",
            "evaluated": result.evaluated}


def literature_report(source, result, completion):
    return job_report(source, result, completion)


def reasoning_report(source, result, completion):
    return job_report(source, result, completion)
