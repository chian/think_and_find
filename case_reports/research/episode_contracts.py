"""Closed commissions for Root's two arms and their concrete child episodes."""
from dataclasses import dataclass, field
from copy import deepcopy

from ...method_loop.communication import ReportContract
from .episode_memory import RETURN_FIELDS, job_report
from .causal_commission import SheetCommission


DISCOVERY_GOALS_RETURN = {
    "discovery_goals": "Exact newly created information goals, their definitions, answer types and executable criteria."}


def _scope(question, inputs, snapshots):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A commission needs its scientific question")
    if not isinstance(inputs, (list, tuple)) or any(not isinstance(key, str) for key in inputs):
        raise ValueError("Commission inputs are scientific record keys")
    if not isinstance(snapshots, (list, tuple)) or any(
            not isinstance(ref, dict) or set(ref) != {"key", "version"} for ref in snapshots):
        raise ValueError("Commission snapshots identify exact scientific record versions")


@dataclass(frozen=True)
class ScientificCommission:
    question: str
    inputs: tuple[str, ...]
    snapshots: tuple[dict, ...] = ()

    def __post_init__(self):
        _scope(self.question, self.inputs, self.snapshots)
        object.__setattr__(self, "inputs", tuple(dict.fromkeys(self.inputs)))
        object.__setattr__(self, "snapshots", tuple(deepcopy(self.snapshots)))

    def as_record(self):
        return {"question": self.question, "inputs": list(self.inputs), "snapshots": list(self.snapshots)}

    def request(self, sheet):
        return {"question": self.question, "inputs": sheet.view(self.inputs),
                "historical_inputs": sheet.snapshot_view(self.snapshots)}

    def report_contract(self):
        return ReportContract(decision=self.question, measurements={"scope": self.as_record()},
            information={**RETURN_FIELDS, **DISCOVERY_GOALS_RETURN,
                "assessment": "Scientific findings, formalized theories/Gs, comparisons and repairs with their evidence and qualifications.",
                "pending_discovery": "Findings whose scientific correspondence needs further reconstruction or comparison."})


@dataclass(frozen=True)
class LiteratureCommission:
    question: str
    inputs: tuple[str, ...]
    snapshots: tuple[dict, ...] = ()
    discovery_goal: str = field(kw_only=True)
    assessment_meaning = "Source-grounded information answering the commissioned discovery goal."
    return_fields = {}

    def __post_init__(self):
        _scope(self.question, self.inputs, self.snapshots)
        if not isinstance(self.discovery_goal, str) or not self.discovery_goal.strip():
            raise ValueError("Acquisition must identify its exact discovery goal")
        object.__setattr__(self, "inputs", tuple(dict.fromkeys(self.inputs)))
        object.__setattr__(self, "snapshots", tuple(deepcopy(self.snapshots)))

    def as_record(self):
        return {"question": self.question, "inputs": list(self.inputs), "snapshots": list(self.snapshots),
                "discovery_goal": self.discovery_goal}

    def request(self, sheet):
        sheet._require_kind(self.discovery_goal, "discovery_goal")
        return {"question": self.question, "inputs": sheet.view(self.inputs),
                "historical_inputs": sheet.snapshot_view(self.snapshots),
                "discovery_goal": sheet.view([self.discovery_goal])[0]}

    def report_contract(self):
        return ReportContract(decision=self.question, measurements={"scope": self.as_record()},
            information={**{key: value for key, value in RETURN_FIELDS.items() if key != "score_component_changes"},
                         "assessment": self.assessment_meaning, **self.return_fields})


class DiscoveryCommission(LiteratureCommission):
    assessment_meaning = "Literal native matches, separately verified scientific requirements, source-grounded findings and pending scientific questions returned to Root."
    return_fields = {"discovery_goal": "Exact commissioned information goal.",
                     "request_credit": "Native matches and scientific fulfillment at their declared verification scopes, with repeated request slots deduplicated.",
                     "scientific_questions": "Pointed unresolved questions for Scientific through Root.",
                     "searches": "Literal search queries selected for the goal."}


class SearchCommission(LiteratureCommission):
    assessment_meaning = "Information supplied by sources examined for one literal query, with remaining criteria."
    return_fields = {"query": "The literal query whose sources were examined.",
                     "examined_sources": "Source identities and retrieved versions, with access failures retained."}


class PageCommission(LiteratureCommission):
    assessment_meaning = "Source information, qualifications and remaining reading relevant to the goal."
    return_fields = {"source": "Document identity and exact version.",
                     "remaining_source": "Unexamined document coverage and available tables."}


class ReadingCommission(LiteratureCommission):
    def report_contract(self):
        contract = super().report_contract()
        return ReportContract(decision=self.question, measurements={"scope": self.as_record()},
            information={**contract.information,
                "extracted_records": "Exact source records extracted by this reader.",
                "extraction_credit": "Local credit for distinct returned source information.",
                "extraction_errors": "Unresolved extraction errors returned to Page."})


class ReportCommission(ReadingCommission):
    assessment_meaning = "Document-order source extraction, qualifications and unresolved links."
    return_fields = {"source_anchors": "Source/version and exact retained spans.",
                     "document_position": "Cursor and remaining document windows."}


class LexicalCommission(ReadingCommission):
    assessment_meaning = "Query-ranked source passages, qualifications and unresolved links."
    return_fields = {"source_anchors": "Source/version and exact retained spans.",
                     "passage_selection": "Literal query, selected passages and remaining coverage."}


class TableCommission(ReadingCommission):
    assessment_meaning = "Native table rows, headers and conditions returned to Page."
    return_fields = {"source_anchors": "Source/version and exact retained spans.",
                     "table": "Native table identity, headers, rows and provenance.",
                     "table_queries": "Row-query outcomes and remaining rows."}


class ExtractionRepairCommission(ReadingCommission):
    assessment_meaning = "Corrected rejected extraction, valid retained findings and unresolved exact errors."
    return_fields = {"source_anchors": "Source/version and exact spans supplied for correction.",
                     "repair_status": "Host validation, rejected fields and remaining errors."}


THEORY_RETURN_FIELDS = {
    "theory_fragment": "Exact key/version reference to the shared typed reconstruction.",
    "theory_contributions": "Added component IDs and unresolved representation requirements; definitions remain in the scientific sheet."}


@dataclass(frozen=True)
class TheoryReconstructionCommission(ScientificCommission):
    theory_request: dict = field(kw_only=True)
    seed: dict | None = field(kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        from .theory_runtime import validate_request
        object.__setattr__(self, "theory_request", validate_request(self.theory_request))

    def as_record(self):
        return {**super().as_record(), "theory_request": deepcopy(self.theory_request), "seed": deepcopy(self.seed)}

    def request(self, sheet):
        from .causal_sheet import CausalSheet
        from .theory_runtime import validate_request
        key = self.theory_request["reconstruction"]
        if key not in sheet.rows:
            if self.seed is None:
                raise ValueError("A new reconstruction needs its explicit declaration seed")
            sheet = CausalSheet(sheet.history)
            sheet.put(key, "theory_fragment", self.seed)
        elif self.seed is not None:
            raise ValueError("Existing reconstructions receive additions; continue with seed=null")
        validate_request(self.theory_request, sheet)
        return {**super().request(sheet), "theory_request": deepcopy(self.theory_request)}

    def report_contract(self):
        return ReportContract(decision=self.question, measurements={"scope": self.as_record()},
            information={**RETURN_FIELDS, **THEORY_RETURN_FIELDS, **DISCOVERY_GOALS_RETURN,
                         "assessment": "Connected formalized theory, its Gs, conditions, derivations and unresolved scientific premises."})


class TheoryRepairCommission(TheoryReconstructionCommission):
    pass


class TheoryComparisonCommission(SheetCommission):
    def report_contract(self):
        contract = super().report_contract()
        return ReportContract(decision=contract.decision, measurements=contract.measurements,
            information={**contract.information, **DISCOVERY_GOALS_RETURN})


@dataclass(frozen=True)
class TheoryFormatRepairCommission(TheoryReconstructionCommission):
    def report_contract(self):
        return ReportContract(decision=self.question, measurements={"scope": self.as_record()},
            information={**RETURN_FIELDS, **THEORY_RETURN_FIELDS,
                "assessment": "Corrections of rejected typed theory additions using validator feedback and supplied evidence.",
                "repair_status": "Host validation outcome and unresolved malformed additions."})


def completion_fields(result, completion):
    return {"ended_by": getattr(completion, "ended_by", result.stop_reason) or "in_progress",
            "evaluated": result.evaluated}


def literature_report(source, result, completion):
    return job_report(source, result, completion)
