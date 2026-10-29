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
                              information={**COMMON_RETURN, **self.return_fields})


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
    return_fields = {"document_position": "Document-order cursor and remaining windows; cross-window context stays with the reader."}


class LexicalCommission(LiteratureCommission):
    return_fields = {"passage_selection": "Literal source-local query, number of ranked passages considered and remaining."}


class TableCommission(LiteratureCommission):
    return_fields = {"table_queries": "Last eight row queries, selected row keys and remaining row count; findings retain headers and exact row anchors."}


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
    delta, sheet = changed_sheet(source, result)
    return {"completion": completion_fields(result, completion),
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
            "operations": operation_fields(delta)}
