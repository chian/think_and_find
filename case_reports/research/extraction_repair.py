"""A reader's iterative repair of one rejected source extraction."""
from copy import deepcopy

from .acquisition import ReadingSource
from .episode_contracts import ExtractionRepairCommission
from .episode_credit import _extraction_gain, _within_scope


class ExtractionRepairSource(ReadingSource):
    """Correct source-output defects without taking over theory reconstruction."""

    commission_type = ExtractionRepairCommission
    assessment_fields = {
        "reading_answer": "Valid source information retained while correcting the rejected extraction.",
        "open_questions": "Unresolved source-output defects and source-context questions for the reader."}
    instructions = (
        "Repair the supplied rejected source extraction using its exact validator errors and source passages. "
        "Return corrections only for the host-indexed rejected targets, exactly once per issue. Accepted "
        "findings and fields are fixed; do not reword or replace them. A rejected claim or observation may "
        "be removed with null when it cannot be represented, retaining its finding prose. Replace the whole "
        "response only when the host declares a root target after an original parse or top-level shape failure. "
        "Do not invent source information to satisfy the schema. Preserve source-specific "
        "qualifications and report unresolved defects. The commissioning reader owns further passage selection; "
        "Page and its acquisition parents own source coverage; Scientific owns theory reconstruction. "
        "Credit measures newly returned source information. "
        "The numerical controller decides whether another corrective attempt is worthwhile.")
    credit_columns = ("extraction_repair_gain",)

    def __init__(self, run, inherited, *, document, repair, memory, seen_spans):
        super().__init__(run, inherited, document=document)
        self.spans = tuple(repair["spans"])
        self.extraction_repair = deepcopy(repair)
        self.memory = deepcopy(memory)
        self.seen_spans = dict(seen_spans)
        self.known_records.update((ref["key"], ref["version"])
                                  for ref in self.memory["retained_evidence"])

    def credit(self, result):
        return _extraction_gain(result, self.initial, "extraction_repair_gain",
            lambda row: _within_scope(row, self.document, self.spans))

    def propose(self, view):
        repair = self.extraction_repair
        if repair is None:
            # The one commissioned extraction has been repaired; no further
            # source window belongs to this child.
            return None
        return self.leaf(lambda: self.read(self.spans, repair["question"]),
                         f"correct_source_extraction:{view.units_consumed + 1}")

    def parent_report(self, result, completion=None, request=None):
        report = super().parent_report(result, completion, request)
        repair = self.extraction_repair
        report["repair_status"] = {
            "repaired": repair is None,
            "reading_memory": deepcopy(self.memory),
            "rejected_response": deepcopy(repair["response"]) if repair is not None else None,
            "errors": deepcopy(repair["errors"]) if repair is not None else [],
        }
        return report
