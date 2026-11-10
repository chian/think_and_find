"""Exact source-anchor builders for retained and authored verifier inputs."""
from copy import deepcopy
from types import SimpleNamespace

from episodic_workflow.case_reports.research import source_anchors
from episodic_workflow.case_reports.research.documents import Document, Span
from episodic_workflow.case_reports.research.records import Ledger, ResearchResult


def anchor(document, passages, quote, *, unit_id=None, preferred_unit_id=None):
    """Use a fixture-specified exact quotation, never infer scientific identity."""
    choices = []
    for key, span in passages.items():
        for unit in source_anchors.source_units(document):
            if unit_id is not None and unit["id"] != unit_id:
                continue
            left, right = max(unit["start"], span.start), min(unit["end"], span.end)
            if left >= right:
                continue
            fragment = span.text[left - span.start:right - span.start]
            offset = fragment.find(quote)
            if offset >= 0 and fragment.find(quote, offset + 1) < 0:
                choices.append({"span": key, "unit": unit["id"], "quote": quote})
    assert choices, f"No exact single-unit fixture anchor for {quote!r}"
    preferred = [choice for choice in choices if choice["unit"] == preferred_unit_id]
    if preferred:
        return preferred[0]
    return choices[0]


def result_anchor(document, passages, literal, occurrence, *, unit_id):
    """Explicit value-only quote and 1-based occurrence, never inferred context."""
    assert isinstance(literal, str) and literal and type(occurrence) is int and occurrence > 0
    unit = next(unit for unit in source_anchors.source_units(document) if unit["id"] == unit_id)
    for key, span in passages.items():
        left, right = max(unit["start"], span.start), min(unit["end"], span.end)
        if left >= right:
            continue
        fragment = span.text[left - span.start:right - span.start]
        offset = -1
        for _ in range(occurrence):
            offset = fragment.find(literal, offset + 1)
            if offset < 0:
                break
        if offset >= 0:
            return {"span": key, "unit": unit_id, "quote": literal, "occurrence": occurrence}
    raise AssertionError(f"No occurrence {occurrence} of explicit result literal {literal!r} in selected unit")


def observation_binding(document, passages, statement, quantity, result, occurrence=1, *conditions, quantity_unit_id=None):
    statement_anchor = anchor(document, passages, statement)
    return {"statement": statement_anchor,
            "quantity": anchor(document, passages, quantity, unit_id=quantity_unit_id,
                               preferred_unit_id=statement_anchor["unit"]),
            "result": result_anchor(document, passages, result, occurrence, unit_id=statement_anchor["unit"]),
            "conditions": [anchor(document, passages, value) for value in conditions]}


def claim_binding(document, passages, statement, subjects, conditions=()):
    return {"statement": anchor(document, passages, statement),
            "subjects": [anchor(document, passages, value) for value in subjects],
            "conditions": [anchor(document, passages, value) for value in conditions]}


def stored_sidecar(document, passages, observations=None, claims=None, findings=(), ledger=None):
    reader = SimpleNamespace(document=document, ledger=ledger or Ledger(ResearchResult()))
    known = source_anchors.registry(reader)
    sidecar = {"observations": {}, "claims": {}, "finding": [], "pending": False}
    for category, values in (("observations", observations or {}), ("claims", claims or {})):
        for selector, value in values.items():
            sidecar[category][selector] = source_anchors.resolve_entry(reader, passages, category, selector, value, known)
    sidecar["finding"] = source_anchors.resolve_finding(reader, passages,
        [anchor(document, passages, quote) for quote in findings], known)
    return source_anchors.finalize(reader, passages, sidecar)


def retained_document(source, passage_groups):
    """Reconstruct supplied source spans at their actual offsets; gaps are blank.

    This is explicitly a retained-span fixture, not the full publication body.
    The reconstructed document has its own SHA rather than impersonating the
    original full document's version.
    """
    spans = [Span(**value) for passages in passage_groups for value in passages.values()]
    text = [" "] * max(span.end for span in spans)
    for span in spans:
        previous = "".join(text[span.start:span.end])
        assert all(old == " " or old == new for old, new in zip(previous, span.text))
        text[span.start:span.end] = span.text
    return Document(source, "Retained exact publication passages", "".join(text), 12000)


def unit_passages(passages):
    """Reconstruct explicit unit-fragment fixtures at their retained coordinates."""
    spans = {}
    for key, supplied in passages.items():
        text = [" "] * (supplied["end"] - supplied["start"])
        for unit in supplied["source_units"]:
            text[unit["start"] - supplied["start"]:unit["end"] - supplied["start"]] = unit["text"]
        spans[key] = Span(key, supplied["start"], supplied["end"], "".join(text))
    return spans


def attach_response_anchors(document, passages, response, observations, claims, prose):
    """Bindings are explicit fixture quotations indexed by native selectors."""
    result = deepcopy(response)
    for index, finding in enumerate(result["findings"]):
        native = finding["measurements"]
        finding["source_anchors"] = {
            "observations": {key: observation_binding(document, passages, *observations[key])
                             for key in native["observations"]},
            "claims": {key: claim_binding(document, passages, *claims[key]) for key in native["claims"]},
            "finding": [anchor(document, passages, quote) for quote in prose.get(index, ())]}
    return result
