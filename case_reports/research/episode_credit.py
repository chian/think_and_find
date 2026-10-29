"""Episode-scoped projections of executed scientific contributions.

Scientific points have one global identity. Source scopes select contributions
through scientific provenance; activity, passage length and source count are not
rewards. Completed calculation coverage is a separate Reasoning measurement.
"""
from fractions import Fraction

from .records import Credit, Ledger
from ...credit_assignment.causal_sheet import investigation_credit
from ...credit_assignment.causal_support import compile_support, controller_projection


def _source_evidence(sheet, predicate):
    return {key for key, row in sheet.of_kind("evidence") if predicate(row["data"])}


def _touches_evidence(sheet, references, evidence):
    todo, seen = list(references), set()
    while todo:
        key = todo.pop()
        if key in evidence:
            return True
        if key in seen or key not in sheet.rows:
            continue
        seen.add(key)
        todo.extend(sheet.rows[key]["dependencies"])
    return False


def _scientific_gain(result, baseline, column, evidence):
    sheet = Ledger(result).causal_sheet
    score = compile_support(sheet)
    before = compile_support(Ledger(baseline).causal_sheet)["points"]
    points = {}
    for key, value in score["points"].items():
        # Subtract the whole inherited point before provenance filtering. Adding
        # another source for an existing atom cannot earn that atom again.
        gain = max(Fraction(0), Fraction(value) - Fraction(before.get(key, "0")))
        if gain and _touches_evidence(sheet, score["point_dependencies"].get(key, ()), evidence):
            points[key] = gain
    projection = controller_projection(points, score["resolution"])
    if not projection["supported"]:
        return Credit(evaluated=False)
    return Credit(evaluated=result.evaluated, quality_resolution=projection["resolution"],
        qualities=tuple((column, key, value) for key, value in sorted(projection["values"].items())))


def search_strategy_credit(result, baseline):
    sheet = Ledger(result).causal_sheet
    original = Ledger(baseline).causal_sheet
    returned = {key for key, row in sheet.of_kind("evidence")
                if key not in original.rows or original.rows[key]["version"] != row["version"]}
    return _scientific_gain(result, baseline, "strategy_scientific_evidence_gain", returned)


def search_credit(result, baseline, source_urls):
    sheet = Ledger(result).causal_sheet
    evidence = _source_evidence(sheet, lambda row: row["source"] in source_urls)
    return _scientific_gain(result, baseline, "query_scientific_evidence_gain", evidence)


def page_credit(result, baseline, source_url):
    sheet = Ledger(result).causal_sheet
    evidence = _source_evidence(sheet, lambda row: row["source"] == source_url)
    return _scientific_gain(result, baseline, "page_scientific_evidence_gain", evidence)


def _within_scope(row, document, spans):
    if row["source"] != document.source or row["measurements"].get("source_version") != document.sha256:
        return False
    locator = row["locator"]
    if not locator.startswith("characters "):
        return False
    try:
        start, end = map(int, locator.removeprefix("characters ").split(":"))
    except ValueError:
        return False
    return any(start < span.end and end > span.start for span in spans)


def report_credit(result, baseline, document, spans):
    sheet = Ledger(result).causal_sheet
    evidence = _source_evidence(sheet, lambda row: _within_scope(row, document, spans))
    return _scientific_gain(result, baseline, "report_scientific_evidence_gain", evidence)


def lexical_credit(result, baseline, document, parts):
    sheet = Ledger(result).causal_sheet
    spans = tuple(span for group in parts.values() for span in group)
    evidence = _source_evidence(sheet, lambda row: _within_scope(row, document, spans))
    return _scientific_gain(result, baseline, "lexical_scientific_evidence_gain", evidence)


def table_credit(result, baseline, document, table):
    sheet = Ledger(result).causal_sheet
    evidence = _source_evidence(sheet, lambda row: _within_scope(row, document, table.rows))
    return _scientific_gain(result, baseline, "table_scientific_evidence_gain", evidence)


def reasoning_credit(result, baseline, commission):
    sheet = Ledger(result).causal_sheet
    score = compile_support(sheet)
    before = compile_support(Ledger(baseline).causal_sheet)["points"]
    points = {key: max(Fraction(0), Fraction(value) - Fraction(before.get(key, "0")))
              for key, value in score["points"].items()}
    projection = controller_projection(points, score["resolution"])
    coverage = investigation_credit(sheet, commission.assignments, context=commission.calculation_context)
    if not projection["supported"] or not coverage.evaluated:
        return Credit(evaluated=False)
    return Credit(evaluated=result.evaluated,
        quality_resolution=min(projection["resolution"], coverage.quality_resolution or 1.0),
        qualities=tuple(("reasoning_scientific_gain", key, value) for key, value in projection["values"].items()) +
                  tuple(("assigned_calculation_coverage", key, value) for _, key, value in coverage.qualities))
