"""Fixed parameter-evidence credit, independent of generated search objectives."""

from urllib.parse import urlsplit, urlunsplit

from ...task_executors.scenario_metrics.dependencies import input_fields
from ...task_executors.scenario_metrics.evaluate import CONDITIONS, UNITS, validate_parameter
from .modeling import DERIVATIONS, derivation_inputs
from .records import Credit, identity


PARAMETER_EVIDENCE = "parameter_evidence"
ADDITIONAL_SUPPORT = 0.1


def parameter_requirements(parameter):
    """Use executable dependencies, never a search-generated checklist."""
    if parameter not in UNITS and parameter not in CONDITIONS:
        raise ValueError("Investigation parameter must be a declared calculation input")
    names = {parameter}
    if parameter == "quanta_concentration_exhaled_m3":
        for operator in DERIVATIONS:
            if operator != "transfer_parameter":
                names.update(derivation_inputs(operator, {}))
                names.update(derivation_inputs(operator, {"attack_probability": 0}))
    return frozenset(input_fields(names, parameters=UNITS) & (UNITS.keys() | CONDITIONS.keys()))


def _source(record):
    source, quote = record.get("source", ""), record.get("quote", "")
    if not source or not quote or source == "request":
        return None
    if record.get("kind") not in {"source_observation", "source_model_parameter", "source_evidence"}:
        return None
    parts = urlsplit(source)
    locator = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))
    return locator, " ".join(quote.split())


def evidence_key(record):
    return identity({k: record.get(k) for k in (
        "subject", "field", "fields", "value", "unit", "kind", "source", "quote")})


def _distinct_sources(records):
    grouped = {}
    for record in records:
        support = _source(record)
        if support:
            source, quote = support
            grouped.setdefault(source, set()).add(quote)
    sources, seen_quotes = [], set()
    for source, quotes in grouped.items():
        if not seen_quotes.intersection(quotes):
            sources.append(source)
        seen_quotes.update(quotes)
    return sources


def parameter_support(result, requirements, *, hypotheses_only=False, subjects=()):
    """Count filled elements, retaining alternatives rather than selecting a value."""
    from .target_values import invalidated_source_records

    values = {field: {} for field in requirements}
    learning = {field: {} for field in requirements}
    invalid = {evidence_key(record) for record in invalidated_source_records(result)}

    def add(field, subject, value, unit, records, *, relation="supports", reviewed=False):
        if field not in values:
            return
        key = identity([subject, field, value, unit])
        bucket = values[field].setdefault(key, {"subject": subject, "value": value,
            "unit": unit, "supports": [], "contradicts": []})
        for record in records:
            if evidence_key(record) not in invalid and _source(record):
                bucket[relation].append(record)
                if reviewed:
                    learning[field].setdefault(subject, []).append(record)

    if not hypotheses_only:
        for fact in result.facts:
            if fact.kind in {"source_observation", "source_model_parameter"}:
                add(fact.field, fact.subject, fact.value, fact.unit, (fact.semantic_record(),), reviewed=True)
    for assessment in result.assumptions:
        if subjects and assessment["subject"] not in subjects:
            continue
        if assessment.get("role") == "request_constraint":
            continue
        for row in assessment.get("learning_evidence", ()):
            add(assessment["field"], assessment["subject"],
                assessment["value"] if assessment["value"] is not None else assessment["statement"],
                assessment["unit"], (row["evidence"],), relation=row["relation"], reviewed=True)

    rows = []
    for field, candidates in sorted(values.items()):
        alternatives, corroboration = [], set()
        for key, candidate in sorted(candidates.items()):
            sources = _distinct_sources(candidate["supports"])
            contrary = _distinct_sources(candidate["contradicts"])
            if not sources and not contrary:
                continue
            alternatives.append({"subject": candidate["subject"], "value": candidate["value"],
                                 "unit": candidate["unit"], "supporting_sources": sources,
                                 "contradicting_sources": contrary})
        for records in learning[field].values():
            corroboration.update(_distinct_sources(records)[1:])
        if any(row["supporting_sources"] for row in alternatives):
            rows.append({"field": field, "alternatives": alternatives,
                         "points": round(1 + ADDITIONAL_SUPPORT * len(corroboration), 10)})
    return rows


def progress_credit(rows, *, evaluated):
    return Credit(evaluated=evaluated, quality_resolution=ADDITIONAL_SUPPORT,
                  qualities=tuple((PARAMETER_EVIDENCE, row["field"], row["points"]) for row in rows))


def found_parameter_progress(result, *, ledger=None):
    """One discovery floor per independent target value, separate from its support.

    Source records alone are evidence, not target assignments. Review grades and
    applicability qualify an assigned value; they do not erase its discovery.
    """
    from .records import Ledger
    from .scenario_sharing import independent_groups
    from .target_values import invalidated_source_records, target_value_preference

    ledger = ledger if ledger is not None else Ledger(result)
    invalid = {evidence_key(record) for record in invalidated_source_records(ledger)}
    rows = []
    for group in independent_groups(ledger):
        field = group["parameter"]
        candidates = []
        for assessment in ledger.assumptions.values():
            if (assessment.get("subject") not in group["subjects"]
                    or assessment.get("field") != field
                    or assessment.get("source_candidate_unavailable")
                    or assessment.get("role") in
                    {"scenario_constraint", "request_constraint", "model_default"}):
                continue
            try:
                validate_parameter(field, assessment.get("value"), assessment.get("unit"))
            except ValueError:
                continue
            candidates.append(assessment)
        reviewed = [row for row in candidates
                    if row.get("review", {}).get("verdict") in {"supported", "provisional"}]
        selected = max(candidates, key=target_value_preference) if candidates else None
        # Preserve the existing evidence-learning increment without creating a
        # second discovery point when evidence later supports the same value.
        learning = [row for assessment in candidates for row in assessment.get("learning_evidence", ())
                    if evidence_key(row["evidence"]) not in invalid and _source(row["evidence"])]
        sources = _distinct_sources(row["evidence"] for row in learning)
        bonus = (ADDITIONAL_SUPPORT * max(0, len(sources) - 1)
                 if any(row["relation"] == "supports" for row in learning) else 0.0)
        rows.append({"group_id": group["group_id"], "parameter": field,
            "subjects": group["subjects"], "found": bool(candidates),
            "reviewed": bool(reviewed),
            "origin_subject": selected["subject"] if selected else None,
            "selected_assumption_key": selected["key"] if selected else None,
            "value": selected["value"] if selected else None,
            "unit": selected["unit"] if selected else UNITS.get(field),
            "source_applicability": selected.get("source_applicability") if selected else None,
            "review_verdict": selected.get("review", {}).get("verdict", "unassessed") if selected else None,
            "discovery_points": 1.0 if candidates else 0.0,
            "learning_bonus": round(bonus, 10),
            "points": round((1.0 if candidates else 0.0) + bonus, 10)})
    return rows
