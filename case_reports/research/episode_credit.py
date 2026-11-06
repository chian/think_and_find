"""Local extraction yield for Page/readers; scientific credit for investigators."""
from fractions import Fraction

from .records import Credit, Ledger, identity
from .support_observations import observation_identity
from .source_claims import claim_identity
from ...credit_assignment.causal_progress import compile_episode_points
from ...credit_assignment.causal_support import controller_projection


def _positive_delta(points, before):
    return {key: max(Fraction(0), Fraction(value) - Fraction(before.get(key, "0")))
            for key, value in points.items()}


def page_credit(result, baseline, source_url):
    return _extraction_gain(result, baseline, "page_extraction_gain",
                            lambda row: row["source"] == source_url)


def _claim_extraction_identity(data, selector):
    """Deduplicate source claims independently of extraction-window boundaries."""
    _, claim = claim_identity(data, selector)
    study = " ".join(data["study_id"].split())
    source = [claim["source"], claim["source_version"], study]
    if claim["statement_id"]:
        return identity([*source, "statement_id", claim["statement_id"]])
    # A paragraph or row can contain several claims. Without a source-native
    # identifier, retain the validated structure to distinguish those claims,
    # but never let the host's window boundaries create a new identity.
    return identity([*source, "source_locator", claim["native_locator"], claim["structure"]])


def _extracted_items(result, predicate):
    """Identities of returned source information, independent of scientific merit."""
    items = set()
    for _, row in Ledger(result).causal_sheet.of_kind("evidence"):
        data = row["data"]
        if not predicate(data):
            continue
        measurements = data["measurements"]
        native = measurements["native"]
        table = native.get("table_row")
        if table is not None:
            for column, value in enumerate(table["cells"]):
                if value.strip():
                    items.add("cell:" + identity([data["source"], measurements["source_version"],
                        table["table"], table["row"], column]))
            continue
        for selector in native["observations"]:
            key, _ = observation_identity(data, selector)
            items.add("observation:" + key)
        for selector in native["claims"]:
            items.add("claim:" + _claim_extraction_identity(data, selector))
        if not native["observations"] and not native["claims"] and data["finding"].strip():
            # A qualitative finding still supplies information. Rephrasing the
            # same source fragment cannot earn another extraction item.
            items.add("finding:" + identity([data["source"], measurements["source_version"], data["locator"]]))
    return items


def _extraction_gain(result, baseline, column, predicate):
    returned = _extracted_items(result, predicate)
    inherited = _extracted_items(baseline, predicate)
    return Credit(evaluated=result.evaluated, quality_resolution=1.0,
        qualities=tuple((column, key, 1.0) for key in sorted(returned - inherited)))


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
    return _extraction_gain(result, baseline, "report_extraction_gain",
                            lambda row: _within_scope(row, document, spans))


def lexical_credit(result, baseline, document, parts):
    spans = tuple(span for group in parts.values() for span in group)
    return _extraction_gain(result, baseline, "lexical_extraction_gain",
                            lambda row: _within_scope(row, document, spans))


def table_credit(result, baseline, document, table):
    return _extraction_gain(result, baseline, "table_extraction_gain",
                            lambda row: _within_scope(row, document, table.rows))


def reasoning_credit(result, baseline, commission):
    sheet = Ledger(result).causal_sheet
    score = compile_episode_points(sheet)
    before = compile_episode_points(Ledger(baseline).causal_sheet)["points"]
    points = _positive_delta(score["points"], before)
    projection = controller_projection(points, score["resolution"])
    if not projection["supported"]:
        return Credit(evaluated=False)
    return Credit(evaluated=result.evaluated, quality_resolution=projection["resolution"],
        qualities=tuple(("reasoning_scientific_gain", key, value)
                        for key, value in projection["values"].items()))


def scientific_attempt_credit(result, baseline):
    """Project only new global scientific points earned during this attempt."""
    score = compile_episode_points(Ledger(result).causal_sheet)
    before = compile_episode_points(Ledger(baseline).causal_sheet)["points"]
    points = _positive_delta(score["points"], before)
    projection = controller_projection(points, score["resolution"])
    if not projection["supported"]:
        return Credit(evaluated=False)
    return Credit(evaluated=result.evaluated, quality_resolution=projection["resolution"],
                  qualities=tuple(("attempt_scientific_gain", key, value)
                                  for key, value in sorted(projection["values"].items())))
