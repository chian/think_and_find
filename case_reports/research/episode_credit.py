"""Local extraction yield for Page/readers; scientific credit for investigators."""
from fractions import Fraction

from .records import Credit, Ledger, identity
from .support_observations import observation_identity
from .source_claims import claim_identity
from .source_anchors import finding_identities
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
    return claim_identity(data, selector)[0]


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
            items.update("finding:" + key for key in finding_identities(data))
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


def scientific_credit(result, baseline):
    """Project the Scientific commission's actual new scientific findings."""
    score = compile_episode_points(Ledger(result).causal_sheet)
    before = compile_episode_points(Ledger(baseline).causal_sheet)["points"]
    points = _positive_delta(score["points"], before)
    projection = controller_projection(points, score["resolution"])
    if not projection["supported"]:
        return Credit(evaluated=False)
    return Credit(evaluated=result.evaluated, quality_resolution=projection["resolution"],
                  qualities=tuple(("scientific_gain", key, value)
                                  for key, value in sorted(projection["values"].items())))
