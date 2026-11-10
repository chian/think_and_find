"""Host source-occurrence identities, independent of model locator wording.

Units are deterministic text boundaries, not scientific claims. Overlapping
same-role mentions in one unit reuse an occurrence identity; this is a source
identity convention, not a proof of semantic equivalence. Ambiguous broad
mentions require a more precise source anchor instead of earning another item.
"""
from copy import deepcopy
import re

from .records import identity


class SourceAnchorError(ValueError):
    def __init__(self, path, message):
        super().__init__(message)
        self.path = list(path)


_BOUNDARY = re.compile(r"[.!?;](?=\s|$)|:(?=\s)|\s(?:and|but|whereas|while)\s+")
_NUMBER = re.compile(r"[+\-−]?(?:\d+/\d+|(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+\-−]?\d+)?)")


def source_units(document):
    """Stable document-wide syntactic units; never split at reader windows."""
    if hasattr(document, "source_identity_units"):
        return document.source_identity_units
    cuts = [0]
    for match in _BOUNDARY.finditer(document.text):
        cut = match.start() if match.group()[0].isspace() else match.end()
        if cut > cuts[-1]:
            cuts.append(cut)
    cuts.append(len(document.text))
    units = []
    for start, end in zip(cuts, cuts[1:]):
        while start < end and document.text[start].isspace():
            start += 1
        while end > start and document.text[end - 1].isspace():
            end -= 1
        if start < end:
            units.append({"id": "source-unit:" + identity([document.source, document.sha256, start, end]),
                          "start": start, "end": end})
    document.source_identity_units = tuple(units)
    return document.source_identity_units


def _stored_roles(sidecar):
    for name in ("observations", "claims"):
        for entry in sidecar.get(name, {}).values():
            if name == "observations":
                yield "quantity", entry["quantity"]
                yield "result", entry["result"]
            else:
                yield from (("subject", anchor) for anchor in entry["subjects"])
            yield from (("condition", anchor) for anchor in entry["conditions"])


def registry(reader):
    found = {}
    for row in reader.ledger.causal_sheet.history:
        if row["kind"] != "evidence":
            continue
        data = row["data"]
        if data["source"] != reader.document.source or data["measurements"].get("source_version") != reader.document.sha256:
            continue
        retain_roles(found, data["measurements"].get("source_anchors", {}))
    return found


def context(reader, allowed):
    units = [unit for unit in source_units(reader.document)
             if any(unit["start"] < span.end and unit["end"] > span.start for span in allowed.values())]
    ids = {unit["id"] for unit in units}
    known = [{"role": role, "unit": unit, **value}
             for (role, unit, _), value in registry(reader).items() if unit in ids]
    return {"units": units, "known_mentions": known,
        "instruction": "Source anchors identify occurrences in this exact document version. Choose a supplied "
            "unit and an exact quote unique within that unit's portion of a supplied passage. A statement quote must lie inside its "
            "host unit; longer/shorter quotes there retain that unit identity. Each observation also needs a result "
            "anchor quoting only its reported value, without units, uncertainty or surrounding context, and a positive "
            "1-based occurrence selecting that literal within the selected unit's supplied portion. Keep the source's "
            "literal wording for text values. Anchor quantity and descriptive conditions separately; adding or dropping "
            "condition descriptions does not create a new result occurrence. Claims use native subject/object mentions in subjects when one "
            "unit contains several assertions. Reuse known source mentions; do not use prose locator wording, "
            "extraction keys or paraphrased quantities as identity. Source occurrence identity does not establish "
            "scientific truth or equivalence across separate occurrences."}


def passages(reader, allowed):
    """Label the source text once, with units usable without character counting."""
    units = source_units(reader.document)
    return {key: {"key": key, "start": span.start, "end": span.end,
        "source_units": [{"unit": unit["id"], "start": max(unit["start"], span.start),
            "end": min(unit["end"], span.end),
            "text": reader.document.text[max(unit["start"], span.start):min(unit["end"], span.end)]}
            for unit in units if unit["start"] < span.end and unit["end"] > span.start]}
        for key, span in allowed.items()}


def retain_roles(known, sidecar):
    for role, anchor in _stored_roles(sidecar):
        canonical = deepcopy(anchor["identity"])
        key = (role, anchor["unit"]["id"], canonical["id"])
        previous = known.get(key, canonical)
        ranges = previous["occurrences"]
        for occurrence in canonical["occurrences"]:
            if occurrence not in ranges:
                ranges.append(occurrence)
        canonical["occurrences"] = deepcopy(ranges)
        known[key] = canonical


def input_schema(allowed, units):
    text = {"type": "string", "minLength": 1,
        "description": "Verbatim source text, unique within the selected unit's portion of the supplied span."}
    anchor = {"type": "object", "additionalProperties": False,
        "required": ["span", "unit", "quote"], "properties": {
            "span": {"enum": list(allowed)}, "unit": {"enum": [unit["id"] for unit in units]}, "quote": text}}
    conditions = {"type": "array", "items": anchor, "uniqueItems": True}
    result = {**anchor, "required": [*anchor["required"], "occurrence"],
        "properties": {**anchor["properties"], "quote": {"type": "string", "minLength": 1,
            "description": "Only the exact reported value literal, without units, uncertainty or context; repeated equal values are selected by occurrence."},
            "occurrence": {"type": "integer", "minimum": 1,
            "description": "1-based exact value occurrence inside the selected unit's supplied portion; repeated equal-valued cells select their own positions."}}}
    observation = {"type": "object", "additionalProperties": False,
        "required": ["statement", "quantity", "result", "conditions"],
        "properties": {"statement": anchor, "quantity": anchor, "result": result, "conditions": conditions}}
    claim = {"type": "object", "additionalProperties": False,
        "required": ["statement", "subjects", "conditions"],
        "properties": {"statement": anchor, "subjects": {**conditions, "minItems": 1}, "conditions": conditions}}
    return {"type": "object", "additionalProperties": False,
        "required": ["observations", "claims", "finding"], "properties": {
            "observations": {"type": "object", "additionalProperties": observation},
            "claims": {"type": "object", "additionalProperties": claim},
            "finding": {"type": "array", "items": anchor, "uniqueItems": True}}}


def _resolve(reader, allowed, value, path, role, known, *, occurrence=None):
    if not isinstance(value, dict) or set(value) != {"span", "unit", "quote"}:
        raise SourceAnchorError(path, "Source anchor requires exactly span, unit and exact quote")
    if not isinstance(value["span"], str) or value["span"] not in allowed:
        raise SourceAnchorError(path + ["span"], "Choose an exact passage supplied to this extraction")
    span = allowed[value["span"]]
    quote = value["quote"]
    if not isinstance(quote, str) or not quote.strip() or quote not in span.text:
        raise SourceAnchorError(path + ["quote"], "Anchor quote must be an exact nonempty substring of its supplied passage")
    units = {unit["id"]: unit for unit in source_units(reader.document)}
    unit = units.get(value["unit"]) if isinstance(value["unit"], str) else None
    if unit is None:
        raise SourceAnchorError(path, "Choose a host source unit containing the exact quoted occurrence")
    selected_start = max(unit["start"], span.start)
    selected_end = min(unit["end"], span.end)
    selected = span.text[selected_start - span.start:selected_end - span.start]
    if selected_start >= selected_end or quote not in selected:
        raise SourceAnchorError(path, "Quote must lie within the selected host source unit; select its exact clause instead of a broad multi-unit quote")
    offsets, offset = [], selected.find(quote)
    while offset >= 0:
        offsets.append(offset)
        offset = selected.find(quote, offset + 1)
    if occurrence is None:
        if len(offsets) != 1:
            raise SourceAnchorError(path + ["quote"], "Anchor quote has multiple occurrences within the selected source unit; include enough exact context to identify one")
        offset = offsets[0]
    else:
        if type(occurrence) is not int or not 1 <= occurrence <= len(offsets):
            raise SourceAnchorError(path + ["occurrence"], "Choose an actual positive 1-based result occurrence within this selected unit portion")
        offset = offsets[occurrence - 1]
    start = selected_start + offset
    end = start + len(quote)
    if role == "result":
        if not _NUMBER.fullmatch(quote):
            if ((start > unit["start"] and reader.document.text[start - 1].isalnum() and quote[0].isalnum())
                    or (end < unit["end"] and reader.document.text[end].isalnum() and quote[-1].isalnum())):
                raise SourceAnchorError(path + ["quote"], "Result text must select a complete source value, not part of a word")
        else:
            tokens = {(unit["start"] + match.start(), unit["start"] + match.end())
                for match in _NUMBER.finditer(reader.document.text[unit["start"]:unit["end"]])}
            if (start, end) not in tokens:
                raise SourceAnchorError(path + ["quote"], "Result quote must select the complete source number, not a digit or partial numeric literal")
    result = {"unit": deepcopy(unit), "start": start, "end": end, "quote": quote,
              "span": {"start": span.start, "end": span.end}}
    if role is not None:
        overlap = {key: candidate for key, candidate in known.items()
                   if key[0] == role and key[1] == unit["id"]
                   and any(left < end and right > start for left, right in
                           candidate["occurrences"])}
        if len(overlap) > 1:
            raise SourceAnchorError(path, "This quote overlaps several distinct source mentions. Choose the precise quantity/result/condition/subject occurrence; a broad quote cannot mint or merge identities")
        canonical = deepcopy(next(iter(overlap.values()))) if overlap else {
            "id": "source-mention:" + identity([reader.document.source, reader.document.sha256, unit["id"], role, start, end]),
            "start": start, "end": end, "occurrences": [[start, end]]}
        if [start, end] not in canonical["occurrences"]:
            canonical["occurrences"].append([start, end])
        known[(role, unit["id"], canonical["id"])] = canonical
        result["identity"] = canonical
    return result


def resolve_entry(reader, allowed, category, selector, value, known):
    path = [category, selector]
    fields = {"statement", "quantity", "result", "conditions"} if category == "observations" else {"statement", "subjects", "conditions"}
    if not isinstance(value, dict) or set(value) != fields:
        raise SourceAnchorError(path, f"Source anchors for {category} require exactly {sorted(fields)}")
    working = deepcopy(known)
    entry = {"statement": _resolve(reader, allowed, value["statement"], path + ["statement"], None, working)}
    if category == "observations":
        entry["quantity"] = _resolve(reader, allowed, value["quantity"], path + ["quantity"], "quantity", working)
        result = value["result"]
        if not isinstance(result, dict) or set(result) != {"span", "unit", "quote", "occurrence"}:
            raise SourceAnchorError(path + ["result"], "A result anchor requires span, unit, exact value quote and occurrence")
        if type(result["occurrence"]) is not int or result["occurrence"] < 1:
            raise SourceAnchorError(path + ["result", "occurrence"], "Result occurrence must be a positive 1-based integer")
        entry["result"] = _resolve(reader, allowed,
            {key: result[key] for key in ("span", "unit", "quote")}, path + ["result"], "result", working,
            occurrence=result["occurrence"])
        if entry["result"]["unit"]["id"] != entry["statement"]["unit"]["id"]:
            raise SourceAnchorError(path + ["result"], "The result occurrence must belong to this observation's statement unit")
    else:
        if not isinstance(value["subjects"], list) or not value["subjects"]:
            raise SourceAnchorError(path + ["subjects"], "Source subject mentions must be a nonempty array")
        entry["subjects"] = [_resolve(reader, allowed, item, path + ["subjects", index], "subject", working)
                             for index, item in enumerate(value["subjects"])]
    if not isinstance(value["conditions"], list):
        raise SourceAnchorError(path + ["conditions"], "Source condition mentions must be an array")
    entry["conditions"] = [_resolve(reader, allowed, item, path + ["conditions", index], "condition", working)
                            for index, item in enumerate(value["conditions"])]
    known.clear()
    known.update(working)
    return entry


def validate_result_value(value, anchor):
    """Check literal fidelity; source coordinates, not the numeric value, identify a result."""
    quote = anchor["quote"]
    if type(value) is bool:
        if quote.strip().casefold() != str(value).casefold():
            raise ValueError("Boolean observations require a literal source true/false result; otherwise retain the source's exact text value")
    elif type(value) in (int, float):
        from .importance import rational
        try:
            matches = rational(quote.replace("−", "-")) == rational(str(value))
        except ValueError:
            matches = False
        if not matches:
            raise ValueError("Result quote must be the reported numeric value alone; retain nonnumeric or unsupported source forms as exact text")
    elif not isinstance(value, str) or " ".join(value.split()) != " ".join(quote.split()):
        raise ValueError("Text observation values must reproduce the exact source result wording")


def resolve_finding(reader, allowed, values, known):
    if not isinstance(values, list):
        raise SourceAnchorError(["finding"], "Source prose anchors must be an array")
    return [_resolve(reader, allowed, item, ["finding", index], None, known) for index, item in enumerate(values)]


def finalize(reader, allowed, sidecar):
    spans = {}
    def collect(value):
        if isinstance(value, dict):
            if "quote" in value and "span" in value:
                key = (value["span"]["start"], value["span"]["end"])
                spans[key] = next(span for span in allowed.values() if (span.start, span.end) == key)
            for item in value.values():
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(sidecar)
    return {**sidecar, "source_spans": [{"start": start, "end": end, "text": span.text}
            for (start, end), span in sorted(spans.items())]}


def _validate_anchor(data, anchor, role):
    sidecar = data["measurements"].get("source_anchors")
    if not isinstance(sidecar, dict):
        raise ValueError("Source occurrence needs host-validated source_anchors; prose locators are not identities")
    try:
        unit, start, end = anchor["unit"], anchor["start"], anchor["end"]
        if type(start) is not int or type(end) is not int or not unit["start"] <= start < end <= unit["end"]:
            raise ValueError("Source anchor has invalid document coordinates")
        expected = "source-unit:" + identity([data["source"], data["measurements"]["source_version"], unit["start"], unit["end"]])
        if unit["id"] != expected:
            raise ValueError("Source unit identity does not match its document/version/coordinates")
        span = next(span for span in sidecar["source_spans"]
                    if span["start"] == anchor["span"]["start"] and span["end"] == anchor["span"]["end"])
        if span["end"] - span["start"] != len(span["text"]) or not span["start"] <= start < end <= span["end"]:
            raise ValueError("Source anchor is outside its retained source fragment")
        if span["text"][start - span["start"]:end - span["start"]] != anchor["quote"]:
            raise ValueError("Source anchor quote does not reproduce retained source text")
        if role is None:
            return unit["id"]
        canonical = anchor["identity"]
        left, right = canonical["start"], canonical["end"]
        if not unit["start"] <= left < right <= unit["end"]:
            raise ValueError("Canonical source mention has invalid coordinates")
        for occurrence in canonical["occurrences"]:
            first, last = occurrence
            if not unit["start"] <= first < last <= unit["end"] or first >= right or last <= left:
                raise ValueError("Canonical occurrence chain must remain connected within one source unit")
            left, right = min(left, first), max(right, last)
        if [start, end] not in canonical["occurrences"]:
            raise ValueError("Canonical source mention must retain this accepted exact occurrence")
        expected = "source-mention:" + identity([data["source"], data["measurements"]["source_version"], unit["id"], role, canonical["start"], canonical["end"]])
        if canonical["id"] != expected:
            raise ValueError("Canonical source mention identity does not match its coordinates")
        return canonical["id"]
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError("Source occurrence needs complete host-validated anchors and retained source fragments") from exc


def entry_identity(data, category, selector):
    try:
        entry = data["measurements"]["source_anchors"][category][selector]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{category}[{selector!r}] needs host-validated source anchors; prose locators cannot supply identity") from exc
    statement = _validate_anchor(data, entry["statement"], None)
    for value in entry["conditions"]:
        _validate_anchor(data, value, "condition")
    if category == "observations":
        roles = [_validate_anchor(data, entry["quantity"], "quantity")]
        result = _validate_anchor(data, entry["result"], "result")
        if entry["result"]["unit"]["id"] != statement:
            raise ValueError("Observation result occurrence is outside its statement unit")
        validate_result_value(data["measurements"]["native"]["observations"][selector]["value"], entry["result"])
        roles.append(result)
    else:
        roles = sorted({_validate_anchor(data, value, "subject") for value in entry["subjects"]})
    return identity([data["source"], data["measurements"]["source_version"], statement, roles])[:32]


def finding_identities(data):
    sidecar = data["measurements"].get("source_anchors", {})
    if sidecar.get("pending"):
        return ()
    return tuple(identity([data["source"], data["measurements"]["source_version"],
                           _validate_anchor(data, anchor, None)])
                 for anchor in sidecar.get("finding", ()))
