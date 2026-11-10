"""Source-grounded first examinations of a selected G in theoretical accounts.

An examination records completed public scientific reasoning, including a
qualified or unresolved conclusion. It is registration progress, not a host
verdict that the interpretation is correct and not support/change credit.
"""
from jsonschema import Draft202012Validator


FIELDS = ("inference", "theories", "question", "argument", "grounds")
_TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
_SNAPSHOT = {"type": "object", "additionalProperties": False,
    "required": ["key", "version"], "properties": {
        "key": _TEXT, "version": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}}


def record_schema():
    return {"type": "object", "additionalProperties": False,
        "required": list(FIELDS), "properties": {
            "inference": {**_SNAPSHOT,
                "description": "Exact G snapshot examined by the public argument."},
            "theories": {"type": "array", "minItems": 1, "uniqueItems": True,
                "items": _SNAPSHOT,
                "description": "Exact theory or theory_fragment snapshots used in the examination."},
            "question": {**_TEXT,
                "description": "The scientific question the completed reasoning addresses."},
            "argument": {**_TEXT,
                "description": "Key of a completed public reasoning record, not prose or a plan to examine G later."},
            "grounds": {"type": "array", "minItems": 1, "uniqueItems": True,
                "items": _TEXT,
                "description": "Scientific premises used by the argument; their dependency closure must include actual source evidence."}}}


def references(data):
    # G and theory versions are explicit snapshots rather than dependencies on
    # whichever current versions happen to occupy the main sheet later.
    return [data["argument"], *data["grounds"]]


def snapshot_references(data):
    return [data["inference"], *data["theories"]]


def context_pins(data):
    pins = {}
    for ref in snapshot_references(data):
        pins.setdefault(ref["key"], set()).add(ref["version"])
    return pins


def _closure(sheet, row):
    """Resolve the exact premise versions actually used by a public argument."""
    found, todo = {}, [row]
    while todo:
        current = todo.pop()
        reference = (current["key"], current["version"])
        if reference in found:
            continue
        found[reference] = current
        for key, version in current["dependencies"].items():
            todo.append(sheet.snapshot({"key": key, "version": version}))
    return found


def validate(sheet, key, data):
    errors = list(Draft202012Validator(record_schema()).iter_errors(data))
    if errors:
        error = errors[0]
        raise ValueError(f"Examination {error.json_path}: {error.message}")
    inference = sheet.snapshot(data["inference"])
    if inference["kind"] != "inference":
        raise ValueError("An examination must identify an exact inference/G snapshot")
    theories = [sheet.snapshot(ref) for ref in data["theories"]]
    if any(row["kind"] not in {"theory", "theory_fragment"} for row in theories):
        raise ValueError("Examination theories must be theory or theory_fragment snapshots")

    sheet._require_kind(data["argument"], "reasoning")
    argument = sheet.rows[data["argument"]]
    public = argument["data"]
    if (not public["steps"] or any(not isinstance(step, str) or not step.strip()
                                   for step in public["steps"])
            or not isinstance(public["conclusion"], str) or not public["conclusion"].strip()):
        raise ValueError("An examination needs actual public reasoning steps and a conclusion, including qualified or unresolved conclusions")
    closure = _closure(sheet, argument)
    for ref in snapshot_references(data):
        if (ref["key"], ref["version"]) not in closure:
            raise ValueError("The examination argument must address the selected exact G and every selected theoretical input")

    source_grounded = False
    for ground in data["grounds"]:
        row = sheet.rows.get(ground)
        if row is None:
            raise ValueError(f"Unknown examination ground: {ground}")
        if (ground, row["version"]) not in closure:
            raise ValueError("Every stated examination ground must be an actual premise of its public argument")
        source_grounded |= any(item["kind"] == "evidence"
                               for item in _closure(sheet, row).values())
    if not source_grounded:
        raise ValueError("A first examination needs source-grounded scientific premises; a proposed task or source-free argument is not a completed examination")
    previous = sheet.rows.get(key)
    if previous and previous["data"]["inference"] != data["inference"]:
        raise ValueError("An examination retains its selected exact G snapshot")

