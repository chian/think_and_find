"""Source-anchored scientific statement operands; extraction earns zero credit.

These are registrations of what a source appears to state. Structural checks do
not establish source fidelity, causal truth, or support for a candidate G.
"""

from hashlib import sha256
import json


CLAIM_FIELDS = frozenset({"kind", "source_id", "locator", "left", "relation",
                          "right", "conditions", "symbols"})
CLAIM_KINDS = frozenset({"equation", "causal_relation", "boundary_condition"})
CLAIM_RELATIONS = {
    "equation": frozenset({"="}),
    "causal_relation": frozenset({"causes", "increases", "decreases", "inhibits", "enables",
                                  "prevents", "has_no_effect_on", "reverses_effect_on", "depends_on",
                                  "mediates", "moderates", "affects"}),
    "boundary_condition": frozenset({"applies_when", "fails_when", "changes_when", "requires"}),
}


def _text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()) or len(value) > 2000:
        raise ValueError(f"Source claim {name} must be bounded {'possibly empty' if empty else 'nonempty'} text")
    return " ".join(value.split())


def claims_schema():
    text = {"type": "string", "minLength": 1, "maxLength": 2000}
    properties = {"kind": {"enum": sorted(CLAIM_KINDS)},
                  "source_id": {"type": "string", "maxLength": 2000},
                  "locator": text, "left": text,
                  "relation": {"enum": sorted(set().union(*CLAIM_RELATIONS.values()))}, "right": text,
                  "conditions": text,
                  "symbols": {"type": "object",
                              "additionalProperties": text}}
    return {"type": "object", "additionalProperties": {
        "type": "object", "additionalProperties": False,
        "required": list(properties), "properties": properties}}


def source_anchor(evidence_data):
    """Check the host-selected exact source fragment retained with an operand."""
    if not isinstance(evidence_data, dict):
        raise ValueError("Source operand needs an evidence record")
    _text(evidence_data.get("source"), "source")
    _text(evidence_data.get("study_id"), "study_id")
    measurements = evidence_data.get("measurements")
    if not isinstance(measurements, dict):
        raise ValueError("Source operand needs native measurements and quote")
    quote = measurements.get("quote")
    _text(quote, "exact source quote")
    version = _text(measurements.get("source_version"), "source version")
    if len(version) != 64 or any(ch not in "0123456789abcdef" for ch in version):
        raise ValueError("Source operand needs the document SHA-256 version")
    locator = _text(evidence_data.get("locator"), "evidence locator")
    if not locator.startswith("characters "):
        raise ValueError("Source operand needs a character-anchored source span")
    try:
        start, end = map(int, locator.removeprefix("characters ").split(":"))
    except ValueError as exc:
        raise ValueError("Malformed source character span") from exc
    if start < 0 or end - start != len(quote):
        raise ValueError("Source quote and character span disagree")
    return quote, version


def claim_identity(evidence_data, claim_id):
    """Identify a recorded source claim for inspection, without assigning credit."""
    _text(claim_id, "selector")
    if not isinstance(evidence_data, dict):
        raise ValueError("Source claim needs an evidence record")
    measurements = evidence_data.get("measurements")
    native = measurements.get("native") if isinstance(measurements, dict) else None
    claims = native.get("claims") if isinstance(native, dict) else None
    if not isinstance(claims, dict) or claim_id not in claims:
        raise ValueError("Select an actual typed native.claims entry")
    claim = claims[claim_id]
    if not isinstance(claim, dict) or set(claim) != CLAIM_FIELDS:
        raise ValueError("Source claim needs exactly its declared relation fields")
    if claim["kind"] not in CLAIM_KINDS:
        raise ValueError("Unknown source claim kind")
    quote, version = source_anchor(evidence_data)
    source_id = _text(claim["source_id"], "source_id", empty=True)
    locator = _text(claim["locator"], "native locator")
    if not source_id and not locator:
        raise ValueError("Source claim needs a source-native identifier or locator")
    structural = {"kind": claim["kind"]}
    for name in ("left", "relation", "right", "conditions"):
        structural[name] = _text(claim[name], name)
    if structural["relation"] not in CLAIM_RELATIONS[claim["kind"]]:
        raise ValueError("Source claim relation does not match its scientific kind")
    symbols = claim["symbols"]
    if not isinstance(symbols, dict):
        raise ValueError("Source claim symbols must be a definition map")
    structural["symbols"] = {_text(key, "symbol"): _text(value, "symbol definition")
                             for key, value in symbols.items()}
    if len(structural["symbols"]) != len(symbols):
        raise ValueError("Source claim has duplicate normalized symbols")
    if claim["kind"] == "equation":
        if not symbols:
            raise ValueError("Equation claim needs equality and defined symbols")
    elif symbols:
        raise ValueError("Symbol definitions belong to equation claims")
    provenance = {"source": evidence_data["source"], "source_version": version,
                  "evidence_locator": evidence_data["locator"], "structure": structural}
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    ident = sha256(canonical.encode()).hexdigest()[:32]
    detail = {"claim": claim_id, "source_id": source_id, "native_locator": locator,
              "source": evidence_data["source"], "source_version": version,
              "evidence_locator": evidence_data["locator"], "study_id": evidence_data["study_id"],
              "quote": quote, "structure": structural}
    return ident, detail
