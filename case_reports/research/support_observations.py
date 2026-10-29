"""Typed source observations used as support operands, separate from metadata.

An extraction map key is only a selector. It is never the observation's identity.
The source-established identifier (or precise source locator when none exists)
must be reused across extractions. This validates structure and deduplicates those
identities; it does not prove source fidelity or detect every semantic alias.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math


OBSERVATION_FIELDS = {"source_id", "locator", "quantity", "conditions", "value", "units", "uncertainty"}


def _text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"Observation {name} must be {'possibly empty' if empty else 'nonempty'} text")
    if len(value) > 2000:
        raise ValueError(f"Observation {name} exceeds the bounded source description")
    return " ".join(value.split())


def _value(value):
    # Preserve source-native value text, including intervals and detection limits.
    # Boolean observations retain their own type, never 0/1 numerical surrogates.
    if type(value) is bool:
        return
    if type(value) in (int, float):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("Observation value must be finite")
        return
    _text(value, "value")


def observation_identity(evidence_data, observation_id):
    """Return (stable identity, source-provenance detail) for one selected observation.

    Two actual observations from the same study require different source IDs or
    precise locators. Rewording a finding or renaming its extraction field does
    not create an observation. Units, uncertainty and sample-size metadata are
    attributes of a measurement, never separately counted top-level operands.
    """
    if not isinstance(observation_id, str) or not observation_id.strip():
        raise ValueError("Select a nonempty observation key")
    if not isinstance(evidence_data, dict):
        raise ValueError("Observation selection requires an evidence record")
    study = _text(evidence_data.get("study_id"), "study_id")
    source = _text(evidence_data.get("source"), "source")
    evidence_locator = _text(evidence_data.get("locator"), "evidence locator")
    measurements = evidence_data.get("measurements")
    native = measurements.get("native") if isinstance(measurements, dict) else None
    observations = native.get("observations") if isinstance(native, dict) else None
    if not isinstance(observations, dict) or observation_id not in observations:
        raise ValueError("Select an actual typed native.observations entry; metadata and raw field names are not observations")
    observation = observations[observation_id]
    if not isinstance(observation, dict) or set(observation) != OBSERVATION_FIELDS:
        raise ValueError("Source observation needs exactly source_id, locator, quantity, conditions, value, units and uncertainty")
    source_id = _text(observation["source_id"], "source_id", empty=True)
    locator = _text(observation["locator"], "locator")
    for name in ("quantity", "units"):
        _text(observation[name], name)
    conditions = observation["conditions"]
    if isinstance(conditions, dict):
        from .importance import rational
        if len(conditions) > 32 or any(not isinstance(name, str) or not name.isidentifier() for name in conditions):
            raise ValueError("Structured source conditions must be named numeric inputs")
        for value in conditions.values():
            rational(str(value) if type(value) is float else value)
    else:
        _text(conditions, "conditions")
    _value(observation["value"])
    uncertainty = observation["uncertainty"]
    if isinstance(uncertainty, dict):
        if set(uncertainty) != {"kind", "lower", "upper"} or uncertainty["kind"] != "interval":
            raise ValueError("Structured source uncertainty must be a reported inclusive interval")
        from .dependence import validate_target
        validate_target(uncertainty)
    else:
        _text(uncertainty, "uncertainty")
    # Whitespace normalization is conservative: case and punctuation can identify
    # genuinely different samples, rows or columns, so neither is discarded.
    canonical = {"study": study, "identity_kind": "source_id" if source_id else "source_locator",
                 "identity": source_id or locator}
    identity = sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]
    detail = {"study_id": evidence_data["study_id"], "canonical_observation": canonical,
              "source": evidence_data["source"], "evidence_locator": evidence_data["locator"],
              "observation": observation_id, "measurement": deepcopy(observation)}
    for name in ("quote", "source_version"):
        if name in measurements:
            detail[name] = deepcopy(measurements[name])
    return identity, detail


def observations_schema():
    """An empty map is appropriate when a source establishes no identifiable measurement."""
    text = {"type": "string", "minLength": 1, "maxLength": 2000}
    properties = {"source_id": {"type": "string", "maxLength": 2000}, "locator": text,
                  "quantity": text, "conditions": {"oneOf": [text, {"type": "object", "maxProperties": 32,
                      "additionalProperties": {"oneOf": [{"type": "number"}, text]}}]},
                  "value": {"oneOf": [{"type": "number"}, {"type": "boolean"}, text]},
                  "units": text, "uncertainty": {"oneOf": [text,
                      {"type": "object", "additionalProperties": False,
                       "required": ["kind", "lower", "upper"], "properties": {
                           "kind": {"const": "interval"}, "lower": {"oneOf": [{"type": "integer"}, text]},
                           "upper": {"oneOf": [{"type": "integer"}, text]}}}]}}
    return {"type": "object", "maxProperties": 16, "additionalProperties": {
        "type": "object", "additionalProperties": False,
        "required": list(properties), "properties": properties}}


def observation_target(observation):
    """Read a comparison target from source operands, never a scoring tolerance.

    Free-text uncertainty that is not an exact/no-uncertainty declaration stays
    unresolved. Structured intervals must have been reported by the source;
    this function does not estimate intervals, confidence or noise.
    """
    from .importance import rational
    from .dependence import validate_target
    uncertainty = observation["uncertainty"]
    value = observation["value"]
    if isinstance(uncertainty, dict):
        validate_target(uncertainty)
        if type(value) is bool:
            return None, "Boolean observation cannot use numeric uncertainty"
        try:
            numeric = rational(str(value) if type(value) is float else value)
        except ValueError:
            return None, "Source observation value has no supported numerical representation"
        if not rational(uncertainty["lower"]) <= numeric <= rational(uncertainty["upper"]):
            return None, "Reported central value is outside the supplied reported uncertainty interval"
        return deepcopy(uncertainty), "source_reported_interval"
    declaration = " ".join(uncertainty.strip().lower().split())
    if declaration not in {"exact", "exact stipulation", "none", "0", "not reported", "not applicable"}:
        return None, "Source uncertainty is unstructured; no comparison tolerance was inferred"
    if type(value) is bool:
        return {"kind": "exact", "answer": {"type": "boolean", "value": value}}, "reported_boolean_equality"
    try:
        numeric = rational(str(value) if type(value) is float else value)
    except ValueError:
        return None, "Source observation value has no supported numerical representation"
    return {"kind": "exact", "answer": {"type": "rational", "value": str(numeric)}}, (
        "literal_reported_value_agreement_uncertainty_unknown" if declaration == "not reported" else "reported_value_equality")
