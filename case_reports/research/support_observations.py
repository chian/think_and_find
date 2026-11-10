"""Typed source observations used as support operands, separate from metadata.

An extraction map key is only a selector. It is never the observation's identity.
Source-established observation IDs retain their meaning. Without one, host-validated
source occurrence anchors supply identity; prose locator wording cannot mint it.
This does not prove scientific equivalence across separate source occurrences.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math


OBSERVATION_KINDS = {"measurement", "fit_parameter", "simulation_input", "simulation_output",
                     "model_prediction", "reported_property", "unknown"}
OBSERVATION_FIELDS = {"source_id", "locator", "kind", "quantity", "conditions", "value", "units", "uncertainty"}


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


def _conditions(value):
    if isinstance(value, dict):
        from .importance import rational
        if len(value) > 32 or any(not isinstance(name, str) or not name.isidentifier() for name in value):
            raise ValueError("Structured source conditions must be named numeric inputs")
        return {name: str(rational(str(item) if type(item) is float else item))
                for name, item in value.items()}
    return _text(value, "conditions")


def validate_observation_fields(evidence_data, observation_id):
    """Validate native fields without granting a prose locator identity."""
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
        raise ValueError("Source observation needs exactly source_id, locator, kind, quantity, conditions, value, units and uncertainty")
    if not isinstance(observation["kind"], str) or observation["kind"] not in OBSERVATION_KINDS:
        raise ValueError("Observation kind must identify measurement, fit_parameter, simulation_input, "
                         "simulation_output, model_prediction, reported_property or unknown")
    source_id = _text(observation["source_id"], "source_id", empty=True)
    locator = _text(observation["locator"], "locator")
    if source_id and source_id in {source, study}:
        raise ValueError("Observation source_id must be the source-given OBSERVATION identifier, "
            "not the document URL or study ID. Use an empty source_id when the source gives no observation ID, "
            "and supply a precise locator identifying this quantity and condition within the source.")
    for name in ("quantity", "units"):
        _text(observation[name], name)
    conditions = _conditions(observation["conditions"])
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
    canonical = {"study": study, "identity_kind": "source_id", "identity": source_id}
    scope = (_text(observation["quantity"], "quantity"), conditions)
    for selector, other in observations.items():
        if selector == observation_id or not isinstance(other, dict) or set(other) != OBSERVATION_FIELDS:
            continue
        try:
            other_id = _text(other["source_id"], "source_id", empty=True)
            other_scope = (_text(other["quantity"], "quantity"), _conditions(other["conditions"]))
        except (ValueError, TypeError, KeyError):
            # That entry receives its own field validation; it cannot make an
            # otherwise well-formed observation invalid merely by being malformed.
            continue
        if source_id and other_id == source_id and other_scope != scope:
            raise ValueError(f"Observation identity is shared with native.observations[{selector!r}] "
                "for a different quantity or condition. Reuse the source's full observation ID, "
                "or leave source_id empty and anchor each actual quantity and condition precisely. "
                "Do not use a document/study identifier or invent identity from value, units or uncertainty.")
    detail = {"study_id": evidence_data["study_id"], "canonical_observation": canonical,
              "source": evidence_data["source"], "evidence_locator": evidence_data["locator"],
              "observation": observation_id, "measurement": deepcopy(observation)}
    for name in ("quote", "source_version"):
        if name in measurements:
            detail[name] = deepcopy(measurements[name])
    return detail


def observation_identity(evidence_data, observation_id):
    """Native IDs or host source occurrences, never extraction wording or values."""
    detail = validate_observation_fields(evidence_data, observation_id)
    canonical = detail["canonical_observation"]
    if canonical["identity"]:
        ident = sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]
    else:
        from .source_anchors import entry_identity
        ident = entry_identity(evidence_data, "observations", observation_id)
        detail["canonical_observation"] = {"identity_kind": "source_occurrence", "identity": ident}
    return ident, detail


def observations_schema(*, source=None):
    """An empty map is appropriate when a source establishes no identifiable measurement."""
    text = {"type": "string", "minLength": 1, "maxLength": 2000}
    source_id = {"type": "string", "maxLength": 2000,
        "description": "Optional source-given OBSERVATION identifier, such as a complete result/table-row ID. "
            "Use an empty string if none exists; host source_anchors identify its quantity and condition. "
            "Never use the document URL, study ID or an invented label. Different quantities/conditions "
            "need distinct complete native IDs or exact source result occurrences; descriptive condition changes, "
            "values/units/uncertainty do not define identity."}
    if source is not None:
        source_id["not"] = {"const": source}
    properties = {"source_id": source_id,
                  "kind": {"enum": sorted(OBSERVATION_KINDS), "description":
                      "Provenance of this value. measurement is an empirical observation or sample statistic; "
                      "a fitted saturation/asymptotic/kinetic parameter is fit_parameter, not a measured finite-condition "
                      "endpoint. Distinguish simulation inputs/outputs, model predictions and reported background properties. "
                      "Use unknown when the supplied source cannot establish the kind."},
                  "locator": {**text,
        "description": "Precise source locator for this actual observation, including the relevant row, clause "
            "or condition when one passage reports several observations. Locator prose is provenance, not identity."},
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
