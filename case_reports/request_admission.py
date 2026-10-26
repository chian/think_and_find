"""Interpret the original prompt without injecting researched parameter values."""

import json
from collections.abc import Mapping

from .research.admission import extract_and_admit, exact_span


TARGET_FIELDS = (
    "removal_capacity_m3_s", "breathing_flow_m3_s", "mask_penetration",
    "risk_tolerance", "susceptible_count", "occupant_count", "exposure_time_s",
    "floor_area_m2", "room_height_m", "room_volume_m3", "air_exchange_rate_s",
    "ventilation_flow_m3_s", "recirculation_flow_m3_s", "filter_efficiency",
    "deactivation_rate_s", "settling_speed_m_s", "plume_encounter_probability",
    "mouth_area_m2", "entrainment_coefficient", "neighbor_distance_m",
)


def interpret_guideline_request(llm, request, models, *, requested_model=None):
    """Called within the existing root's first unit, before outbreak discovery."""
    if not isinstance(request, str) or not request.strip():
        raise ValueError("Supply the original natural-language request")

    def agent(payload):
        if llm is None:
            raise ValueError("Request interpretation requires an LLM")
        result = llm(payload)
        if isinstance(result, str):
            result = json.loads(result)
        if not isinstance(result, Mapping):
            raise ValueError("Request interpretation must return an object")
        return result

    scope = agent({
        "operation": "interpret_guideline_request", "request": request,
        "available_models": dict(models), "requested_model": requested_model,
        "instruction": (
            "Interpret only this original request. Name its disease and setting using exact request spans. "
            "Select one implemented output matching the request; do not silently narrow a request for "
            "multiple outputs or invent a disease. If this interface cannot represent it, return a nonempty "
            "scope_error. Do not supply numeric assumptions, source URLs or answers from memory. "
            "Request constraints are extracted separately; missing scientific inputs belong to evidence admission."
        ),
        "response_contract": {
            "disease_name": "exact disease text from request", "setting": "exact setting text, or empty",
            "transmission_model": "one available model", "scope_error": "empty or explicit unsupported scope",
        },
    })
    error = scope.get("scope_error")
    if not isinstance(error, str):
        raise ValueError("Scope interpretation requires scope_error")
    if error:
        raise ValueError(error)
    disease = exact_span(scope.get("disease_name"), request)
    setting = scope.get("setting", "")
    if not isinstance(setting, str):
        raise ValueError("Setting must be text")
    if setting:
        setting = exact_span(setting, request)
    model = requested_model or scope.get("transmission_model")
    if model not in models:
        raise ValueError("Request did not select an implemented transmission model")
    extracted = extract_and_admit(
        agent, text=request, source="request",
        objective="Interpret explicit target constraints only; do not answer the scientific question.",
        context={"research_request": request},
        question="Extract only explicit numeric conditions for subject target. Do not turn requested answers, reference examples or outbreak observations into target conditions.",
    )
    if extracted.issues or not extracted.evaluated:
        raise ValueError("Requester constraints were not admitted: " + "; ".join(extracted.issues))
    values = {}
    for fact in extracted.facts:
        if fact.subject != "target" or fact.field not in TARGET_FIELDS or fact.kind != "request_constraint":
            raise ValueError("Request extraction assigned a value outside the explicit target constraints")
        if fact.field in values and values[fact.field] != fact.value:
            raise ValueError(f"Conflicting requester constraints for {fact.field}")
        values[fact.field] = fact.value
    for source, destination, factor in (
        ("air_exchange_rate_s", "air_exchange_rate_h", 3600),
        ("deactivation_rate_s", "deactivation_rate_h", 3600),
        ("settling_speed_m_s", "settling_speed_m_h", 3600),
        ("ventilation_flow_m3_s", "outdoor_air_flow_m3_s", 1),
    ):
        if source in values:
            values[destination] = values.pop(source) * factor
    return {
        "disease_name": disease, "setting": setting, "transmission_model": model,
        "target_scenario": {**values, "description": request},
        "request_constraints": [fact.as_record() for fact in extracted.facts],
        "original_request": request,
    }
