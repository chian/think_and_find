"""Synthetic evidence and an in-memory HTTP transport, never real research."""

from __future__ import annotations

import io
import json
from urllib.parse import urlsplit

FACTS = {
    "attendee_count": (11, "count", "There were 11 attendees."),
    "index_case_count": (1, "count", "There was 1 index case."),
    "secondary_case_count": (2, "count", "There were 2 secondary cases."),
    "removal_capacity_m3_s": (0.5, "m3/s", "Clean-air removal capacity was 0.5 m3/s."),
    "exposure_time_s": (3600, "s", "Exposure duration was 3600 s."),
    "breathing_flow_m3_s": (0.0001, "m3/s", "Breathing flow was 0.0001 m3/s."),
    "mask_penetration": (1.0, "fraction", "Mask penetration was 1.0."),
}
SOURCE_URL = "https://evidence.example/synthetic-outbreak"
SOURCE_TEXT = " ".join(item[2] for item in FACTS.values())
PROVIDER_ENV = {
    "FIRECRAWL_API_KEY": "test-firecrawl-key",
    "TYPESAFE_API_KEY": "test-typesafe-key",
    "EPISODIC_LLM_BASE_URL": "https://llm.example/v1",
    "EPISODIC_LLM_MODEL": "test-model",
    "EPISODIC_LLM_USER": "test-user",
    "FIRECRAWL_BASE_URL": "https://firecrawl.example/v2",
    "TYPESAFE_ENDPOINT": "https://jev.example/v1/systemone",
}
SCENARIO = {
    "floor_area_m2": 100,
    "room_height_m": 3,
    "air_exchange_rate_h": 6,
    "recirculation_flow_m3_s": 0,
    "filter_efficiency": 0,
    "deactivation_rate_h": 0,
    "settling_speed_m_h": 0,
    "breathing_flow_m3_s": 0.0001,
    "mask_penetration": 1,
    "risk_tolerance": 0.1,
    "susceptible_count": 10,
    "occupant_count": 11,
    "exposure_time_s": 3600,
    "plume_encounter_probability": 0.1,
    "mouth_area_m2": 0.0002,
    "entrainment_coefficient": 0.1,
    "neighbor_distance_m": 2,
    "well_mixed_dilution_factor": 100,
}


def applicability_response(request, **overrides):
    """Explicitly synthetic acceptance; real applicability uses the provider."""
    target_ref, target_text = next(iter(request["target_evidence"].items()))
    return {
        "applicable": True,
        "scale_compatible": True,
        "mechanism_compatible": True,
        "evidence_role": "event_observation",
        "rationale": "Synthetic fixture describes the same test event.",
        "source_ref": request["source"]["source_ref"],
        "source_evidence_span": request["source"]["text"],
        "target_ref": target_ref,
        "target_evidence_span": target_text,
        **overrides,
    }


class ProviderTransport:
    """Respond to the real adapters' HTTP requests with deterministic fixtures."""

    def __init__(self):
        self.calls = []

    def __call__(self, request, *, timeout):
        payload = json.loads(request.data)
        path = urlsplit(request.full_url).path
        self.calls.append((path, payload, dict(request.headers)))
        if path == "/v2/search":
            result = {
                "url": SOURCE_URL,
                "title": "Synthetic outbreak fixture",
                "description": "Synthetic outbreak counts, clean-air flow, and duration.",
            }
            if payload.get("scrapeOptions"):
                result["markdown"] = SOURCE_TEXT
            body = {"success": True, "data": {"web": [result]}}
        elif path == "/v2/scrape":
            body = {"success": True, "data": {"markdown": SOURCE_TEXT, "metadata": {"sourceURL": SOURCE_URL}}}
        elif path == "/v1/systemone":
            body = {"answers": {"decision": {"type": "noul", "noul": 0.9}}}
        elif path == "/v1/chat/completions":
            prompt = payload["messages"][-1]["content"]
            data = json.loads(prompt.split("\n\n", 1)[1])
            if "candidates" in data:
                result = {"option_id": data["candidates"][0]["option_id"], "objective": "Find outbreak evidence", "rationale": "Synthetic selection"}
            elif data.get("operation") == "review_evidence_applicability":
                result = applicability_response(data)
            else:
                result = {"values": [
                    {
                        "field": name,
                        "value": FACTS[name][0],
                        "unit": FACTS[name][1],
                        "normalized_value": FACTS[name][0],
                        "normalized_unit": FACTS[name][1],
                        "source_ref": data["source"]["source_ref"],
                        "evidence_span": FACTS[name][2],
                        "confidence": 1.0,
                        "notes": [],
                    }
                    for name in data["required_fields"] if name in FACTS
                ]}
            body = {"choices": [{"message": {"content": json.dumps(result)}, "finish_reason": "stop"}]}
        else:
            raise AssertionError(f"Unexpected provider request: {path}")
        return io.BytesIO(json.dumps(body).encode("utf-8"))
