"""Conditional physical baselines reviewed before target-value scope is constructed."""

from dataclasses import replace

from ...task_executors.effective_removal_rate.rates import (
    AIR_DENSITY_KG_M3, AIR_DYNAMIC_VISCOSITY_PA_S, GRAVITY_M_S2, WATER_DENSITY_KG_M3,
)
from .records import Fact, identity


_PROVENANCE = "task_executors/effective_removal_rate/rates.py: existing Stokes-settling baseline constants"
MODEL_DEFAULTS = {
    "gravity_m_s2": {"value": GRAVITY_M_S2, "unit": "m/s2",
        "basis": "Standard terrestrial gravitational acceleration.",
        "material_factors": ["nonterrestrial or materially different gravity"]},
    "air_density_kg_m3": {"value": AIR_DENSITY_KG_M3, "unit": "kg/m3",
        "basis": "Ordinary indoor air baseline, approximately 20 degrees C and atmospheric pressure.",
        "material_factors": ["materially different air temperature, pressure, or gas composition"]},
    "air_dynamic_viscosity_pa_s": {"value": AIR_DYNAMIC_VISCOSITY_PA_S, "unit": "Pa*s",
        "basis": "Ordinary indoor air viscosity baseline, approximately 20 degrees C.",
        "material_factors": ["materially different air temperature or gas composition"]},
    "particle_density_kg_m3": {"value": WATER_DENSITY_KG_M3, "unit": "kg/m3",
        "basis": "Water-like particle material baseline; not a claim of measured aerosol composition.",
        "material_factors": ["explicitly different particle material or composition"]},
}
MODEL_DEFAULTS = {field: {**row, "provenance": _PROVENANCE}
                  for field, row in MODEL_DEFAULTS.items()}


def retained_defaults(scenario):
    """Automatically supplied values; promoted and caller-fixed values are absent."""
    return {field: row for field, row in scenario.get("model_defaults", {}).items()
            if row.get("decision") == "retain"}


def model_default_request(item):
    """One constants judge, separate from the subsequent sharing judge."""
    prose = {"type": "string", "minLength": 1}
    row = {"type": "object", "additionalProperties": False,
        "required": ["decision", "reason", "scenario_evidence"],
        "properties": {"decision": {"enum": ["retain", "revisit"]}, "reason": prose,
            "scenario_evidence": {"type": "array", "uniqueItems": True, "items": prose}},
        "allOf": [{"if": {"properties": {"decision": {"const": "revisit"}}},
                   "then": {"properties": {"scenario_evidence": {"minItems": 1}}}}]}
    scenarios = item["scenarios"]
    decisions = {subject: {"type": "object", "additionalProperties": False,
        "required": list(fields), "properties": {field: row for field in fields}}
        for subject, fields in item["default_candidates"].items()}
    return {"operation": "assess_scenario_model_defaults",
        "disease_of_interest": item.get("disease_of_interest"),
        "scenarios": scenarios, "model_defaults": MODEL_DEFAULTS,
        "default_candidates": item["default_candidates"],
        "instruction": (
            "Before parameter sharing, credit initialization or searching, review only the supplied "
            "physical baseline constants against each caller-selected scenario. Retain each baseline "
            "unless a concrete, explicitly stated scenario feature makes its variation material to "
            "the calculation. Missing measurements, unspecified temperature/humidity, general "
            "uncertainty, or a desire for more evidence are NOT grounds to revisit a baseline. "
            "Ordinary changes in room size, occupancy, age, respiratory activity, ventilation rate "
            "or face coverings do not by themselves justify changing these four physical constants. "
            "Use the listed material_factors to distinguish changes in a physical property from "
            "changes in outputs that merely depend on it. Water-like density is an explicit default "
            "premise; routine breathing/speech, particle size differences or hypothetical evaporation "
            "alone do not establish an exceptional material composition. For revisit, quote exact "
            "nonempty text from that scenario's description, fixed conditions, or supplied fixed_input_text "
            "and explain the concrete material change. Do not invent features or values. For retain, "
            "explain why no material exception is specified; scenario_evidence may be empty. "
            "Do not estimate replacement values or decide scenario sharing. A separate judge handles "
            "sharing after promoted fields enter the initial input list. Return every requested "
            "scenario/field decision exactly once. These are conditional model premises, not "
            "literature discoveries or measurements of the selected scenarios."
        ),
        "response_contract": {"type": "object", "additionalProperties": False,
            "required": ["decisions"], "properties": {"decisions": {
                "type": "object", "additionalProperties": False,
                "required": list(decisions), "properties": decisions}}}}


def initialize_model_defaults(initial, assessor=None, *, disease=None):
    """Promote material exceptions before constructing scenario sharing or credit."""
    scenarios = {row["subject"]: row for row in initial.scenarios
                 if row.get("role", "target") == "target"}
    candidates = {subject: [field for field in MODEL_DEFAULTS
                           if field not in row.get("fixed_parameters", {})]
                  for subject, row in scenarios.items()}
    inputs = {subject: {"description": row["description"],
        "fixed_parameters": row.get("fixed_parameters", {}),
        "fixed_conditions": row.get("fixed_conditions", {}),
        "fixed_input_text": [f"{field} = {value['value']} {value['unit']}"
            for field, value in row.get("fixed_parameters", {}).items()]}
        for subject, row in scenarios.items()}
    item = {"operation": "assess_scenario_model_defaults", "disease_of_interest": disease,
            "scenarios": inputs, "default_candidates": candidates}
    response = assessor(item) if assessor and any(candidates.values()) else {"decisions": {
        subject: {field: {"decision": "retain", "reason": "Configured physical baseline; no constants judge supplied.",
                          "scenario_evidence": []} for field in fields}
        for subject, fields in candidates.items()}}
    decisions = response.get("decisions") if isinstance(response, dict) else None
    if not isinstance(decisions, dict) or set(decisions) != set(candidates):
        raise ValueError("Model-default review must cover every selected target")
    planned, facts, assumptions = {}, [], []
    for subject, fields in candidates.items():
        selected = decisions[subject]
        if not isinstance(selected, dict) or set(selected) != set(fields):
            raise ValueError("Model-default review must cover every unfixed default exactly once")
        scenario = scenarios[subject]
        available = [inputs[subject]["description"], *inputs[subject]["fixed_conditions"].values(),
                     *inputs[subject]["fixed_input_text"]]
        rows = {}
        for field, decision in selected.items():
            if (not isinstance(decision, dict) or set(decision) != {"decision", "reason", "scenario_evidence"}
                    or decision["decision"] not in {"retain", "revisit"}
                    or not isinstance(decision["reason"], str) or not decision["reason"].strip()
                    or not isinstance(decision["scenario_evidence"], list)
                    or any(not isinstance(quote, str) or not quote.strip()
                           or not any(quote in text for text in available)
                           for quote in decision["scenario_evidence"])
                    or (decision["decision"] == "revisit" and not decision["scenario_evidence"])):
                raise ValueError("A material default exception needs a reason and exact evidence from its target scenario")
            rows[field] = {**MODEL_DEFAULTS[field], **decision,
                           "method": "scenario_default_judge" if assessor else "configured_baseline"}
            if decision["decision"] == "revisit":
                continue
            row = rows[field]
            statement = f"Retained model baseline: {field} = {row['value']} {row['unit']}. {row['basis']}"
            key = identity(["model_default", subject, field, row["value"]])
            assumption = {"subject": subject, "field": field, "value": row["value"],
                "unit": row["unit"], "statement": statement, "basis": row["basis"],
                "provenance": row["provenance"], "default_decision": dict(decision),
                "key": key, "role": "model_default", "reasonableness": 1.0,
                "evidence": {}, "review": {"completed": True, "verdict": "fixed_condition",
                    "criteria": {}, "request_constraint": None, "explanation": decision["reason"],
                    "limitations": ["Conditional physical baseline, not a source observation or discovery."],
                    "alternatives": [], "missing_evidence": []}}
            assumption["version"] = identity(assumption)
            assumptions.append(assumption)
            facts.append(Fact(subject, field, row["value"], row["unit"], "model_default",
                              row["basis"], context=statement, parents=(key,)))
        planned[subject] = {**scenario, "model_defaults": rows}
    from .scenario_sharing import needed_fields
    for scenario in planned.values():
        scenario["assumption_needs"] = sorted(needed_fields(scenario))
    # Replace a previous defaults plan, while preserving acquired scientific records.
    old_keys = {a["key"] for a in initial.assumptions if a.get("role") == "model_default"}
    return replace(initial,
        scenarios=tuple(planned.get(row["subject"], row) for row in initial.scenarios),
        facts=tuple(f for f in initial.facts if f.kind != "model_default") + tuple(facts),
        assumptions=tuple(a for a in initial.assumptions if a["key"] not in old_keys) + tuple(assumptions))
