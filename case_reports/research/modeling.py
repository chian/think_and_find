"""A bounded executable model catalog with open-ended agent-proposed assumptions."""

from __future__ import annotations

import math

from ...task_executors.scenario_metrics.evaluate import CONDITIONS, UNITS, evaluate_parameters
from ...task_executors.scenario_metrics.physics import integrated_concentration
from .records import Fact, ResearchResult, current_support


DERIVATIONS = {
    "cq_from_steady_outbreak": "Infer native Cq from attack probability or secondary cases, Ns, I, t, Qeff, Qb, pm and sr. Exponential dose response; steady concentration.",
    "cq_from_transient_outbreak": "Infer native Cq from the same counts and exact clean-start concentration integral, V and total removal rate. Initial concentration must explicitly be zero.",
    "transfer_parameter": "Transfer one existing parameter to another subject with a named compatibility hypothesis. Independent review determines support, not execution. Never invent a scale factor.",
    "cq_from_emission": "Cq=lambda_q/Qb for one named activity/population.",
    "cq_from_microphysics": "Cq=ci*cv*phi_a, with explicitly supported infectious-virion concentration, infectivity and aerosol volume fraction.",
    "cq_activity_rescaling": "Cq_baseline*phi_activity/phi_baseline, with compatible pathogen, humidity, size-cutoff and activity evidence.",
}


def derivation_inputs(operator, values):
    """The numeric dependencies of the selected deterministic operator."""
    if operator not in DERIVATIONS:
        raise ValueError("Unknown derivation; arbitrary model code is not executed")
    if operator == "transfer_parameter":
        return set(values)
    direct = {
        "cq_from_emission": {"quanta_emission_rate_s", "breathing_flow_m3_s"},
        "cq_from_microphysics": {"microscopic_infectivity_per_virion", "pathogen_concentration_virions_m3", "aerosol_volume_fraction"},
        "cq_activity_rescaling": {"quanta_concentration_exhaled_m3", "activity_aerosol_volume_fraction", "baseline_aerosol_volume_fraction"},
    }
    if operator in direct:
        return direct[operator]
    fields = {"index_case_count", "breathing_flow_m3_s", "mask_penetration", "relative_transmissibility", "exposure_time_s"}
    fields.update({"attack_probability"} if "attack_probability" in values else {"secondary_case_count", "susceptible_count"})
    fields.update({"initial_concentration_quanta_m3", "room_volume_m3", "removal_rate_s"}
                  if operator == "cq_from_transient_outbreak" else {"removal_capacity_m3_s"})
    return fields


def calibrate_cq_from_outbreak(values, *, transient):
    """Deterministic outbreak inversion within the infectiousness tool."""
    attack = values.get("attack_probability")
    if attack is None:
        attack = values["secondary_case_count"] / values["susceptible_count"]
    if not 0 <= attack < 1:
        raise ValueError("Finite Cq inference requires 0 <= attack probability < 1")
    factor = (values["index_case_count"] * values["breathing_flow_m3_s"]**2
              * values["mask_penetration"]**2 * values["relative_transmissibility"])
    if transient:
        if values["initial_concentration_quanta_m3"] != 0:
            raise ValueError("This calibration operator requires an explicit clean start")
        factor *= integrated_concentration(values["exposure_time_s"], 1,
            values["room_volume_m3"], values["removal_rate_s"], 0)
    else:
        factor *= values["exposure_time_s"] / values["removal_capacity_m3_s"]
    if factor <= 0:
        raise ValueError("No identifiable emission parameter under these assumptions")
    return -math.log1p(-attack) / factor


def source_calculation_inputs(action, ledger):
    """Resolve cited native records for a registered source/calibration event."""
    from .records import numeric_references

    subject = action.get("subject")
    scenario = ledger.scenarios.get(subject)
    if scenario is None or scenario.get("role", "target") == "target":
        raise ValueError("Source input references require a registered source/calibration scenario")
    if "facts" in action:
        raise ValueError("The facts selection interface has been removed")
    keys = action.get("source_inputs")
    if not isinstance(keys, list) or not keys or any(not isinstance(k, str) for k in keys):
        raise ValueError("Source calculations require source_inputs containing the supplied local native-input references")
    records = numeric_references(ledger)
    invalid = [key for key in keys if key not in records]
    if invalid:
        raise ValueError(f"Unknown numeric source records: {invalid!r}")
    facts = tuple(records[k] for k in keys)
    if action.get("operator") == "transfer_parameter":
        if any(fact.subject in ledger.scenarios
               and ledger.scenarios[fact.subject].get("role", "target") == "target"
               for fact in facts):
            raise ValueError("Source transfer cannot select target inputs")
    elif any(fact.subject != subject for fact in facts):
        raise ValueError("Source calculation inputs must belong to the registered source scenario")
    if len({f.field for f in facts}) != len(facts):
        duplicates = {field: [key for key, fact in zip(keys, facts) if fact.field == field]
                      for field in {f.field for f in facts}
                      if sum(f.field == field for f in facts) > 1}
        raise ValueError(f"Source calculation contains repeated numeric fields: {duplicates!r}")
    return facts




def derive_parameter(agent, action, facts, context):
    operator = action.get("operator")
    if operator not in DERIVATIONS:
        raise ValueError("Unknown derivation; arbitrary model code is not executed")
    subject = action.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("A derived parameter needs a subject")
    if operator != "transfer_parameter" and {f.subject for f in facts} != {subject}:
        raise ValueError("Calibration inputs must refer to the same event; explicit transfer is a separate hypothesis")
    v = {f.field: f.value for f in facts}
    required = derivation_inputs(operator, v)
    missing = required - v.keys()
    if missing:
        raise ValueError(f"{operator} requires numeric inputs: {sorted(missing)!r}")
    field = "quanta_concentration_exhaled_m3"
    kind = "derived"
    if operator == "transfer_parameter":
        if len(facts) != 1:
            raise ValueError("Transfer takes exactly one fact and preserves its numeric value")
        field, value = facts[0].field, facts[0].value
    elif operator == "cq_from_emission":
        value = v["quanta_emission_rate_s"] / v["breathing_flow_m3_s"]
    elif operator == "cq_from_microphysics":
        value = v["microscopic_infectivity_per_virion"] * v["pathogen_concentration_virions_m3"] * v["aerosol_volume_fraction"]
    elif operator == "cq_activity_rescaling":
        value = v["quanta_concentration_exhaled_m3"] * v["activity_aerosol_volume_fraction"] / v["baseline_aerosol_volume_fraction"]
    else:
        value = calibrate_cq_from_outbreak(v, transient=operator == "cq_from_transient_outbreak")
    if not math.isfinite(value):
        raise ValueError("Derived parameter is not finite")
    fact = Fact(subject, field, value, UNITS[field], kind,
                f"{operator}: {action['basis']}",
                context="Conditional on the selected input provenance and model",
                parents=tuple(f.key for f in facts if f.field in required))
    return ResearchResult((fact,), findings=(f"Derived {field}={value:.8g} for {subject}. "
        "An outbreak inversion is calibration, not independent validation.",))


def evaluate_model(agent, action, facts, context, metrics, *, assumptions=()):
    subject = action.get("subject")
    if {f.subject for f in facts} != {subject}:
        raise ValueError("All scenario inputs must belong to this subject; transfer must be explicit")
    selected = {f.field: f.value for f in facts}
    selected_assumptions = []
    support_complete = True
    for field, value in selected.items():
        supported = [a for a in assumptions if a["subject"] == subject and a["field"] == field
                     and a["value"] == value]
        if not supported:
            support_complete = False
        else:
            selected_assumptions.append(max(supported, key=lambda a: a["reasonableness"]))
    conditions = action.get("conditions", {})
    if not isinstance(conditions, dict) or any(k not in CONDITIONS for k in conditions):
        raise ValueError("Unknown model condition")
    for field, statement in conditions.items():
        supported = [a for a in assumptions if a["subject"] == subject and a["field"] == field
                     and a["statement"] == statement]
        if not supported:
            support_complete = False
        else:
            selected_assumptions.append(max(supported, key=lambda a: a["reasonableness"]))
    result = evaluate_parameters({f.field: f.value for f in facts}, conditions, metrics,
                                subject=subject, label=action["basis"],
                                provenance={f.field: f.as_record() for f in facts})
    result["assumption_keys"] = [a["key"] for a in selected_assumptions]
    result["assumption_support"] = selected_assumptions
    result["model_review"], result["support_complete"] = {}, support_complete
    result["reasonableness"] = current_support(result, {a["key"]: a for a in assumptions})
    return ResearchResult(estimates=(result,))
