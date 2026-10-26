"""Numerically parameterized duration relationships using the scalar kernels."""

import math

from .catalog import METRICS
from .physics import integrated_concentration, threshold_distance, well_mixed_distance
from .._validation import non_negative, positive


CONDITIONAL_INPUTS = {
    "indoor_reproductive_number": ("susceptible_count", "transmission_rate_s"),
    "transient_indoor_reproductive_number": (
        "susceptible_count", "breathing_flow_m3_s", "mask_penetration", "relative_transmissibility",
        "emission_rate_quanta_s", "room_volume_m3", "removal_rate_s", "initial_concentration_quanta_m3"),
    "risk_threshold_distance_m": ("risk_tolerance", "susceptible_count", "transmission_rate_s",
                                  "plume_encounter_probability", "critical_distance_m"),
}


def conditional_inputs(name, *, regime=None):
    if name == "risk_threshold_distance_m" and regime == "well_mixed_distance":
        return "risk_tolerance", "susceptible_count", "transmission_rate_s"
    return CONDITIONAL_INPUTS[name]


def bind_duration(name, inputs, *, regime=None):
    """Calculate fixed coefficients, leaving the caller's duration variable free."""
    background_distance = name == "risk_threshold_distance_m" and regime == "well_mixed_distance"
    values = {key: inputs[key] for key in conditional_inputs(name, regime=regime)}
    for key, value in values.items():
        non_negative(value, key)
    boundaries = []
    if name == "transient_indoor_reproductive_number":
        factor = (values["susceptible_count"] * values["breathing_flow_m3_s"]
                  * values["mask_penetration"] * values["relative_transmissibility"])
        a = factor * values["initial_concentration_quanta_m3"]
        b = factor * values["emission_rate_quanta_s"] / positive(values["room_volume_m3"], "room_volume_m3")
        rate = values["removal_rate_s"]
        coefficients = {"initial_risk_rate_s": a, "risk_acceleration_s2": b, "removal_rate_s": rate}
        if rate == 0:
            expression = f"R_in(t) = {a:.12g}*t + {b:.12g}*t^2/2"
        else:
            decay = f"(1-exp(-{rate:.12g}*t))/{rate:.12g}"
            expression = f"R_in(t) = {a:.12g}*{decay} + {b:.12g}/{rate:.12g}*(t-{decay})"
    else:
        risk_rate = values["susceptible_count"] * values["transmission_rate_s"]
        coefficients = {"risk_rate_s": risk_rate}
        if name == "indoor_reproductive_number":
            expression = f"R_in(t) = {risk_rate:.12g}*t"
        else:
            epsilon = values["risk_tolerance"]
            coefficients["risk_tolerance"] = epsilon
            if background_distance:
                expression = ("No positive minimum separation in the spatially uniform steady-background model while "
                              f"R_in(t) = {risk_rate:.12g}*t <= {epsilon:.12g}; "
                              "no finite separation meets the criterion above that boundary. "
                              "Conditional on negligible near-field enhancement, not a contact-safety claim.")
            else:
                positive(values["susceptible_count"], "susceptible_count")
                positive(values["critical_distance_m"], "critical_distance_m")
                numerator = (values["transmission_rate_s"] * values["plume_encounter_probability"]
                             * values["critical_distance_m"])
                coefficients["distance_rate_m_s"] = numerator
                if risk_rate == 0:
                    expression = "d_min(t) = 0 (zero modeled hazard)"
                elif numerator == 0:
                    expression = (f"d_min(t) = 0 while {risk_rate:.12g}*t <= {epsilon:.12g}; "
                                  "no finite separation above that room-risk boundary")
                else:
                    expression = (f"d_min(t) = {numerator:.12g}*t/({epsilon:.12g}-{risk_rate:.12g}*t) "
                                  "for t>0 below the room-risk boundary; 0 at t=0; "
                                  "no finite separation at or above the positive-time boundary")
            if risk_rate > 0:
                boundary = epsilon / risk_rate
                if not math.isfinite(boundary):
                    raise ValueError("Duration boundary exceeds representable range")
                boundaries.append({"time_s": boundary, "meaning": "Background R_in equals the risk criterion"})
    if not all(math.isfinite(value) for value in coefficients.values()):
        raise ValueError("Conditional coefficients exceed representable range")
    return {"version": 2 if background_distance else 1, "kernel": name,
            **({"regime": regime} if background_distance else {}),
            "variable": "exposure_time_s", "variable_unit": "s",
            "domain": {"minimum": 0, "minimum_inclusive": True}, "inputs": values,
            "coefficients": coefficients, "expression": expression, "boundaries": boundaries,
            "interpretation": "Numerically bound conditional relationship, not a result at an assumed event duration. "
                "t is in seconds. Distance is in metres; a zero bound means no positive minimum in this model, "
                "not a universal safety guarantee. " + (
                "Spatially uniform risk, conditional on negligible near-field enhancement; no local flow or contact model is included."
                if background_distance else "The unmasked-jet approximation is not validated below its near-mouth scale.")
                if name == "risk_threshold_distance_m" else
                "Numerically bound conditional relationship with t in seconds, not a result at an assumed event duration."}


def is_duration_relation(row):
    relation = row.get("function")
    if (not isinstance(relation, dict) or row.get("status") != "calculated"
            or row.get("limit") != "function" or row.get("value") is not None):
        return False
    name = row.get("name")
    inputs = relation.get("inputs")
    coefficients = relation.get("coefficients")
    regime = row.get("regime")
    background_distance = name == "risk_threshold_distance_m" and regime == "well_mixed_distance"
    version_matches = (relation.get("version") == 2 and relation.get("regime") == regime
                       if background_distance else relation.get("version") == 1 and "regime" not in relation)
    return (name in CONDITIONAL_INPUTS and version_matches
            and relation.get("kernel") == name and relation.get("variable") == "exposure_time_s"
            and isinstance(inputs, dict) and set(inputs) == set(conditional_inputs(name, regime=regime))
            and isinstance(coefficients, dict) and bool(coefficients)
            and all(type(v) in (int, float) and math.isfinite(v) and v >= 0
                    for v in (*inputs.values(), *coefficients.values())))


def evaluate_duration_relation(row, time_s):
    """Evaluate a retained relationship with the same kernels as scalar execution."""
    if not is_duration_relation(row):
        raise ValueError("A complete calculated duration relationship is required")
    time_s = non_negative(time_s, "exposure_time_s")
    values = row["function"]["inputs"]
    name = row["name"]
    if name == "transient_indoor_reproductive_number":
        concentration = integrated_concentration(time_s, **{key: values[key] for key in (
            "emission_rate_quanta_s", "room_volume_m3", "removal_rate_s", "initial_concentration_quanta_m3")})
        dose = METRICS["transient_individual_dose"].function({
            "integrated_concentration_quanta_s_m3": concentration,
            **{key: values[key] for key in ("breathing_flow_m3_s", "mask_penetration", "relative_transmissibility")}})
        return METRICS[name].function({"susceptible_count": values["susceptible_count"], "transient_individual_dose": dose})
    dose = METRICS["individual_dose"].function({"transmission_rate_s": values["transmission_rate_s"], "exposure_time_s": time_s})
    risk = METRICS["indoor_reproductive_number"].function({"susceptible_count": values["susceptible_count"], "individual_dose": dose})
    if name == "indoor_reproductive_number":
        return risk
    if row.get("regime") == "well_mixed_distance":
        return well_mixed_distance(values["risk_tolerance"], risk)
    return threshold_distance(indoor_reproductive_number=risk, **{key: values[key] for key in (
        "risk_tolerance", "susceptible_count", "plume_encounter_probability", "critical_distance_m")})
