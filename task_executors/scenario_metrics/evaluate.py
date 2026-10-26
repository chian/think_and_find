"""Evaluate one named parameterization, preserving missingness and provenance."""

from __future__ import annotations

import math
import operator
import hashlib
import json
from typing import Mapping

from .catalog import METRICS, metric_conditions, select_metric
from .dependencies import PARAMETER_DERIVATIONS, input_requirements
from .conditional import CONDITIONAL_INPUTS, bind_duration, conditional_inputs
from .exposure import EXPOSURE_OUTPUTS, FIXED_FIELDS, model_inputs


UNITS = {
    "breathing_flow_m3_s": "m3/s", "quanta_concentration_exhaled_m3": "quanta/m3",
    "mask_penetration": "fraction", "relative_transmissibility": "scalar",
    "removal_capacity_m3_s": "m3/s", "removal_rate_s": "1/s",
    "room_volume_m3": "m3", "room_height_m": "m", "floor_area_m2": "m2",
    "susceptible_count": "person", "occupant_count": "person", "index_case_count": "person",
    "secondary_case_count": "person", "attack_probability": "fraction",
    "exposure_time_s": "s", "risk_tolerance": "scalar",
    "air_exchange_rate_s": "1/s", "ventilation_flow_m3_s": "m3/s",
    "recirculation_flow_m3_s": "m3/s", "filter_efficiency": "fraction",
    "deactivation_rate_s": "1/s", "settling_speed_m_s": "m/s",
    "effective_radius_m": "m", "particle_density_kg_m3": "kg/m3",
    "air_density_kg_m3": "kg/m3", "air_dynamic_viscosity_pa_s": "Pa*s",
    "gravity_m_s2": "m/s2", "initial_concentration_quanta_m3": "quanta/m3",
    "mouth_area_m2": "m2", "entrainment_coefficient": "scalar",
    "neighbor_distance_m": "m", "plume_encounter_probability": "fraction",
    "breath_volume_m3": "m3", "clearance_fraction": "fraction",
    "quanta_emission_rate_s": "quanta/s",
    "initial_radius_m": "m", "solute_volume_fraction": "fraction", "relative_humidity": "fraction",
    "evaporation_coefficient_m2_s": "m2/s", "microscopic_infectivity_per_virion": "1/virion",
    "pathogen_concentration_virions_m3": "virion/m3", "aerosol_volume_fraction": "fraction",
    "activity_aerosol_volume_fraction": "fraction", "baseline_aerosol_volume_fraction": "fraction",
}

POSITIVE = frozenset((
    "breathing_flow_m3_s", "room_volume_m3", "room_height_m", "floor_area_m2",
    "particle_density_kg_m3", "air_dynamic_viscosity_pa_s", "gravity_m_s2",
    "mouth_area_m2", "entrainment_coefficient", "neighbor_distance_m",
    "occupant_count", "index_case_count",
    "initial_radius_m", "evaporation_coefficient_m2_s", "baseline_aerosol_volume_fraction",
))
FRACTIONS = frozenset(k for k, unit in UNITS.items() if unit == "fraction")
CONDITIONS = {
    "well_mixed": "Concentration is treated as spatially uniform outside the explicitly modeled jet.",
    "single_infectious_source": "One continuously present infectious person emits at a constant rate.",
    "emissions_stopped": "No further emissions or other concentration inputs occur during the modeled clearance interval.",
    "uniform_breathing_and_masks": "Emitter and recipients share the specified Qb and pm; susceptibility is represented by sr.",
    "continuous_unmasked_jet": "Continuous unmasked respiratory jet; not masked flow, a cough or a sneeze.",
    "quasi_steady_local_plume": "A continuous local jet is treated as immediately established while the background concentration evolves; propagation delay and transient local-flow dynamics are neglected. This is an additional approximation, not implied by a clean room.",
    "negligible_near_field": "Near-field enhancement is neglected throughout the modeled occupied region, so separation does not change airborne risk. This is a distinct hypothesis, not implied by mask penetration or by a well-mixed background.",
    "stokes_particles": "Spherical particles at low Reynolds number; radius is an effective model input.",
    "random_orientation_and_spacing": "Random relative orientations and uniform planar occupancy.",
    "equilibrium_evaporation": "Equilibrium hygroscopic approximation at stated humidity; theta corresponds to the assumed temperature.",
    "uniform_virion_infectivity": "Uniform infectious virion concentration/infectivity across the specified aerosol distribution; RNA is not silently equated with infectious virions.",
}

DEFAULT_METRICS = tuple(EXPOSURE_OUTPUTS)


def validate_parameter(field, value, unit):
    if field not in UNITS or unit != UNITS[field]:
        raise ValueError(f"{field} requires canonical unit {UNITS.get(field)}")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field} requires a finite number")
    if value < 0 or (field in POSITIVE and value == 0):
        raise ValueError(f"{field} is outside its physical domain")
    if field in FRACTIONS and value > 1:
        raise ValueError(f"{field} must be <= 1")
    if field in {"occupant_count", "index_case_count", "secondary_case_count"} and int(value) != value:
        raise ValueError(f"{field} must be an integer count")


def output_is_resolved(row):
    """Excluding one kernel does not answer a requested numerical quantity."""
    if row.get("status") != "calculated":
        return False
    if row.get("name") in EXPOSURE_OUTPUTS:
        operation, _ = EXPOSURE_OUTPUTS[row["name"]]
        return (row.get("limit") in {"finite", "unbounded"}
                and bool(row.get("model_id")) and row.get("operation") == operation
                and set(row.get("fixed_inputs", {})) == set(FIXED_FIELDS[operation]))
    return True


def output_meets_contract(row, contract):
    if not output_is_resolved(row):
        return False
    if not contract:
        return True
    if contract.get("result_form") == "scalar" and row.get("limit") not in {"finite", "unbounded"}:
        return False
    if row.get("operation") != contract.get("operation"):
        return False
    if row.get("background") != contract.get("background"):
        return False
    return (set(row.get("fixed_inputs", {})) == set(contract["fixed_fields"])
            and all(row["fixed_inputs"].get(field) == value
                    for field, value in contract.get("fixed_inputs", {}).items()))


def exposure_outputs_agree(outputs):
    """A set of operations cannot claim coherence across different risk models."""
    models, thresholds = {}, {}
    for name, row in outputs.items():
        if name not in EXPOSURE_OUTPUTS or row.get("status") != "calculated":
            continue
        _, background = EXPOSURE_OUTPUTS[name]
        model = row.get("model_id")
        if not model or models.setdefault(background, model) != model:
            return False
        if "risk_threshold" in row and thresholds.setdefault(background, row["risk_threshold"]) != row["risk_threshold"]:
            return False
    return True


def output_has_exclusion(row):
    """Retain an explicit domain exclusion for dependency propagation and feedback."""
    applicability = row.get("applicability", {})
    return (row.get("status") == "inapplicable" and bool(applicability.get("reason"))
            and bool(applicability.get("inputs")))


def output_dependencies(estimate, name, *, derived=False):
    """Read provenance recorded by execution, including exclusions and derivations."""
    trace = estimate["derived_dependencies" if derived else "metric_dependencies"][name]
    return trace["input_fields"], trace["condition_fields"]


def evaluate_parameters(parameters, conditions, metrics, *, subject, label, provenance=None, variables=()):
    """All results describe the selected assumptions, not an unconditional safety claim."""
    for name, value in parameters.items():
        validate_parameter(name, value, UNITS.get(name))
    if set(variables).intersection(parameters):
        raise ValueError("A calculation variable cannot also be assigned a scalar value")
    if not isinstance(conditions, Mapping) or any(k not in CONDITIONS for k in conditions):
        raise ValueError("Undeclared model condition")
    for name, basis in conditions.items():
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError(f"Named basis required for {name}")
    values, outputs = dict(parameters), {}
    derivations = {}
    dependencies = {k: {"input_fields": [k], "condition_fields": []} for k in parameters}
    metric_dependencies, derived_dependencies = {}, {}

    def trace(inputs, conditions=()):
        fields, required = set(), set(conditions)
        for field in inputs:
            dependency = dependencies.get(field, {"input_fields": [field], "condition_fields": []})
            fields.update(dependency["input_fields"])
            required.update(dependency["condition_fields"])
        return {"input_fields": sorted(fields), "condition_fields": sorted(required)}

    def derive(name, inputs, formula, function, required_conditions=()):
        # Bound inputs and results already computed in this call are reused.
        if name in values:
            return
        if all(k in values for k in inputs):
            answer = function(*(values[k] for k in inputs))
            values[name] = answer
            derivations[name] = {"inputs": list(inputs), "formula": formula, "value": answer}
            dependencies[name] = derived_dependencies[name] = trace(inputs, required_conditions)

    for _ in range(len(PARAMETER_DERIVATIONS) + 1):
        before = len(values)
        for derivation in PARAMETER_DERIVATIONS:
            derive(derivation.output, derivation.inputs, derivation.formula,
                   derivation.function, derivation.conditions)
        if len(values) == before:
            break
    if "single_infectious_source" in conditions:
        if parameters.get("index_case_count", 1) != 1:
            raise ValueError("The requested single-source model conflicts with index_case_count")
    if "susceptible_count" in values and "occupant_count" in values and values["susceptible_count"] > values["occupant_count"]-1:
        raise ValueError("susceptible_count exceeds N-1")
    if {"continuous_unmasked_jet", "negligible_near_field"} <= set(conditions):
        raise ValueError("Continuous-jet and negligible-near-field distance assumptions require separate parameterizations")

    def compute(name):
        if name in outputs:
            return outputs[name]
        if name in values or name not in METRICS:
            metric_dependencies[name] = trace((name,))
            if name not in values:
                return {"status": "missing_inputs", "missing": [name],
                        "requirements": input_requirements((name,), values, conditions, variables)}
            return {"status": "calculated", "value": values[name], "limit": "finite", "unit": UNITS[name]}
        m = select_metric(name, conditions)
        row = {**m.as_record(), "value": None, "limit": None, "status": "missing_inputs", "missing": []}
        outputs[name] = row
        metric_dependencies[name] = trace(m.inputs, metric_conditions(name, regime=m.regime))
        def exclude(reason, inputs):
            metric_dependencies[name] = trace(inputs)
            row.pop("distance_outcome", None)
            row.update(status="inapplicable", missing=[], applicability={"reason": reason, "inputs": inputs})
            return row

        for constraint in m.parameter_constraints():
            field = constraint["field"]
            accepts = {"equal": operator.eq, "minimum": operator.ge}[constraint["operator"]]
            if field in values and not accepts(values[field], constraint["value"]):
                return exclude(constraint["reason"], {field: values[field]})
        needed = metric_conditions(name, regime=m.regime)
        row["missing"] = []
        args = {}
        conditional = name in CONDITIONAL_INPUTS and "exposure_time_s" in variables
        inputs = conditional_inputs(name, regime=m.regime) if conditional else m.inputs
        for dep in inputs:
            if dep in values:
                args[dep] = values[dep]
                continue
            child = compute(dep)
            if child.get("limit") == "function":
                row.update(status="unsupported_variable", missing=[f"conditional_dependency:{dep}"],
                           capability_gap="This output does not implement composition of duration relationships; do not assign an arbitrary duration.")
                return row
            if output_has_exclusion(child):
                return exclude(f"Required quantity {dep} is inapplicable: {child['applicability']['reason']}",
                               child["applicability"]["inputs"])
            if child["status"] != "calculated":
                row["missing"].extend(child.get("missing") or [dep])
            elif child.get("limit") == "unbounded":
                args[dep] = math.inf
            else:
                args[dep] = child["value"]
        guards = tuple(c["field"] for c in m.parameter_constraints())
        metric_dependencies[name] = trace((*inputs, *guards), needed)
        row["missing"] = sorted(set(row["missing"]))
        if row["missing"]:
            row["requirements"] = input_requirements((name,), values, conditions, variables)
            return row
        if m.regime in {"jet", "random_jet"} and "mask_penetration" not in values:
            row.update(status="missing_inputs", missing=["mask_penetration"])
            return row
        try:
            if conditional:
                row.update(status="calculated", value=None, limit="function",
                           function=bind_duration(name, args, regime=m.regime))
                return row
            answer = m.function(args)
            if math.isnan(answer) or answer < 0:
                raise ValueError("Nonphysical or undefined result")
            row.update(status="calculated", value=answer if math.isfinite(answer) else None,
                       limit="finite" if math.isfinite(answer) else "unbounded")
            if name in EXPOSURE_OUTPUTS:
                operation, background = EXPOSURE_OUTPUTS[name]
                near_field = row["near_field"]
                binding = {"family": row["model_family"], "background": background, "near_field": near_field,
                           "parameters": {field: args[field] for field in model_inputs(background, near_field)},
                           "conditions": {field: conditions[field] for field in sorted(needed) if field in conditions}}
                row["model_id"] = hashlib.sha256(json.dumps(binding, sort_keys=True, allow_nan=False).encode()).hexdigest()
                row["model_binding"] = binding
                row["fixed_inputs"] = {field: {"value": args[field], "unit": UNITS[field]}
                                       for field in FIXED_FIELDS[operation]}
                if operation != "evaluate_R_in":
                    row["risk_threshold"] = args["risk_tolerance"]
                if operation == "solve_distance":
                    row["distance_outcome"] = ("no_finite_separation" if not math.isfinite(answer) else
                                               "no_positive_minimum" if answer == 0 else "finite_boundary")
            values[name] = answer
            dependencies[name] = metric_dependencies[name]
            # Preserve computed canonical intermediates for reuse and dependency
            # tracking, including metrics evaluated recursively inside an output.
            if name in UNITS and name not in parameters and math.isfinite(answer):
                derivations[name] = {"inputs": list(inputs), "formula": m.formula, "value": answer}
                derived_dependencies[name] = metric_dependencies[name]
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            row.update(status="undefined", missing=[str(exc)])
        return row

    requested = {name: compute(name) for name in metrics}
    intermediates = {name: row for name, row in outputs.items() if name not in requested}
    return {"subject": subject, "label": label, "metrics": requested,
            "intermediate_outputs": intermediates,
            "intermediate_dependencies": {name: metric_dependencies[name] for name in intermediates},
            "parameters": dict(parameters), "derived_parameters": derivations,
            "metric_dependencies": {name: metric_dependencies[name] for name in metrics},
            "derived_dependencies": derived_dependencies,
            "conditions": dict(conditions), "input_provenance": provenance or {},
            "calculation_variables": list(variables),
            "execution_status": "completed" if all(row["status"] == "calculated" for row in requested.values()) else "not_calculated",
            "interpretation": "Conditional model estimates; crossing a budget boundary is not a guarantee of safety.",
            "validation": "No independent validation is implied by solving or reusing a calibration equation."}
