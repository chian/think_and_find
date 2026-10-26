"""Executable parameter derivations and their parent-facing dependency contracts."""

from dataclasses import dataclass
import operator
from typing import Callable

from .catalog import METRICS, metric_conditions, metric_routes, select_metric
from .conditional import CONDITIONAL_INPUTS, conditional_inputs


@dataclass(frozen=True)
class Derivation:
    output: str
    inputs: tuple[str, ...]
    formula: str
    function: Callable
    conditions: tuple[str, ...] = ()

    def as_record(self):
        return {"output": self.output, "inputs": list(self.inputs),
                "formula": self.formula, "conditions": list(self.conditions)}


PARAMETER_DERIVATIONS = (
    Derivation("room_volume_m3", ("floor_area_m2", "room_height_m"), "A*H", lambda a, h: a*h),
    Derivation("room_height_m", ("room_volume_m3", "floor_area_m2"), "V/A", lambda v, a: v/a),
    Derivation("ventilation_flow_m3_s", ("air_exchange_rate_s", "room_volume_m3"), "lambda_a*V", lambda a, v: a*v),
    Derivation("air_exchange_rate_s", ("ventilation_flow_m3_s", "room_volume_m3"), "Q/V", lambda q, v: q/v),
    Derivation("settling_speed_m_s", METRICS["settling_speed_m_s"].inputs,
               METRICS["settling_speed_m_s"].formula,
               lambda *values: METRICS["settling_speed_m_s"].function(
                   dict(zip(METRICS["settling_speed_m_s"].inputs, values))), ("stokes_particles",)),
    Derivation("removal_capacity_m3_s", ("ventilation_flow_m3_s", "recirculation_flow_m3_s",
               "filter_efficiency", "deactivation_rate_s", "room_volume_m3", "settling_speed_m_s", "floor_area_m2"),
               "Q+pf*Qr+lambda_v*V+vs*A", lambda q, qr, pf, lv, v, vs, a: q+pf*qr+lv*v+vs*a),
    Derivation("removal_rate_s", ("removal_capacity_m3_s", "room_volume_m3"), "Qeff/V", lambda q, v: q/v),
)
DERIVED_NAMES = frozenset(d.output for d in PARAMETER_DERIVATIONS)

# These registered calculations remain usable inputs, but are not initial
# parameter-discovery goals. Reversible geometry inputs are deliberately separate.
CALCULATED_PARAMETER_FIELDS = frozenset({
    "quanta_emission_rate_s", "settling_speed_m_s",
    "removal_capacity_m3_s", "removal_rate_s",
})


def fixed_input_fields(available, conditions=()):
    """Inputs supplied directly or derivable entirely from caller-fixed inputs."""
    fields, conditions = set(available), set(conditions)
    while True:
        derived = {d.output for d in PARAMETER_DERIVATIONS
                   if set(d.inputs) <= fields}
        if derived <= fields:
            return fields
        fields.update(derived)


def input_fields(names, *, parameters=(), fixed_values=None, fixed_conditions=()):
    """All relevant parameter/condition fields across executable dependency routes."""
    seen, fields = set(), set()
    fixed_values = fixed_values or {}
    fixed_conditions = set(fixed_conditions)
    fixed = fixed_input_fields(fixed_values, fixed_conditions) | fixed_conditions

    def compatible(route):
        if fixed_conditions & {"continuous_unmasked_jet", "negligible_near_field"}:
            if route.regime != select_metric(route.name, fixed_conditions).regime:
                return False
        for constraint in route.parameter_constraints():
            field = constraint["field"]
            accepts = {"equal": operator.eq, "minimum": operator.ge}[constraint["operator"]]
            if field in fixed_values and not accepts(fixed_values[field], constraint["value"]):
                return False
        return True

    def visit(name):
        if name in seen or name in fixed:
            return
        seen.add(name)
        if name not in METRICS or name in DERIVED_NAMES or name in parameters:
            fields.add(name)
        if name in METRICS:
            for route in metric_routes(name):
                if not compatible(route):
                    continue
                fields.update(metric_conditions(name, regime=route.regime) - fixed_conditions)
                for dependency in (*route.inputs, *(c["field"] for c in route.parameter_constraints())):
                    visit(dependency)
        for derivation in PARAMETER_DERIVATIONS:
            if derivation.output == name:
                fields.update(set(derivation.conditions) - fixed_conditions)
                for dependency in derivation.inputs:
                    visit(dependency)

    for name in names:
        visit(name)
    return fields


def input_requirements(names, available=(), conditions=(), variables=()):
    """Describe missing inputs and alternative routes; never choose input values."""
    available, conditions, variables = set(available), set(conditions), set(variables)
    unresolved, seen = {}, set()

    def visit(name):
        if name in seen:
            return
        seen.add(name)
        if name in available or name in conditions:
            return
        if name in variables:
            unresolved[name] = {"status": "calculation_variable", "scalar_hypothesis_allowed": False}
            return
        routes = [d.as_record() for d in PARAMETER_DERIVATIONS if d.output == name]
        if name in METRICS and name not in DERIVED_NAMES:
            conditional = name in CONDITIONAL_INPUTS and "exposure_time_s" in variables
            for metric in metric_routes(name):
                routes.append({"output": name, "regime": metric.regime,
                               "inputs": list(conditional_inputs(name, regime=metric.regime) if conditional else metric.inputs),
                               "formula": metric.formula,
                               "conditions": sorted(metric_conditions(name, regime=metric.regime)),
                               "parameter_constraints": metric.parameter_constraints(),
                               "interpretation": metric.interpretation,
                               "result_form": "duration_relationship" if conditional else "scalar"})
        unresolved[name] = {"status": "derivable" if routes else "needs_input",
                            "routes": routes}
        for route in routes:
            required = tuple(dict.fromkeys((*route["inputs"],
                *(c["field"] for c in route.get("parameter_constraints", ())))))
            route["missing_inputs"] = [f for f in required if f not in available]
            route["missing_conditions"] = []
            for dependency in (*required, *route["conditions"]):
                visit(dependency)

    for name in names:
        visit(name)
    return {"requested": list(names), "available_inputs": sorted(available),
            "calculation_variables": sorted(variables),
            "available_conditions": sorted(conditions), "unresolved": unresolved,
            "meaning": "Executable dependency alternatives, not evidence or a prescribed action sequence. "
            "A missing derived quantity can use a recorded hypothesis or an applicable derivation route. "
            "Parameter constraints describe enforced kernel applicability, not evidence thresholds; "
            "a new condition statement cannot override them or add an executable model. "
            "Distance routes are alternatives: negligible_near_field selects spatially uniform risk, "
            "otherwise the continuous-unmasked-jet route applies. These two conditions must not be "
            "combined in one parameterization. Mask penetration alone selects neither hypothesis. "
            "Absent terms are not silently zero. Variables remain variables. Field availability does not "
            "select among alternative values or establish one coherent parameterization."}
