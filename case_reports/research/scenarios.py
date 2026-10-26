"""Caller-owned case definitions, never extracted findings or computed answers."""

from dataclasses import dataclass
import json
from pathlib import Path

from ...task_executors.scenario_metrics.evaluate import CONDITIONS, UNITS, validate_parameter
from ...task_executors.scenario_metrics.exposure import EXPOSURE_OUTPUTS, FIXED_FIELDS
from ...task_executors.unit_conversion.units import METERS_PER_FOOT, SECONDS_PER_MINUTE
from .records import Fact, ResearchResult, identity


REFERENCE_INPUTS = (("neighbor_distance_m", 6 * METERS_PER_FOOT),
                    ("exposure_time_s", 15 * SECONDS_PER_MINUTE))


@dataclass(frozen=True)
class Scenario:
    subject: str
    description: str
    fixed: tuple
    conditions: tuple = ()
    assumption_needs: tuple = ()
    variables: tuple = ("exposure_time_s", "neighbor_distance_m")
    basis: str = "Caller-defined case conditions, not empirical evidence or computed answers."

    def __post_init__(self):
        near_field = {"continuous_unmasked_jet", "negligible_near_field"}.intersection(dict(self.conditions))
        if len(near_field) != 1:
            raise ValueError("Scenario conditions must explicitly select exactly one of continuous_unmasked_jet or negligible_near_field")

    def as_record(self):
        from .scenario_sharing import needed_fields

        fixed = {field: {"value": value, "unit": UNITS[field]} for field, value in self.fixed}
        contracts = {}
        for name, (operation, background) in EXPOSURE_OUTPUTS.items():
            contracts[name] = {"operation": operation, "background": background, "result_form": "scalar",
                "fixed_fields": list(FIXED_FIELDS[operation]),
                "fixed_inputs": {field: fixed[field] for field in FIXED_FIELDS[operation] if field in fixed},
                "solved_variable": {"solve_time": "exposure_time_s", "solve_distance": "neighbor_distance_m",
                                    "evaluate_R_in": None}[operation]}
        record = {"subject": self.subject, "description": self.description, "role": "target",
                "basis": self.basis, "scope_review": "Fixed by the caller's scenario selection.",
                "scope_locked": True,
                "fixed_parameters": fixed, "output_contracts": contracts,
                "fixed_conditions": dict(self.conditions),
                "assumption_needs": list(self.assumption_needs), "calculation_variables": list(self.variables)}
        record["assumption_needs"] = sorted(needed_fields(record))
        return record


def _paper_cases():
    # Fig. 3 cases with the selected six-foot and fifteen-minute reference inputs.
    # These are inputs, not breathing/infectiousness estimates or safety answers.
    settings = (
        ("classroom", "Quiet classroom, 19 students and one teacher", 83.6, 301, 20, .1, "cloth masks"),
        ("nursing-home", "Shared nursing-home room with elderly occupants at rest or quiet speech",
         22.3, 53.5, 3, .01, "surgical or hybrid-fabric masks"),
    )
    cases = []
    for setting, description, area, volume, occupants, tolerance, mask in settings:
        for ventilation, ach in (("natural", .34), ("mechanical", 8.0)):
            for masked in (False, True):
                fixed = (("floor_area_m2", area), ("room_volume_m3", volume),
                         ("occupant_count", occupants), ("index_case_count", 1),
                         ("susceptible_count", occupants - 1), ("risk_tolerance", tolerance),
                         ("air_exchange_rate_s", ach / 3600),
                         *REFERENCE_INPUTS,
                         ("initial_concentration_quanta_m3", 0))
                if not masked:
                    fixed += (("mask_penetration", 1),)
                cases.append(Scenario(
                    f"{setting}-{ventilation}-{'masked' if masked else 'unmasked'}",
                    f"{description}; {ventilation} ventilation; {mask if masked else 'no masks'}. "
                    "Solve allowable time at 6 ft (1.8288 m); solve safe distance at 15 min (900 s); "
                    "evaluate R_in at both reference values. These are inputs, not assumed safe answers. "
                    "One infectious occupant; assess steady background and initially clean-room regimes explicitly.",
                    fixed,
                    (("well_mixed", "Spatially uniform background concentration in the selected case."),
                     ("single_infectious_source", "One continuously present infectious source."),
                     ("continuous_unmasked_jet", "Include the local continuous-jet contribution in this scenario.")),
                    ("breathing_flow_m3_s", "quanta_concentration_exhaled_m3", "relative_transmissibility",
                     "deactivation_rate_s", "effective_radius_m") + (("mask_penetration",) if masked else ()),
                    variables=(),
                    basis="Bazant and Bush (2021), Fig. 3 case definitions, six-foot separation and fifteen-minute reference duration, selected for the requested evaluations; fixed design conditions, not discovered evidence or computed safety answers."))
    return tuple(cases)


BAZANT_BUSH_CASES = _paper_cases()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Scenario {name} must be nonempty text")
    return value


def scenario_from_record(record):
    allowed = {"subject", "description", "fixed", "conditions", "assumption_needs", "variables", "basis"}
    if not isinstance(record, dict) or set(record) - allowed or not {"subject", "description", "fixed"} <= set(record):
        raise ValueError("Scenario requires subject, description and fixed inputs; answers, evidence and extra fields are not accepted")
    subject, description = (_text(record[k], k) for k in ("subject", "description"))
    fixed = record["fixed"]
    if not isinstance(fixed, dict):
        raise ValueError("Scenario fixed inputs must map parameter names to value/unit objects")
    for field, row in fixed.items():
        if not isinstance(row, dict) or set(row) != {"value", "unit"}:
            raise ValueError("Each fixed scenario input requires exactly value and unit")
        validate_parameter(field, row["value"], row["unit"])
    # Reference coordinates define target evaluations, never outbreak durations.
    fixed = {**{field: {"value": value, "unit": UNITS[field]} for field, value in REFERENCE_INPUTS},
             **fixed}
    conditions = record.get("conditions", {})
    if not isinstance(conditions, dict) or set(conditions) - set(CONDITIONS):
        raise ValueError("Scenario conditions must be declared model conditions")
    for name, statement in conditions.items():
        _text(statement, name)
    needs = record.get("assumption_needs", [])
    variables = record.get("variables", [])
    if variables != []:
        raise ValueError("Target scenarios require scalar time, distance and R_in outputs. "
                         "Use fixed reference inputs or omit them to use the declared defaults; "
                         "free calculation variables are not supported by this report contract.")
    for name, values, choices in (("assumption_needs", needs, set(UNITS) | set(CONDITIONS)),
                                  ("variables", variables, {"exposure_time_s", "neighbor_distance_m"})):
        if (not isinstance(values, list) or any(not isinstance(v, str) or v not in choices for v in values)
                or len(set(values)) != len(values) or set(values).intersection(fixed) or set(values).intersection(conditions)):
            raise ValueError(f"Scenario {name} must contain distinct unfixed canonical fields")
    if set(needs).intersection(variables):
        raise ValueError("Calculation variables are not assumptions to assign arbitrary values")
    return Scenario(subject, description, tuple((k, v["value"]) for k, v in fixed.items()),
                    tuple(conditions.items()), tuple(needs), tuple(variables),
                    _text(record.get("basis", Scenario.basis), "basis"))


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate scenario key: {key}")
        result[key] = value
    return result


def load_scenarios(selection="bazant-bush"):
    """Built-in set/name or a strict JSON file of caller-specified case inputs."""
    if selection == "bazant-bush":
        return BAZANT_BUSH_CASES
    if selection in {"classroom", "nursing-home"}:
        return tuple(case for case in BAZANT_BUSH_CASES if case.subject.startswith(selection + "-"))
    matches = tuple(case for case in BAZANT_BUSH_CASES if case.subject == selection)
    if matches:
        return matches
    data = json.loads(Path(selection).read_text(), object_pairs_hook=_object)
    if not isinstance(data, dict) or set(data) != {"scenarios"} or not isinstance(data["scenarios"], list) or not data["scenarios"]:
        raise ValueError("Scenario file requires a nonempty scenarios list, with no answers or evidence")
    cases = tuple(scenario_from_record(row) for row in data["scenarios"])
    if len({case.subject for case in cases}) != len(cases):
        raise ValueError("Scenario subjects must be unique")
    return cases


def initial_scenarios(cases):
    facts, assumptions = [], []
    for case in cases:
        claims = [(field, value, f"Fixed case condition: {field} = {value} {UNITS[field]}.")
                  for field, value in case.fixed]
        claims.extend((field, None, statement) for field, statement in case.conditions)
        for field, value, statement in claims:
            key = identity([case.subject, field, value, statement if value is None else None])
            assumption = {"subject": case.subject, "field": field, "value": value,
                "unit": UNITS.get(field, "condition"), "statement": statement, "basis": case.basis,
                "key": key, "role": "scenario_constraint", "reasonableness": 1.0,
                "evidence": {}, "review": {
                    "completed": True, "verdict": "fixed_condition", "criteria": {}, "request_constraint": None,
                    "explanation": "Defines the selected case, not an empirical finding or discovered support.",
                    "limitations": ["Conditional scope only; no empirical evidence is implied."],
                    "alternatives": [], "missing_evidence": []}}
            assumption["version"] = identity(assumption)
            assumptions.append(assumption)
            if value is not None:
                facts.append(Fact(case.subject, field, value, UNITS[field], "scenario_constraint",
                                  case.basis, context=statement, parents=(key,)))
    return ResearchResult(facts=tuple(facts), assumptions=tuple(assumptions),
                          scenarios=tuple(case.as_record() for case in cases))


def check_case_inputs(definition, parameters, conditions=()):
    """Scope constraints, not an evidence-quality gate."""
    if not definition.get("scope_locked"):
        return
    from .model_defaults import retained_defaults
    defaults = retained_defaults(definition)
    for field, value in parameters.items():
        if field in definition["calculation_variables"]:
            raise ValueError(f"{field} is a calculation variable for this case, not a freely chosen scenario assumption")
        fixed = definition["fixed_parameters"].get(field)
        if fixed is not None and value != fixed["value"]:
            raise ValueError(f"{field} conflicts with the fixed scenario condition")
        if field in defaults and value != defaults[field]["value"]:
            raise ValueError(f"{field} conflicts with the retained model baseline; its scenario plan did not promote it")
    for field, statement in dict(conditions).items():
        fixed = definition["fixed_conditions"].get(field)
        if fixed is not None and statement != fixed:
            raise ValueError(f"Condition {field} conflicts with the fixed scenario definition")
