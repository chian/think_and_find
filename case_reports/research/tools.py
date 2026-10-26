"""Selectable deterministic calculations; admission and credit belong to the parent."""

from ...task_executors.scenario_metrics.catalog import METRICS
from ...task_executors.scenario_metrics.dependencies import input_requirements
from ...task_executors.scenario_metrics.evaluate import UNITS, evaluate_parameters, output_dependencies
from ...task_executors.scenario_metrics.exposure import EXPOSURE_OUTPUTS, MODEL_VERSION
from .modeling import DERIVATIONS, derivation_inputs, derive_parameter, source_calculation_inputs
from .records import Fact, ResearchResult, identity, current_support, current_scenario_links, assumption_claim, assumption_matches, sharing_scope_version
from .model_defaults import retained_defaults
from .scenarios import check_case_inputs
from .scenario_sharing import shared_assumptions


GEOMETRY = frozenset({"removal_rate_s", "settling_speed_m_s", "critical_aerosol_radius_m",
    "air_mixing_time_s", "concentration_relaxation_time_s", "air_clearance_time_s"})
LIMITS = frozenset({"cumulative_exposure_bound_person_s", "exposure_time_bound_s", "susceptible_bound",
    "total_occupancy_bound", "required_removal_capacity_m3_s", "transient_exposure_time_bound_s",
    "transient_susceptible_bound", "transient_total_occupancy_bound", "transient_interpolation_time_bound_s",
    "critical_distance_m", "risk_threshold_distance_m", "jet_corrected_time_bound_s",
    *(name for name, (operation, _) in EXPOSURE_OUTPUTS.items() if operation != "evaluate_R_in")})
TOOLS = {
    "calculate_geometry_and_removal": "Derive geometry and removal quantities from the current main-table values and supplied scenario inputs.",
    "calibrate_infectiousness": "Apply an explicitly selected calibration, activity rescaling, or transfer operator.",
    "calculate_transmission_metrics": "Evaluate the shared risk model at fixed time AND fixed distance. Select scenario_reproductive_number and/or transient_scenario_reproductive_number. The reference inputs are bound from the selected case; numeric inputs are bound automatically from the current main table. Available parameter derivations execute inside this tool.",
    "calculate_conditional_limits": "Invert the shared risk model: allowable_room_time_s (or transient counterpart) fixes distance and solves time; safe_distance_m (or transient counterpart) fixes time and solves distance. Reference inputs are bound from the case. Select either operation or both; no prescribed sequence or separate geometry call is required.",
    "compare_predictions_with_observations": "Compare one versioned prediction with an admitted observation; preserve calibration dependence.",
}
TOOL_METRICS = {
    "calculate_geometry_and_removal": GEOMETRY,
    "calculate_conditional_limits": LIMITS,
    "calculate_transmission_metrics": frozenset(METRICS) - GEOMETRY - LIMITS,
}


def main_calculation_inputs(ledger, subject):
    """Bind actual main-table values, supplied inputs, and current computed facts.

    This is a deterministic binding, not a catalog from which a caller chooses.
    """
    from .calculation_steps import current_calculated_inputs, selected_target_snapshot

    scenario = ledger.scenarios.get(subject)
    if scenario is None or scenario.get("role", "target") != "target":
        raise ValueError("Main-table calculation binding requires a registered target scenario")
    bound = {}
    fixed = scenario.get("fixed_parameters", {})
    defaults = retained_defaults(scenario)
    for field, row in {**defaults, **fixed}.items():
        kind = "scenario_constraint" if field in fixed else "model_default"
        matches = [fact for fact in ledger.facts.values() if fact.subject == subject
                   and fact.field == field and fact.kind == kind
                   and (fact.value, fact.unit) == (row["value"], row["unit"])]
        if not matches:
            # Supplied values are explicit scenario metadata, not inferred claims.
            matches = [Fact(subject, field, row["value"], row["unit"], kind,
                            row.get("basis", "Explicit scenario input"))]
        bound[field] = matches[0]
    shared = {assumption["key"]: relation for assumption, relation in shared_assumptions(ledger, subject)}
    for (target, field), row in selected_target_snapshot(ledger).items():
        if target != subject or not row.get("found") or field in bound:
            continue
        assumption = ledger.assumptions[row["selected_assumption_key"]]
        parents = (assumption["key"],)
        context = assumption["statement"]
        if assumption["subject"] != subject:
            if assumption["key"] not in shared:
                raise ValueError(f"Main-table input {field} has no current sharing relation for {subject}")
            parents += (identity(shared[assumption["key"]]),)
            context = f"Shared from {assumption['subject']}: {context}"
        bound[field] = Fact(subject, field, row["value"], row["unit"], "model_assumption",
                            assumption["basis"], context=context, parents=parents)
    for fact in current_calculated_inputs(ledger, subject):
        if fact.field not in bound:
            bound[fact.field] = fact
    return tuple(bound.values())


def resolve_calculation_inputs(action, ledger, requested_metrics=()):
    """Bind numeric inputs and model conditions from the registered scenario."""
    subject = action.get("subject")
    definition = ledger.scenarios.get(subject)
    if definition is None:
        raise ValueError("Calculations require a reviewed scenario")
    action = {**action, "conditions": dict(definition.get("fixed_conditions", {}))}
    target = definition.get("role", "target") == "target"
    if target:
        action.pop("facts", None)
        action.pop("source_inputs", None)
        if action.get("operator") == "transfer_parameter":
            raise ValueError("Target values are bound from the main table; source transfer cannot override them")
        facts = main_calculation_inputs(ledger, subject)
    else:
        facts = source_calculation_inputs(action, ledger)
        by_field = {fact.field: fact for fact in facts}
        for field, row in {**retained_defaults(definition), **definition.get("fixed_parameters", {})}.items():
            if field not in by_field:
                kind = "scenario_constraint" if field in definition.get("fixed_parameters", {}) else "model_default"
                by_field[field] = next((fact for fact in ledger.facts.values()
                    if fact.subject == subject and fact.field == field and fact.kind == kind
                    and (fact.value, fact.unit) == (row["value"], row["unit"])),
                    Fact(subject, field, row["value"], row["unit"], kind,
                         row.get("basis", "Explicit scenario input")))
        facts = tuple(by_field.values())
    if action.get("action") == "calibrate_infectiousness":
        required = derivation_inputs(action.get("operator"), {fact.field: fact.value for fact in facts})
        facts = tuple(fact for fact in facts if fact.field in required)
    return action, facts


def support_for(ledger, facts, conditions, subject):
    """Describe support gaps without preventing a conditional calculation."""
    selected = {}
    visiting = set()
    complete = True
    invalidated = []

    def require(fact, recorded_links=()):
        nonlocal complete
        if fact.key in visiting:
            raise ValueError("Cyclic parameter provenance")
        if fact.kind == "model_assumption" and fact.parents:
            # A selected hypothesis already identifies its premise. Another
            # same-valued claim must not silently replace that lineage.
            key = fact.parents[0]
            parent = ledger.assumptions.get(key)
            direct = (parent is not None and len(fact.parents) == 1
                      and parent["subject"] == fact.subject
                      and parent["statement"] == fact.context)
            shared = (len(fact.parents) == 2 and any(
                a["key"] == key and (identity(relation) == fact.parents[1] or any(
                    link.get("assumption_key") == key
                    and link.get("target_subject") == fact.subject
                    and link.get("relation_version") == fact.parents[1]
                    and current_scenario_links({"scenario_parameter_links": [link]}, ledger)
                    for link in recorded_links))
                and fact.context == f"Shared from {a['subject']}: {a['statement']}"
                for a, relation in shared_assumptions(ledger, fact.subject)))
            if (parent is not None and not parent.get("source_candidate_unavailable")
                    and (direct or shared) and parent["field"] == fact.field
                    and parent["value"] == fact.value and parent["unit"] == fact.unit):
                selected[key] = parent
            else:
                complete = False
                invalidated.append(key)
            return
        matches = [a for a in ledger.assumptions.values() if a["subject"] == fact.subject
                   and a["field"] == fact.field and a["value"] == fact.value
                   and not a.get("source_candidate_unavailable")]
        if matches:
            a = max(matches, key=lambda a: a["reasonableness"])
            selected[a["key"]] = a
            return
        derivations = [c for c in ledger.calculations.values() if fact.key in c.get("output_fact_keys", ())]
        if fact.kind != "derived" or not derivations:
            complete = False
            return
        calculation = derivations[-1]
        if not calculation.get("model_review", {}).get("completed", True):
            same_model = ("tool", "subject", "input_fact_keys", "conditions", "assumption_versions",
                          "operator", "basis", "parameterization")
            completed = [c for c in derivations if c.get("model_review", {}).get("completed", True)
                         and all(c.get(k) == calculation.get(k) for k in same_model)]
            if completed:
                calculation = completed[-1]
        if calculation.get("operator") == "transfer_parameter":
            complete = False
        dependency = calculation.get("derived_support", {}).get(fact.field, calculation)
        complete = complete and dependency.get("support_complete", True)
        invalidated.extend(dependency.get("invalidated_dependencies", ()))
        if not current_scenario_links(dependency, ledger):
            complete = False
            invalidated.append(calculation["version"])
        if (not calculation.get("model_review", {}).get("admit", True)
                or not calculation.get("model_review", {}).get("completed", True)):
            complete = False
        for key in dependency["assumption_versions"]:
            current = ledger.assumptions.get(key, {})
            if not assumption_matches(dependency, key, current):
                complete = False
                invalidated.append(key)
            if current:
                selected[key] = current
        visiting.add(fact.key)
        # A rationale update can change a fact key without changing its claim.
        # The version/claim checks above still govern whether these inputs apply.
        inputs = {identity(row): Fact(**{**row, "parents": tuple(row.get("parents", ()))})
                  for row in calculation.get("input_records", ())}
        inputs.update(ledger.facts)
        for key in dependency["input_fact_keys"]:
            if key in inputs:
                require(inputs[key], dependency.get("scenario_parameter_links", ()))
            else:
                complete = False
                invalidated.append(key)
        visiting.remove(fact.key)

    for fact in facts:
        require(fact)
    for field, statement in conditions.items():
        matches = [a for a in ledger.assumptions.values() if a["subject"] == subject
                   and a["field"] == field and a["statement"] == statement]
        if not matches:
            complete = False
            continue
        a = max(matches, key=lambda a: a["reasonableness"])
        selected[a["key"]] = a
    return selected, complete, list(dict.fromkeys(invalidated))


def scenario_parameter_links(facts, ledger=None):
    """Keep scope provenance even when a shared input feeds a derived parameter."""
    links = []
    seen = set()
    def visit(fact):
        if fact.key in seen:
            return
        seen.add(fact.key)
        if fact.kind == "model_assumption" and len(fact.parents) == 2:
            link = {"assumption_key": fact.parents[0], "relation_version": fact.parents[1],
                    "target_subject": fact.subject}
            if ledger is not None:
                for assumption, relation in shared_assumptions(ledger, fact.subject):
                    if assumption["key"] == fact.parents[0] and identity(relation) == fact.parents[1]:
                        link.update(scope_version=sharing_scope_version(relation),
                                    assumption_claim=assumption_claim(assumption))
                        break
            links.append(link)
        if ledger is not None and fact.kind == "derived":
            for calculation in ledger.calculations.values():
                if fact.key not in calculation.get("output_fact_keys", ()):
                    continue
                dependency = calculation.get("derived_support", {}).get(fact.field, calculation)
                recorded_links = dependency.get("scenario_parameter_links", ())
                links.extend(recorded_links)
                records = {identity(row): Fact(**{**row, "parents": tuple(row.get("parents", ()))})
                           for row in calculation.get("input_records", ())}
                records.update(ledger.facts)
                for key in dependency.get("input_fact_keys", ()):
                    if key in records:
                        parent = records[key]
                        if parent.kind == "model_assumption" and len(parent.parents) == 2 and any(
                                row.get("assumption_key") == parent.parents[0]
                                and row.get("relation_version") == parent.parents[1]
                                and row.get("target_subject") == parent.subject for row in recorded_links):
                            continue
                        visit(parent)
    for fact in facts:
        visit(fact)
    return list({identity(row): row for row in links}.values())


def calculation_record(tool, subject, facts, support, output, *, conditions, basis, **extra):
    record = {"tool": tool, "subject": subject, "input_fact_keys": [f.key for f in facts],
              "input_records": [f.as_record() for f in facts],
              "input_values": {f.field: f.value for f in facts}, "conditions": dict(conditions),
              "assumption_versions": {k: a["version"] for k, a in support.items()},
              "assumption_claims": {k: assumption_claim(a) for k, a in support.items()},
              "basis": basis, "output": output,
              "scenario_parameter_links": scenario_parameter_links(facts),
              "model_version": MODEL_VERSION if set(output).intersection(EXPOSURE_OUTPUTS) else "bazant-bush-scalar-v1", **extra}
    record["version"] = identity(record)
    return record


def scenario_input_bindings(facts, ledger):
    """Record the supplied values actually used, separately from evidence review."""
    bindings = []
    for fact in facts:
        scenario = ledger.scenarios.get(fact.subject, {})
        fixed = scenario.get("fixed_parameters", {})
        defaults = retained_defaults(scenario)
        kind = ("scenario_constraint" if fact.field in fixed else
                "model_default" if fact.field in defaults else None)
        if kind is not None:
            bindings.append({"subject": fact.subject, "field": fact.field,
                             "value": fact.value, "unit": fact.unit, "kind": kind})
    return bindings


def scenario_condition_bindings(subject, conditions, ledger):
    fixed = ledger.scenarios.get(subject, {}).get("fixed_conditions", {})
    return [{"subject": subject, "field": field, "statement": statement}
            for field, statement in conditions.items() if field in fixed]


def output_support(estimate, field, facts, ledger, review, *, derived=False, selection=None):
    from .calculation_steps import selected_target_bindings, selected_target_snapshot
    if selection is None:
        selection = selected_target_snapshot(ledger)
    inputs, conditions = output_dependencies(estimate, field, derived=derived)
    selected = [fact for fact in facts if fact.field in inputs]
    support, complete, invalidated = support_for(ledger, selected,
        {k: v for k, v in estimate["conditions"].items() if k in conditions}, estimate["subject"])
    complete = (complete and set(inputs) <= {f.field for f in selected}
                and set(conditions) <= estimate["conditions"].keys())
    return {"input_fields": inputs, "condition_fields": conditions,
            "selected_target_bindings": selected_target_bindings(selected, selection),
            "input_fact_keys": [f.key for f in selected],
            "scenario_input_bindings": scenario_input_bindings(selected, ledger),
            "scenario_condition_bindings": scenario_condition_bindings(estimate["subject"],
                {field: estimate["conditions"][field] for field in conditions if field in estimate["conditions"]}, ledger),
            "scenario_parameter_links": scenario_parameter_links(selected, ledger),
            "assumption_versions": {k: a["version"] for k, a in support.items()},
            "assumption_claims": {k: assumption_claim(a) for k, a in support.items()},
            "support_complete": complete, "model_review": review, "invalidated_dependencies": invalidated}


def execute_tool(action, ledger, requested_metrics, *, review=None):
    review = {} if review is None else review
    name = action.get("action")
    if name not in TOOLS:
        raise ValueError("Undeclared deterministic tool")
    if name == "compare_predictions_with_observations":
        return compare(action, ledger, review)
    action, facts = resolve_calculation_inputs(action, ledger, requested_metrics)
    subject = action.get("subject")
    if subject not in ledger.scenarios:
        raise ValueError("Calculations require a reviewed scenario")
    if action.get("operator") != "transfer_parameter" and {f.subject for f in facts} != {subject}:
        raise ValueError("Calculation inputs must belong to the same scenario; transfer must be explicit")
    conditions = action.get("conditions", {})
    if not isinstance(conditions, dict):
        raise ValueError("Conditions must be named statements")
    check_case_inputs(ledger.scenarios[subject], {f.field: f.value for f in facts}, conditions)
    if name == "calibrate_infectiousness":
        required = derivation_inputs(action.get("operator"), {f.field: f.value for f in facts})
        facts = tuple(fact for fact in facts if fact.field in required)
    support, support_complete, invalidated = support_for(ledger, facts, conditions, subject)
    from .calculation_steps import selected_target_bindings, selected_target_snapshot
    selection = selected_target_snapshot(ledger)
    selection_bindings = selected_target_bindings(facts, selection)
    if name == "calibrate_infectiousness":
        result = derive_parameter(None, action, facts, {})
        calculation = calculation_record(name, subject, facts, support,
            {f.field: {"value": f.value, "unit": f.unit} for f in result.facts},
            conditions=conditions, basis=action["basis"], operator=action["operator"],
            model_review=review, support_complete=support_complete,
            selected_target_bindings=selection_bindings,
            scenario_input_bindings=scenario_input_bindings(facts, ledger),
            scenario_condition_bindings=scenario_condition_bindings(subject, conditions, ledger),
            scenario_parameter_links=scenario_parameter_links(facts, ledger),
            invalidated_dependencies=invalidated,
            output_fact_keys=[f.key for f in result.facts],
            calibration_subjects=sorted({f.subject for f in facts}) if "outbreak" in action["operator"] else [])
        if "outbreak" in action["operator"]:
            calculation["subtask"] = "calibrate_cq_from_outbreak"
            calculation["version"] = identity({k: v for k, v in calculation.items() if k != "version"})
        return ResearchResult(facts=result.facts, findings=result.findings, calculations=(calculation,))
    allowed = TOOL_METRICS[name]
    metrics = action.get("metrics", [m for m in requested_metrics if m in allowed])
    if not isinstance(metrics, list) or any(not isinstance(m, str) or m not in allowed for m in metrics):
        invalid = [m for m in metrics if not isinstance(m, str) or m not in allowed] if isinstance(metrics, list) else metrics
        raise ValueError(f"{name} received invalid metrics {invalid!r}; allowed metric names are {sorted(allowed)!r}. "
                         "Omit metrics to select this tool's requested outputs; no calculation was executed. "
                         "Geometry-derived input parameters are emitted automatically, not selected as metrics.")
    if not metrics and name != "calculate_geometry_and_removal":
        raise ValueError("Select at least one quantity to calculate")
    estimate = evaluate_parameters({f.field: f.value for f in facts}, conditions, tuple(metrics),
        subject=subject, label=action["basis"], provenance={f.field: f.as_record() for f in facts},
        variables=ledger.scenarios[subject].get("calculation_variables", ()))
    estimate["output_contracts"] = {m: ledger.scenarios[subject]["output_contracts"][m]
        for m in metrics if m in ledger.scenarios[subject].get("output_contracts", {})}
    parameterization = action.get("parameterization", action["basis"])
    if not isinstance(parameterization, str) or not parameterization.strip():
        raise ValueError("A calculation needs a named parameterization")
    estimate.update(parameterization=parameterization, label=parameterization)
    estimate.update(assumption_keys=list(support), assumption_versions={k: a["version"] for k, a in support.items()},
                    selected_target_bindings=selection_bindings,
                    assumption_claims={k: assumption_claim(a) for k, a in support.items()},
                    assumption_support=list(support.values()),
                    model_review=review, support_complete=support_complete, invalidated_dependencies=invalidated,
                    scenario_input_bindings=scenario_input_bindings(facts, ledger),
                    scenario_condition_bindings=scenario_condition_bindings(subject, conditions, ledger),
                    scenario_parameter_links=scenario_parameter_links(facts, ledger))
    estimate["metric_support"] = {m: output_support(estimate, m, facts, ledger, review, selection=selection) for m in metrics}
    intermediate_view = {**estimate, "metrics": estimate["intermediate_outputs"],
                         "metric_dependencies": estimate["intermediate_dependencies"]}
    estimate["intermediate_support"] = {
        field: output_support(intermediate_view, field, facts, ledger, review, selection=selection)
        for field in estimate["intermediate_outputs"]}
    estimate["reasonableness"] = current_support(estimate, ledger.assumptions, ledger=ledger)
    derived = tuple(Fact(subject, field, row["value"], UNITS[field], "derived", row["formula"],
                        context=action["basis"], parents=tuple(f.key for f in facts
                            if f.field in output_dependencies(estimate, field, derived=True)[0]))
                    for field, row in estimate["derived_parameters"].items())
    calculation = calculation_record(name, subject, facts, support, estimate["metrics"],
        conditions=conditions, basis=action["basis"], output_fact_keys=[f.key for f in derived],
        model_review=review, support_complete=support_complete,
        selected_target_bindings=selection_bindings,
        scenario_input_bindings=scenario_input_bindings(facts, ledger),
        scenario_condition_bindings=scenario_condition_bindings(subject, conditions, ledger),
        scenario_parameter_links=scenario_parameter_links(facts, ledger),
        invalidated_dependencies=invalidated,
        derived_parameters=estimate["derived_parameters"], parameterization=parameterization,
        execution_status=estimate["execution_status"],
        calculation_variables=estimate["calculation_variables"],
        output_contracts=estimate["output_contracts"],
        metric_support=estimate["metric_support"],
        intermediate_outputs=estimate["intermediate_outputs"],
        intermediate_support=estimate["intermediate_support"],
        derived_support={field: output_support(estimate, field, facts, ledger, review, derived=True, selection=selection)
                         for field in estimate["derived_parameters"]})
    estimate["calculation_versions"] = [calculation["version"]]
    return ResearchResult(facts=derived, estimates=(estimate,), calculations=(calculation,),
                           findings=tuple(review.get("limitations", [])))


def compare(action, ledger, review):
    options = ledger.calculation_options()
    key, observation_key, metric = action.get("prediction"), action.get("observation"), action.get("metric")
    if key not in options or observation_key not in ledger.options():
        raise ValueError("Comparison must select an existing calculation and fact")
    prediction, observation = options[key], ledger.options()[observation_key]
    if observation.kind != "source_observation":
        raise ValueError("A model choice or assumption is not an observed validation outcome")
    if observation.subject != prediction["subject"]:
        raise ValueError("Prediction and observation must describe the same scenario")
    comparable = {"attack_probability": {"individual_infection_probability", "transient_individual_infection_probability"},
                  "secondary_case_count": {"expected_secondary_cases", "transient_expected_secondary_cases"}}
    if metric not in comparable.get(observation.field, set()):
        raise ValueError("Prediction and observed quantity have incompatible meanings")
    row = prediction["output"].get(metric, {})
    if row.get("status") != "calculated" or row.get("limit") != "finite":
        raise ValueError("Comparison requires a finite calculated prediction")
    dependency = prediction.get("metric_support", {}).get(metric, prediction)
    support_complete = dependency.get("support_complete", True)
    if (not prediction.get("model_review", {}).get("admit", True)
            or not prediction.get("model_review", {}).get("completed", True)):
        support_complete = False
    lineage, pending, calibration_subjects = set(), list(dependency["input_fact_keys"]), set()
    while pending:
        fact_key = pending.pop()
        if fact_key in lineage:
            continue
        lineage.add(fact_key)
        for calculation in ledger.calculations.values():
            if fact_key in calculation.get("output_fact_keys", ()):
                fact = ledger.facts.get(fact_key)
                provenance = calculation.get("derived_support", {}).get(fact.field, calculation) if fact else calculation
                pending.extend(provenance["input_fact_keys"])
                calibration_subjects.update(calculation.get("calibration_subjects", ()))
    dependent = observation.key in lineage or observation.subject in calibration_subjects
    comparison = {"metric": metric, "predicted": row["value"], "observed": observation.value,
                  "residual": row["value"] - observation.value,
                  "absolute_error": abs(row["value"] - observation.value),
                  "observation": observation.semantic_record(),
                  "prediction_version": prediction["version"],
                  "category": "calibration_consistency" if dependent else "out_of_calibration_comparison",
                  "interpretation": review["reason"],
                  "independent_validation": False,
                  "limitation": "No preregistered held-out protocol or independence from outcome-informed modeling is established."}
    calculation = {"tool": action["action"], "subject": prediction["subject"], "output": comparison,
                   "input_fact_keys": [*dependency["input_fact_keys"], observation.key],
                   "assumption_versions": dependency["assumption_versions"], "basis": action["basis"],
                   "assumption_claims": dependency.get("assumption_claims", {}),
                   "model_review": review, "support_complete": support_complete}
    calculation["version"] = identity(calculation)
    return ResearchResult(calculations=(calculation,), findings=(
        f"{comparison['category']}: {metric} residual={comparison['residual']:.8g}. {review['reason']}",))


def tool_options(ledger):
    from .calculation_steps import calculation_is_up_to_date
    return {key: {"version": c["version"], "tool": c["tool"], "subject": c["subject"],
                  "output": c["output"], "basis": c["basis"], "parameterization": c.get("parameterization"),
                  "derived_parameters": c.get("derived_parameters", {}),
                  "intermediate_outputs": c.get("intermediate_outputs", {}),
                  "current_intermediate_metrics": [m for m in c.get("intermediate_outputs", {})
                       if calculation_is_up_to_date(c, m, ledger)],
                  "current_support": current_support(c, ledger.assumptions, ledger=ledger),
                   "current_computed_metrics": [m for m in c["output"] if calculation_is_up_to_date(c, m, ledger)],
                  "metric_support": {m: current_support(c, ledger.assumptions, metric=m, ledger=ledger)
                                     for m in c.get("metric_support", {})},
                  "model_review": c.get("model_review"), "support_complete": c.get("support_complete", True),
                  "input_values": c.get("input_values", {}), "assumption_versions": c["assumption_versions"],
                  "interpretation": "Computed consequences, not observations. Current computed metrics may have zero evidence support. metric_support qualifies each output; "
                  "current_support describes all selected assumptions, including inputs unused by an individual output."}
            for key, c in ledger.calculation_options().items()}
