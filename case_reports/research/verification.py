"""Read-only calibration provenance used by the existing credit projection."""

from ...task_executors.scenario_metrics.evaluate import exposure_outputs_agree, output_meets_contract
from .records import cited_assessment_evidence, current_computation, current_premises, current_support, identity, Ledger


CALIBRATION_PROGRESS = "outbreak_calibration"
CALIBRATION_EVIDENCE_FIELDS = ("attack_probability", "secondary_case_count", "susceptible_count", "exposure_time_s")
OUTBREAK_OPERATORS = {"cq_from_steady_outbreak", "cq_from_transient_outbreak"}


def provenance_index(result):
    records = {f.key: f.as_record() for f in result.facts}
    records.update((identity(row), row) for row in result.evidence)
    for calculation in result.calculations:
        records.update((identity(row), row) for row in calculation.get("input_records", ()))
    for assessment in result.assumptions:
        evidence = cited_assessment_evidence(assessment)
        records.update((identity(row), row) for row in evidence.values())
        # A transferred calculation keeps its numerical ancestry even before a
        # support review cites evidence. This link does not establish support.
        records[assessment["key"]] = {**assessment, "kind": "model_assumption",
            "parents": list(dict.fromkeys([
                *assessment.get("derivation", {}).get("input_fact_keys", ()),
                *assessment.get("derivation", {}).get("source_evidence_ids", ()),
                *(identity(row) for row in evidence.values()),
            ]))}
    return records


def observation_fields(record):
    if record.get("kind") == "source_observation":
        return {record["field"]}
    if record.get("kind") == "source_evidence" and record.get("evidence_role") == "observation":
        return set(record.get("observed_fields", ()))
    return set()


def ancestors(keys, records):
    seen, pending = set(), list(keys)
    while pending:
        key = pending.pop()
        if key not in seen:
            seen.add(key)
            pending.extend(records.get(key, {}).get("parents", ()))
    return seen


def calibration_progress(result):
    records = provenance_index(result)
    assessments = {a["key"]: a for a in result.assumptions}
    ledger = Ledger(result)
    calibrations = []
    observed = {}
    for fact in result.facts:
        if fact.kind == "source_observation":
            observed.setdefault(fact.subject, set()).add(fact.field)
    for evidence in result.evidence:
        fields = observation_fields(evidence)
        if fields:
            observed.setdefault(evidence["subject"], set()).update(fields)
    for calculation in result.calculations:
        if calculation.get("operator") not in OUTBREAK_OPERATORS:
            continue
        lineage = ancestors(calculation["input_fact_keys"], records)
        observations = [records[k] for k in lineage if k in records
                        and observation_fields(records[k])]
        fields = set().union(*(observation_fields(row) for row in observations))
        inputs = {row["field"]: row for row in calculation.get("input_records", ())}

        def observed_basis(field):
            selected = inputs.get(field)
            if selected is None:
                return set()
            keys = ancestors((identity(selected),), records)
            return set().union(*(observation_fields(records[k]) for k in keys if k in records))

        # Follow the inputs actually used by the inversion, not unrelated facts
        # that happen to accompany them. Derived denominators retain their basis.
        if "attack_probability" in inputs:
            attack_basis = observed_basis("attack_probability")
            has_attack = ("attack_probability" in attack_basis or
                ("secondary_case_count" in attack_basis and bool({"susceptible_count", "occupant_count"}.intersection(attack_basis))))
        else:
            has_attack = ("secondary_case_count" in observed_basis("secondary_case_count")
                and bool({"susceptible_count", "occupant_count"}.intersection(observed_basis("susceptible_count"))))
        has_duration = "exposure_time_s" in observed_basis("exposure_time_s")
        grounded = has_attack
        calibrations.append({"version": calculation["version"], "subject": calculation["subject"],
            "subtask": "calibrate_cq_from_outbreak", "operator": calculation["operator"],
            "output_fact_keys": calculation.get("output_fact_keys", []),
            "observed_fields": sorted(fields), "outbreak_evidence_acquired": grounded,
            "duration_observation_available": has_duration,
            "source_refs": sorted({row["source"] for row in observations}),
            "current": current_premises(calculation, assessments, ledger=ledger),
            "support": current_support(calculation, assessments, ledger=ledger),
            "missing_observations": ([] if has_attack else ["Runtime outbreak outcomes supporting attack probability or secondary/susceptible counts, including observed ranges"])})
    return {"observed_event_fields": {subject: sorted(fields) for subject, fields in observed.items()
                                     if fields.intersection({"attack_probability", "secondary_case_count", "exposure_time_s"})},
            "calibrations": calibrations,
            "outbreak_calibration_performed": any(row["outbreak_evidence_acquired"] and row["current"]
                                                  for row in calibrations),
            "meaning": "A guessed or published Cq is not a workflow-performed outbreak inversion. "
                       "Outbreak outcomes may be scalar observations or independently admitted observational "
                       "ranges supporting reviewed hypotheses. Their runtime source provenance is required. "
                       "Duration and other calibration/transfer inputs may be inferred or assumed; their "
                       "independent reasonableness judgments determine support, not permission to calculate."}


def target_calibration_progress(result, metrics, *, calibration=None):
    """Project existing calibration lineage for the calibration credit component."""
    progress = calibration if calibration is not None else calibration_progress(result)
    calibrated = {key: row for row in progress["calibrations"]
                  if row["outbreak_evidence_acquired"] and row["current"] for key in row["output_fact_keys"]}
    records = provenance_index(result)
    assessments = {a["key"]: a for a in result.assumptions}
    ledger = Ledger(result)
    targets = {}
    for definition in result.scenarios:
        if definition.get("role", "target") != "target":
            continue
        subject = definition["subject"]
        contracts = definition.get("output_contracts", {})
        alternatives = []
        for estimate in result.estimates:
            if estimate["subject"] != subject:
                continue
            computed = [name for name in metrics if current_computation(estimate, assessments, name, ledger=ledger)
                        and output_meets_contract(estimate["metrics"][name], contracts.get(name))]
            linked = {name: sorted(ancestors(estimate.get("metric_support", {}).get(name, {}).get("input_fact_keys", ()),
                                                   records).intersection(calibrated))
                      for name in computed}
            support = {name: min(current_support(estimate, assessments, metric=name, ledger=ledger),
                                min((calibrated[key]["support"] for key in keys), default=0))
                       for name, keys in linked.items()}
            coherent = exposure_outputs_agree(estimate["metrics"])
            complete = coherent and all(linked.get(name) for name in metrics)
            alternatives.append({"parameterization": estimate["label"], "coherent": coherent,
                "computed_quantities": computed, "calibrated_inputs_by_quantity": linked,
                "calibrated_support_by_quantity": support,
                "missing_calibration_linkage": [name for name in metrics if not linked.get(name)],
                "coherent_calibrated_solution": complete,
                "supported_coherent_calibrated_solution": complete and all(support.get(name, 0) > 0 for name in metrics)})
        targets[subject] = {"alternatives": alternatives,
            "coherent_calibrated_solution_available": any(a["coherent_calibrated_solution"] for a in alternatives),
            "supported_coherent_calibrated_solution_available": any(a["supported_coherent_calibrated_solution"] for a in alternatives)}
    return targets
