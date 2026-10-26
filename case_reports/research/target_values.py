"""One relevance and target-selection pipeline for recorded numeric candidates."""

from .progress import evidence_key, parameter_requirements
from .records import Fact, Ledger, ResearchResult, assumption_claim, current_premises, identity
from .relevance import source_profiles, target_comparison_context
from .scenario_sharing import independent_groups
from ...task_executors.scenario_metrics.evaluate import UNITS, validate_parameter


def invalidated_source_records(result):
    """Exact recorded-input errors, including corrections made before execution."""
    assumptions = result.assumptions.values() if isinstance(result.assumptions, dict) else result.assumptions
    records = [row for assumption in assumptions for row in assumption.get("invalidated_evidence", ())]
    records.extend(row for diagnostic in result.diagnostics
                   if diagnostic.get("stage") == "input_record_correction"
                   for row in diagnostic.get("invalidated_evidence", ()))
    return tuple({evidence_key(row): row for row in records}.values())


def _source_value_lineage(ledger, fact, invalid, visiting=frozenset()):
    """Return current calculation/source provenance, independently of support grades."""
    if fact.key in visiting or evidence_key(fact.semantic_record()) in invalid:
        return None
    if fact.kind in {"source_observation", "source_model_parameter"}:
        return {"sources": [fact.semantic_record()], "calculations": [], "assumptions": []}
    if fact.kind == "model_assumption" and fact.parents:
        from .records import numeric_provenance

        assumption = ledger.assumptions.get(fact.parents[0])
        if (assumption is None or assumption.get("source_candidate_unavailable")
                or any(assumption.get(name) != getattr(fact, name) for name in ("field", "value", "unit"))):
            return None
        context = (assumption["statement"] if assumption["subject"] == fact.subject else
                   f"Shared from {assumption['subject']}: {assumption['statement']}")
        if fact.context != context or assumption["key"] in visiting:
            return None
        derivation = assumption.get("derivation", {})
        if not current_premises({
                "assumption_versions": derivation.get("input_assumption_versions", {}),
                "assumption_claims": derivation.get("input_assumption_claims", {})},
                ledger.assumptions, ledger=ledger):
            return None
        options = numeric_provenance(ledger)
        selected = [options.get(key) for key in derivation.get("input_fact_keys", ())]
        if any(item is None for item in selected):
            return None
        ancestry = [_source_value_lineage(ledger, item, invalid,
                    visiting | {fact.key, assumption["key"]}) for item in selected]
        if any(item is None for item in ancestry):
            return None
        evidence = {identity(row): row for row in ledger.evidence_options().values()}
        evidence.update((identity(row), row) for row in assumption.get("evidence", {}).values())
        cited = [evidence.get(key) for key in derivation.get("source_evidence_ids", ())]
        if any(row is None or evidence_key(row) in invalid for row in cited):
            return None
        premise = {name: assumption[name] for name in
                   ("subject", "field", "value", "unit", "statement", "basis")}
        premise["added_assumptions"] = list(derivation.get("added_assumptions", ()))
        premise["review"] = assumption.get("review", {})
        return {"sources": list({identity(row): row for row in
                    (*cited, *(row for item in ancestry for row in item["sources"]))}.values()),
                "calculations": [row for item in ancestry for row in item["calculations"]],
                "assumptions": [*(row for item in ancestry for row in item["assumptions"]), premise]}
    if fact.kind != "derived":
        return {"sources": [], "calculations": [], "assumptions": []}
    producers = [row for row in ledger.calculations.values()
                 if fact.key in row.get("output_fact_keys", ())]
    for record in reversed(producers):
        output = record.get("output", {}).get(fact.field,
                     record.get("derived_parameters", {}).get(fact.field, {}))
        if (output.get("value") != fact.value
                or output.get("unit", UNITS.get(fact.field)) != fact.unit):
            continue
        dependency = record.get("derived_support", {}).get(fact.field,
                         record.get("metric_support", {}).get(fact.field, record))
        if not current_premises(dependency, ledger.assumptions, ledger=ledger):
            continue
        inputs = {identity(row): Fact(**{**row, "parents": tuple(row.get("parents", ()))})
                  for row in record.get("input_records", ())}
        inputs.update(ledger.facts)
        selected = [inputs.get(key) for key in dependency.get("input_fact_keys", ())]
        if not selected or any(item is None for item in selected):
            continue
        lineage = [_source_value_lineage(ledger, item, invalid, visiting | {fact.key})
                   for item in selected]
        if any(item is None for item in lineage):
            continue
        assumptions = list({identity(row): row for item in lineage for row in item["assumptions"]}.values())
        return {"sources": list({identity(source): source for item in lineage
                                  for source in item["sources"]}.values()),
                "assumptions": assumptions,
                "calculations": [*list({identity(calculation): calculation for item in lineage
                                        for calculation in item["calculations"]}.values()),
                    {"calculation_version": record["version"],
                     "tool": record["tool"], "subject": record["subject"],
                     "operator": record.get("operator"), "basis": record["basis"],
                     "conditions": record.get("conditions", {}),
                     "input_assumptions": assumptions,
                     "inputs": [item.semantic_record() for item in selected],
                     "output": {"field": fact.field, "value": fact.value, "unit": fact.unit},
                     "model_review": record.get("model_review", {}),
                     "support_complete": record.get("support_complete", False)}]}
    return None


def eligible_source_values(ledger):
    """Native values and current source calculations can fill existing target needs."""
    if not isinstance(ledger, Ledger):
        ledger = Ledger(ledger)
    invalid = {evidence_key(row) for row in invalidated_source_records(ledger)}
    eligible = []
    for fact in ledger.facts.values():
        if fact.kind not in {"source_observation", "source_model_parameter", "derived"}:
            continue
        if fact.kind == "derived" and ledger.scenarios.get(fact.subject, {}).get("role", "target") == "target":
            continue
        if fact.field not in UNITS or fact.unit != UNITS[fact.field]:
            continue
        try:
            validate_parameter(fact.field, fact.value, fact.unit)
        except (ValueError, TypeError):
            continue
        if _source_value_lineage(ledger, fact, invalid) is not None:
            eligible.append(fact)
    return tuple(eligible)


def source_value_profile(ledger, fact):
    """Exact value context submitted for applicability, including conditional lineage."""
    profile = {"contexts": [fact.context], "fields": [fact.field], "kinds": [fact.kind],
        "facts": [{"field": fact.field, "value": fact.value, "unit": fact.unit,
                   "kind": fact.kind, "quote": fact.quote, "source_version": fact.source_version}]}
    if fact.kind == "derived":
        invalid = {evidence_key(row) for row in invalidated_source_records(ledger)}
        profile["derivation"] = _source_value_lineage(ledger, fact, invalid)
        profile["source_scenario"] = ledger.scenarios.get(fact.subject, {})
    return profile


def source_value_version(ledger, fact):
    return (relevance_profile_version(source_value_profile(ledger, fact)) if fact.kind == "derived"
            else identity(fact.semantic_record()))


def relevance_profile_version(profile):
    """Review grades qualify support; they are not new applicability premises."""
    def premises(value):
        if isinstance(value, dict):
            return {key: premises(item) for key, item in value.items()
                    if key not in {"review", "model_review", "reasonableness", "support_complete"}}
        if isinstance(value, (list, tuple)):
            return [premises(item) for item in value]
        return value
    return identity(premises(profile))


def target_value_candidates(ledger):
    """Normalize provenance, not priority, for every way of obtaining a value.

    Descriptors are transient views of existing records. Native and computed
    source values can serve every declared group needing that quantity. An
    inferred target claim retains its identity and its declared shared scope;
    it is never copied into a competing automatic claim.
    """
    if not isinstance(ledger, Ledger):
        ledger = Ledger(ledger)
    groups = independent_groups(ledger)
    candidates = []
    for fact in eligible_source_values(ledger):
        candidates.append({"key": fact.key, "source": fact.source, "subject": fact.subject,
            "field": fact.field, "value": fact.value, "unit": fact.unit, "kind": fact.kind,
            "profile": source_value_profile(ledger, fact),
            "version": source_value_version(ledger, fact), "fact": fact,
            "assumption": None, "target_groups": tuple(group["group_id"] for group in groups
                if group["parameter"] == fact.field)})
    evidence = {identity(row): row for row in ledger.evidence_options().values()}
    from .records import numeric_provenance
    numeric = numeric_provenance(ledger)
    invalid = {evidence_key(row) for row in invalidated_source_records(ledger)}
    for assumption in ledger.assumptions.values():
        if (assumption.get("entry_type") == "source_candidate"
                or assumption.get("source_candidate_unavailable")
                or assumption.get("role") in {"scenario_constraint", "request_constraint", "model_default"}
                or assumption.get("field") not in UNITS):
            continue
        try:
            validate_parameter(assumption["field"], assumption.get("value"), assumption.get("unit"))
        except (ValueError, TypeError):
            continue
        derivation = assumption.get("derivation", {})
        if not current_premises({
                "assumption_versions": derivation.get("input_assumption_versions", {}),
                "assumption_claims": derivation.get("input_assumption_claims", {})},
                ledger.assumptions, ledger=ledger):
            continue
        inputs = [numeric.get(key) for key in derivation.get("input_fact_keys", ())]
        if any(item is None or _source_value_lineage(ledger, item, invalid) is None for item in inputs):
            continue
        known_evidence = {**evidence, **{identity(row): row for row in assumption.get("evidence", {}).values()}}
        sources = [known_evidence.get(key) for key in derivation.get("source_evidence_ids", ())]
        if any(row is None or evidence_key(row) in invalid for row in sources):
            continue
        subject, field = assumption["subject"], assumption["field"]
        target = ledger.scenarios.get(subject, {}).get("role", "target") == "target"
        eligible = tuple(group["group_id"] for group in groups if group["parameter"] == field
                         and (not target or subject in group["subjects"]))
        profile = {"contexts": [assumption["statement"]], "fields": [field],
            "kinds": ["inferred_parameter"],
            "facts": [{"field": field, "value": assumption["value"], "unit": assumption["unit"],
                       "kind": "inferred_parameter", "quote": "", "source_version": ""}],
            "inference": {"statement": assumption["statement"], "basis": assumption["basis"],
                "added_assumptions": list(derivation.get("added_assumptions", ())),
                "sources": sources, "inputs": [item.semantic_record() for item in inputs],
                "input_lineage": [_source_value_lineage(ledger, item, invalid) for item in inputs],
                "scope": ledger.scenarios.get(subject, {}),
                "meaning": "An inferred conditional value, not an observation. Judge its actual evidence and inference; naming a target does not establish applicability."}}
        candidates.append({"key": assumption["key"], "source": "", "subject": subject,
            "field": field, "value": assumption["value"], "unit": assumption["unit"],
            "kind": "inferred_parameter", "profile": profile, "version": relevance_profile_version(profile),
            "fact": None, "assumption": assumption, "target_groups": eligible})
    return tuple(candidates)


def source_value_profiles(ledger, requirements, *, candidates=None):
    """Canonical profile identity shared by grading and target projection."""
    candidates = target_value_candidates(ledger) if candidates is None else candidates
    profiles = source_profiles(ledger.result(), requirements)
    for candidate in candidates:
        if candidate["kind"] not in {"derived", "inferred_parameter"} or candidate["field"] not in requirements:
            continue
        key = (candidate["source"], candidate["subject"])
        row = profiles.setdefault(key, {"contexts": [], "fields": [], "kinds": [], "facts": [], "derivations": []})
        value = candidate["profile"]
        for name in ("contexts", "fields", "kinds", "facts"):
            row[name].extend(item for item in value[name] if item not in row[name])
        if "derivation" in value:
            row.setdefault("derivations", []).append(value["derivation"])
        if "inference" in value:
            row.setdefault("inferences", []).append(value["inference"])
    return profiles


def target_value_preference(assumption):
    """Rank alternatives by applicability without turning its grade into a gate."""
    grade = assumption.get("source_applicability", {}).get("score")
    # max() retains the first existing alternative on a tie. Scientific review
    # confidence is separate and must not outweigh source-target relevance.
    return (grade if isinstance(grade, (int, float)) and not isinstance(grade, bool) else -1,)


def _applicability(ledger, candidates, group, profiles):
    records = []
    for candidate in candidates:
        profile = profiles.get((candidate["source"], candidate["subject"]))
        for subject in group["subjects"]:
            relation = ledger.source_target_relationships.get(
                (candidate["source"], candidate["subject"], subject, candidate["field"]))
            target = target_comparison_context(ledger.scenarios[subject], ledger.assumptions.values())
            if (relation is None or profile is None
                    or relation.get("target_version") != identity(target)
                    or relation.get("source_profile_version") != relevance_profile_version(profile)):
                continue
            value_scores = relation.get("source_value_scores")
            if value_scores is not None:
                if (candidate["key"] not in value_scores
                        or relation.get("source_value_versions", {}).get(candidate["key"]) != candidate["version"]):
                    continue
                score = value_scores[candidate["key"]]
            else:
                if candidate["kind"] == "inferred_parameter":
                    continue
                native_values = {(row["value"], row["unit"]) for row in profile["facts"]
                                 if row["field"] == candidate["field"]}
                if len(native_values) != 1:
                    # A whole-source score cannot distinguish different native
                    # values/activities recorded under one source subject.
                    continue
                score = relation["score"]
            records.append({**relation, "source_fact_key": candidate["key"],
                            "score": score})
    records = list({identity(row): row for row in records}.values())
    score = max((row["score"] for row in records), default=None)
    return {"status": "assessed" if records else "unknown", "score": score,
            "group_id": group["group_id"], "relationships": records,
            "meaning": "Source-to-target applicability for this parameter; not empirical support or a permission threshold."}


def project_source_target_values(ledger, *, candidates=None):
    """Project every normalized candidate through the same target-value entry.

    Existing inferred target records are updated in place. Source values become
    conditional target records without changing their quantity or unit. Review
    and relevance never determine whether an otherwise current value is found.
    """
    if not isinstance(ledger, Ledger):
        ledger = Ledger(ledger)
    candidates = target_value_candidates(ledger) if candidates is None else candidates
    deltas, current_keys = [], set()
    profile_cache = {}
    for group in independent_groups(ledger):
        field = group["parameter"]
        eligible = [candidate for candidate in candidates if group["group_id"] in candidate["target_groups"]]
        if not eligible:
            continue
        if field not in profile_cache:
            profile_cache[field] = source_value_profiles(ledger, parameter_requirements(field), candidates=candidates)
        profiles = profile_cache[field]
        subject = min(group["subjects"])
        entries = {}
        for candidate in eligible:
            own = candidate["assumption"]
            own_target = own is not None and own["subject"] in group["subjects"]
            key = own["key"] if own_target else identity([
                "source_candidate", group["group_id"], field, candidate["value"], candidate["unit"]])
            entry = entries.setdefault(key, {"original": own if own_target else None, "candidates": []})
            entry["candidates"].append(candidate)
        for key, entry in entries.items():
            values = sorted(entry["candidates"], key=lambda candidate: candidate["key"])
            value, unit = values[0]["value"], values[0]["unit"]
            current_keys.add(key)
            previous = ledger.assumptions.get(key, {})
            sources = {}
            derivations = []
            input_keys = []
            for candidate in values:
                fact = candidate["fact"]
                if candidate["kind"] == "derived":
                    lineage = candidate["profile"]["derivation"]
                    sources.update(("source:" + identity(row), row) for row in lineage["sources"])
                    derivations.extend(lineage["calculations"])
                elif candidate["kind"] == "inferred_parameter":
                    inference = candidate["profile"]["inference"]
                    sources.update(("source:" + identity(row), row) for row in inference["sources"])
                    for lineage in inference["input_lineage"]:
                        sources.update(("source:" + identity(row), row) for row in lineage["sources"])
                        derivations.extend(lineage["calculations"])
                else:
                    sources["source:" + fact.key] = fact.semantic_record()
                if fact is not None:
                    input_keys.append(fact.key)
                elif entry["original"] is None:
                    # A source-scope hypothesis is an explicit computational
                    # parent, even when its scientific review is incomplete.
                    from .records import numeric_provenance
                    input_keys.extend(option.key for option in numeric_provenance(ledger).values()
                        if option.subject == candidate["subject"] and option.parents
                        and option.parents[0] == candidate["key"])
            applicability = _applicability(ledger, values, group, profiles)
            statement = (f"Conditional {field} = {value:g} {unit} for the target parameter group; "
                         "using the recorded source value is an explicit transfer assumption.")
            basis = "Candidate transferred without changing quantity or unit. Source contexts: " + "; ".join(
                dict.fromkeys(context for candidate in values for context in candidate["profile"]["contexts"]))
            review = {}
            assumption = {"key": key, "subject": subject, "field": field, "value": value, "unit": unit,
                "statement": statement, "basis": basis, "role": "inferred_assumption",
                "entry_type": "source_candidate", "source_candidate_group": group["group_id"],
                "source_fact_keys": [candidate["key"] for candidate in values],
                "source_applicability": applicability,
                "derivation": {"source_evidence_ids": [identity(row) for row in sources.values()],
                    "input_fact_keys": list(dict.fromkeys(input_keys)),
                    "input_assumption_versions": {candidate["key"]: candidate["assumption"]["version"]
                        for candidate in values if candidate["assumption"] is not None},
                    "input_assumption_claims": {candidate["key"]: assumption_claim(candidate["assumption"])
                        for candidate in values if candidate["assumption"] is not None},
                    "added_assumptions": ["Transfer the recorded source value to this target parameter group as a conditional alternative."]},
                "reasonableness": 0.0, "evidence": sources, "review": review,
                "learning_evidence": [],
                "invalidated_evidence": list(previous.get("invalidated_evidence", ())) }
            if derivations:
                assumption["source_derivations"] = list({identity(row): row for row in derivations}.values())
                assumption["basis"] += "; Conditional source calculations: " + "; ".join(
                    dict.fromkeys(row["basis"] for row in derivations))
            if entry["original"] is not None:
                # Preserve the exact inferred claim, review and dependency
                # identity. It participates in the same grading and selection
                # without an additional automatic target record.
                assumption = {**entry["original"], "source_applicability": applicability,
                              "target_candidate_group": group["group_id"]}
            # A later scientific review is independent of relevance refreshes.
            if previous.get("review", {}).get("completed", True) and not previous.get("source_candidate_unavailable"):
                for name in ("review", "reasonableness", "learning_evidence", "invalidated_evidence", "evidence"):
                    if name in previous:
                        assumption[name] = previous[name]
            assumption.pop("version", None)
            assumption["version"] = identity(assumption)
            if assumption != previous:
                deltas.append(assumption)
    # Invalidation and scope changes withdraw obsolete entries of either origin.
    for key, previous in ledger.assumptions.items():
        if ((previous.get("entry_type") != "source_candidate" and not previous.get("target_candidate_group"))
                or key in current_keys
                or previous.get("source_candidate_unavailable")):
            continue
        revised = {**previous, "source_candidate_unavailable": True}
        revised.pop("version", None)
        revised["version"] = identity(revised)
        deltas.append(revised)
    return ResearchResult(assumptions=tuple(deltas))
