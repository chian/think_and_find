"""Source-event grouping and target-relative acquisition credit."""

from collections import defaultdict
from collections.abc import Mapping

from ...task_executors._http import ProviderError
from .progress import ADDITIONAL_SUPPORT, evidence_key, parameter_requirements, parameter_support
from .model_defaults import retained_defaults
from .records import Credit, ResearchResult, identity


RELEVANCE_LEVELS = (0, 0.25, 0.5, 0.75, 1)
CONTEXT_VARIABLES = ("disease_context", "respiratory_activity", "population", "physical_setting")
RELATIONS = ("match", "partial", "mismatch", "unknown", "not_applicable")
TARGET_RELEVANCE_CREDIT_MEANING = (
    "Evidence-filled inputs earn up to one point each within a related source-event type, plus 0.1 "
    "for distinct corroboration of the same source-event subject. Each field uses its strongest current "
    "event-to-target relevance and comparison coverage; only the strongest type contributes for that target. "
    "Original observations retain their event identity. Eligible source values automatically "
    "become conditional target alternatives; applicability grades select among them, while "
    "scientific support is reviewed separately."
)


def source_profiles(result, requirements):
    profiles = defaultdict(lambda: {"contexts": set(), "fields": set(), "kinds": set(), "facts": []})
    for fact in result.facts:
        if fact.kind not in {"source_observation", "source_model_parameter"} or fact.field not in requirements:
            continue
        row = profiles[(fact.source, fact.subject)]
        row["contexts"].add(fact.context)
        row["fields"].add(fact.field)
        row["kinds"].add(fact.kind)
        row["facts"].append({"field": fact.field, "value": fact.value, "unit": fact.unit,
                             "kind": fact.kind, "quote": fact.quote,
                             "source_version": fact.source_version})
    for evidence in result.evidence:
        if (evidence.get("kind") != "source_evidence"
                or evidence.get("evidence_role") != "observation"
                or not requirements.intersection(evidence.get("fields", ()))):
            continue
        key = (evidence.get("source"), evidence.get("subject"))
        row = profiles[key]
        row["contexts"].add(evidence.get("context", ""))
        row["fields"].update(requirements.intersection(evidence.get("fields", ())))
        row["kinds"].add("source_observation_context")
    return {key: {name: sorted(values, key=lambda item: str(item)) for name, values in row.items()}
            for key, row in profiles.items()}


def comparison_variables(profile, target):
    """Compare context plus numeric fields known on both sides of this application."""
    target_fields = set(target.get("fixed_parameters", {})) | set(target.get("recorded_hypotheses", {}))
    return (*CONTEXT_VARIABLES, *sorted(set(profile["fields"]) & target_fields))


def _plain_value(value):
    """Copy closed contract mappings into JSON-compatible assessment context."""
    if isinstance(value, Mapping):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_value(item) for item in value]
    return value


def target_comparison_context(definition, assumptions):
    """Compare source applicability to scenario inputs, not candidate wording."""
    return {name: _plain_value(definition.get(name, {} if name != "assumption_needs" else []))
            for name in ("description", "fixed_parameters", "fixed_conditions", "assumption_needs", "model_defaults")}


def _unchanged_relationships(source, profiles, targets, parameter, relationships):
    """Reuse an assessment only for the same admitted profile and assigned target."""
    return {(row["source_subject"], row["target_subject"]): row for row in relationships
            if row.get("source") == source and row.get("parameter") == parameter
            and (source, row.get("source_subject")) in profiles
            and row.get("target_subject") in targets
            and row.get("source_profile_version") == identity(profiles[(source, row["source_subject"])])
            and row.get("target_version") == identity(targets[row["target_subject"]])}


def assess_source_types_with_jev(agent, jev_assess, source, profiles, targets,
                                 known_groups, disease=None, *, parameter=None,
                                 previous_relationships=()):
    """Group observed events, then score each source/target pair with a Noul."""
    subjects = sorted(subject for locator, subject in profiles if locator == source)
    if not subjects:
        return [], []
    existing = sorted(known_groups)
    response = agent({
        "operation": "classify_source_event_types",
        "source": source,
        "source_subjects": {subject: profiles[(source, subject)] for subject in subjects},
        "disease_of_interest": disease,
        "known_types": {key: {name: group[name] for name in ("description", "basis")}
                        for key, group in known_groups.items()},
        "instruction": (
            "Group source subjects by the actual exposure event or modeled source context. "
            "Keep distinct events separate; reuse a known type only for comparable disease, "
            "activity, population, and physical setting. This classifies source identity only. "
            "Target-scenario applicability is measured separately by Jev."
        ),
        "response_contract": {"type": "object", "additionalProperties": False,
            "required": ["groups"], "properties": {"groups": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["type_id", "description", "subjects", "basis"],
                "properties": {"type_id": {"type": "string", "enum": [*existing, "new"]},
                    "description": {"type": "string", "minLength": 1},
                    "subjects": {"type": "array", "minItems": 1, "uniqueItems": True,
                                 "items": {"type": "string", "enum": subjects}},
                    "basis": {"type": "string", "minLength": 1}}}}}},
    })
    groups = response.get("groups")
    if not isinstance(groups, list) or any(not isinstance(group, dict)
            or not isinstance(group.get("subjects"), list) for group in groups):
        raise ValueError("Source-event classification requires groups with subjects")
    if sorted(subject for group in groups for subject in group["subjects"]) != subjects:
        raise ValueError("Source-event groups must partition source subjects exactly")
    reused = [group.get("type_id") for group in groups if group.get("type_id") != "new"]
    if (len(set(reused)) != len(reused)
            or any(group.get("type_id") not in {*existing, "new"}
                   or not isinstance(group.get("description"), str) or not group["description"].strip()
                   or not isinstance(group.get("basis"), str) or not group["basis"].strip()
                   for group in groups)):
        raise ValueError("Source-event classification has an invalid type or basis")

    relationships = assess_jev_relationships(jev_assess, source, profiles, targets,
        disease=disease, parameter=parameter, previous_relationships=previous_relationships)
    scores = {(row["source_subject"], row["target_subject"]): row["score"] for row in relationships}
    return [{**group, "target_relevance": {target: max(scores[(subject, target)]
            for subject in group["subjects"]) for target in targets}} for group in groups], relationships


def assess_jev_relationships(jev_assess, source, profiles, targets, *, disease=None,
                             parameter=None, previous_relationships=()):
    """Grade source/target applicability without requiring event reclassification."""
    subjects = sorted(subject for locator, subject in profiles if locator == source)
    unchanged = _unchanged_relationships(source, profiles, targets, parameter, previous_relationships)
    relationships = []
    for subject in subjects:
        profile = profiles[(source, subject)]
        changed_targets = {name: target for name, target in targets.items() if (subject, name) not in unchanged}
        probabilities = (jev_assess({"source": source, "source_subject": subject,
            "source_profile": profile, "target_scenarios": changed_targets,
            "disease_of_interest": disease, "parameter": parameter}) if changed_targets else {})
        if not isinstance(probabilities, Mapping) or set(probabilities) != set(changed_targets):
            raise ValueError("Jev source-target assessment must cover selected targets")
        for target_subject, target in targets.items():
            if (subject, target_subject) in unchanged:
                relation = unchanged[(subject, target_subject)]
                relationships.append(relation)
                continue
            probability = probabilities[target_subject]
            if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                    or not 0 <= probability <= 1):
                raise ValueError("Jev source-target probability must be between zero and one")
            score = float(probability)
            relationships.append({"source": source, "source_subject": subject,
                "target_subject": target_subject, "parameter": parameter,
                "target_version": identity(target),
                "source_profile_version": identity(profile),
                "score": score, "jev_probability": score,
                "assessment_method": "jev_noul",
                "basis": "Jev Noul probability that this source event is applicable to this selected target for the assigned parameter.",
                "source_role": sorted(profile["kinds"]),
                "source_contexts": list(profile["contexts"]),
                "target_description": target.get("description", ""),
                "target_fixed_conditions": target.get("fixed_conditions", {}),
                "comparisons": [{"field": "source_target_applicability",
                    "relation": "jev_noul", "basis": "Source and target contexts were submitted together as one Noul question.",
                    "source_values": profile["facts"],
                    "target_value": {"fixed_parameters": target.get("fixed_parameters", {}),
                        "fixed_conditions": target.get("fixed_conditions", {}),
                        "recorded_hypotheses": target.get("recorded_hypotheses", {})}}],
                "known": 1, "considered": 1})
    return relationships


def assess_numeric_target_relationships(ledger, jev_assess, *, disease=None, candidates=None):
    """Grade every normalized numeric candidate with the same per-value mechanism.

    A source's local research assignment does not restrict this projection.
    Current relationship versions are reused on subsequent child returns.
    Without an assessor, values remain eligible with unknown relevance.
    """
    if jev_assess is None:
        return ResearchResult()
    from .scenario_sharing import independent_groups
    from .target_values import target_value_candidates, source_value_profiles, relevance_profile_version

    result = ledger.result()
    candidates = target_value_candidates(ledger) if candidates is None else candidates
    subjects_by_field = {}
    groups = independent_groups(ledger)
    subjects_by_group = {group["group_id"]: set(group["subjects"]) for group in groups}
    for group in groups:
        subjects_by_field.setdefault(group["parameter"], set()).update(group["subjects"])
    changes, diagnostics = [], []
    for field in sorted(subjects_by_field):
        eligible = [candidate for candidate in candidates if candidate["field"] == field
                    and candidate["target_groups"]]
        source_keys = {(candidate["source"], candidate["subject"]) for candidate in eligible}
        if not source_keys:
            continue
        targets = {subject: target_comparison_context(ledger.scenarios[subject], result.assumptions)
                   for subject in sorted(subjects_by_field[field])}
        profiles = {key: profile for key, profile in source_value_profiles(
                        ledger, parameter_requirements(field), candidates=candidates).items()
                    if key in source_keys}
        for source, source_subject in sorted(profiles):
            profile = profiles[(source, source_subject)]
            values = [candidate for candidate in eligible
                      if (candidate["source"], candidate["subject"]) == (source, source_subject)]
            rows = {}
            for target_subject, target in targets.items():
                target_values = [candidate for candidate in values if any(target_subject in subjects_by_group[group]
                                 for group in candidate["target_groups"])]
                if not target_values:
                    continue
                previous = ledger.source_target_relationships.get((source, source_subject, target_subject, field), {})
                current = previous.get("target_version") == identity(target)
                valid_keys = {candidate["key"] for candidate in target_values}
                versions = {candidate["key"]: candidate["version"] for candidate in target_values}
                scores = {key: score for key, score in previous.get("source_value_scores", {}).items()
                          if current and key in valid_keys
                          and previous.get("source_value_versions", {}).get(key) == versions[key]}
                rows[target_subject] = {**previous, "source_value_scores": scores,
                    "source_value_versions": versions,
                    "source_value_contexts": {candidate["key"]: "; ".join(candidate["profile"]["contexts"])
                                              for candidate in target_values}}
            for candidate in values:
                missing = {subject: target for subject, target in targets.items()
                           if subject in rows and candidate["key"] in rows[subject]["source_value_versions"]
                           and candidate["key"] not in rows[subject]["source_value_scores"]}
                if not missing:
                    continue
                value_profile = candidate["profile"]
                try:
                    reviewed = assess_jev_relationships(jev_assess, source,
                        {(source, source_subject): value_profile}, missing, disease=disease, parameter=field)
                except (ProviderError, ValueError, TypeError, KeyError, OSError) as exc:
                    why = f"Source-value relevance remains unknown: {type(exc).__name__}: {exc}"
                    diagnostics.append({"stage": "source_value_relevance", "category": "assessment_incomplete", "status": "failed",
                        "source": source, "source_subject": source_subject, "field": field,
                        "value": candidate["value"], "unit": candidate["unit"],
                        "target_subjects": list(missing), "why": why})
                    continue
                for relation in reviewed:
                    subject = relation["target_subject"]
                    row = rows[subject]
                    scores = row["source_value_scores"]
                    scores[candidate["key"]] = relation["score"]
                    rows[subject] = {**relation, **row, "source_value_scores": scores}
            for subject, row in rows.items():
                if not row["source_value_scores"]:
                    continue
                row.update(source=source, source_subject=source_subject, target_subject=subject, parameter=field,
                    target_version=identity(targets[subject]), source_profile_version=relevance_profile_version(profile),
                    source_role=sorted(profile["kinds"]), source_contexts=list(profile["contexts"]),
                    score=max(row["source_value_scores"].values()),
                    jev_probability=max(row["source_value_scores"].values()),
                    assessment_method="jev_noul_per_value", known=1, considered=1,
                    basis="Each recorded value and its provenance are graded for this field against the target scenario; the aggregate is only a summary.")
                key = (source, source_subject, subject, field)
                if ledger.source_target_relationships.get(key) != row:
                    changes.append(row)
    return ResearchResult(source_target_relationships=tuple(changes),
                          diagnostics=tuple(diagnostics))


def assess_source_types(agent, source, profiles, targets, known_groups, disease=None, *, parameter=None,
                        previous_relationships=()):
    """Classify one source's observed subjects against reusable event types."""
    subjects = sorted(subject for locator, subject in profiles if locator == source)
    if not subjects:
        return []
    existing = sorted(known_groups)
    unchanged = _unchanged_relationships(source, profiles, targets, parameter, previous_relationships)
    expected = {(subject, target) for subject in subjects for target in targets} - unchanged.keys()
    variables = {(subject, target): comparison_variables(profiles[(source, subject)], targets[target])
                 for subject, target in expected}
    response = agent({
        "operation": "assess_source_event_types",
        "source": source,
        "source_subjects": {subject: profiles[(source, subject)] for subject in subjects},
        "target_scenarios": targets,
        "disease_of_interest": disease,
        "assigned_parameter": parameter,
        "comparison_variables": {subject: {target: list(variables[(subject, target)]) for target in targets
                                           if (subject, target) in expected}
                                 for subject in subjects},
        "unchanged_relationships": list(unchanged.values()),
        "known_types": {key: {name: group[name] for name in ("description", "target_relevance", "basis")}
                        for key, group in known_groups.items()},
        "instruction": (
            "Group the supplied source subjects by the kind of exposure event or modeled source context, "
            "reusing a known type only when its disease context, activity, population and physical setting are comparable. "
            "The same actual event may have several source subject labels; retain their separate records. "
            "Do not merge distinct events into one observation or treat a model sensitivity case as a measured outbreak. "
            "For each type, assess how relevant its context is to each target scenario. Also assess "
            "each source-subject/target pair listed in comparison_variables for the assigned parameter. "
            "Unchanged relationships already have an assessment; retain them without reassessment. For each "
            "pair, compare exactly its declared comparison_variables. Use match, partial, mismatch, "
            "unknown or not_applicable and explain each comparison from the supplied source and "
            "target context. Unknown is not a mismatch. Do not invent values or treat target inputs "
            "as source observations. Source facts, target givens, and recorded target hypotheses "
            "with review status are supplied separately. "
            "Relevance is a multiplier for learning from these sources, not a claim of scientific confidence. "
            "Use 0 for unrelated contexts, 0.25 for a remote analogy, "
            "0.5 for a related mechanism with major transfer differences, 0.75 for a close but qualified match, "
            "and 1 only for a directly matching context. A related type remains available even at low relevance; "
            "eligible source values automatically become conditional target alternatives without a relevance "
            "threshold or a separate transfer action. Their scientific support is reviewed separately. "
            "Return every supplied subject exactly once in a group and every pair listed in "
            "comparison_variables exactly once in relationships, excluding unchanged_relationships."
        ),
        "response_contract": {"type": "object", "additionalProperties": False,
            "required": ["groups", "relationships"], "properties": {"groups": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["type_id", "description", "subjects", "target_relevance", "basis"],
                "properties": {
                    "type_id": {"type": "string", "enum": [*existing, "new"]},
                    "description": {"type": "string", "minLength": 1},
                    "subjects": {"type": "array", "minItems": 1, "uniqueItems": True,
                                 "items": {"type": "string", "enum": subjects}},
                    "target_relevance": {"type": "object", "additionalProperties": False,
                        "required": list(targets), "properties": {
                            target: {"type": "number", "enum": list(RELEVANCE_LEVELS)} for target in targets}},
                    "basis": {"type": "string", "minLength": 1},
                }}}, "relationships": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["source_subject", "target_subject", "score", "basis", "comparisons"],
                    "properties": {
                        "source_subject": {"type": "string", "enum": subjects},
                        "target_subject": {"type": "string", "enum": list(targets)},
                        "score": {"type": "number", "enum": list(RELEVANCE_LEVELS)},
                        "basis": {"type": "string", "minLength": 1},
                        "comparisons": {"type": "array", "items": {"type": "object",
                            "additionalProperties": False, "required": ["field", "relation", "basis"],
                            "properties": {"field": {"type": "string"},
                                           "relation": {"type": "string", "enum": list(RELATIONS)},
                                           "basis": {"type": "string", "minLength": 1}}}},
                    }} }},
            },
    })
    groups = response.get("groups")
    if not isinstance(groups, list):
        raise ValueError("Source-event assessment requires groups")
    if any(not isinstance(group, dict) or not isinstance(group.get("subjects"), list)
           for group in groups):
        raise ValueError("Source-event groups must contain subject lists")
    assigned = [subject for group in groups for subject in group["subjects"]]
    if sorted(assigned) != subjects:
        raise ValueError("Source-event groups must partition the source subjects exactly")
    reused = [group["type_id"] for group in groups if group.get("type_id") != "new"]
    if len(set(reused)) != len(reused):
        raise ValueError("A known source-event type may appear only once per assessment")
    for group in groups:
        if group.get("type_id") not in {*existing, "new"}:
            raise ValueError("Source-event group references an unknown type")
        if not isinstance(group.get("description"), str) or not group["description"].strip():
            raise ValueError("Source-event type requires a description")
        if not isinstance(group.get("basis"), str) or not group["basis"].strip():
            raise ValueError("Source-event relevance requires a basis")
        relevance = group.get("target_relevance")
        if not isinstance(relevance, dict) or set(relevance) != set(targets):
            raise ValueError("Source-event relevance must cover every selected target")
        if any(type(value) not in (int, float) or value not in RELEVANCE_LEVELS
               for value in relevance.values()):
            raise ValueError("Source-event relevance must use a declared multiplier")
    relationships = response.get("relationships")
    if not isinstance(relationships, list) or len(relationships) != len(expected):
        raise ValueError("Source-event relationships must cover every source/target pair")
    seen = set()
    admitted = list(unchanged.values())
    for row in relationships:
        if not isinstance(row, dict):
            raise ValueError("Source-event relationship must be an object")
        key = (row.get("source_subject"), row.get("target_subject"))
        if key not in expected or key in seen:
            raise ValueError("Source-event relationships must be unique declared pairs")
        seen.add(key)
        if type(row.get("score")) not in (int, float) or row["score"] not in RELEVANCE_LEVELS:
            raise ValueError("Source-event relationship score must use a declared level")
        if not isinstance(row.get("basis"), str) or not row["basis"].strip():
            raise ValueError("Source-event relationship needs a basis")
        comparisons = row.get("comparisons")
        if not isinstance(comparisons, list) or {item.get("field") for item in comparisons
                if isinstance(item, dict)} != set(variables[key]) or len(comparisons) != len(variables[key]):
            raise ValueError("Source-event comparisons must cover the declared variables exactly")
        enriched = []
        profile = profiles[(source, key[0])]
        target = targets[key[1]]
        for item in comparisons:
            if (not isinstance(item, dict) or item.get("relation") not in RELATIONS
                    or not isinstance(item.get("basis"), str) or not item["basis"].strip()):
                raise ValueError("Source-event comparison needs a relation and basis")
            field = item["field"]
            source_values = [{k: fact[k] for k in ("value", "unit", "kind", "quote", "source_version")}
                             for fact in profile["facts"] if fact["field"] == field]
            target_value = (target.get("fixed_parameters", {}).get(field)
                            or target.get("recorded_hypotheses", {}).get(field))
            enriched.append({**item, "source_values": source_values,
                             "target_value": target_value})
        considered = sum(item["relation"] != "not_applicable" for item in enriched)
        known = sum(item["relation"] in {"match", "partial", "mismatch"} for item in enriched)
        admitted.append({"source": source, "source_subject": key[0], "target_subject": key[1],
                         "parameter": parameter, "target_version": identity(target),
                         "source_profile_version": identity(profile),
                         "score": row["score"], "basis": row["basis"],
                         "source_role": sorted(profile["kinds"]),
                         "source_contexts": list(profile["contexts"]),
                         "target_description": target.get("description", ""),
                         "target_fixed_conditions": target.get("fixed_conditions", {}),
                         "comparisons": enriched,
                         "known": known, "considered": considered})
    scores = {(row["source_subject"], row["target_subject"]): row["score"] for row in admitted}
    return [{**group, "target_relevance": {target: max(scores[(subject, target)]
            for subject in group["subjects"]) for target in targets}} for group in groups], admitted


def target_relevance_credit(result, requirements, groups, targets, *, parameter=None):
    """Credit the strongest related event type per target, never a field pool across types."""
    assessments = []
    invalid = {evidence_key(evidence) for assessment in result.assumptions
               for evidence in assessment.get("invalidated_evidence", ())}
    relationships = {(row["source"], row["source_subject"], row["target_subject"], row["parameter"]): row
                     for row in result.source_target_relationships}
    profile_fields = parameter_requirements(parameter) if parameter is not None else requirements
    profiles = source_profiles(result, profile_fields) if relationships else {}
    for target in targets:
        target_requirements = set(requirements) - retained_defaults(targets[target]).keys()
        choices = []
        for type_id, group in groups.items():
            if not group["members"]:
                continue
            members = sorted({tuple(member) for member in group["members"]})
            assessed_members = [relationships[(source, subject, target, parameter)] for source, subject in members
                                if (source, subject, target, parameter) in relationships]
            if assessed_members:
                target_context = target_comparison_context({**targets[target], "subject": target}, result.assumptions)
                field_points = {}
                multipliers = []
                for source, subject in members:
                    relation = relationships.get((source, subject, target, parameter))
                    profile = profiles.get((source, subject))
                    if (relation is None or profile is None
                            or relation.get("target_version") != identity(target_context)
                            or relation.get("source_profile_version") != identity(profile)):
                        continue
                    multiplier = relation["score"] * (relation["known"] / relation["considered"]
                                                      if relation["considered"] else 0)
                    multipliers.append(multiplier)
                    facts = tuple(fact for fact in result.facts if (fact.source, fact.subject) == (source, subject)
                                  and evidence_key(fact.semantic_record()) not in invalid)
                    for row in parameter_support(ResearchResult(facts=facts), target_requirements):
                        field_points[row["field"]] = max(field_points.get(row["field"], 0),
                                                         multiplier * row["points"])
                rows = [{"field": field, "points": round(value, 10)} for field, value in sorted(field_points.items())]
                multiplier = max(multipliers, default=0)
                points = round(sum(field_points.values()), 10)
            else:
                facts = tuple(fact for fact in result.facts if (fact.source, fact.subject) in members
                              and evidence_key(fact.semantic_record()) not in invalid)
                rows = parameter_support(ResearchResult(facts=facts), target_requirements)
                multiplier = group["target_relevance"].get(target, 0)
                points = round(multiplier * sum(row["points"] for row in rows), 10)
                rows = [{"field": row["field"], "points": round(multiplier * row["points"], 10)}
                        for row in rows]
            choices.append({"target": target, "type_id": type_id,
                            "description": group["description"], "relevance": multiplier,
                            "points": points,
                            "fields": rows,
                            "member_count": len(group["members"]),
                            "basis": group["basis"]})
        if choices:
            assessments.append(max(choices, key=lambda row: (row["points"], row["relevance"])))
    credit = Credit(evaluated=result.evaluated, quality_resolution=ADDITIONAL_SUPPORT / 4,
                    qualities=tuple(("parameter_evidence", row["target"], row["points"])
                                    for row in assessments))
    return assessments, credit
