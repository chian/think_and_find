"""Structured, revisable causal arguments and their numerical progress measure.

The model chooses which argument to build and which gaps to investigate. This
module validates references and dependencies, then scores the argument that was
actually supplied; it does not prescribe an order of research operations.
"""

from __future__ import annotations

import math


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


TEXT = {"type": "string", "minLength": 1}
MAYBE_TEXT = {"type": "string"}


def argument_contract(reference_items):
    """One response shape for complete and provisional reversal arguments."""
    refs = {"type": "array", "uniqueItems": True, "items": reference_items}
    step = _object({
        "id": TEXT,
        "role": {"type": "string", "enum": ["source_report", "inference"]},
        "claim": TEXT,
        "premise_ids": {"type": "array", "uniqueItems": True, "items": TEXT},
        "warrant_refs": refs,
        "reasoning": TEXT,
    })
    frame = _object({
        "question": TEXT,
        "actors_and_categories": TEXT,
        "boundary": TEXT,
        "intervention": TEXT,
        "outcome": TEXT,
        "scope": TEXT,
        "steps": {"type": "array", "items": step},
        "conclusion_step_id": MAYBE_TEXT,
        "pragmatic_use": TEXT,
    })
    translation = _object({
        "source_step_id": MAYBE_TEXT,
        "target_step_id": MAYBE_TEXT,
        "actor_map": TEXT,
        "intervention_map": TEXT,
        "outcome_map": TEXT,
        "relation_map": TEXT,
        "bridge_premise": TEXT,
        "transported_inference": TEXT,
        "target_inference": TEXT,
        "failure_mechanism": TEXT,
        "consequence": TEXT,
        "evidence_refs": refs,
    })
    alternative = _object({
        "fixed_vocabulary": TEXT,
        "added_context_or_dependencies": TEXT,
        "reconciliation": TEXT,
        "status": {"type": "string", "enum": ["reconciles", "challenged", "unresolved"]},
        "evidence_refs": refs,
    })
    shared_system = _object({
        "focal_system": TEXT,
        "organisms_frame_one": {"type": "array", "minItems": 1,
                                "uniqueItems": True, "items": TEXT},
        "organisms_frame_two": {"type": "array", "minItems": 1,
                                "uniqueItems": True, "items": TEXT},
        "overlap_reasoning": TEXT,
        "status": {"type": "string", "enum": ["same", "different", "unresolved"]},
        "evidence_refs": refs,
    })
    return _object({"frame_one": frame, "frame_two": frame,
                    "translation": translation,
                    "fixed_category_alternative": alternative,
                    "shared_system": shared_system})


def normalize_argument(value, available_refs):
    """Check source identities and an acyclic local premise graph."""
    if not isinstance(value, dict) or set(value) != {
            "frame_one", "frame_two", "translation", "fixed_category_alternative",
            "shared_system"}:
        raise ValueError("A causal argument needs two frames, translation, a fixed-category alternative, and a shared system")
    refs = set(available_refs)
    result = {}
    for frame_name in ("frame_one", "frame_two"):
        frame = value[frame_name]
        if not isinstance(frame, dict) or set(frame) != {
                "question", "actors_and_categories", "boundary", "intervention",
                "outcome", "scope", "steps", "conclusion_step_id", "pragmatic_use"}:
            raise ValueError(f"{frame_name} has incomplete frame fields")
        for field in ("question", "actors_and_categories", "boundary", "intervention",
                      "outcome", "scope", "pragmatic_use"):
            if not isinstance(frame[field], str) or not frame[field].strip():
                raise ValueError(f"{frame_name}.{field} needs text")
        steps = frame["steps"]
        if not isinstance(steps, list):
            raise ValueError(f"{frame_name}.steps must be an array")
        seen = set()
        for step in steps:
            if not isinstance(step, dict) or set(step) != {
                    "id", "role", "claim", "premise_ids", "warrant_refs", "reasoning"}:
                raise ValueError(f"{frame_name} has an incomplete inference step")
            identity = step["id"]
            if not isinstance(identity, str) or not identity.strip() or identity in seen:
                raise ValueError(f"{frame_name} step IDs must be unique nonempty text")
            if step["role"] not in ("source_report", "inference"):
                raise ValueError(f"{frame_name} step role is unknown")
            if any(not isinstance(step[field], str) or not step[field].strip()
                   for field in ("claim", "reasoning")):
                raise ValueError(f"{frame_name} steps need a claim and reasoning")
            premises = step["premise_ids"]
            if (not isinstance(premises, list) or any(not isinstance(item, str) for item in premises)
                    or len(premises) != len(set(premises))
                    or any(premise not in seen for premise in premises)):
                raise ValueError(f"{frame_name} step premises must name earlier steps")
            citations = step["warrant_refs"]
            if (not isinstance(citations, list) or any(not isinstance(item, str) for item in citations)
                    or len(citations) != len(set(citations))
                    or any(ref not in refs for ref in citations)):
                raise ValueError(f"{frame_name} step warrants must name available sources")
            if step["role"] == "source_report" and not citations:
                raise ValueError(f"{frame_name} source reports require an exact source reference")
            seen.add(identity)
        conclusion = frame["conclusion_step_id"]
        if not isinstance(conclusion, str) or (conclusion and conclusion not in seen):
            raise ValueError(f"{frame_name} conclusion must name one of its steps or remain open")
        result[frame_name] = frame
    translation = value["translation"]
    if not isinstance(translation, dict) or set(translation) != {
            "source_step_id", "target_step_id", "actor_map", "intervention_map",
            "outcome_map", "relation_map", "bridge_premise", "transported_inference",
            "target_inference", "failure_mechanism", "consequence", "evidence_refs"}:
        raise ValueError("The translation has incomplete fields")
    for field, frame_name in (("source_step_id", "frame_one"),
                              ("target_step_id", "frame_two")):
        identity = translation[field]
        if not isinstance(identity, str) or (identity and identity not in {
                step["id"] for step in result[frame_name]["steps"]}):
            raise ValueError(f"The translation's {field} must name a frame step or remain open")
    for field in ("actor_map", "intervention_map", "outcome_map", "relation_map",
                  "bridge_premise", "transported_inference", "target_inference",
                  "failure_mechanism", "consequence"):
        if not isinstance(translation[field], str) or not translation[field].strip():
            raise ValueError(f"The translation's {field} needs text")
    _check_refs(translation["evidence_refs"], refs, "translation")
    alternative = value["fixed_category_alternative"]
    if not isinstance(alternative, dict) or set(alternative) != {
            "fixed_vocabulary", "added_context_or_dependencies", "reconciliation",
            "status", "evidence_refs"}:
        raise ValueError("The fixed-category alternative has incomplete fields")
    for field in ("fixed_vocabulary", "added_context_or_dependencies", "reconciliation"):
        if not isinstance(alternative[field], str) or not alternative[field].strip():
            raise ValueError(f"The fixed-category alternative's {field} needs text")
    if alternative["status"] not in ("reconciles", "challenged", "unresolved"):
        raise ValueError("The fixed-category alternative has an unknown status")
    _check_refs(alternative["evidence_refs"], refs, "fixed-category alternative")
    system = value["shared_system"]
    if not isinstance(system, dict) or set(system) != {
            "focal_system", "organisms_frame_one", "organisms_frame_two",
            "overlap_reasoning", "status", "evidence_refs"}:
        raise ValueError("The shared-system comparison has incomplete fields")
    for field in ("focal_system", "overlap_reasoning"):
        if not isinstance(system[field], str) or not system[field].strip():
            raise ValueError(f"The shared system's {field} needs text")
    for field in ("organisms_frame_one", "organisms_frame_two"):
        names = system[field]
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not name.strip() for name in names)
                or len(names) != len(set(names))):
            raise ValueError(f"The shared system's {field} needs organism names")
    if system["status"] not in ("same", "different", "unresolved"):
        raise ValueError("The shared system has an unknown status")
    _check_refs(system["evidence_refs"], refs, "shared system")
    return result | {"translation": translation,
                     "fixed_category_alternative": alternative,
                     "shared_system": system}


def _check_refs(values, available, name):
    if (not isinstance(values, list) or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or any(value not in available for value in values)):
        raise ValueError(f"{name} references must name available source claims")


def argument_refs(argument):
    refs = []
    for frame_name in ("frame_one", "frame_two"):
        for step in argument[frame_name]["steps"]:
            refs.extend(step["warrant_refs"])
    refs.extend(argument["translation"]["evidence_refs"])
    refs.extend(argument["fixed_category_alternative"]["evidence_refs"])
    refs.extend(argument["shared_system"]["evidence_refs"])
    return list(dict.fromkeys(refs))


def step_ids(argument):
    return {frame_name: tuple(step["id"] for step in argument[frame_name]["steps"])
            for frame_name in ("frame_one", "frame_two")}


def argument_brief(argument):
    """Retain conclusion dependencies while omitting unrelated steps in a handoff."""
    result = {}
    for frame_name in ("frame_one", "frame_two"):
        frame = argument[frame_name]
        steps = {step["id"]: step for step in frame["steps"]}
        relevant = set()

        def visit(identity):
            if identity in relevant:
                return
            relevant.add(identity)
            for premise in steps[identity]["premise_ids"]:
                visit(premise)

        if frame["conclusion_step_id"]:
            visit(frame["conclusion_step_id"])
        else:
            relevant.update(steps)
        result[frame_name] = {
            key: frame[key] for key in ("question", "actors_and_categories", "boundary",
                                   "intervention", "outcome", "conclusion_step_id",
                                   "pragmatic_use")}
        result[frame_name]["steps"] = [
            ({key: step[key] for key in ("id", "claim", "premise_ids", "warrant_refs")}
             if step["role"] == "inference"
             else {key: step[key] for key in ("id", "role", "claim", "warrant_refs")})
            for step in frame["steps"] if step["id"] in relevant]
        result[frame_name]["omitted_step_ids"] = [
            step["id"] for step in frame["steps"] if step["id"] not in relevant]
    result["translation"] = {
        key: argument["translation"][key] for key in (
            "source_step_id", "target_step_id", "actor_map", "intervention_map",
            "relation_map", "bridge_premise", "transported_inference",
            "target_inference", "failure_mechanism", "consequence")}
    result["fixed_category_alternative"] = {
        key: argument["fixed_category_alternative"][key] for key in (
            "status", "reconciliation")}
    result["shared_system"] = {
        key: argument["shared_system"][key] for key in (
            "focal_system", "organisms_frame_one", "organisms_frame_two",
            "status", "evidence_refs")}
    return result


def same_system(argument):
    """Only an explicitly evidenced, stable organism set can support a reversal."""
    system = argument["shared_system"]
    def names(field):
        return {" ".join(name.casefold().split()) for name in system[field]}
    return (system["status"] == "same" and bool(system["evidence_refs"])
            and names("organisms_frame_one") == names("organisms_frame_two"))


def frame_strength(frame, ratings):
    """A conclusion is no stronger than any step it depends on."""
    conclusion = frame["conclusion_step_id"]
    if not conclusion:
        return 0.0
    steps = {step["id"]: step for step in frame["steps"]}
    required = set()

    def visit(identity):
        if identity in required:
            return
        required.add(identity)
        for premise in steps[identity]["premise_ids"]:
            visit(premise)

    visit(conclusion)
    return min(float(ratings[identity]["score"]) for identity in required)


def argument_progress(argument, fit, step_warrants):
    """Bounded partial progress plus the larger complete-witness component."""
    strengths = {}
    for frame_name, warrant_name in (("frame_one", "first_frame_warrant"),
                                     ("frame_two", "second_frame_warrant")):
        ratings = step_warrants[frame_name]
        if set(ratings) != set(step_ids(argument)[frame_name]):
            raise ValueError(f"Every {frame_name} step needs a warrant measurement")
        if any(not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1
               for row in ratings.values()):
            raise ValueError("Step warrant scores must lie in [0,1]")
        strengths[frame_name] = min(frame_strength(argument[frame_name], ratings),
                                    fit[warrant_name]["score"])
    first, second = strengths["frame_one"], strengths["frame_two"]
    witness = min(fit[name]["score"] for name in (
        "translation_specificity", "inference_reversal", "category_relation_break",
        "fixed_category_challenge", "pragmatic_pluralism"))
    if argument["fixed_category_alternative"]["status"] == "reconciles":
        witness = 0.0
    system_match = same_system(argument)
    system_strength = fit["same_system_warrant"]["score"] if system_match else 0.0
    score = system_strength * (0.1 * max(first, second) + 0.1 * min(first, second)
                               + 0.8 * min(first, second, witness))
    return {"score": score, "frame_one_strength": first,
            "frame_two_strength": second, "witness_strength": witness,
            "system_match": system_match, "system_strength": system_strength}
