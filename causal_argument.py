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
    graph_edit = _object({
        "operation": {"type": "string", "enum": [
            "add_variable", "add_edge", "change_mapping", "constrain_parameter"]},
        "subject": TEXT, "object": TEXT, "expression": TEXT,
    })
    addition = _object({
        "kind": {"type": "string", "enum": ["none", "context_variable",
                    "dependency", "scale_mapping", "parameter_assumption"]},
        "edits": {"type": "array", "items": graph_edit},
        "formal_change": TEXT,
        "preservation_argument": TEXT,
    })
    support = _object({
        "id": TEXT, "question": TEXT, "theory": TEXT, "consequence": TEXT,
        "derivation": TEXT, "addition": addition,
        "distinct_consequence": TEXT,
        "relation_to_rule": {"type": "string", "enum": [
            "application", "component", "restatement", "speculative"]},
        "relation_reasoning": TEXT,
        "status": {"type": "string", "enum": ["warranted", "provisional", "unsupported"]},
        "status_reasoning": TEXT, "evidence_refs": refs,
    })
    repair = _object({
        "id": TEXT, "addition": addition,
        "target_prediction": TEXT, "prior_uses_after_repair": TEXT,
        "tested_support_ids": {"type": "array", "uniqueItems": True, "items": TEXT},
        "independent_constraint": TEXT,
        "status": {"type": "string", "enum": ["works", "fails", "unresolved"]},
        "evidence_refs": refs,
    })
    generalization = _object({
        "source_step_id": MAYBE_TEXT,
        "rule": TEXT, "premises": TEXT, "causal_conclusion": TEXT,
        "variable_meanings": TEXT, "proposed_domain": TEXT,
        "domain_warrant": TEXT, "abstraction_map": TEXT,
        "domain_evidence_refs": refs,
        "supports": {"type": "array", "items": support},
        "target_domain_status": {"type": "string", "enum": [
            "supported", "unsupported", "unresolved"]},
        "target_domain_reasoning": TEXT,
        "target_domain_refs": refs,
        "target_theory": TEXT, "target_theory_refs": refs,
        "repair_tests": {"type": "array", "items": repair},
    })
    return _object({"frame_one": frame, "frame_two": frame,
                    "translation": translation,
                    "fixed_category_alternative": alternative,
                    "generalization": generalization})


def normalize_argument(value, available_refs):
    """Check source identities and an acyclic local premise graph."""
    if not isinstance(value, dict) or set(value) != {
            "frame_one", "frame_two", "translation", "fixed_category_alternative",
            "generalization"}:
        raise ValueError("A causal argument needs two frames, translation, a fixed-category alternative, and a generalization")
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
    generalization = _normalize_generalization(value["generalization"], result, refs)
    return result | {"translation": translation,
                     "fixed_category_alternative": alternative,
                     "generalization": generalization}


def _normalize_addition(value, label):
    if not isinstance(value, dict) or set(value) != {
            "kind", "edits", "formal_change", "preservation_argument"}:
        raise ValueError(f"{label} needs a typed formal change and preservation argument")
    if value["kind"] not in ("none", "context_variable", "dependency",
                              "scale_mapping", "parameter_assumption"):
        raise ValueError(f"{label} has an unknown addition kind")
    for field in ("formal_change", "preservation_argument"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"{label}.{field} needs text")
    edits=value["edits"]
    expected={"context_variable":"add_variable", "dependency":"add_edge",
              "scale_mapping":"change_mapping",
              "parameter_assumption":"constrain_parameter"}
    if not isinstance(edits,list) or (value["kind"]=="none") != (len(edits)==0):
        raise ValueError(f"{label} needs graph edits matching its addition kind")
    seen=set()
    for edit in edits:
        if not isinstance(edit,dict) or set(edit)!={
                "operation","subject","object","expression"}:
            raise ValueError(f"{label} has an incomplete graph edit")
        if edit["operation"] not in expected.values():
            raise ValueError(f"{label} has an unknown graph edit")
        if any(not isinstance(edit[field],str) or not edit[field].strip()
               for field in ("subject","object","expression")):
            raise ValueError(f"{label} graph edits need named terms and an expression")
        identity=tuple(edit[field] for field in ("operation","subject","object","expression"))
        if identity in seen:
            raise ValueError(f"{label} repeats a graph edit")
        seen.add(identity)
    if value["kind"]!="none" and expected[value["kind"]] not in {
            edit["operation"] for edit in edits}:
        raise ValueError(f"{label} has no edit matching its addition kind")
    return value


def _normalize_generalization(value, frames, refs):
    fields = {"source_step_id", "rule", "premises", "causal_conclusion",
              "variable_meanings", "proposed_domain", "domain_warrant",
              "abstraction_map", "domain_evidence_refs", "supports",
              "target_domain_status", "target_domain_reasoning",
              "target_domain_refs", "target_theory", "target_theory_refs",
              "repair_tests"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("The generalization has incomplete fields")
    source_id = value["source_step_id"]
    if not isinstance(source_id, str) or (source_id and source_id not in {
            step["id"] for step in frames["frame_one"]["steps"]}):
        raise ValueError("The generalization source step must name a first-frame step or remain open")
    for field in ("rule", "premises", "causal_conclusion", "variable_meanings",
                  "proposed_domain", "domain_warrant", "abstraction_map",
                  "target_domain_reasoning", "target_theory"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"The generalization's {field} needs text")
    for field in ("domain_evidence_refs", "target_domain_refs", "target_theory_refs"):
        _check_refs(value[field], refs, f"generalization {field}")
    if value["target_domain_status"] not in ("supported", "unsupported", "unresolved"):
        raise ValueError("The target domain status is unknown")
    for group, expected in (("supports", {"id", "question", "theory", "consequence",
            "derivation", "addition", "distinct_consequence",
            "relation_to_rule", "relation_reasoning", "status",
            "status_reasoning", "evidence_refs"}),
            ("repair_tests", {"id", "addition", "target_prediction",
            "prior_uses_after_repair", "tested_support_ids",
            "independent_constraint", "status",
            "evidence_refs"})):
        items = value[group]
        if not isinstance(items, list):
            raise ValueError(f"The generalization's {group} must be an array")
        seen = set()
        for item in items:
            if not isinstance(item, dict) or set(item) != expected:
                raise ValueError(f"The generalization's {group} has incomplete fields")
            identity = item["id"]
            if not isinstance(identity, str) or not identity.strip() or identity in seen:
                raise ValueError(f"The generalization's {group} IDs must be unique")
            seen.add(identity)
            for field in expected - {"id", "addition", "evidence_refs", "status",
                                     "relation_to_rule",
                                     "tested_support_ids"}:
                if not isinstance(item[field], str) or not item[field].strip():
                    raise ValueError(f"The generalization's {group}.{field} needs text")
            _normalize_addition(item["addition"], f"{group}.{identity}.addition")
            _check_refs(item["evidence_refs"], refs, f"{group}.{identity}")
            if group == "repair_tests" and item["status"] not in (
                    "works", "fails", "unresolved"):
                raise ValueError("A repair status is unknown")
            if group == "supports" and item["status"] not in (
                    "warranted", "provisional", "unsupported"):
                raise ValueError("A supported-theory status is unknown")
            if group == "supports" and item["relation_to_rule"] not in (
                    "application", "component", "restatement", "speculative"):
                raise ValueError("A supported-theory relation is unknown")
            if group == "repair_tests":
                prior = item["tested_support_ids"]
                support_ids = {support["id"] for support in value["supports"]}
                if (not isinstance(prior,list)
                        or any(not isinstance(identity,str) for identity in prior)
                        or len(prior)!=len(set(prior))
                        or any(identity not in support_ids for identity in prior)):
                    raise ValueError("A repair must name available supported theories it tests")
    return value


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
    generalization = argument["generalization"]
    for field in ("domain_evidence_refs", "target_domain_refs", "target_theory_refs"):
        refs.extend(generalization[field])
    for group in ("supports", "repair_tests"):
        for item in generalization[group]:
            refs.extend(item["evidence_refs"])
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
    generalization = argument["generalization"]
    result["generalization"] = {
        key: generalization[key] for key in (
            "rule", "proposed_domain", "domain_warrant", "abstraction_map",
            "target_domain_status", "target_domain_reasoning", "target_theory")}
    result["generalization"]["supports"] = [{
        key: item[key] for key in ("id", "question", "theory", "consequence",
                              "derivation", "addition", "relation_to_rule",
                              "relation_reasoning", "status", "evidence_refs")}
        for item in generalization["supports"]]
    result["generalization"]["repair_tests"] = [{
        key: item[key] for key in ("id", "addition", "status", "target_prediction",
                              "prior_uses_after_repair", "tested_support_ids",
                              "evidence_refs")}
        for item in generalization["repair_tests"]]
    return result


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


def argument_progress(argument, fit, step_warrants, support_warrants, repair_warrants):
    """Reward warranted reuse of one inference and a harder failed repair."""
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
    generalization = argument["generalization"]
    def scores_for(group, ratings):
        identities = {item["id"] for item in generalization[group]}
        if set(ratings) != identities:
            raise ValueError(f"Every {group} item needs a warrant measurement")
        if any(not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1
               for row in ratings.values()):
            raise ValueError(f"{group} warrant scores must lie in [0,1]")
    scores_for("supports", support_warrants)
    scores_for("repair_tests", repair_warrants)
    source_id = generalization["source_step_id"]
    local = (min(first, fit["generalization_warrant"]["score"])
             if source_id and generalization["domain_evidence_refs"] else 0.0)
    candidates = []
    for index, item in enumerate(generalization["supports"]):
        key = " ".join(item["consequence"].casefold().split())
        theory_key = " ".join(item["theory"].casefold().split())
        strength = (min(local, support_warrants[item["id"]]["score"])
                    if item["status"] == "warranted"
                    and item["relation_to_rule"] == "application"
                    and item["evidence_refs"] else 0.0)
        candidates.append((strength, index, item["id"], key, theory_key))
    seen_consequences = set()
    seen_theories = set()
    support_strengths = {item["id"]: 0.0 for item in generalization["supports"]}
    for strength, _, identity, consequence, theory in sorted(
            candidates, key=lambda row: (-row[0], row[1])):
        if not strength or consequence in seen_consequences or theory in seen_theories:
            continue
        support_strengths[identity] = strength
        seen_consequences.add(consequence)
        seen_theories.add(theory)
    breadth = 1.0 - math.exp(-math.fsum(support_strengths.values()))
    target = (min(second, fit["target_theory_warrant"]["score"])
              if generalization["target_theory_refs"] else 0.0)
    domain = (fit["target_domain_warrant"]["score"]
              if generalization["target_domain_status"] == "supported"
              and generalization["target_domain_refs"] else 0.0)
    repairs = generalization["repair_tests"]
    repair_strength = (min(fit["repair_audit"]["score"],
                           *(repair_warrants[item["id"]]["score"] for item in repairs))
                       if repairs and all(item["status"] == "fails" and item["evidence_refs"]
                                          and item["tested_support_ids"]
                                          and any(support_strengths[identity] > 0 for identity in
                                                  item["tested_support_ids"])
                                          for item in repairs) else 0.0)
    witness = min(fit[name]["score"] for name in (
        "translation_specificity", "inference_reversal", "category_relation_break",
        "fixed_category_challenge", "pragmatic_pluralism", "predictive_loss"))
    if argument["fixed_category_alternative"]["status"] != "challenged":
        witness = 0.0
    challenge = min(local, breadth, target, domain, repair_strength, witness)
    score = 0.05 * local + 0.20 * breadth + 0.75 * challenge
    return {"score": score, "frame_one_strength": first,
            "frame_two_strength": second, "generalization_strength": local,
            "support_breadth": breadth, "support_strengths": support_strengths,
            "target_strength": target, "domain_strength": domain,
            "repair_strength": repair_strength, "witness_strength": witness,
            "challenge_strength": challenge}
