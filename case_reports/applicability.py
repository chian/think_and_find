"""Source-grounded review of whether extracted evidence fits the target event."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from episodic_workflow.credit_assignment.simple import CalculationError


def review_applicability(
    llm: Callable,
    *,
    source: Mapping[str, Any],
    values: Mapping[str, Mapping[str, Any]],
    target_context: Mapping[str, Any],
    allow_comparable: bool,
    removal_evidence: bool,
) -> tuple[dict[str, Any], tuple[CalculationError, ...]]:
    targets = {
        record["source_ref"]: record["text"]
        for record in target_context.get("outbreak_sources", ())
        if isinstance(record, Mapping)
        and isinstance(record.get("source_ref"), str)
        and isinstance(record.get("text"), str)
    }
    if not allow_comparable or source["source_ref"] in target_context.get("attack_source_refs", ()):
        targets[source["source_ref"]] = source["text"]
    if "target_scenario" in target_context:
        try:
            targets["runtime_target"] = json.dumps({
                name: target_context[name]
                for name in ("disease_name", "setting", "target_scenario")
                if name in target_context
            }, sort_keys=True, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            return {}, (_error("invalid_target_evidence", "Target evidence must be JSON-serializable with finite numbers."),)
    if not targets:
        return {}, (_error("missing_target_evidence", "Applicability needs the outbreak text or an explicit target scenario."),)
    request = {
        "operation": "review_evidence_applicability",
        "source": dict(source),
        "required_fields": list(values),
        "extracted_values": dict(values),
        "target_evidence": targets,
        "target_context": dict(target_context),
        "allow_comparable_observations": allow_comparable,
        "removal_evidence": removal_evidence,
    }
    try:
        response = llm(request)
        if isinstance(response, str):
            response = json.loads(response)
        if not isinstance(response, Mapping):
            raise ValueError("Applicability review must be an object")
        result = dict(response)
    except Exception as exc:
        return {}, (_error("applicability_review_error", str(exc)),)

    bool_fields = ("applicable", "scale_compatible", "mechanism_compatible")
    text_fields = ("evidence_role", "rationale")
    if any(not isinstance(result.get(name), bool) for name in bool_fields) or any(
        not isinstance(result.get(name), str) or not result[name].strip()
        for name in text_fields
    ):
        return result, (_error("invalid_applicability_review", "Review requires typed decisions, an evidence role, and a rationale."),)

    allowed_roles = {"event_observation", "event_model_parameter"}
    if allow_comparable:
        allowed_roles.add("comparable_observation")
    if result["evidence_role"] not in allowed_roles or not all(result[name] for name in bool_fields):
        return result, (_error("inapplicable_evidence", result["rationale"]),)

    errors = []
    if result.get("source_ref") != source["source_ref"]:
        errors.append(_error("source_ref_mismatch", "Applicability review must cite the extracted source."))
    source_span = result.get("source_evidence_span")
    if not isinstance(source_span, str) or not source_span.strip() or source_span not in source["text"]:
        errors.append(_error("unsupported_evidence_span", "Applicability source span must be exact retrieved text."))
    target_ref = result.get("target_ref")
    target_span = result.get("target_evidence_span")
    if not isinstance(target_ref, str) or target_ref not in targets:
        errors.append(_error("source_ref_mismatch", "Applicability review must cite a supplied target."))
    elif not isinstance(target_span, str) or not target_span.strip() or target_span not in targets[target_ref]:
        errors.append(_error("unsupported_evidence_span", "Applicability target span must be exact supplied target text."))
    return result, tuple(errors)


def _error(code: str, message: str) -> CalculationError:
    return CalculationError(code=code, field="evidence_applicability", message=message)
