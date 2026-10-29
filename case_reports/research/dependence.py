"""Executable dependence witnesses and target comparisons over declared models.

This is a subscoring-sheet projection over existing immutable importance
protocols and reproducible host receipts, not another episode or model grader.
It establishes consequences of the supplied formalization, not the truth of its
empirical premises or a universal natural-language equivalence to G.
"""
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import json

from .importance import rational, result_key, validate_result, score_importance, protocol_identity, answer, validate_protocol


def _pins_for(sheet, context, pins):
    merged = {key: set(versions) for key, versions in (pins or {}).items()}
    for key, versions in sheet.calculation_pins(context).items():
        merged.setdefault(key, set()).update(versions)
    return merged


def paired_case_result(sheet, assessment, case_id, *, context=None, _pins=None):
    """Read a current, reproducible result or None; never calculate missing work.

    An execution failure is a real receipt with a missing answer and error text.
    It remains unresolved rather than being converted to unchanged/zero impact.
    """
    sheet._require_kind(assessment, "importance")
    protocol = sheet.rows[assessment]["data"]
    if case_id not in {case["id"] for case in protocol["domain"]["cases"]}:
        raise ValueError("Dependence must select a declared circumstance")
    pins = _pins_for(sheet, context, _pins)
    key = result_key(assessment, case_id)
    row = sheet.rows.get(key)
    if (not sheet.usable(assessment, _pins=pins) or row is None
            or not sheet.usable(key, _pins=pins)):
        return None
    validate_result(row["data"], protocol)
    return deepcopy(row["data"])


@lru_cache(maxsize=1024)
def _focal_answers_json(protocol_json, inputs_json):
    protocol, inputs = json.loads(protocol_json), json.loads(inputs_json)
    model = protocol["model"]
    model["answer"] = model["focal"]
    results, errors = {}, []
    for label, altered in (("baseline", False), ("altered", True)):
        try:
            results[label] = answer(protocol, inputs, altered=altered)
        except (ValueError, KeyError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
            results[label] = None
            errors.append(label + ": " + str(exc))
    return json.dumps({"focal": model["focal"], **results, "error": "; ".join(errors)},
                      sort_keys=True, separators=(",", ":"))


def focal_answers(protocol, inputs):
    """Execute the actual focal equation outputs, before downstream masking.

    The original model is not changed. This is a host calculation of the focal
    variable under the same paired models, not a substitute for the declared
    final prediction or its importance denominator.
    """
    validate_protocol(protocol)
    return json.loads(_focal_answers_json(json.dumps(protocol, sort_keys=True), json.dumps(inputs, sort_keys=True)))


def focal_case_result(sheet, assessment, case_id, *, context=None, _pins=None):
    """Return host focal values only for an admitted current paired execution."""
    receipt = paired_case_result(sheet, assessment, case_id, context=context, _pins=_pins)
    if receipt is None:
        return None
    protocol = sheet.rows[assessment]["data"]
    case = next(case for case in protocol["domain"]["cases"] if case["id"] == case_id)
    return {"assessment": assessment, "circumstance": case_id, **focal_answers(protocol, case["inputs"])}


def dependence_result(sheet, assessment, *, context=None, _pins=None, case_ids=None):
    """Project measured dependence, exact claim bindings and bounded witnesses."""
    sheet._require_kind(assessment, "importance")
    assessment_row = sheet.rows[assessment]
    protocol = assessment_row["data"]
    application_key = protocol["application"]
    application_ref = {"key": application_key, "version": assessment_row["dependencies"][application_key]}
    application = sheet.snapshot(application_ref)
    inference_key = application["data"]["inference"]
    inference_ref = {"key": inference_key, "version": application["dependencies"][inference_key]}
    pins = _pins_for(sheet, context, _pins)
    score = score_importance(sheet, assessment, _pins=pins)
    domain_cases = protocol["domain"]["cases"]
    declared_ids = {case["id"] for case in domain_cases}
    selected = [case["id"] for case in domain_cases[:8]] if case_ids is None else case_ids
    if (not isinstance(selected, (list, tuple)) or len(selected) > 8
            or len(set(selected)) != len(selected) or not set(selected) <= declared_ids):
        raise ValueError("Select at most eight distinct declared dependence circumstances")
    witnesses, cases = [], []
    for case in domain_cases:
        receipt = paired_case_result(sheet, assessment, case["id"], _pins=pins)
        resolved = receipt is not None and not receipt["error"]
        changed = receipt["baseline"] != receipt["altered"] if resolved else None
        if changed:
            witnesses.append(case["id"])
        if case["id"] in selected:
            cases.append({"id": case["id"], "inputs": deepcopy(case["inputs"]),
                "weight": str(rational(case["weight"])),
                "status": "resolved" if resolved else "unresolved",
                "baseline": receipt["baseline"] if receipt else None,
                "altered": receipt["altered"] if receipt else None,
                "changed": changed, "error": receipt["error"] if receipt else "No current calculation receipt"})
    if Fraction(score["changed_mass"]) > 0:
        status = "demonstrated"
    elif not Fraction(score["unresolved_mass"]):
        status = "not_demonstrated"
    else:
        status = "unresolved"
    model = protocol["model"]
    return {"assessment": assessment, "assessment_version": assessment_row["version"],
        "protocol_identity": protocol_identity(protocol), "application": application_ref,
        "inference": inference_ref, "current": score["current"], "calculation": score,
        "formal_relationship": {"method": protocol["method"], "focal": model["focal"],
            "original": model["equations"][model["focal"]], "altered": model["replacement"],
            "answer": model["answer"], "edge": list(model["edge"]), "steps": model["steps"]},
        "dependence": {"status": status, "witness_cases": witnesses[:8],
            "witness_count": len(witnesses),
            "meaning": "A changed paired answer is a witness within the declared formal model and circumstance domain. "
                       "No changed result after complete evaluation establishes zero measured dependence only in that domain."},
        "cases": cases, "total_cases": len(domain_cases),
        "limitations": "Formal realization and circumstance measure are explicit scientific inputs. "
                       "This calculation does not certify their empirical adequacy, G's expected generalizability bounds, "
                       "absence of other derivations, or a continuum claim from finite samples."}


def validate_target(target):
    """Exact answer or inclusive numeric interval; no model relevance score."""
    if not isinstance(target, dict):
        raise ValueError("An answer target must be an explicit typed object")
    if target.get("kind") == "exact" and set(target) == {"kind", "answer"}:
        answer = target["answer"]
        if not isinstance(answer, dict) or set(answer) != {"type", "value"}:
            raise ValueError("An exact target needs a typed answer")
        if answer["type"] == "boolean" and type(answer["value"]) is bool:
            return
        if answer["type"] == "rational":
            rational(answer["value"])
            return
        raise ValueError("Exact answer targets are Boolean or rational")
    if target.get("kind") == "interval" and set(target) == {"kind", "lower", "upper"}:
        if rational(target["lower"]) > rational(target["upper"]):
            raise ValueError("An answer interval needs lower <= upper")
        return
    raise ValueError("Target must declare exact answer or inclusive interval")


def answer_matches(answer, target):
    """Return a calculated match, mismatch, or None for a missing answer."""
    validate_target(target)
    if answer is None:
        return None
    if not isinstance(answer, dict) or set(answer) != {"type", "value"}:
        raise ValueError("An answer must retain its calculation type and value")
    if answer["type"] == "boolean":
        if type(answer["value"]) is not bool:
            raise ValueError("Boolean answers must not be numeric surrogates")
    elif answer["type"] == "rational":
        rational(answer["value"])
    else:
        raise ValueError("Unsupported calculated answer type")
    if target["kind"] == "exact":
        expected = target["answer"]
        if expected["type"] != answer["type"]:
            return False
        if expected["type"] == "boolean":
            return answer["value"] == expected["value"]
        return rational(answer["value"]) == rational(expected["value"])
    if answer["type"] != "rational":
        return False
    return rational(target["lower"]) <= rational(answer["value"]) <= rational(target["upper"])


def target_identity(target):
    """Semantic identity ignores equivalent rational spellings, never its type."""
    validate_target(target)
    normalized = deepcopy(target)
    if target["kind"] == "interval":
        normalized.update(lower=str(rational(target["lower"])), upper=str(rational(target["upper"])))
    elif target["answer"]["type"] == "rational":
        normalized["answer"]["value"] = str(rational(target["answer"]["value"]))
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def target_schema():
    number = {"oneOf": [{"type": "integer"}, {"type": "string", "minLength": 1, "maxLength": 256}]}
    return {"oneOf": [
        {"type": "object", "additionalProperties": False, "required": ["kind", "answer"],
         "properties": {"kind": {"const": "exact"}, "answer": {"oneOf": [
             {"type": "object", "additionalProperties": False, "required": ["type", "value"],
              "properties": {"type": {"const": "boolean"}, "value": {"type": "boolean"}}},
             {"type": "object", "additionalProperties": False, "required": ["type", "value"],
              "properties": {"type": {"const": "rational"}, "value": number}}]}}},
        {"type": "object", "additionalProperties": False, "required": ["kind", "lower", "upper"],
         "properties": {"kind": {"const": "interval"}, "lower": number, "upper": number}}]}
