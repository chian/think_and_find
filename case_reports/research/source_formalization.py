"""A sourced theoretical statement expressed in the same calculus as G.

The mapping from source prose to equations is a public scientific proposal. The
host can check its consequences and correspondence to G, not transcription
fidelity. Empirical measurements remain observations rather than invented models.
"""
from copy import deepcopy

from .importance import METHODS, validate_protocol, paired_answers
from .bounds import grade_bounds, checked_claim_relation
from .dependence import paired_case_result, answer_matches


FIELDS = ("evidence", "assessment", "method", "model", "quantity", "units", "derivation", "limitations")


def references(data):
    return [data["evidence"], data["assessment"]]


def protocol(sheet, data):
    comparison = sheet.rows[data["assessment"]]["data"]
    return {"application": comparison["application"], "method": data["method"],
            "question": data["derivation"], "contrast": "Source statement under the same declared G alteration",
            "domain": deepcopy(comparison["domain"]), "model": deepcopy(data["model"]),
            "comparator": "exact_typed_answer"}


def validate(sheet, key, data):
    if not isinstance(data, dict) or set(data) != set(FIELDS):
        raise ValueError("Source formalization needs its source, G comparison and complete model")
    sheet._require_kind(data["evidence"], "evidence")
    sheet._require_kind(data["assessment"], "importance")
    for name in ("quantity", "units", "derivation", "limitations"):
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError(f"Source formalization {name} must be explicit")
    comparison = sheet.rows[data["assessment"]]["data"]
    if data["method"] != comparison["method"]:
        raise ValueError("Source statement and G must use the same formal method")
    validate_protocol(protocol(sheet, data))
    prior = sheet.rows.get(key)
    if prior and any(prior["data"][name] != data[name] for name in ("evidence", "assessment")):
        raise ValueError("Reinterpret a different source or G protocol under a new formalization key")


def compare(sheet, key, bounds_test, *, pins=None):
    """Execute paired source and G models on one fixed circumstance domain.

    This checks a declared correspondence, not whether the source actually says
    the proposed equations. Every in-scope case is retained, including conflicts.
    """
    sheet._require_kind(key, "source_formalization")
    sheet._require_kind(bounds_test, "expectation_test")
    data = sheet.rows[key]["data"]
    test = sheet.rows[bounds_test]["data"]
    if data["assessment"] != test["assessment"]:
        raise ValueError("Source and G comparison must share the same formal protocol")
    source_protocol = protocol(sheet, data)
    cases = grade_bounds(sheet, bounds_test)["cases"]
    declared = {case["id"]: case for case in sheet.rows[data["assessment"]]["data"]["domain"]["cases"]}
    current = sheet.usable(key, _pins=pins) and sheet.usable(bounds_test, _pins=pins)
    compared = []
    for classified in cases:
        case_id = classified["case_id"]
        inputs = declared[case_id]["inputs"]
        result = {"case_id": case_id, "in_scope": classified["in_scope"],
                  "source_baseline": None, "source_altered": None,
                  "G_baseline": classified["baseline"], "G_altered": classified["altered"],
                  "source_G_relation": None, "source_expected_answer": None,
                  "exact_output_agreement": None,
                  "status": "outside_expected_bounds" if classified["in_scope"] is False else "unresolved"}
        if current and classified["in_scope"] is True:
            source = paired_answers(source_protocol, case_id)
            result["source_baseline"], result["source_altered"] = source["baseline"], source["altered"]
            if source["error"]:
                result["error"] = source["error"]
            else:
                relation, reason = checked_claim_relation(test["claim_relation"], source, inputs,
                                                          source_protocol, test["claim_intervention"])
                result["source_G_relation"] = relation
                if reason:
                    result["error"] = reason
                G = paired_case_result(sheet, data["assessment"], case_id, _pins=pins)
                if G is not None and not G["error"] and relation is not None:
                    result["source_expected_answer"] = answer_matches(source["baseline"], test["target"])
                    result["exact_output_agreement"] = (source["baseline"] == G["baseline"] and
                                                        source["altered"] == G["altered"])
                    if (classified["original_G_relation"] is not True or
                            classified["baseline_fits"] is not True or G["baseline"] == G["altered"]):
                        result["status"] = "not_a_G_witness"
                    elif result["source_expected_answer"] is not None:
                        result["status"] = ("aligned" if relation is True and
                            result["source_expected_answer"] is True and
                            source["baseline"] != source["altered"] else "conflict")
        compared.append(result)
    in_scope = [case for case in compared if case["in_scope"] is True]
    return {"formalization": key, "evidence": data["evidence"], "method": data["method"],
            "assessment": data["assessment"], "bounds_test": bounds_test,
            "status": "stale_operands" if not current else
                "conflict" if any(case["status"] == "conflict" for case in in_scope) else
                "partial_alignment" if any(case["status"] == "aligned" for case in in_scope) and
                    any(case["status"] == "unresolved" for case in in_scope) else
                "aligned" if any(case["status"] == "aligned" for case in in_scope) else "unresolved",
            "cases": compared,
            "qualification": "Finite or sampled agreement of authored formalisms; source fidelity and semantic equivalence are not proved"}


def record_schema():
    from .importance import protocol_schema
    text = {"type": "string", "minLength": 1}
    model = protocol_schema()["properties"]["model"]
    properties = {"evidence": text, "assessment": text, "method": {"enum": list(METHODS)},
                  "model": model, "quantity": text, "units": text, "derivation": text,
                  "limitations": text}
    return {"type": "object", "additionalProperties": False, "required": list(FIELDS),
            "properties": properties}
