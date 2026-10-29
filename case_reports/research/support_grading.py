"""Executable support operands linked to exact G and B_G versions.

Hypothesis dependence is calculated from paired formal-model answers. Source
observations are compared with computed predictions using their reported values
and uncertainty, never a model-issued grade. Interpretation of a source/model
mapping remains explicit and is not certified merely by arithmetic.
"""
import ast
from copy import deepcopy
from hashlib import sha256
import json

from .dependence import dependence_result, paired_case_result, answer_matches
from .importance import rational
from .support_observations import observation_identity, observation_target

FIELDS = ("support", "application", "importance", "bounds_test", "prediction", "facts", "limitations")


def references(data):
    return [data[name] for name in ("support", "application", "importance", "bounds_test")] + [item["evidence"] for item in data["facts"]]


def context_pins(sheet, data):
    support = sheet.rows[data["support"]]["data"]
    pins = {}
    for ref in (support["inference"], support["expectation"]):
        pins.setdefault(ref["key"], set()).add(ref["version"])
    return pins


def validate(sheet, key, data):
    if not isinstance(data, dict) or set(data) != set(FIELDS):
        raise ValueError("Support assessment needs its exact executable operands")
    for name in ("support", "application", "importance", "bounds_test", "limitations"):
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError("Support assessment reference/limitations must be explicit")
    for name, kind in (("support", "expectation_support"), ("application", "application"),
                       ("importance", "importance"), ("bounds_test", "expectation_test")):
        sheet._require_kind(data[name], kind)
    support = sheet.rows[data["support"]]["data"]
    application = sheet.rows[data["application"]]
    protocol = sheet.rows[data["importance"]]["data"]
    bounds_test = sheet.rows[data["bounds_test"]]["data"]
    if (application["data"]["inference"] != support["inference"]["key"] or
            application["dependencies"][support["inference"]["key"]] != support["inference"]["version"]):
        raise ValueError("Support assessment must use the exact G version named in its support claim")
    if protocol["application"] != data["application"]:
        raise ValueError("Support dependence protocol must address the selected application")
    if bounds_test["expectation"] != support["expectation"] or bounds_test["assessment"] != data["importance"]:
        raise ValueError("Support and B_G subscoring sheets must share the exact expected bounds and formal protocol")
    prediction = data["prediction"]
    if not isinstance(prediction, dict) or set(prediction) != {"quantity", "units"} or any(
            not isinstance(value, str) or not value.strip() for value in prediction.values()):
        raise ValueError("Prediction quantity and units must be explicit; no implicit conversions")
    if not isinstance(data["facts"], list) or len(data["facts"]) > 16:
        raise ValueError("Select at most sixteen observations per support assessment")
    case_ids = {case["id"] for case in protocol["domain"]["cases"]}
    seen = set()
    for fact in data["facts"]:
        if not isinstance(fact, dict) or set(fact) != {"evidence", "observation", "case_id"}:
            raise ValueError("An observation comparison names evidence, observation and declared circumstance")
        sheet._require_kind(fact["evidence"], "evidence")
        observation_identity(sheet.rows[fact["evidence"]]["data"], fact["observation"])
        if fact["case_id"] not in case_ids:
            raise ValueError("Observation comparison must use a predeclared circumstance")
        identity = (fact["evidence"], fact["observation"])
        if identity in seen:
            raise ValueError("An observation can be selected once per support assessment")
        seen.add(identity)
    prior = sheet.rows.get(key)
    if prior and any(prior["data"][name] != data[name] for name in FIELDS if name != "limitations"):
        raise ValueError("Support comparison operands are fixed; create an explicit new assessment for a changed comparison")


def _expression_identity(source):
    return ast.dump(ast.parse(source, mode="eval"), annotate_fields=True, include_attributes=False)


def hypothesis_identity(protocol, prediction=None):
    """Structural formal identity, with static substitution and alpha-renaming.

    Label/domain/contrast changes cannot create hypotheses. Static equations and
    SCMs use one representation. This is not general algebraic or dynamical
    equivalence; that remaining identity uncertainty is exposed in every result.
    """
    model = protocol["model"]
    names = {}
    active = set()
    def parsed(source):
        tree = ast.parse(source, mode="eval").body
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                node.exact_literal = ast.get_source_segment(source, node)
        return tree

    def normalize(source, expand):
        class Normalize(ast.NodeTransformer):
            def visit_Constant(self, node):
                if type(node.value) in (int, float):
                    return ast.Constant(value="rational:" + str(rational(node.exact_literal)))
                return node

            def visit_Call(self, node):
                node.args = [self.visit(argument) for argument in node.args]
                return node

            def visit_Name(self, node):
                if expand and node.id in model["equations"]:
                    if node.id in active:
                        raise ValueError("Cannot canonicalize a cyclic static hypothesis")
                    active.add(node.id)
                    result = self.visit(parsed(model["equations"][node.id]))
                    active.remove(node.id)
                    return result
                alias = names.setdefault(node.id, "v" + str(len(names)))
                return ast.Name(id=alias, ctx=ast.Load())
        return ast.dump(Normalize().visit(parsed(source)), include_attributes=False)
    if protocol["method"] != "simulation":
        representation = {"method": "static", "answer": normalize(model["answer"], True),
                          "focal": normalize(model["equations"][model["focal"]], True)}
    else:
        # Rooted names fix the focal output before traversing the dynamics.
        answer = normalize(model["answer"], False)
        normalize(model["focal"], False)
        equations = []
        remaining = dict(model["equations"])
        while remaining:
            name = min(remaining, key=lambda item: (item not in names, names.get(item, item)))
            lhs = names.setdefault(name, "v" + str(len(names)))
            rhs = normalize(remaining.pop(name), False)
            equations.append((lhs, rhs))
        representation = {"method": "discrete_simulation", "answer": answer, "equations": sorted(equations),
            "initial": sorted((names[name], normalize(source, False)) for name, source in model["initial"].items()),
            "steps": model["steps"]}
    return sha256(json.dumps(representation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def grade_support(sheet, key):
    """Calculate reusable hypothesis and observation support operands."""
    from .bounds import grade_bounds
    sheet._require_kind(key, "support_assessment")
    row = sheet.rows[key]
    data = row["data"]
    support = sheet.snapshot({"key": data["support"], "version": row["dependencies"][data["support"]]})["data"]
    pins = context_pins(sheet, data)
    current = sheet.usable(key, _pins=pins)
    if not current:
        return {"assessment": key, "version": row["version"], "current": False,
            "support": data["support"], "inference": support["inference"], "expectation": support["expectation"],
            "application": data["application"], "importance": data["importance"], "bounds_test": data["bounds_test"],
            "hypothesis": {"identity": None, "status": "stale_operands", "witness_cases": [], "unresolved_cases": []},
            "facts": [{**fact, "identity": None, "status": "stale_operands"} for fact in data["facts"]],
            "dependence": None, "limitations": ["Inputs changed; reconsider the scientific mapping and rerun its calculations"]}
    dependence = dependence_result(sheet, data["importance"], _pins=pins)
    bounds = grade_bounds(sheet, data["bounds_test"])
    cases = {case["case_id"]: case for case in bounds["cases"]}
    protocol = sheet.rows[data["importance"]]["data"]
    witnesses, pending_cases = [], []
    for case in protocol["domain"]["cases"]:
        classification = cases[case["id"]]
        if classification["in_scope"] is None:
            pending_cases.append(case["id"])
            continue
        if classification["in_scope"] is not True:
            continue
        receipt = paired_case_result(sheet, data["importance"], case["id"], _pins=pins)
        if receipt is None or receipt["error"] or classification.get("original_G_relation") is None:
            pending_cases.append(case["id"])
        elif classification["original_G_relation"] is True and classification["baseline_fits"] is True and receipt["baseline"] != receipt["altered"]:
            witnesses.append(case["id"])
    hypothesis_status = ("stale_operands" if not current else "demonstrated" if witnesses else
                         "unresolved" if pending_cases else "not_demonstrated_in_expected_bounds")
    formal_identity = None
    identity_error = ""
    if hypothesis_status == "demonstrated":
        try:
            formal_identity = hypothesis_identity(protocol, data["prediction"])
        except (ValueError, SyntaxError, RecursionError) as exc:
            hypothesis_status = "unresolved_formal_identity"
            identity_error = str(exc)
    fact_results = []
    for fact in data["facts"]:
        evidence = sheet.rows[fact["evidence"]]["data"]
        oid, source = observation_identity(evidence, fact["observation"])
        observed = source["measurement"]
        classification = cases[fact["case_id"]]
        target, target_status = observation_target(observed)
        receipt = paired_case_result(sheet, data["importance"], fact["case_id"], _pins=pins)
        base_match = altered_match = None
        status = "unresolved"
        if not current:
            status = "stale_operands"
        elif hypothesis_status != "demonstrated":
            status = "application_dependence_not_demonstrated"
        elif classification["in_scope"] is not True:
            status = "outside_expected_bounds"
        elif classification.get("original_G_relation") is None:
            status = "unresolved_G_correspondence"
        elif classification["original_G_relation"] is False:
            status = "G_relation_not_satisfied"
        elif observed["quantity"] != data["prediction"]["quantity"] or observed["units"] != data["prediction"]["units"]:
            status = "quantity_or_units_mismatch"
        elif not isinstance(observed["conditions"], dict):
            status = "unresolved_source_conditions"
        elif set(observed["conditions"]) != set(next(case["inputs"] for case in protocol["domain"]["cases"] if case["id"] == fact["case_id"])):
            status = "source_conditions_mismatch"
        elif any(rational(str(observed["conditions"][name]) if type(observed["conditions"][name]) is float else observed["conditions"][name]) != rational(value)
                 for name, value in next(case["inputs"] for case in protocol["domain"]["cases"] if case["id"] == fact["case_id"]).items()):
            status = "source_conditions_mismatch"
        elif target is None:
            status = "unresolved_source_value_or_uncertainty"
        elif receipt is not None and not receipt["error"]:
            base_match = answer_matches(receipt["baseline"], target)
            altered_match = answer_matches(receipt["altered"], target)
            status = "aligned" if base_match else "not_aligned"
        fact_results.append({**fact, "identity": oid, "status": status,
            "baseline_matches": base_match, "altered_matches": altered_match,
            "discriminating": base_match is True and altered_match is False,
            "target": target, "comparison_basis": target_status,
            "source": source, "case_inputs": deepcopy(next(case["inputs"] for case in protocol["domain"]["cases"] if case["id"] == fact["case_id"]))})
    return {"assessment": key, "version": row["version"], "current": current,
        "support": data["support"], "inference": support["inference"], "expectation": support["expectation"],
        "application": data["application"], "importance": data["importance"], "bounds_test": data["bounds_test"],
        "hypothesis": {"identity": formal_identity, "status": hypothesis_status, "identity_error": identity_error,
            "witness_cases": witnesses, "unresolved_cases": pending_cases},
        "facts": fact_results, "dependence": dependence["calculation"],
        "limitations": [data["limitations"], "Executed consequences are conditional on the supplied formal realization, units and circumstance-to-source mapping.",
            "Formal fingerprints substitute static equations and normalize variable names, but do not prove every algebraic or dynamical equivalence; source extraction fidelity remains auditable.",
            "Aligned observations count in a dependent application even if the altered prediction also aligns; discrimination is reported separately."]}


def record_schema(snapshot_schema=None):
    text = {"type": "string", "minLength": 1}
    def obj(properties):
        return {"type": "object", "additionalProperties": False, "required": list(properties), "properties": properties}
    return obj({"support": text, "application": text, "importance": text, "bounds_test": text,
        "prediction": obj({"quantity": text, "units": text}),
        "facts": {"type": "array", "maxItems": 16, "items": obj({"evidence": text, "observation": text, "case_id": text})},
        "limitations": text})
