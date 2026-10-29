"""Paired evaluations of a focal relationship, never model-assigned grades.

The built-in evaluators cover explicit equations, acyclic deterministic SCMs,
and discrete-time master equations. Fractions preserve exact finite arithmetic.
They do not claim to solve arbitrary implicit equations, continuous-time systems,
or prove properties of an entire continuous parameter space from a grid.
"""

import ast
from functools import lru_cache
from fractions import Fraction
from hashlib import sha256
import json
import re

from .grading import GRADING_FIELDS


METHODS = ("equations", "scm", "simulation")
PROTOCOL_FIELDS = ("application", "method", "question", "contrast", "domain", "model", "comparator")
RESULT_FIELDS = ("assessment", "circumstance", "baseline", "altered", "error")


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("Use integer or rational/decimal text, not floating-point grades")
    if len(str(value)) > 256:
        raise ValueError("Numeric operand exceeds the supported precision")
    exponent = re.search(r"[eE]([+-]?\d+)$", str(value))
    if exponent and abs(int(exponent.group(1))) > 1200:
        raise ValueError("Numeric exponent exceeds the supported exact-arithmetic size")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Invalid rational operand") from exc
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > 4096:
        raise ValueError("Numeric operand exceeds the supported exact-arithmetic size")
    return result


def _bounded(value):
    if isinstance(value, bool):
        return value
    value = Fraction(value)
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
        raise ValueError("Calculation exceeds the supported exact-arithmetic size")
    return value


def expression(source, variables):
    """Interpret a small mathematical expression; never eval/exec model code."""
    if not isinstance(source, str) or not source.strip() or len(source) > 2048:
        raise ValueError("Supply a bounded mathematical expression")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Invalid mathematical expression") from exc
    if len(list(ast.walk(tree))) > 256:
        raise ValueError("Expression exceeds the supported operation count")

    def visit(node):
        if isinstance(node, ast.Constant):
            if type(node.value) is bool:
                return node.value
            if type(node.value) not in (int, float):
                raise ValueError("Only numeric and Boolean constants are supported")
            return _bounded(rational(ast.get_source_segment(source, node)))
        if isinstance(node, ast.Name):
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            if isinstance(node.op, ast.USub):
                return _bounded(-value)
            if isinstance(node.op, ast.UAdd):
                return value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return _bounded(left + right)
            if isinstance(node.op, ast.Sub):
                return _bounded(left - right)
            if isinstance(node.op, ast.Mult):
                return _bounded(left * right)
            if isinstance(node.op, ast.Div):
                return _bounded(Fraction(left) / right)
            if isinstance(node.op, ast.Pow):
                right = Fraction(right)
                if right.denominator != 1 or abs(right) > 16:
                    raise ValueError("Only integer powers in [-16,16] are supported")
                return _bounded(Fraction(left) ** int(right))
        if isinstance(node, ast.Compare):
            left = visit(node.left)
            for op, operand in zip(node.ops, node.comparators):
                right = visit(operand)
                operations = {ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                              ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right,
                              ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right}
                if type(op) not in operations:
                    raise ValueError("Unsupported comparison")
                if not operations[type(op)]():
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(visit(value) for value in node.values)
            if isinstance(node.op, ast.Or):
                return any(visit(value) for value in node.values)
        if isinstance(node, ast.IfExp):
            return visit(node.body if visit(node.test) else node.orelse)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            functions = {"abs": abs, "min": min, "max": max}
            if node.func.id in functions:
                return _bounded(functions[node.func.id](*(visit(arg) for arg in node.args)))
        raise ValueError("Unsupported expression operation")

    return visit(tree.body)


def validate_protocol(data):
    if set(data) != set(PROTOCOL_FIELDS) or data["method"] not in METHODS:
        raise ValueError("Importance requires a supported method and exactly its declared fields")
    for name in ("application", "question", "contrast"):
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError(f"Importance {name} must be explicit")
    if data["comparator"] != "exact_typed_answer":
        raise ValueError("The supported comparator is exact_typed_answer; define a Boolean predicate in answer for a threshold question")
    domain = data["domain"]
    if not isinstance(domain, dict) or set(domain) != {"kind", "description", "measure", "cases"}:
        raise ValueError("Domain requires kind, description, measure and cases")
    if domain["kind"] not in {"finite", "sample"}:
        raise ValueError("Declare finite-domain calculation or sample; no implicit continuum coverage")
    if any(not isinstance(domain[n], str) or not domain[n].strip() for n in ("description", "measure")):
        raise ValueError("Declare the circumstance domain and the meaning of its weights")
    if not isinstance(domain["cases"], list) or not 1 <= len(domain["cases"]) <= 256:
        raise ValueError("Declare 1..256 circumstances in the fixed denominator before evaluating")
    ids, points, names = set(), set(), None
    for case in domain["cases"]:
        if not isinstance(case, dict) or set(case) != {"id", "inputs", "weight"}:
            raise ValueError("Each circumstance needs id, inputs and weight")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in ids:
            raise ValueError("Circumstance identities must be nonempty and unique")
        ids.add(case["id"])
        if not isinstance(case["inputs"], dict) or len(case["inputs"]) > 32 or any(not isinstance(n, str) or not n.isidentifier() for n in case["inputs"]):
            raise ValueError("Circumstance inputs must be named numeric parameters")
        canonical = tuple(sorted((n, str(rational(v))) for n, v in case["inputs"].items()))
        if canonical in points:
            raise ValueError("Duplicate circumstances do not enlarge the denominator")
        points.add(canonical)
        current_names = set(case["inputs"])
        if names is not None and current_names != names:
            raise ValueError("All circumstances must specify the same input variables")
        names = current_names
        if rational(case["weight"]) <= 0:
            raise ValueError("Circumstance weights must be positive")
    model = data["model"]
    if not isinstance(model, dict) or set(model) != {
            "equations", "focal", "replacement", "answer", "initial", "steps", "edge"}:
        raise ValueError("Model needs equations, focal, replacement, answer, initial, steps and edge")
    equations = model["equations"]
    if not isinstance(equations, dict) or not equations or len(equations) > 32:
        raise ValueError("Provide 1..32 explicit equations")
    if any(not isinstance(n, str) or not n.isidentifier() for n in equations) or set(equations) & names:
        raise ValueError("Equation outputs must be identifiers distinct from circumstance inputs")
    if model["focal"] not in equations:
        raise ValueError("Identify the equation realizing G")
    if any(not isinstance(e, str) or not e.strip() or len(e) > 2048
           for e in [*equations.values(), model["replacement"], model["answer"]]):
        raise ValueError("Provide explicit original, replacement and answer expressions")
    if type(model["steps"]) is not int or not isinstance(model["initial"], dict):
        raise ValueError("Initial state and step count must be explicit")
    if data["method"] == "simulation":
        if not 1 <= model["steps"] <= 1024 or set(model["initial"]) != set(equations):
            raise ValueError("Discrete simulation needs an initial expression for every state and 1..1024 steps")
        if any(not isinstance(v, str) or not v.strip() for v in model["initial"].values()):
            raise ValueError("Initial conditions must be explicit mathematical expressions")
    elif model["steps"] != 0 or model["initial"]:
        raise ValueError("Static equations and SCMs use steps=0 and initial={}")
    edge = model["edge"]
    if data["method"] == "scm":
        if not isinstance(edge, list) or len(edge) != 2 or edge[1] != model["focal"]:
            raise ValueError("SCM must name the focal [parent, child] edge")
        try:
            parents = {node.id for node in ast.walk(ast.parse(equations[edge[1]], mode="eval"))
                       if isinstance(node, ast.Name)}
        except SyntaxError as exc:
            raise ValueError("Invalid structural equation") from exc
        if edge[0] not in parents or edge[0] not in names | set(equations):
            raise ValueError("Focal edge must occur in its structural equation")
    elif edge != []:
        raise ValueError("Only the SCM representation declares an edge; otherwise use []")


def answer(protocol, inputs, *, altered=False):
    model = protocol["model"]
    equations = dict(model["equations"])
    if altered:
        equations[model["focal"]] = model["replacement"]
    env = {name: rational(value) for name, value in inputs.items()}
    if protocol["method"] == "simulation":
        state = {name: expression(value, env) for name, value in model["initial"].items()}
        for _ in range(model["steps"]):
            before = {**env, **state}
            state = {name: expression(value, before) for name, value in equations.items()}
        env.update(state)
    else:
        pending = dict(equations)
        while pending:
            completed = []
            for name, value in pending.items():
                try:
                    env[name] = expression(value, env)
                except KeyError:
                    continue
                completed.append(name)
            if not completed:
                raise ValueError("Unresolved/cyclic equations: this evaluator requires an explicit acyclic system")
            for name in completed:
                del pending[name]
    result = expression(model["answer"], env)
    return {"type": "boolean", "value": result} if isinstance(result, bool) else {
        "type": "rational", "value": str(_bounded(result))}


def evaluate_cases(sheet, assessment, case_ids, *, context=None):
    """Host-selected ordinary calculation, returning version-bound sheet updates."""
    sheet._require_kind(assessment, "importance")
    pins = {}
    if context:
        if context not in sheet.rows or sheet.rows[context]["kind"] not in {"expectation_support", "support_contribution", "causal_change", *GRADING_FIELDS}:
            raise ValueError("Historical calculation context must be a scoped support or change record")
        pins = sheet._claim_pins(sheet.rows[context], {})
        if not sheet.usable(context, _pins=pins):
            raise ValueError("Historical calculation context has stale or refuted scientific grounds")
    if not sheet.usable(assessment, _pins=pins):
        raise ValueError("Reconsider the protocol against its current scientific dependencies or select its historical claim context")
    protocol = sheet.rows[assessment]["data"]
    validate_protocol(protocol)
    if not isinstance(case_ids, list) or not 1 <= len(case_ids) <= 8 or len(set(case_ids)) != len(case_ids):
        raise ValueError("Select one to eight distinct circumstances per calculation")
    cases = {case["id"]: case for case in protocol["domain"]["cases"]}
    if not set(case_ids) <= cases.keys():
        raise ValueError("Only predeclared circumstances may be evaluated")
    staged = type(sheet)(sheet.history)
    updates = []
    for case_id in case_ids:
        key = result_key(assessment, case_id)
        updates.append(staged.put(key, "importance_result", {
            "assessment": assessment, "circumstance": case_id,
            **paired_answers(protocol, case_id)}))
    return updates


@lru_cache(maxsize=512)
def _paired_answers_json(protocol_json, case_id):
    """Cache immutable host output; repeated history admission cannot mutate it."""
    protocol = json.loads(protocol_json)
    case = next((case for case in protocol["domain"]["cases"] if case["id"] == case_id), None)
    if case is None:
        raise ValueError("Only a declared circumstance can be calculated")
    outputs, errors = {}, []
    for label, altered in (("baseline", False), ("altered", True)):
        try:
            outputs[label] = answer(protocol, case["inputs"], altered=altered)
        except (ValueError, KeyError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
            outputs[label] = None
            errors.append(label + ": " + str(exc))
    return json.dumps({**outputs, "error": "; ".join(errors)}, sort_keys=True, separators=(",", ":"))


def paired_answers(protocol, case_id):
    """Execute the two fixed models for one declared case, without scoring it.

    A new mapping is returned on every call. The canonical calculation cache
    retains only immutable serialized operands/results, never caller objects.
    """
    validate_protocol(protocol)
    protocol_json = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    return json.loads(_paired_answers_json(protocol_json, case_id))


def result_key(assessment, circumstance):
    value = json.dumps([assessment, circumstance], separators=(",", ":"))
    return "importance-result:" + sha256(value.encode()).hexdigest()[:24]


def validate_result(data, protocol):
    if set(data) != set(RESULT_FIELDS) or data["circumstance"] not in {
            case["id"] for case in protocol["domain"]["cases"]}:
        raise ValueError("Result must name a declared circumstance")
    for label in ("baseline", "altered"):
        value = data[label]
        if value is None:
            continue
        if not isinstance(value, dict) or set(value) != {"type", "value"}:
            raise ValueError("Answer needs a semantic type and value")
        if value["type"] == "boolean" and type(value["value"]) is bool:
            continue
        if value["type"] != "rational" or str(rational(value["value"])) != value["value"]:
            raise ValueError("Answer must be a Boolean or canonical exact rational")
    if not isinstance(data["error"], str) or bool(data["error"]) != (
            data["baseline"] is None or data["altered"] is None):
        raise ValueError("Unresolved calculations need an error; complete paired results cannot have one")
    expected = paired_answers(protocol, data["circumstance"])
    if {name: data[name] for name in ("baseline", "altered", "error")} != expected:
        raise ValueError("Calculation receipt does not reproduce the declared protocol and circumstance")


def score_importance(sheet, assessment, *, _pins=None):
    protocol = sheet.rows[assessment]["data"]
    cases = protocol["domain"]["cases"]
    total = sum((rational(case["weight"]) for case in cases), Fraction())
    changed = unchanged = Fraction()
    unresolved = []
    for case in cases:
        key = result_key(assessment, case["id"])
        row = sheet.rows.get(key)
        if row is None or not sheet.usable(key, _pins=_pins) or row["data"]["error"]:
            unresolved.append(case["id"])
            continue
        if row["data"]["baseline"] != row["data"]["altered"]:
            changed += rational(case["weight"])
        else:
            unchanged += rational(case["weight"])
    unknown = total - changed - unchanged
    return {"application": protocol["application"], "method": protocol["method"],
            "quantity": "finite_domain_importance" if protocol["domain"]["kind"] == "finite" else "sample_change_frequency",
            "total_mass": str(total), "changed_mass": str(changed), "unchanged_mass": str(unchanged),
            "unresolved_mass": str(unknown), "unresolved_cases": unresolved,
            "lower": str(changed / total), "upper": str((changed + unknown) / total),
            "value": str(changed / total) if not unknown else None,
            "resolved_fraction": str((changed + unchanged) / total),
            "resolution": str(min(rational(case["weight"]) for case in cases) / total),
            "scope": protocol["domain"]["description"], "measure": protocol["domain"]["measure"],
            "question": protocol["question"], "contrast": protocol["contrast"],
            "comparator": protocol["comparator"], "current": sheet.usable(assessment, _pins=_pins),
            "meaning": "Conditional on the declared model and comparison. Bounds cover unresolved declared circumstances only; "
                       "a sample frequency is not a certified continuous-space volume or population confidence interval."}


def protocol_schema():
    """Concrete model-authored operands; calculated results are never a model field."""
    text = {"type": "string", "minLength": 1}
    number = {"oneOf": [{"type": "integer"}, {"type": "string", "minLength": 1, "maxLength": 256}]}
    def obj(properties):
        return {"type": "object", "additionalProperties": False,
                "required": list(properties), "properties": properties}
    return obj({
        "application": text, "method": {"enum": list(METHODS)},
        "question": text, "contrast": text, "comparator": {"const": "exact_typed_answer"},
        "domain": obj({"kind": {"enum": ["finite", "sample"]}, "description": text, "measure": text,
            "cases": {"type": "array", "minItems": 1, "maxItems": 256, "items": obj({
                "id": text, "inputs": {"type": "object", "maxProperties": 32, "additionalProperties": number},
                "weight": number})}}),
        "model": obj({"equations": {"type": "object", "minProperties": 1, "maxProperties": 32,
                                    "additionalProperties": {"type": "string", "minLength": 1, "maxLength": 2048}},
            "focal": text, "replacement": text, "answer": text,
            "initial": {"type": "object", "maxProperties": 32, "additionalProperties": text},
            "steps": {"type": "integer", "minimum": 0, "maximum": 1024},
            "edge": {"type": "array", "maxItems": 2, "items": text}})})


def protocol_identity(protocol):
    """Ignore labels, prose and uniform weight scaling when deduplicating calculations."""
    total = sum((rational(c["weight"]) for c in protocol["domain"]["cases"]), Fraction())
    cases = [{"inputs": {k: str(rational(v)) for k, v in c["inputs"].items()},
              "weight": str(rational(c["weight"]) / total)} for c in protocol["domain"]["cases"]]
    cases.sort(key=lambda c: json.dumps(c, sort_keys=True))
    operands = {"application": protocol["application"], "method": protocol["method"],
                "kind": protocol["domain"]["kind"], "cases": cases,
                "model": protocol["model"], "comparator": protocol["comparator"]}
    return sha256(json.dumps(operands, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
