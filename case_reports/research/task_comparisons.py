"""Executed, task-specific consequences of retaining or altering G.

ODE event times and explicit equations share a declared observable and units.
Numerical refinement estimates are not rigorous enclosures or empirical truth.
Expected generalizability bounds label the comparison; they do not exclude useful
extensions. No model-authored success label or numerical merit grade is accepted.
"""
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache
import ast
import json
import math
import operator

from jsonschema import Draft202012Validator, ValidationError

from .importance import rational
from .records import ResearchResult, identity


FIELDS = ("inference", "expectation", "original", "candidate", "reference", "task",
          "domain", "scope", "argument", "limitations")
RESULT_FIELDS = ("assessment", "case_id", "original", "candidate", "reference", "error")
ROLES = ("original", "candidate", "reference")
_TEXT = {"type": "string", "minLength": 1}
_FUNCTIONS = {"abs": abs, "min": min, "max": max, "sqrt": math.sqrt,
              "exp": math.exp, "log": math.log, "tanh": math.tanh}
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Pow: operator.pow}


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def record_schema(snapshot):
    expressions = {"type": "object", "additionalProperties": _TEXT}
    model = {"oneOf": [
        _object({"kind": {"const": "equations"}, "source": snapshot,
                 "equations": expressions, "answer": _TEXT}),
        _object({"kind": {"const": "ode"}, "source": snapshot,
                 "equations": expressions, "initial": expressions,
                 "event": _object({"expression": _TEXT, "direction": {"enum": [-1, 0, 1]}}),
                 "time": _object({"start": _TEXT, "end": _TEXT}),
                 "solver": _object({"rtol": _TEXT, "atol": _TEXT})})]}
    return _object({"inference": snapshot, "expectation": snapshot,
        **{role: model for role in ROLES},
        "task": _object({"quantity": _TEXT, "units": _TEXT,
                         "absolute_tolerance": _TEXT, "relative_tolerance": _TEXT}),
        "domain": _object({"kind": {"enum": ["finite", "sample"]},
            "description": _TEXT, "measure": _TEXT,
            "cases": {"type": "array", "minItems": 1, "items": _object({
                "id": _TEXT, "inputs": {"type": "object", "additionalProperties": {
                    "oneOf": [{"type": "integer"}, _TEXT]}},
                "weight": {"oneOf": [{"type": "integer"}, _TEXT]}})}}),
        "scope": {"type": "string", "description":
            "Optional Boolean expression for membership in the ORIGINAL expected generalizability bounds. "
            "Empty means not numerically classified. Membership never gates task credit; do not invent a cutoff for <<."},
        "argument": _TEXT, "limitations": _TEXT})


@lru_cache(maxsize=256)
def _tree(text):
    try: tree = ast.parse(text.strip(), mode="eval").body
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Task expressions must be valid mathematical expressions") from exc
    def check(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, bool):
            return
        if isinstance(node, ast.Name) and node.id.isidentifier():
            return
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            check(node.left); check(node.right); return
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
            check(node.operand); return
        if isinstance(node, ast.Compare) and all(isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)) for op in node.ops):
            check(node.left)
            for child in node.comparators: check(child)
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            for child in node.values: check(child)
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
            for child in node.args: check(child)
            return
        raise ValueError("Task expressions accept only declared scalars and mathematical operations")
    check(tree)
    return tree


def _value(text, values):
    def visit(node):
        if isinstance(node, ast.Constant): return node.value
        if isinstance(node, ast.Name): return values[node.id]
        if isinstance(node, ast.BinOp): return _BINOPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else +value if isinstance(node.op, ast.UAdd) else not value
        if isinstance(node, ast.Call): return _FUNCTIONS[node.func.id](*(visit(child) for child in node.args))
        if isinstance(node, ast.BoolOp):
            return all(visit(child) for child in node.values) if isinstance(node.op, ast.And) else any(visit(child) for child in node.values)
        left = visit(node.left)
        comparisons = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
                       ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne}
        for op, child in zip(node.ops, node.comparators):
            right = visit(child)
            if not comparisons[type(op)](left, right): return False
            left = right
        return True
    value = visit(_tree(text))
    if isinstance(value, complex) or not math.isfinite(value):
        raise ValueError("Task expression produced a nonfinite or nonreal answer")
    return value


def references(data):
    return [data["argument"]]


def snapshot_references(data):
    return [data["inference"], data["expectation"], *(data[role]["source"] for role in ROLES)]


def context_pins(sheet, data):
    pins = {}
    for ref in snapshot_references(data): pins.setdefault(ref["key"], set()).add(ref["version"])
    return pins


def validate(sheet, key, data):
    try:
        Draft202012Validator(record_schema({"type": "object", "additionalProperties": False,
            "required": ["key", "version"], "properties": {"key": _TEXT, "version": _TEXT}})).validate(data)
    except ValidationError as exc:
        raise ValueError(f"Task comparison typing at {list(exc.absolute_path)}: {exc.message}") from exc
    g = sheet.snapshot(data["inference"])
    expected = sheet.snapshot(data["expectation"])
    if g["kind"] != "inference" or expected["kind"] != "expectation" or expected["data"]["inference"] != g["key"]:
        raise ValueError("Task comparison requires the exact G and its original expected generalizability bounds")
    if expected["dependencies"][g["key"]] != g["version"]:
        raise ValueError("Task comparison G and expected-bounds snapshots must agree")
    sheet._require_kind(data["argument"], "reasoning")
    for ref in snapshot_references(data):
        row = sheet.snapshot(ref)
        if row["kind"] not in {"inference", "expectation", "theory_fragment", "source_formalization", "premise", "reasoning"}:
            raise ValueError("Task models require scientific definitions, not a verdict or raw evidence")
        if not sheet._depends_on(data["argument"], ref["key"]):
            raise ValueError("The public alteration argument must address G, its bounds and every model source")
    for field in ("absolute_tolerance", "relative_tolerance"):
        if rational(data["task"][field]) < 0:
            raise ValueError("Task accuracy tolerances must be nonnegative")
    ids, cases, names = set(), set(), None
    for case in data["domain"]["cases"]:
        inputs = case["inputs"]
        canonical = tuple(sorted((name, str(rational(value))) for name, value in inputs.items()))
        if case["id"] in ids or canonical in cases:
            raise ValueError("Duplicate cases or aliases cannot create another task comparison")
        if any(not name.isidentifier() or name in _FUNCTIONS or name == "t" for name in inputs):
            raise ValueError("Case inputs must be named scalars distinct from mathematical functions and t")
        if names is not None and set(inputs) != names:
            raise ValueError("Every task circumstance supplies the same input names")
        names = set(inputs); ids.add(case["id"]); cases.add(canonical)
        if rational(case["weight"]) <= 0:
            raise ValueError("Case weights must be positive")
    if data["scope"]: _tree(data["scope"])
    for role in ROLES:
        model = data[role]
        equations = model["equations"]
        if any(not name.isidentifier() or name in names or name in _FUNCTIONS or name == "t" for name in equations):
            raise ValueError("Model quantities must be identifiers distinct from parameters and functions")
        expressions = list(equations.values())
        if model["kind"] == "ode":
            if not equations or set(model["initial"]) != set(equations):
                raise ValueError("Every ODE state requires one derivative and one initial expression")
            expressions += [*model["initial"].values(), model["event"]["expression"], *model["time"].values()]
            if any(not 0 < float(rational(value)) < 1 for value in model["solver"].values()):
                raise ValueError("ODE tolerances must be positive precision controls smaller than one")
        else:
            expressions.append(model["answer"])
        for text in expressions: _tree(text)
    prior = sheet.rows.get(key)
    if prior and prior["data"] != data:
        raise ValueError("A task comparison fixes its models, cases, targets and original bounds before execution")


def _solve(model, parameters):
    if model["kind"] == "equations":
        values, pending = dict(parameters), dict(model["equations"])
        while pending:
            progressed = False
            for name, text in list(pending.items()):
                try: value = _value(text, values)
                except KeyError: continue
                values[name] = value; del pending[name]; progressed = True
            if not progressed: raise ValueError("Static task equations have an unresolved dependency or cycle")
        value = float(_value(model["answer"], values))
        return {"status": "computed", "value": value, "numerical_uncertainty": 1e-14 * max(1, abs(value)),
                "method": "explicit_equations"}
    from scipy.integrate import solve_ivp
    names = list(model["equations"])
    initial = [float(_value(model["initial"][name], parameters)) for name in names]
    start, end = (float(_value(model["time"][name], parameters)) for name in ("start", "end"))
    if end <= start: raise ValueError("An ODE experiment requires an increasing physical-time interval")
    def values(t, y): return {**parameters, "t": t, **dict(zip(names, y))}
    def rhs(t, y): return [_value(model["equations"][name], values(t, y)) for name in names]
    def event(t, y): return _value(model["event"]["expression"], values(t, y))
    direction = model["event"]["direction"]
    initial_event = event(start, initial)
    if direction and initial_event * direction > 0:
        return {"status": "not_represented", "reason": "The reduced initial state is already beyond the requested directional crossing"}
    event.terminal = True
    event.direction = direction
    times = []
    rtol, atol = (float(rational(model["solver"][name])) for name in ("rtol", "atol"))
    for precision in (1, 0.1):
        solved = solve_ivp(rhs, (start, end), initial, events=event, method="DOP853",
                          rtol=rtol * precision, atol=atol * precision)
        if not solved.success: return {"status": "unresolved", "reason": solved.message}
        if not len(solved.t_events[0]):
            return {"status": "unresolved", "reason": "No crossing was resolved in the declared physical experiment interval; this is not evidence of absence"}
        times.append(float(solved.t_events[0][0]) - start)
    error = abs(times[1] - times[0]) + atol + rtol * abs(times[1])
    return {"status": "computed", "value": times[1], "numerical_uncertainty": error,
            "method": "DOP853_event_with_tolerance_refinement"}


@lru_cache(maxsize=256)
def _execute(encoded):
    data, case = json.loads(encoded)
    parameters = {name: float(rational(value)) for name, value in case["inputs"].items()}
    results = {}
    for role in ROLES:
        try: results[role] = _solve(data[role], parameters)
        except (ValueError, KeyError, ZeroDivisionError, OverflowError, TypeError) as exc:
            results[role] = {"status": "unresolved", "reason": str(exc)}
    return results


def execute_case(data, case):
    # Prose, IDs and record versions do not alter the actual numerical operation.
    models = {role: {name: value for name, value in data[role].items() if name != "source"} for role in ROLES}
    return deepcopy(_execute(json.dumps([models, {"inputs": case["inputs"]}], sort_keys=True)))


def evaluate_cases(sheet, assessment, case_ids):
    data = sheet.rows[assessment]["data"]
    sheet._require_kind(assessment, "causal_change")
    cases = {case["id"]: case for case in data["domain"]["cases"]}
    if not case_ids or len(set(case_ids)) != len(case_ids) or not set(case_ids) <= cases.keys():
        raise ValueError("Select distinct circumstances declared in the task comparison")
    updates = []
    for case_id in case_ids:
        key = "task-result:" + identity([sheet.anchor(assessment), case_id])
        if key in sheet.rows:
            updates.append(sheet.rows[key]); continue
        answers = execute_case(data, cases[case_id])
        updates.append(sheet.put(key, "task_result", {"assessment": assessment, "case_id": case_id,
            **answers, "error": ""}))
    return tuple(updates)


def validate_result(sheet, data):
    if set(data) != set(RESULT_FIELDS): raise ValueError("Task results retain exactly the executed fields")
    sheet._require_kind(data["assessment"], "causal_change")
    assessment = sheet.rows[data["assessment"]]["data"]
    cases = {case["id"]: case for case in assessment["domain"]["cases"]}
    if data["case_id"] not in cases or data["error"]:
        raise ValueError("Task result must identify a declared executed circumstance")
    if execute_case(assessment, cases[data["case_id"]]) != {role: data[role] for role in ROLES}:
        raise ValueError("Task result does not reproduce its declared scientific models")


def _error(prediction, reference, task):
    if reference["status"] != "computed": return None
    if prediction["status"] == "not_represented": return {"passes": False, "relative": None}
    if prediction["status"] != "computed": return None
    ref, value = reference["value"], prediction["value"]
    uncertainty = reference["numerical_uncertainty"] + prediction["numerical_uncertainty"]
    tolerance = float(rational(task["absolute_tolerance"])) + float(rational(task["relative_tolerance"])) * abs(ref)
    distance = abs(value - ref)
    lower, upper = max(0.0, distance - uncertainty), distance + uncertainty
    passes = True if upper <= tolerance else False if lower > tolerance else None
    return {"passes": passes, "relative": distance / abs(ref) if ref else None,
            "distance": distance, "lower": lower, "upper": upper, "tolerance": tolerance}


def model_identity(model):
    def canonical(item):
        if isinstance(item, dict):
            return {key: canonical({"start": value["start"]} if key == "time" else value)
                    for key, value in item.items() if key not in {"source", "solver"}}
        if isinstance(item, str):
            try: return ast.dump(_tree(item))
            except ValueError: return item
        return item
    return canonical(model)


def point_identity(sheet, data, case):
    from ...credit_assignment.causal_progress import _inference_identity
    original, reference = model_identity(data["original"]), model_identity(data["reference"])
    expressions = [*data["original"]["equations"].values(), *data["reference"]["equations"].values()]
    for role in ("original", "reference"):
        value = data[role]
        expressions += ([value["answer"]] if value["kind"] == "equations" else
                        [*value["initial"].values(), value["event"]["expression"], *value["time"].values()])
    used = {node.id for text in expressions for node in ast.walk(_tree(text)) if isinstance(node, ast.Name)}
    return identity([_inference_identity(sheet.snapshot(data["inference"])), data["task"], original, reference,
                     {name: str(rational(value)) for name, value in case["inputs"].items() if name in used}])


def grade(sheet, key):
    data = sheet.rows[key]["data"]
    rows, points, masses = [], {}, {}
    for case in data["domain"]["cases"]:
        result_key = "task-result:" + identity([sheet.anchor(key), case["id"]])
        result = sheet.rows[result_key]["data"] if result_key in sheet.rows else None
        original = _error(result["original"], result["reference"], data["task"]) if result else None
        candidate = _error(result["candidate"], result["reference"], data["task"]) if result else None
        membership = None
        if data["scope"]:
            try:
                answer = _value(data["scope"], {name: float(rational(value)) for name, value in case["inputs"].items()})
                membership = answer if type(answer) is bool else None
            except (ValueError, KeyError, ZeroDivisionError): pass
        difference, improvement = Fraction(), Fraction()
        if original and original["passes"] is False:
            difference = Fraction(1)
            if candidate and candidate["passes"] is not None:
                if original["relative"] is None:
                    improvement = Fraction(int(candidate["passes"] is True))
                elif candidate.get("upper") is not None and original["lower"] > 0:
                    # Conservative progress relative to the original task error.
                    gain = max(0.0, min(1.0, 1 - candidate["upper"] / original["lower"]))
                    improvement = Fraction(str(gain))
        ident = point_identity(sheet, data, case)
        if difference: points["difference:" + ident] = str(difference)
        if improvement: points["change:" + ident] = str(improvement)
        classification = ("unresolved" if not original or not candidate or original["passes"] is None or candidate["passes"] is None else
            "retained" if original["passes"] and candidate["passes"] else
            "lost" if original["passes"] else "restored" if candidate["passes"] else
            "improved_still_failed" if improvement else "still_failed")
        rows.append({"case_id": case["id"], "original_expected_scope": membership,
            "original": original, "candidate": candidate, "classification": classification,
            "representation": {role: result[role]["status"] for role in ROLES} if result else {},
            "difference_credit": str(difference), "improvement_credit": str(improvement)})
        masses[classification] = masses.get(classification, Fraction()) + rational(case["weight"])
    return {"assessment": key, "cases": rows, "points": points,
            "candidate_identity": identity(model_identity(data["candidate"])),
            "masses": {name: str(value) for name, value in masses.items()},
            "total": str(sum(map(Fraction, points.values()), Fraction())),
            "qualification": "Executed model comparison with numerical refinement estimates, not rigorous integration bounds or empirical validation. "
                "Original expected-scope membership is reported without excluding extensions. Unrepresented and unresolved targets are distinct. "
                "Case counts describe only the declared finite or sampled circumstances, not continuous-domain coverage."}
