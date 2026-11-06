"""Typed, append-only fragments of a scientific theory.

This is a representation and validation library, not a research controller or
an evidential grader. References identify declarations, never names guessed
from prose. Exact rational arithmetic has a conservative normal form; a failure
to match normal forms does not establish scientific or algebraic disagreement.
"""
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import json

from jsonschema import Draft202012Validator, ValidationError


GROUPS = ("entities", "quantities", "statements", "gaps")
STATEMENT_KINDS = ("equation", "causal_relation", "mechanism", "assumption",
                   "derivation", "approximation")
COMPONENT_KINDS = ("entity", "quantity", *STATEMENT_KINDS, "gap")
REFERENCE_KINDS = ("entity", "quantity", "statement", "gap")
RELATIONS = ("eq", "ne", "lt", "le", "gt", "ge")
CAUSAL_RELATIONS = ("increases", "decreases", "enables", "inhibits",
                    "necessary_for", "sufficient_for", "affects")
_GROUP_KIND = dict(zip(GROUPS, REFERENCE_KINDS))
_TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
_RATIONAL = {"oneOf": [{"type": "integer"}, {"type": "string", "minLength": 1}]}


def _object(properties, optional=()):
    return {"type": "object", "additionalProperties": False,
            "required": [key for key in properties if key not in optional],
            "properties": properties}


def _array(items, *, nonempty=False):
    result = {"type": "array", "items": items, "uniqueItems": True}
    if nonempty:
        result["minItems"] = 1
    return result


def _definitions():
    reference = _object({"kind": {"enum": list(REFERENCE_KINDS)}, "id": _TEXT})
    evidence = _object({"key": _TEXT,
                        "version": {"type": "string", "pattern": "^[0-9a-f]{64}$"}})
    refs = _array({"$ref": "#/$defs/reference"})
    expr = {"$ref": "#/$defs/expression"}
    boolean = {"$ref": "#/$defs/boolean"}
    args = {"type": "object", "propertyNames": _TEXT, "additionalProperties": expr}
    expressions = [
        _object({"op": {"const": "quantity"}, "ref": _TEXT, "bindings": args}),
        _object({"op": {"const": "call"}, "function": _TEXT, "bindings": args}),
        _object({"op": {"const": "number"}, "value": _RATIONAL}),
        _object({"op": {"enum": ["add", "multiply"]},
                 "args": {"type": "array", "minItems": 2, "items": expr}}),
        _object({"op": {"enum": ["subtract", "divide", "power"]},
                 "left": expr, "right": expr}),
        _object({"op": {"const": "derivative"}, "expression": expr,
                 "coordinate": _TEXT, "order": {"type": "integer", "minimum": 1}}),
    ]
    booleans = [
        _object({"op": {"const": "boolean"}, "value": {"type": "boolean"}}),
        _object({"op": {"const": "compare"}, "relation": {"enum": list(RELATIONS)},
                 "left": expr, "right": expr}),
        _object({"op": {"enum": ["and", "or"]},
                 "args": {"type": "array", "minItems": 2, "items": boolean}}),
        _object({"op": {"const": "not"}, "argument": boolean}),
    ]
    common = {"definition": _TEXT, "native_id": {"type": "string"},
              "evidence": _array({"$ref": "#/$defs/evidence"}),
              "supersedes": _array(_TEXT)}
    entity = _object({**common, "kind": _TEXT})
    quantity = _object({**common,
        "role": {"enum": ["coordinate", "state", "parameter", "constant"]},
        "entity": _TEXT, "property": _TEXT,
        "value_type": {"enum": ["scalar", "field", "function"]},
        "units": _TEXT,
        "bindings": {"type": "object", "propertyNames": _TEXT, "additionalProperties": _TEXT},
        "value": _RATIONAL}, optional=("value",))
    quantity["allOf"] = [{"if": {"properties": {"role": {"const": "constant"}}},
                           "then": {"required": ["value"]},
                           "else": {"not": {"required": ["value"]}}}]
    scope = _object({"entities": _array(_TEXT), "conditions": _array(_TEXT),
                     "description": {"type": "string"}})
    domain = _array(_object({"quantity": _TEXT, "relation": {"enum": list(RELATIONS)},
                             "value": _RATIONAL}))
    statement = {**common, "scope": scope, "domain": domain, "references": refs}
    assertions = {"oneOf": [
        _object({"form": {"const": "expression"}, "expression": boolean}),
        _object({"form": {"const": "prose"}, "text": _TEXT, "references": refs}),
    ]}
    variants = {
        "equation": {"lhs": expr, "rhs": expr},
        "causal_relation": {"cause": _TEXT, "effect": _TEXT,
                            "relation": {"enum": list(CAUSAL_RELATIONS)}},
        "mechanism": {"inputs": refs,
                      "outputs": _array(_object({
                          "kind": {"enum": ["entity", "quantity", "statement"]},
                          "id": _TEXT}), nonempty=True),
                      "steps": _array(_TEXT, nonempty=True)},
        "assumption": {"assertion": assertions},
        "derivation": {"premises": _array(_TEXT, nonempty=True), "conclusion": _TEXT,
                       "steps": _array(_TEXT)},
        "approximation": {"original": _TEXT, "approximation": _TEXT,
                          "regime": _array(_TEXT, nonempty=True)},
    }
    statements = [_object({**statement, "kind": {"const": kind}, **fields})
                  for kind, fields in variants.items()]
    gap = _object({**common, "question": _TEXT, "components": refs,
                   "requested_kinds": _array({"enum": list(COMPONENT_KINDS)}, nonempty=True),
                   "resolutions": refs})
    return {"reference": reference, "evidence": evidence,
            "expression": {"oneOf": expressions}, "boolean": {"oneOf": booleans},
            "entity": entity, "quantity": quantity,
            "statement": {"oneOf": statements}, "gap": gap}


def _schema(*, fragment):
    properties = {group: {"type": "object", "propertyNames": _TEXT,
                          "additionalProperties": {"$ref": f"#/$defs/{kind}"}}
                  for group, kind in _GROUP_KIND.items()}
    if fragment:
        properties = {"reconstruction": _TEXT, "question": _TEXT, **properties}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema",
            "$id": "urn:think-and-find:theory-fragment" if fragment else "urn:think-and-find:theory-additions",
            **_object(properties), "$defs": _definitions()}


def fragment_schema():
    """A closed JSON schema; empty maps are valid incomplete reconstructions."""
    return _schema(fragment=True)


def additions_schema():
    """The same language, with only the four append-only component maps."""
    return _schema(fragment=False)


def empty_fragment(reconstruction, question):
    return validate_fragment({"reconstruction": reconstruction, "question": question,
                              **{group: {} for group in GROUPS}})


def component_kind(component, map_name):
    if map_name not in _GROUP_KIND:
        raise ValueError(f"Unknown component map: {map_name}")
    return component["kind"] if map_name == "statements" else _GROUP_KIND[map_name]


def _validate_shape(value, schema):
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as exc:
        path = ".".join(map(str, exc.absolute_path)) or "fragment"
        raise ValueError(f"{path}: {exc.message}") from exc


def _rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("Exact numbers must be integers or rational/decimal strings")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid exact rational: {value!r}") from exc


def _dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return sha256(_dump(value).encode()).hexdigest()


def _text(value):
    return " ".join(value.split())


def _index(fragment):
    return {key: (group, value) for group in GROUPS for key, value in fragment[group].items()}


def collect_component_refs(component):
    """Return exact typed graph edges, including expression/signature bindings.

    Supersedes is deliberately omitted: it is audit ancestry, not a scientific
    premise or a link that makes an unrelated request relevant.
    """
    result = {}

    def add(kind, key):
        result[(kind, key)] = {"kind": kind, "id": key}

    def walk(value):
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            if set(value) == {"kind", "id"}:
                add(value["kind"], value["id"])
                return
            op = value.get("op")
            if op == "quantity":
                add("quantity", value["ref"])
            elif op == "call":
                add("quantity", value["function"])
            elif op == "derivative":
                add("quantity", value["coordinate"])
            for key, item in value.items():
                if key not in {"evidence", "supersedes"}:
                    walk(item)

    walk(component)
    if "role" in component:
        add("entity", component["entity"])
        for key in component["bindings"].values():
            add("quantity", key)
    if "scope" in component:
        for key in component["scope"]["entities"]:
            add("entity", key)
        for key in component["scope"]["conditions"]:
            add("statement", key)
        for constraint in component["domain"]:
            add("quantity", constraint["quantity"])
    kind = component.get("kind")
    if kind == "causal_relation" and "cause" in component:
        for key in (component["cause"], component["effect"]):
            add("quantity", key)
    elif kind == "mechanism" and "steps" in component:
        for key in component["steps"]:
            add("statement", key)
    elif kind == "derivation" and "premises" in component:
        for key in [*component["premises"], component["conclusion"], *component["steps"]]:
            add("statement", key)
    elif kind == "approximation" and "original" in component:
        for key in [component["original"], component["approximation"], *component["regime"]]:
            add("statement", key)
    return [result[key] for key in sorted(result)]


def evidence_references(fragment):
    """Read every declared immutable evidence reference without resolving it."""
    _validate_shape(fragment, fragment_schema())
    refs = {(ref["key"], ref["version"]): ref for group in GROUPS
            for component in fragment[group].values() for ref in component["evidence"]}
    return [deepcopy(refs[key]) for key in sorted(refs)]


def _evidence_data(reference, evidence_lookup):
    data = evidence_lookup(deepcopy(reference))
    if not isinstance(data, dict):
        raise ValueError("Evidence lookup must return source evidence data")
    # Kind/version validation belongs to the sheet's immutable snapshot lookup.
    # These fields prevent a truth flag or a fabricated empty record becoming
    # an evidential anchor merely because a callback returned it.
    for key in ("source", "locator", "study_id", "finding"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"Evidence data requires nonempty {key}")
    if not isinstance(data.get("measurements"), dict):
        raise ValueError("Evidence data requires source measurements")
    return data


def validate_fragment(fragment, evidence_lookup=None):
    """Check syntax, exact references and signatures; return an independent copy.

    This validates a representation, not whether a paper supports its content.
    When supplied, evidence_lookup must resolve immutable evidence key/version
    references and reject references to other record kinds or unknown versions.
    """
    _validate_shape(fragment, fragment_schema())
    ids = [key for group in GROUPS for key in fragment[group]]
    if len(ids) != len(set(ids)):
        raise ValueError("Component IDs must be unique across all four maps")
    index = _index(fragment)
    quantities = fragment["quantities"]

    def require(key, kind):
        if key not in index or _GROUP_KIND[index[key][0]] != kind:
            raise ValueError(f"Reference {key!r} must identify a declared {kind}")
        return index[key][1]

    def expression(node):
        op = node["op"]
        if op == "number":
            _rational(node["value"])
        elif op in {"quantity", "call"}:
            quantity = require(node["ref" if op == "quantity" else "function"], "quantity")
            if (op == "call") != (quantity["value_type"] == "function"):
                raise ValueError("Function quantities use call; scalar/field quantities use quantity")
            if set(node["bindings"]) != set(quantity["bindings"]):
                raise ValueError("Expression bindings must exactly match the quantity signature")
            for argument in node["bindings"].values():
                expression(argument)
        elif op in {"add", "multiply"}:
            for argument in node["args"]:
                expression(argument)
        elif op in {"subtract", "divide", "power"}:
            expression(node["left"])
            expression(node["right"])
            if op == "divide" and node["right"]["op"] == "number" and not _rational(node["right"]["value"]):
                raise ValueError("Division by zero is not a theory expression")
        elif op == "derivative":
            expression(node["expression"])
            if type(node["order"]) is not int:
                raise ValueError("Derivative order must be an integer")
            if require(node["coordinate"], "quantity")["role"] != "coordinate":
                raise ValueError("Derivative coordinate must have role coordinate")

    def boolean(node):
        op = node["op"]
        if op == "compare":
            expression(node["left"])
            expression(node["right"])
        elif op in {"and", "or"}:
            for child in node["args"]:
                boolean(child)
        elif op == "not":
            boolean(node["argument"])

    for key, (group, component) in index.items():
        for ref in collect_component_refs(component):
            require(ref["id"], ref["kind"])
        for old in component["supersedes"]:
            if old == key or old not in index or index[old][0] != group:
                raise ValueError("Supersedes must reference an older, different component of the same map")
            if group == "statements" and index[old][1]["kind"] != component["kind"]:
                raise ValueError("A statement revision retains its statement kind")
        if group == "quantities":
            bindings = component["bindings"]
            if component["value_type"] == "scalar" and bindings:
                raise ValueError("Scalar quantities have no coordinate bindings")
            if component["value_type"] != "scalar" and not bindings:
                raise ValueError("Field/function quantities declare their coordinate bindings")
            if component["role"] in {"coordinate", "constant"} and component["value_type"] != "scalar":
                raise ValueError("Coordinates and constants are scalar quantities")
            if len(bindings.values()) != len(set(bindings.values())):
                raise ValueError("Each formal argument binds a distinct declared coordinate")
            for coordinate in bindings.values():
                if require(coordinate, "quantity")["role"] != "coordinate":
                    raise ValueError("Quantity bindings must identify coordinate quantities")
            if component["role"] == "constant":
                _rational(component["value"])
        elif group == "statements":
            for constraint in component["domain"]:
                if quantities[constraint["quantity"]]["value_type"] != "scalar":
                    raise ValueError("Domain bounds reference scalar quantities")
                _rational(constraint["value"])
            if component["kind"] == "equation":
                expression(component["lhs"])
                expression(component["rhs"])
            elif component["kind"] == "assumption" and component["assertion"]["form"] == "expression":
                boolean(component["assertion"]["expression"])
        elif group == "gaps":
            for ref in component["resolutions"]:
                resolved = require(ref["id"], ref["kind"])
                if ref["kind"] == "gap" or not resolved["evidence"]:
                    raise ValueError("A gap resolution must reference a source-grounded non-gap component")
                resolved_kind = component_kind(resolved, index[ref["id"]][0])
                if resolved_kind not in component["requested_kinds"]:
                    raise ValueError("Gap resolutions must have a requested component kind")

    visiting, visited = set(), set()

    def revision(key):
        if key in visiting:
            raise ValueError("Supersedes links must not contain a cycle")
        if key in visited:
            return
        visiting.add(key)
        for old in index[key][1]["supersedes"]:
            revision(old)
        visiting.remove(key)
        visited.add(key)

    for key in index:
        revision(key)
    if evidence_lookup is not None:
        for reference in evidence_references(fragment):
            _evidence_data(reference, evidence_lookup)
    # Reject algebraically undefined operands before admitting a fragment, not
    # later when a contribution projection happens to encounter them.
    normalizer = _Normalizer(fragment, evidence_lookup)
    for key in fragment["statements"]:
        normalizer.component(key)
    return deepcopy(fragment)


def apply_patch(fragment, additions, evidence_lookup=None):
    """Append declarations atomically; an existing definition cannot be replaced."""
    validate_fragment(fragment, evidence_lookup=evidence_lookup)
    _validate_shape(additions, additions_schema())
    updated = deepcopy(fragment)
    for group in GROUPS:
        for key, value in additions[group].items():
            if key in updated[group] and updated[group][key] != value:
                raise ValueError(f"Cannot overwrite {key!r}; add a new ID with an explicit supersedes link")
            updated[group][key] = deepcopy(value)
    return validate_fragment(updated, evidence_lookup=evidence_lookup)


# Polynomials are {monomial: Fraction}; monomials are sorted (symbol, power)
# tuples. Symbols are complete bound-quantity meanings, not display names.
def _clean(poly):
    return {monomial: coefficient for monomial, coefficient in poly.items() if coefficient}


def _add(left, right):
    result = dict(left)
    for monomial, coefficient in right.items():
        result[monomial] = result.get(monomial, Fraction()) + coefficient
    return _clean(result)


def _scale(poly, coefficient):
    return _clean({monomial: value * coefficient for monomial, value in poly.items()})


def _multiply(left, right):
    result = {}
    for a, x in left.items():
        for b, y in right.items():
            powers = dict(a)
            for symbol, exponent in b:
                powers[symbol] = powers.get(symbol, 0) + exponent
            monomial = tuple(sorted(powers.items()))
            result[monomial] = result.get(monomial, Fraction()) + x * y
    return _clean(result)


def _power(poly, exponent):
    result = {(): Fraction(1)}
    while exponent:
        if exponent % 2:
            result = _multiply(result, poly)
        poly = _multiply(poly, poly) if exponent > 1 else poly
        exponent //= 2
    return result


def _poly_data(poly):
    return [[[[symbol, power] for symbol, power in monomial], str(coefficient)]
            for monomial, coefficient in sorted(poly.items())]


def _unit(poly):
    """Remove only a nonzero scalar factor, preserving the zero polynomial."""
    return _scale(poly, 1 / poly[sorted(poly)[-1]]) if poly else {}


def _divide_exact(numerator, denominator):
    """Exact polynomial long division, or None; no numeric approximations."""
    if not denominator:
        raise ValueError("Zero polynomial denominator")
    symbols = sorted({symbol for p in (numerator, denominator) for m in p for symbol, _ in m})

    def order(monomial):
        powers = dict(monomial)
        return sum(powers.values()), tuple(powers.get(symbol, 0) for symbol in symbols)

    lead = max(denominator, key=order)
    lead_powers = dict(lead)
    remainder, quotient = dict(numerator), {}
    while remainder:
        current = max(remainder, key=order)
        powers = dict(current)
        if any(powers.get(symbol, 0) < exponent for symbol, exponent in lead):
            return None
        monomial = tuple((symbol, exponent - lead_powers.get(symbol, 0))
                         for symbol, exponent in current if exponent != lead_powers.get(symbol, 0))
        coefficient = remainder[current] / denominator[lead]
        term = {monomial: coefficient}
        quotient = _add(quotient, term)
        remainder = _add(remainder, _scale(_multiply(term, denominator), -1))
    return quotient


def _rational_form(numerator, denominator, guards):
    if not denominator:
        raise ValueError("Expression denominator is identically zero")
    guard_map = {}
    for guard in guards:
        if not guard:
            raise ValueError("Expression requires an identically zero divisor to be nonzero")
        if set(guard) != {()}:
            normalized = _unit(guard)
            guard_map[_dump(_poly_data(normalized))] = normalized
    # Cancel factors whose nonzero requirement remains explicitly recorded.
    # Exact division also handles whole-denominator cancellation. No general
    # multivariate factorization or unproved analytic identity is claimed.
    for factor in [denominator, *[guard_map[key] for key in sorted(guard_map)]]:
        if set(factor) == {()}:
            continue
        while numerator and denominator:
            divided_denominator = _divide_exact(denominator, factor)
            divided_numerator = _divide_exact(numerator, factor)
            if divided_denominator is None or divided_numerator is None:
                break
            numerator, denominator = divided_numerator, divided_denominator
    if not numerator:
        denominator = {(): Fraction(1)}
    leading = denominator[sorted(denominator)[-1]]
    return (_scale(numerator, 1 / leading), _scale(denominator, 1 / leading),
            [guard_map[key] for key in sorted(guard_map)])


class _Normalizer:
    def __init__(self, fragment, evidence_lookup):
        self.fragment = fragment
        self.index = _index(fragment)
        self.lookup = evidence_lookup
        self.memo = {}
        self.visiting = set()
        self.evidence_cache = {}
        self.unresolved = set()

    def evidence(self, component):
        anchors = []
        for ref in component["evidence"]:
            key = (ref["key"], ref["version"])
            if key not in self.evidence_cache:
                if self.lookup is None:
                    anchor = ["snapshot", *key]
                else:
                    data = _evidence_data(ref, self.lookup)
                    measurements = data["measurements"]
                    anchor = ["source", data["source"], data["study_id"],
                              measurements.get("source_version", ""), data["locator"]]
                    # Without immutable source content identity, the exact
                    # snapshot version prevents silently merging source edits.
                    if not measurements.get("source_version"):
                        anchor.append(ref["version"])
                self.evidence_cache[key] = anchor
            anchors.append(self.evidence_cache[key])
        return sorted({_dump(anchor): anchor for anchor in anchors}.values(), key=_dump)

    def ref(self, reference):
        return self.component(reference["id"])

    def component(self, key):
        if key in self.memo:
            return self.memo[key]
        group, component = self.index[key]
        native = component["native_id"]
        if key in self.visiting:
            self.unresolved.add("cyclic_reference_equivalence")
            # A grounded native identifier can anchor a cycle. Otherwise keep
            # the ID explicitly: do not assert that two unnamed cycles agree.
            return ["cycle", _GROUP_KIND[group], native or key, self.evidence(component)]
        self.visiting.add(key)
        if group == "entities":
            meaning = ["entity", _text(component["kind"]), native or _text(component["definition"])]
            if native:
                meaning.append(self.evidence(component))
        elif group == "quantities":
            meaning = ["quantity", component["role"], self.component(component["entity"]),
                       native or _text(component["definition"]),
                       _text(component["property"]), component["value_type"], _text(component["units"]),
                       sorted([self.component(coordinate) for coordinate in component["bindings"].values()], key=_dump)]
            if native:
                meaning.append(self.evidence(component))
            if component["role"] == "constant":
                meaning.append(str(_rational(component["value"])))
        elif group == "gaps":
            meaning = ["gap", native or _text(component["question"]),
                       sorted([self.ref(ref) for ref in component["components"]], key=_dump),
                       sorted(component["requested_kinds"])]
        else:
            meaning = self.statement(component)
        self.visiting.remove(key)
        self.memo[key] = meaning
        return meaning

    def expression(self, node):
        """Return numerator, denominator, and nonzero domain requirements."""
        op = node["op"]
        one = {(): Fraction(1)}
        if op == "number":
            return _clean({(): _rational(node["value"])}), one, []
        if op in {"quantity", "call"}:
            key = node["ref" if op == "quantity" else "function"]
            quantity = self.fragment["quantities"][key]
            if quantity["role"] == "constant":
                return _clean({(): _rational(quantity["value"])}), one, []
            bindings, guards = [], []
            for name, argument in node["bindings"].items():
                a, b, extra = self.expression(argument)
                guards.extend(extra)
                bindings.append([self.component(quantity["bindings"][name]),
                                 [_poly_data(a), _poly_data(b)]])
            atom = [op, self.component(key), sorted(bindings, key=_dump)]
            if op == "call":
                self.unresolved.add("opaque_function_equivalence")
            return {((_dump(atom), 1),): Fraction(1)}, one, guards
        if op == "derivative":
            a, b, guards = self.expression(node["expression"])
            atom = [op, [_poly_data(a), _poly_data(b)],
                    self.component(node["coordinate"]), node["order"]]
            self.unresolved.add("derivative_equivalence")
            return {((_dump(atom), 1),): Fraction(1)}, one, guards
        if op in {"add", "multiply"}:
            numerator = {} if op == "add" else one
            denominator, guards = one, []
            for child in node["args"]:
                a, b, extra = self.expression(child)
                numerator = (_add(_multiply(numerator, b), _multiply(a, denominator))
                             if op == "add" else _multiply(numerator, a))
                denominator = _multiply(denominator, b)
                guards.extend(extra)
            return _rational_form(numerator, denominator, guards)
        a, b, left_guards = self.expression(node["left"])
        c, d, right_guards = self.expression(node["right"])
        guards = [*left_guards, *right_guards]
        if op == "subtract":
            return _rational_form(_add(_multiply(a, d), _scale(_multiply(c, b), -1)),
                                  _multiply(b, d), guards)
        if op == "divide":
            return _rational_form(_multiply(a, d), _multiply(b, c), [*guards, c])
        # Only exact, constant integral powers have rational polynomial meaning.
        exponent = (c.get((), Fraction()) / d[()] if set(c) <= {()} and set(d) == {()} else None)
        if exponent is not None and exponent.denominator == 1:
            n = int(exponent)
            if n < 0:
                return _rational_form(_power(b, -n), _power(a, -n), [*guards, a])
            return _rational_form(_power(a, n), _power(b, n), guards)
        self.unresolved.add("nonintegral_or_symbolic_power_equivalence")
        atom = ["power", [_poly_data(a), _poly_data(b)], [_poly_data(c), _poly_data(d)]]
        return {((_dump(atom), 1),): Fraction(1)}, one, guards

    def equation(self, lhs, rhs):
        a, b, left_guards = self.expression(lhs)
        c, d, right_guards = self.expression(rhs)
        numerator, denominator, guards = _rational_form(
            _add(_multiply(a, d), _scale(_multiply(c, b), -1)),
            _multiply(b, d), [*left_guards, *right_guards])
        # On the explicitly retained nonzero domain, a denominator and its
        # factors can be removed from an equation's zero residual.
        for factor in guards:
            while numerator:
                quotient = _divide_exact(numerator, factor)
                if quotient is None:
                    break
                numerator = quotient
        return ["zero", _poly_data(_unit(numerator)),
                [_poly_data(guard) for guard in guards]]

    def boolean(self, node):
        op = node["op"]
        if op == "boolean":
            return [op, node["value"]]
        if op == "not":
            return [op, self.boolean(node["argument"])]
        if op in {"and", "or"}:
            return [op, sorted([self.boolean(child) for child in node["args"]], key=_dump)]
        relation = node["relation"]
        if relation in {"eq", "ne"}:
            return [relation, self.equation(node["left"], node["right"])]
        # Ordering over rational functions depends on denominator signs; retain
        # the normalized sides rather than multiplying by an unknown sign.
        a, b, ag = self.expression(node["left"])
        c, d, cg = self.expression(node["right"])
        left, right = [_poly_data(a), _poly_data(b)], [_poly_data(c), _poly_data(d)]
        if relation in {"gt", "ge"}:
            relation, left, right = {"gt": "lt", "ge": "le"}[relation], right, left
        return [relation, left, right,
                sorted({_dump(_poly_data(g)): _poly_data(g) for g in [*ag, *cg]}.values(), key=_dump)]

    def statement(self, component):
        scope = component["scope"]
        scope_key = [sorted([self.component(key) for key in scope["entities"]], key=_dump),
                     sorted([self.component(key) for key in scope["conditions"]], key=_dump),
                     _text(scope["description"])]
        if scope["description"]:
            self.unresolved.add("prose_scope_equivalence")
        domain = sorted([[self.component(item["quantity"]), item["relation"],
                          str(_rational(item["value"]))] for item in component["domain"]], key=_dump)
        kind = component["kind"]
        if kind == "equation":
            body = self.equation(component["lhs"], component["rhs"])
        elif kind == "causal_relation":
            body = [self.component(component["cause"]), component["relation"], self.component(component["effect"])]
        elif kind == "mechanism":
            body = [sorted([self.ref(ref) for ref in component["inputs"]], key=_dump),
                    sorted([self.ref(ref) for ref in component["outputs"]], key=_dump),
                    [self.component(key) for key in component["steps"]]]
        elif kind == "assumption":
            assertion = component["assertion"]
            if assertion["form"] == "expression":
                body = self.boolean(assertion["expression"])
            else:
                self.unresolved.add("prose_assertion_equivalence")
                body = ["prose", _text(assertion["text"]),
                        sorted([self.ref(ref) for ref in assertion["references"]], key=_dump)]
        elif kind == "derivation":
            body = [sorted([self.component(key) for key in component["premises"]], key=_dump),
                    self.component(component["conclusion"]),
                    [self.component(key) for key in component["steps"]]]
        else:
            body = [self.component(component["original"]), self.component(component["approximation"]),
                    sorted([self.component(key) for key in component["regime"]], key=_dump)]
        # Cancellation must not erase which declared quantities the statement
        # was about (including named constants with explicit values).
        bindings = sorted([self.component(ref["id"]) for ref in collect_component_refs(component)
                           if ref["kind"] == "quantity"], key=_dump)
        return [kind, body, scope_key, domain, bindings,
                sorted([self.ref(ref) for ref in component["references"]], key=_dump)]


def contribution_identities(fragment, evidence_lookup=None):
    """Return unweighted source-grounded contributions keyed by component ID.

    Empty/evidence-free declarations and open questions earn no identity. Each
    explicit resolution contributes separately, so extending a gap does not
    turn its old resolutions into new information. These are representation
    identities, not claims of scientific validity or exhaustive equivalence.
    """
    fragment = validate_fragment(fragment, evidence_lookup=evidence_lookup)
    normalizer = _Normalizer(fragment, evidence_lookup)
    result = {}
    for key, (group, component) in normalizer.index.items():
        if not component["evidence"]:
            result[key] = ()
            continue
        if group == "gaps":
            values = {"theory:" + _digest(["resolution", normalizer.component(key),
                                           normalizer.ref(ref)])
                      for ref in component["resolutions"]}
        else:
            # Grounding is required, but additional citations are not additional
            # reconstructed laws. Evidence remains attached to the component.
            values = {"theory:" + _digest(normalizer.component(key))}
        result[key] = tuple(sorted(values))
    return result


def normalization_report(fragment, evidence_lookup=None):
    """Expose exact forms and unresolved operations instead of validity flags."""
    fragment = validate_fragment(fragment, evidence_lookup=evidence_lookup)
    reports = {}
    for key in fragment["statements"]:
        normalizer = _Normalizer(fragment, evidence_lookup)
        form = normalizer.component(key)
        reports[key] = {"normal_form": form,
                        "unresolved_equivalence": sorted(normalizer.unresolved),
                        "comparison": "Matching forms establish only the represented identity on the retained domain. "
                        "Different forms leave equivalence unresolved; general polynomial factorization, "
                        "analytic identities and source fidelity are not decided here."}
    return reports


def language_reference():
    """Concise meanings shared by parent, search and source reconstruction."""
    return {
        "fragment": "reconstruction and question identify the scientific reconstruction; entities, quantities, "
                    "statements and gaps are maps whose component IDs are unique across all four maps. "
                    "Empty maps are allowed. Additions contain only those four maps.",
        "common": "Every component has definition, native_id (source identifier or empty), evidence "
                  "([{key,version}] immutable source-evidence snapshots), supersedes (older IDs or []). "
                  "Do not invent native IDs or evidence. Existing components cannot be overwritten. "
                  "Revise using a new ID and supersedes; old definitions remain available.",
        "entity": "An entity denotes what the theory models or explains. kind describes its scientific type; "
                  "definition declares its meaning. Citation records, cited publications and bibliographic "
                  "metadata belong in source leads and provenance, not in the theory's component maps.",
        "quantity": "role is coordinate/state/parameter/constant. entity binds an entity ID; property and "
                    "definition state what is measured or modeled; units are explicit (use dimensionless when appropriate). "
                    "value_type is scalar/field/function. bindings maps formal argument names to distinct coordinate IDs. "
                    "Scalar quantities have {}; field/function quantities declare nonempty bindings. Coordinates and "
                    "constants are scalar. Only constants have value, an exact integer or rational/decimal string.",
        "expression": "Closed numeric AST: {op:number,value}; {op:quantity,ref,bindings}; "
                      "{op:call,function,bindings}; {op:add|multiply,args:[expressions]}; "
                      "{op:subtract|divide|power,left,right}; {op:derivative,expression,coordinate,order}. "
                      "Quantity uses scalar/field declarations; call uses a function declaration. Every application "
                      "supplies exactly its declared argument names, mapped to numeric AST expressions. "
                      "A derivative coordinate has role coordinate. Unknown functions remain typed opaque calls.",
        "boolean": "Closed Boolean AST: {op:boolean,value}; {op:compare,relation:eq|ne|lt|le|gt|ge,left,right}; "
                   "{op:and|or,args:[Boolean AST]}; {op:not,argument:Boolean AST}. Comparisons take numeric ASTs.",
        "statement": "All statements have kind, scope:{entities:[IDs],conditions:[statement IDs],description}, "
                     "domain:[{quantity:scalar ID,relation:eq|ne|lt|le|gt|ge,value:exact rational}], "
                     "references:[{kind:entity|quantity|statement|gap,id}]. Put scientific constraints in scope/domain; "
                     "prose-only distinctions remain unresolved, not silently equated.",
        "statement_fields": {
            "equation": "lhs and rhs numeric ASTs",
            "causal_relation": "cause/effect quantity IDs; relation increases/decreases/enables/inhibits/necessary_for/sufficient_for/affects",
            "mechanism": "inputs are typed references; outputs must identify at least one declared entity, "
                         "quantity or statement that the process produces or explains. A gap is a research "
                         "question, not a mechanism output; link it through references. Steps are nonempty "
                         "ordered statement IDs. Qualitative source-grounded outputs are allowed; an output "
                         "reference does not verify the mechanism or its algebra.",
            "assumption": "assertion:{form:expression,expression:Boolean AST} or {form:prose,text,references:[typed refs]}",
            "derivation": "nonempty premises:[statement IDs], conclusion:statement ID, ordered steps:[statement IDs]",
            "approximation": "original and approximation statement IDs; nonempty regime:[statement IDs]",
        },
        "gap": "question states what remains unknown; components are typed references; requested_kinds selects "
               "entity/quantity/equation/causal_relation/mechanism/assumption/derivation/approximation/gap. "
               "resolutions are explicit typed references to source-grounded non-gap components of requested kinds, or []. "
               "A resolution is a proposed sourced answer, not a model-authored validity flag.",
        "normalization": "Rational arithmetic uses exact bound quantity meanings and explicit nonzero domains. "
                         "Integral powers, expansion, scalar normalization and exact division by known nonzero "
                         "factors are supported. Derivatives, opaque functions, general powers, prose semantics "
                         "and unhandled factorization retain explicit unresolved equivalence. No numeric sampling "
                         "or string similarity establishes equivalence. Validation does not certify source support.",
    }
