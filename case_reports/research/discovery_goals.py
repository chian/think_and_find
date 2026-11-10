"""Scientific discovery commissions and executable, source-relative fulfillment.

Source claims, formal equivalence and causal dependence are different checks.
An unknown scientific relationship stays pending; a model verdict cannot settle it.
"""
from copy import deepcopy
from fractions import Fraction
import json
import re
from urllib.parse import unquote

from jsonschema import Draft202012Validator, SchemaError, ValidationError
from referencing.exceptions import Unresolvable

from .records import Credit, Ledger, ResearchResult, identity
from .source_claims import CLAIM_FIELDS, CLAIM_KINDS, CLAIM_RELATIONS, claim_identity, claims_schema
from .support_observations import OBSERVATION_FIELDS, OBSERVATION_KINDS, observation_identity, observations_schema


TEXT = {"type": "string", "minLength": 1}
SNAPSHOT = {"type": "object", "additionalProperties": False,
            "required": ["key", "version"],
            "properties": {"key": TEXT, "version": TEXT}}
GOAL_FIELDS = ("question", "inputs", "answer_schema", "criteria")
FINDING_FIELDS = ("goal", "goal_version", "answer", "source_claims", "source_observations", "scientific_links")
CHECK_FIELDS = ("goal", "finding", "results")

DISCOVERY_GOAL_INSTRUCTION = (
    " Each criterion has exactly id and requirement. A native requirement has kind='native' and predicate "
    "(source_claim or source_value); it has no scientific question or semantic assertion. Its receipt certifies "
    "native_fields_only with native_match/native_no_match, never scientific fulfillment. A scientific "
    "requirement has kind='scientific', question, candidate (a native predicate or null), and predicate "
    "(theory_structure, formal_equivalence, causal_dependence or distinct_theory). Candidate matches provide "
    "Search information only; scientific fulfillment requires the predicate's actual linked scientific operands. "
    "theory_structure uses component_schema over one typed component with id/group and structure_schema over "
    "its typed reference closure: root:{kind,id}, entities/quantities/statements/gaps arrays with component IDs. "
    "It checks source-grounded scientific representation, not source fidelity, empirical truth or equivalence. "
    " A source_claim predicate constrains the retained native source claim, not the custom answer object. "
    "Its only fields are kind, statement_id, locator, left, relation, right, conditions and symbols. "
    "kind is equation, causal_relation or boundary_condition. Equation relation is '='; causal relations are "
    + ", ".join(sorted(CLAIM_RELATIONS["causal_relation"]))
    + "; boundary relations are " + ", ".join(sorted(CLAIM_RELATIONS["boundary_condition"])) + ". "
    "For a requested observation value, use source_value with field_path=[] and a schema over ONE complete "
    "observation: require kind, quantity, conditions, value and units together. Constrain kind, quantity and units "
    "with explicit const/enum values, and encode the requested circumstances in conditions. Define quantity names "
    "and condition keys/units in the request so readers can use that vocabulary while retaining exact source anchors. "
    "A requested empirical endpoint requires kind='measurement', its defined quantity, conditions and units. "
    "A fit_parameter, simulation_output or model_prediction cannot fulfill an empirical measurement request. "
    "A scalar value-only criterion is invalid: any number is not a measurement request. Use separate criteria for "
    "separate requested circumstances; every criterion's constraints must match the same source observation. "
    "Unresolved answers can return empty source_claims/source_observations and earn no witness credit; never attach "
    "an explicitly excluded operand merely to submit a partial answer. "
    "Custom fields such as claim_type, native_claim or symbol_definitions belong in answer_schema, "
    "not claim_schema. A source_value field_path starts at one selected native observation: source_id, "
    "locator, kind, quantity, conditions, value, units or uncertainty. Structured uncertainty has kind/lower/upper; "
    "structured conditions have named scalar quantities. Native value is a scalar, not a custom answer object."
    " Observation source_id is optional: host-validated source-occurrence anchors identify unlabeled observations. "
    "Locator prose describes provenance, not identity. Do not require a nonempty "
    "native ID unless the source-given observation ID itself is part of the scientific information need. "
    "Document URLs and study IDs are not observation IDs; use empty source_id when no native ID exists."
)


def native_operand_reference():
    claim = deepcopy(claims_schema(source="")["additionalProperties"])
    claim["properties"]["statement_id"].pop("not", None)
    return {"source_claim": claim,
            "source_observation": deepcopy(observations_schema()["additionalProperties"]),
            "claim_relations_by_kind": {kind: sorted(values) for kind, values in CLAIM_RELATIONS.items()}}


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def _array(items, *, unique=True):
    return {"type": "array", "items": items, "uniqueItems": unique}


def _path(parts):
    return "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in parts)


def _validate(value, schema, field):
    try:
        Draft202012Validator(schema).validate(value)
    except ValidationError as exc:
        raise ValueError(f"{field}{_path(exc.absolute_path)}: {exc.message}") from exc
    except (Unresolvable, re.error, RecursionError) as exc:
        raise ValueError(f"{field}: unresolved local scientific type reference: {exc}") from exc


def _matches(validator, value, field):
    try:
        return validator.is_valid(value)
    except (Unresolvable, re.error, RecursionError) as exc:
        raise ValueError(f"{field}: cannot evaluate the declared scientific type: {exc}") from exc


def _selection_schema(selector):
    return _object({"evidence": SNAPSHOT, selector: TEXT})


def _finding_properties():
    return {"answer": {"type": "object"},
            "source_claims": _array(_selection_schema("claim")),
            "source_observations": _array(_selection_schema("observation")),
            "scientific_links": {**_array(SNAPSHOT), "description":
                "Exact snapshots of existing scientific accounts, comparisons, assessments or formal results. "
                "Excludes evidence, discovery_goal, discovery_finding and discovery_check. Use [] when none exists. "
                "Put native evidence selectors in source_claims/source_observations."}}


def _finding_data_schema():
    return _object({"goal": TEXT, "goal_version": TEXT, **_finding_properties()})


def _check_data_schema():
    return _object({"goal": TEXT, "finding": TEXT, "results": _array(_object({
        "criterion": TEXT,
        "requirement": requirement_schema(),
        "verified_scope": {"enum": ["native_fields_only", "scientific_requirement"]},
        "status": {"enum": ["native_match", "native_no_match", "fulfilled", "not_fulfilled", "needs_scientific_assessment"]},
        "witnesses": _array(TEXT), "reason": TEXT, "inputs": _array(SNAPSHOT),
        "structures": _array(_object({"witness": TEXT, "fragment": SNAPSHOT,
            "component": _object({"kind": {"enum": ["entity", "quantity", "statement", "gap"]}, "id": TEXT}),
            "source_inputs": _array(SNAPSHOT)}))}))})


def _validate_record_shape(kind, data):
    schemas = {"discovery_goal": goal_schema, "discovery_finding": _finding_data_schema,
               "discovery_check": _check_data_schema}
    if kind == "discovery_goal" and isinstance(data, dict):
        if isinstance(data.get("answer_schema"), dict):
            _check_schema(data["answer_schema"], "discovery_goal.data.answer_schema")
        for index, criterion in enumerate(data.get("criteria", ()) if isinstance(data.get("criteria"), list) else ()):
            requirement = criterion.get("requirement") if isinstance(criterion, dict) else None
            if isinstance(requirement, dict):
                for name in ("predicate", "candidate"):
                    predicate = requirement.get(name)
                    if isinstance(predicate, dict):
                        _compile_operand_constraint(predicate,
                            f"discovery_goal.data.criteria[{index}].requirement.{name}")
    _validate(data, schemas[kind](), f"{kind}.data")


def native_predicate_schema():
    native = native_operand_reference()
    claim_meta = {"type": "object", "description":
        "JSON Schema over this exact native source-claim operand, not answer_schema fields: "
        + json.dumps(native["source_claim"], sort_keys=True),
        "properties": {"properties": {"type": "object", "propertyNames": {"enum": sorted(CLAIM_FIELDS)},
            "properties": {
                "kind": {"type": "object", "properties": {
                    "const": {"enum": sorted(CLAIM_KINDS)},
                    "enum": {"type": "array", "items": {"enum": sorted(CLAIM_KINDS)}}}},
                "relation": {"type": "object", "properties": {
                    "const": {"enum": sorted(set().union(*CLAIM_RELATIONS.values()))},
                    "enum": {"type": "array", "items": {"enum": sorted(set().union(*CLAIM_RELATIONS.values()))}}}}}},
            "required": {"type": "array", "items": {"enum": sorted(CLAIM_FIELDS)}}}}
    return {"oneOf": [
        _object({"kind": {"const": "source_claim"},
                 "claim_schema": claim_meta}),
        _object({"kind": {"const": "source_value"},
                 "field_path": {**_array(TEXT, unique=False), "not": {"const": ["value"]},
                    "description": "Path relative to one selected native observation. Root fields: "
                        + ", ".join(sorted(OBSERVATION_FIELDS)) + ". Structured uncertainty keys: kind/lower/upper; "
                        "structured conditions keys are named scalar quantities. Use an empty path for an observation value, "
                        "constraining kind, quantity, conditions, value and units together. Scalar ['value'] is invalid."},
                 "value_schema": {"type": "object", "description":
                    "JSON Schema for the selected native observation field. Observation structure: "
                    + json.dumps(native["source_observation"], sort_keys=True)}})]}


def scientific_predicate_schema():
    return {"oneOf": [
        _object({"kind": {"const": "theory_structure"},
                 "component_schema": {"type": "object", "description":
                    "JSON Schema over one root typed component, including its id and group."},
                 "structure_schema": {"type": "object", "description":
                    "JSON Schema over ONLY that root's transitive typed-reference closure: "
                    "root:{kind,id}; entities, quantities, statements and gaps arrays, each component with id."}}),
        _object({"kind": {"const": "formal_equivalence"},
                 "required_inputs": {**_array(SNAPSHOT), "minItems": 1},
                 "request_schema": {"type": "object",
                    "description": "Local JSON Schema constraining the actual Lean equivalence request: relation, left/right terms and source-bound parameter/assumption mappings. A receipt sharing input IDs alone is insufficient."}}),
        _object({"kind": {"const": "causal_dependence"}, "inference": SNAPSHOT}),
        _object({"kind": {"const": "distinct_theory"},
                 "excluded_theories": {**_array(SNAPSHOT), "minItems": 1}}),
    ]}


def requirement_schema():
    return {"oneOf": [
        _object({"kind": {"const": "native"}, "predicate": native_predicate_schema()}),
        _object({"kind": {"const": "scientific"}, "question": TEXT,
                 "candidate": {"anyOf": [{"type": "null"}, native_predicate_schema()]},
                 "predicate": scientific_predicate_schema()})]}


def goal_schema():
    return _object({"question": TEXT, "inputs": _array(SNAPSHOT),
        "answer_schema": {"type": "object",
            "description": "JSON Schema defining the requested scientific answer, using local $defs for the parent's custom types and quantities."},
        "criteria": {"type": "array", "minItems": 1,
            "items": _object({"id": TEXT, "requirement": requirement_schema()})}})


def discovery_goal_action_schema():
    return _object({"action": {"const": "create_discovery_goal"},
                    "key": TEXT, "goal": goal_schema(), "rationale": TEXT})


def finding_schema():
    return _object({"action": {"const": "submit_finding"},
        "key": {**TEXT, "description":
            "Unused key for this new discovery finding. Do not reuse the commissioned goal key "
            "or overwrite an existing record."}, **_finding_properties(), "rationale": TEXT})


def _schema_children(schema):
    if not isinstance(schema, dict):
        return
    for keyword in ("$defs", "definitions", "properties", "patternProperties", "dependentSchemas"):
        for key, child in schema.get(keyword, {}).items():
            yield (keyword, key), child
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        for index, child in enumerate(schema.get(keyword, ())):
            yield (keyword, index), child
    for keyword in ("items", "contains", "additionalProperties", "unevaluatedProperties", "unevaluatedItems",
                    "propertyNames", "not", "if", "then", "else", "contentSchema"):
        if keyword in schema:
            yield (keyword,), schema[keyword]


def _local_schema(schema, reference, field):
    if not isinstance(reference, str) or not reference.startswith("#"):
        raise ValueError(f"{field}: scientific types use local definitions, not external schema retrieval")
    fragment = unquote(reference[1:])
    if not fragment:
        return schema
    if fragment.startswith("/"):
        value = schema
        try:
            for raw in fragment[1:].split("/"):
                part = raw.replace("~1", "/").replace("~0", "~")
                value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"{field}: unknown local scientific type reference {reference!r}") from exc
        if not isinstance(value, (dict, bool)):
            raise ValueError(f"{field}: local reference {reference!r} does not name a schema")
        return value
    def anchored(value):
        if isinstance(value, dict):
            if value.get("$anchor") == fragment or value.get("$dynamicAnchor") == fragment:
                return value
            for _, child in _schema_children(value):
                found = anchored(child)
                if found is not None:
                    return found
        return None
    found = anchored(schema)
    if found is None:
        raise ValueError(f"{field}: unknown local scientific type reference {reference!r}")
    return found


def _check_schema(schema, field):
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"{field}{_path(exc.absolute_path)}: {exc.message}") from exc
    def local(value, path):
        if not isinstance(value, dict):
            return
        for name in ("$ref", "$dynamicRef"):
            if name in value:
                _local_schema(schema, value[name], path + "." + name)
        for pattern in ([value["pattern"]] if "pattern" in value else ()):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{path}.pattern: invalid scientific type pattern: {exc}") from exc
        for pattern in value.get("patternProperties", {}):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"{path}.patternProperties: invalid scientific property pattern: {exc}") from exc
        for parts, child in _schema_children(value):
            local(child, path + _path(parts))
    local(schema, field)


def _same_instance_schemas(value, root, path):
    if not isinstance(value, dict):
        return
    for name in ("$ref", "$dynamicRef"):
        if name in value:
            yield path + "." + name, _local_schema(root, value[name], path + "." + name)
    for name in ("allOf", "anyOf", "oneOf"):
        for index, child in enumerate(value.get(name, ())):
            yield f"{path}.{name}[{index}]", child
    for name in ("not", "if", "then", "else"):
        if name in value:
            yield path + "." + name, value[name]
    for name, child in value.get("dependentSchemas", {}).items():
        yield f"{path}.dependentSchemas.{name}", child


def _assert_types(schema, allowed, path, root=None, seen=None):
    root = schema if root is None else root
    universe = {"null", "boolean", "object", "array", "number", "integer", "string"}
    def possible(value, visited):
        if not isinstance(value, dict) or id(value) in visited:
            return set(universe)
        visited = {*visited, id(value)}
        declared = value.get("type", universe)
        result = {declared} if isinstance(declared, str) else set(declared)
        if "number" in result:
            result.add("integer")
        for name in ("$ref", "$dynamicRef"):
            if name in value:
                result &= possible(_local_schema(root, value[name], path + "." + name), visited)
        for child in value.get("allOf", ()):
            result &= possible(child, visited)
        for name in ("anyOf", "oneOf"):
            if name in value:
                result &= set().union(*(possible(child, visited) for child in value[name]))
        return result
    declared = possible(schema, set() if seen is None else seen)
    if not declared & set(allowed):
        raise ValueError(f"{path}.type: native operand supports {sorted(allowed)}, not {sorted(declared)}. "
                         "Custom answer structures belong in answer_schema.")


def _assert_native_claim_schema(schema, path):
    native = native_operand_reference()["source_claim"]
    seen = set()
    guidance = ("Custom answer fields belong in answer_schema; claim_schema checks native.claims fields: "
                + ", ".join(sorted(CLAIM_FIELDS)) + ".")
    def walk(value, field):
        if not isinstance(value, dict) or id(value) in seen:
            return
        seen.add(id(value))
        for keyword in ("properties", "dependentSchemas", "dependentRequired"):
            for name in value.get(keyword, {}):
                if name not in CLAIM_FIELDS:
                    raise ValueError(f"{field}.{keyword}.{name}: no such native source-claim field. {guidance}")
        for index, name in enumerate(value.get("required", ())):
            if name not in CLAIM_FIELDS:
                raise ValueError(f"{field}.required[{index}]: {name!r} is not a native source-claim field. {guidance}")
        for name, required in value.get("dependentRequired", {}).items():
            for index, required_name in enumerate(required):
                if required_name not in CLAIM_FIELDS:
                    raise ValueError(f"{field}.dependentRequired.{name}[{index}]: {required_name!r} is not "
                                     f"a native source-claim field. {guidance}")
        for name, constraint in value.get("properties", {}).items():
            expected = native["properties"][name]
            _assert_types(constraint, {"object"} if name == "symbols" else {"string"},
                          f"{field}.properties.{name}", schema)
            def constants(rule, rule_path, visited):
                if not isinstance(rule, dict) or id(rule) in visited:
                    return
                visited.add(id(rule))
                values = [("const", rule["const"])] if "const" in rule else []
                values.extend((f"enum[{i}]", item) for i, item in enumerate(rule.get("enum", ())))
                for suffix, item in values:
                    try:
                        _validate(item, expected, rule_path + "." + suffix)
                    except ValueError as exc:
                        raise ValueError(str(exc) + " " + guidance) from exc
                for child_path, child in _same_instance_schemas(rule, schema, rule_path):
                    constants(child, child_path, visited)
            constants(constraint, f"{field}.properties.{name}", set())
        kind_schema = value.get("properties", {}).get("kind")
        relation_schema = value.get("properties", {}).get("relation")
        kind = kind_schema.get("const") if isinstance(kind_schema, dict) else None
        relation = relation_schema.get("const") if isinstance(relation_schema, dict) else None
        if kind in CLAIM_RELATIONS and relation is not None and relation not in CLAIM_RELATIONS[kind]:
            raise ValueError(f"{field}.properties.relation.const: {relation!r} is not a {kind} relation; "
                             f"use {sorted(CLAIM_RELATIONS[kind])}. {guidance}")
        for child_path, child in _same_instance_schemas(value, schema, field):
            walk(child, child_path)
    _assert_types(schema, {"object"}, path)
    walk(schema, path)


def _common_constraints(schema, root, path, seen=None):
    """Constraints applying to the same instance, including named custom types."""
    seen = set() if seen is None else seen
    if not isinstance(schema, dict) or id(schema) in seen:
        return
    seen = {*seen, id(schema)}
    yield schema
    for name in ("$ref", "$dynamicRef"):
        if name in schema:
            yield from _common_constraints(_local_schema(root, schema[name], path), root, path, seen)
    for item in schema.get("allOf", ()):
        yield from _common_constraints(item, root, path, seen)


def _literal_choices(schema, root, path, seen=None):
    seen = set() if seen is None else seen
    if id(schema) in seen:
        return None
    seen = {*seen, id(schema)}
    choices = None
    for rule in _common_constraints(schema, root, path):
        values = [rule["const"]] if "const" in rule else rule.get("enum")
        if values is not None:
            encoded = {json.dumps(value, sort_keys=True) for value in values}
            choices = encoded if choices is None else choices & encoded
        for name in ("anyOf", "oneOf"):
            if name in rule:
                branches = [_literal_choices(item, root, path, seen) for item in rule[name]]
                if all(branch is not None for branch in branches):
                    union = set().union(*branches)
                    choices = union if choices is None else choices & union
    return None if choices is None else choices


def _condition_selector(schema, root, path, seen=None):
    seen = set() if seen is None else seen
    if id(schema) in seen:
        return False
    seen = {*seen, id(schema)}
    rules = list(_common_constraints(schema, root, path))
    required = {name for rule in rules for name in rule.get("required", ())}
    for rule in rules:
        if any(name in rule for name in ("const", "enum", "pattern", "minimum", "maximum",
                                         "exclusiveMinimum", "exclusiveMaximum")):
            return True
        if any(_condition_selector(child, root, path, seen)
               for name, child in rule.get("properties", {}).items() if name in required):
            return True
        if any(all(_condition_selector(child, root, path, seen) for child in rule[name])
               for name in ("anyOf", "oneOf") if rule.get(name)):
            return True
    return False


def _assert_observation_request(schema, path):
    required, properties = set(), {}
    for rule in _common_constraints(schema, schema, path):
        required.update(rule.get("required", ()))
        for name, value in rule.get("properties", {}).items():
            if name not in OBSERVATION_FIELDS:
                raise ValueError(f"{path}.properties.{name}: no such native observation field")
            properties.setdefault(name, []).append(value)
    fields = {"kind", "quantity", "conditions", "value", "units"}
    if not fields <= required or not fields <= properties.keys():
        raise ValueError(f"{path}: a value request must require kind, quantity, conditions, value and units "
                         "on the same observation; put shared constraints outside anyOf/oneOf branches")
    for name in ("kind", "quantity", "units"):
        choices = _literal_choices({"allOf": properties[name]}, schema, path)
        if not choices or any(not isinstance(json.loads(value), str) or not json.loads(value).strip()
                              for value in choices):
            raise ValueError(f"{path}.properties.{name}: define explicit nonempty const/enum vocabulary")
        if name == "kind" and any(json.loads(value) not in OBSERVATION_KINDS for value in choices):
            raise ValueError(f"{path}.properties.kind: use an actual native observation kind")
    if not _condition_selector({"allOf": properties["conditions"]}, schema, path):
        raise ValueError(f"{path}.properties.conditions: constrain the requested circumstances, "
                         "using named quantities with declared units or exact source condition text")


def _compile_operand_constraint(check, path):
    if check.get("kind") == "source_claim" and isinstance(check.get("claim_schema"), dict):
        field = path + ".claim_schema"
        _check_schema(check["claim_schema"], field)
        _assert_native_claim_schema(check["claim_schema"], field)
    elif check.get("kind") == "source_value" and isinstance(check.get("field_path"), list):
        parts = check["field_path"]
        if not all(isinstance(part, str) for part in parts):
            return
        if parts == ["value"]:
            raise ValueError(f"{path}.field_path: a scalar value cannot establish a requested observation. "
                "Use field_path=[] and constrain kind, quantity, conditions, value and units together.")
        if parts and parts[0] not in OBSERVATION_FIELDS:
            raise ValueError(f"{path}.field_path[0]: {parts[0]!r} is not a native observation field. "
                "Select an observation with source_observations, then address " + ", ".join(sorted(OBSERVATION_FIELDS))
                + "; custom answer fields belong in answer_schema.")
        if len(parts) > 1:
            valid = len(parts) == 2 and (
                parts[0] == "uncertainty" and parts[1] in {"kind", "lower", "upper"}
                or parts[0] == "conditions" and parts[1].isidentifier())
            if not valid:
                raise ValueError(f"{path}.field_path: no such nested native observation field {parts!r}; "
                    "only uncertainty.kind/lower/upper or conditions.<named scalar quantity> have nested fields.")
        if isinstance(check.get("value_schema"), dict):
            field = path + ".value_schema"
            _check_schema(check["value_schema"], field)
            allowed = ({"object"} if not parts else
                {"string", "number", "integer", "boolean"} if parts == ["value"] else
                {"string", "object"} if parts in (["uncertainty"], ["conditions"]) else
                {"string", "integer", "number"} if len(parts) == 2 and parts[-1] != "kind" else {"string"})
            _assert_types(check["value_schema"], allowed, field)
            if not parts:
                _assert_observation_request(check["value_schema"], field)


def references(kind, data):
    # CausalSheet collects references before its full admission validator runs.
    # Validate the closed operand structure before dereferencing model output.
    _validate_record_shape(kind, data)
    if kind == "discovery_goal":
        return [ref["key"] for ref in snapshot_references(kind, data)]
    if kind == "discovery_finding":
        return [data["goal"], *[ref["key"] for ref in snapshot_references(kind, data)]]
    return [data["goal"], data["finding"],
            *(ref["key"] for ref in snapshot_references(kind, data))]


def snapshot_references(kind, data):
    if kind == "discovery_goal":
        refs = list(data["inputs"])
        for criterion in data["criteria"]:
            check = criterion["requirement"]["predicate"]
            refs.extend(check.get("required_inputs", ()))
            refs.extend(check.get("excluded_theories", ()))
            if check["kind"] == "causal_dependence":
                refs.append(check["inference"])
        return refs
    if kind == "discovery_finding":
        return [{"key": data["goal"], "version": data["goal_version"]},
                *(item["evidence"] for item in data["source_claims"]),
                *(item["evidence"] for item in data["source_observations"]),
                *data["scientific_links"]]
    return [ref for result in data["results"] for ref in result["inputs"]]


def context_pins(kind, data):
    pins = {}
    for ref in snapshot_references(kind, data):
        pins.setdefault(ref["key"], set()).add(ref["version"])
    return pins


def validate_record(sheet, key, kind, data):
    if kind == "discovery_goal":
        _validate_record_shape(kind, data)
        _check_schema(data["answer_schema"], "discovery_goal.data.answer_schema")
        ids = [item["id"] for item in data["criteria"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Discovery criterion IDs must be distinct within a goal")
        for index, criterion in enumerate(data["criteria"]):
            check = criterion["requirement"]["predicate"]
            path = f"discovery_goal.data.criteria[{index}].requirement.predicate"
            for name in ("claim_schema", "value_schema", "request_schema", "component_schema", "structure_schema"):
                if name in check:
                    _check_schema(check[name], path + "." + name)
            if check["kind"] == "causal_dependence" and sheet.snapshot(check["inference"])["kind"] != "inference":
                raise ValueError(path + ".inference requires a G inference record")
            if check["kind"] == "distinct_theory" and any(
                    sheet.snapshot(ref)["kind"] != "theory" for ref in check["excluded_theories"]):
                raise ValueError(path + ".excluded_theories requires theory records")
    elif kind == "discovery_finding":
        fields = set(FINDING_FIELDS)
        if not isinstance(data, dict) or set(data) != fields:
            raise ValueError("A discovery finding requires its exact goal, answer and scientific operands")
        _validate_record_shape(kind, data)
        goal = sheet.snapshot({"key": data["goal"], "version": data["goal_version"]})
        if goal["kind"] != "discovery_goal":
            raise ValueError("A finding must answer a recorded discovery goal")
        _validate(data["answer"], goal["data"]["answer_schema"], "discovery_finding.data.answer")
        for item in data["source_claims"]:
            row = sheet.snapshot(item["evidence"])
            if row["kind"] != "evidence":
                raise ValueError("A discovery source operand must be evidence")
            claim_identity(row["data"], item["claim"])
        for item in data["source_observations"]:
            row = sheet.snapshot(item["evidence"])
            if row["kind"] != "evidence":
                raise ValueError("A discovery observation operand must be evidence")
            observation_identity(row["data"], item["observation"])
        for ref in data["scientific_links"]:
            if sheet.snapshot(ref)["kind"] in {"evidence", "discovery_goal", "discovery_finding", "discovery_check"}:
                raise ValueError("scientific_links requires a scientific account, comparison, assessment or formal tool result")
    elif kind == "discovery_check":
        if not isinstance(data, dict) or set(data) != set(CHECK_FIELDS):
            raise ValueError("Discovery checks are host-generated criterion results")
        _validate_record_shape(kind, data)
        sheet._require_kind(data["finding"], "discovery_finding")
        expected = evaluate_finding(sheet, data["finding"])
        if data != {"goal": expected["goal"], "finding": data["finding"], "results": expected["results"]}:
            raise ValueError("Discovery criterion receipt does not reproduce its source/formal checks")
    for ref in snapshot_references(kind, data):
        sheet.snapshot(ref)
    previous = sheet.rows.get(key)
    if previous and previous["data"] != data:
        raise ValueError("Discovery goals, findings and check receipts are immutable; record a new explicit result")


def _native_claims(sheet, finding):
    for item in finding["source_claims"]:
        row = sheet.snapshot(item["evidence"])
        ident, detail = claim_identity(row["data"], item["claim"])
        yield ident, detail, row["data"], item["evidence"]


def _native_observations(sheet, finding):
    for item in finding["source_observations"]:
        row = sheet.snapshot(item["evidence"])
        ident, detail = observation_identity(row["data"], item["observation"])
        yield ident, detail, item["evidence"]


def _field(value, parts):
    for part in parts:
        value = value[int(part)] if isinstance(value, list) and part.isdigit() else value[part]
    return value


def _linked_records(sheet, finding_key, finding):
    selected = {(ref["key"], ref["version"]): sheet.snapshot(ref)
                for ref in finding["scientific_links"]}
    for key, row in sheet.rows.items():
        if row["kind"] not in {"evidence", "discovery_goal", "discovery_finding", "discovery_check"} and sheet._depends_on(key, finding_key):
            selected[(key, row["version"])] = row
    return [(key, row) for (key, _), row in selected.items()]


def _native_matches(check, claims, observations):
    """Return literal source witnesses; this function makes no scientific claim."""
    witnesses, inputs = [], {}
    if check["kind"] == "source_claim":
        validator = Draft202012Validator(check["claim_schema"])
        for ident, detail, _, ref in claims:
            inputs[ref["key"], ref["version"]] = deepcopy(ref)
            operand = {**detail["structure"], "statement_id": detail["statement_id"],
                       "locator": detail["native_locator"]}
            if _matches(validator, operand, "source_claim predicate.claim_schema"):
                witnesses.append("claim:" + ident)
    else:
        validator = Draft202012Validator(check["value_schema"])
        for ident, detail, ref in observations:
            inputs[ref["key"], ref["version"]] = deepcopy(ref)
            try:
                value = _field(detail["measurement"], check["field_path"])
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            if _matches(validator, value, "source_value predicate.value_schema"):
                witnesses.append("observation:" + ident)
    return sorted(set(witnesses)), [inputs[key] for key in sorted(inputs)]


def _linked_fragments(sheet, linked):
    """Resolve only explicitly linked scientific records and their exact premises."""
    seen, todo, fragments = set(), [row for _, row in linked], []
    while todo:
        row = todo.pop()
        pair = row["key"], row["version"]
        if pair in seen:
            continue
        seen.add(pair)
        if row["kind"] == "theory_fragment":
            fragments.append(row)
        for key, version in row["dependencies"].items():
            todo.append(sheet.snapshot({"key": key, "version": version}))
    return sorted(fragments, key=lambda row: (row["key"], row["version"]))


def _theory_structures(sheet, linked, finding, check):
    """Check one root and its actual reference closure, never a fragment-wide bag."""
    from .theory_language import (GROUPS, collect_component_refs, contribution_identities,
                                 validate_fragment)
    from .theory_runtime import evidence_lookup
    lookup = evidence_lookup(sheet)
    selected_sources = {(item["evidence"]["key"], item["evidence"]["version"])
                        for name in ("source_claims", "source_observations") for item in finding[name]}
    groups = dict(zip(GROUPS, ("entity", "quantity", "statement", "gap")))
    component_validator = Draft202012Validator(check["component_schema"])
    structure_validator = Draft202012Validator(check["structure_schema"])
    structures, inputs, examined = [], {}, False
    for row in _linked_fragments(sheet, linked):
        fragment = validate_fragment(row["data"], evidence_lookup=lookup)
        index = {key: (group, value) for group in GROUPS for key, value in fragment[group].items()}
        fragment_ref = {"key": row["key"], "version": row["version"]}
        inputs[row["key"], row["version"]] = fragment_ref
        for root, (group, component) in sorted(index.items()):
            root_operand = {"id": root, "group": group, **deepcopy(component)}
            if not _matches(component_validator, root_operand, "theory_structure predicate.component_schema"):
                continue
            closure, todo = set(), [root]
            while todo:
                key = todo.pop()
                if key in closure:
                    continue
                closure.add(key)
                todo.extend(ref["id"] for ref in collect_component_refs(index[key][1]))
            evidence = {(ref["key"], ref["version"]): deepcopy(ref)
                        for key in closure for ref in index[key][1]["evidence"]}
            grounded = sorted(selected_sources & evidence.keys())
            if not grounded:
                continue
            examined = True
            inputs.update(evidence)
            structure = {"root": {"kind": groups[group], "id": root},
                **{name: [{"id": key, **deepcopy(index[key][1])} for key in sorted(closure)
                           if index[key][0] == name] for name in GROUPS}}
            if not _matches(structure_validator, structure, "theory_structure predicate.structure_schema"):
                continue
            # A derived root can cite its scientific premises rather than source
            # evidence directly. Its closure has already authenticated those
            # exact evidence versions. Supply that existing grounding only to
            # the canonicalizer; do not rewrite the stored scientific record.
            canonical = fragment
            if not component["evidence"]:
                canonical = deepcopy(fragment)
                canonical[group][root]["evidence"] = [evidence[key] for key in sorted(evidence)]
            for witness in contribution_identities(canonical, evidence_lookup=lookup)[root]:
                structures.append({"witness": witness, "fragment": deepcopy(fragment_ref),
                    "component": {"kind": groups[group], "id": root},
                    "source_inputs": [evidence[key] for key in grounded]})
    structures = sorted({identity(item): item for item in structures}.values(), key=identity)
    return structures, [inputs[key] for key in sorted(inputs)], examined


def evaluate_finding(sheet, finding_key):
    finding = sheet.rows[finding_key]["data"]
    goal = sheet.snapshot({"key": finding["goal"], "version": finding["goal_version"]})["data"]
    claims = list(_native_claims(sheet, finding))
    observations = list(_native_observations(sheet, finding))
    linked = _linked_records(sheet, finding_key, finding)
    results = []
    for criterion in goal["criteria"]:
        requirement = criterion["requirement"]
        check = requirement["predicate"]
        witnesses = []
        structures = []
        inputs = {}
        def supplied(ref):
            inputs[(ref["key"], ref["version"])] = deepcopy(ref)
        def examined(key, row):
            supplied({"key": key, "version": row["version"]})
        status = "needs_scientific_assessment"
        reason = "No linked executed scientific result establishes this criterion yet"
        if requirement["kind"] == "native":
            witnesses, native_inputs = _native_matches(check, claims, observations)
            for ref in native_inputs:
                supplied(ref)
            status = "native_match" if witnesses else "native_no_match"
            reason = "Checked only the exact declared native source-field predicate; no scientific correspondence is certified"
        elif check["kind"] == "theory_structure":
            structures, structure_inputs, examined_structure = _theory_structures(sheet, linked, finding, check)
            for ref in structure_inputs:
                supplied(ref)
            witnesses = [item["witness"] for item in structures]
            if witnesses:
                status = "fulfilled"
                reason = "A linked source-grounded typed component and its reference closure match the commissioned scientific structure"
            elif examined_structure:
                status = "not_fulfilled"
                reason = "The examined source-grounded scientific structures do not match the commissioned predicates; source fidelity and empirical truth remain separate"
        elif check["kind"] == "formal_equivalence":
            required = {(ref["key"], ref["version"]) for ref in check["required_inputs"]}
            validator = Draft202012Validator(check["request_schema"])
            for key, row in linked:
                data = row["data"]
                if row["kind"] != "tool_receipt" or data["tool"] != "lean_equivalence":
                    continue
                examined(key, row)
                checked_inputs = {(ref["key"], ref["version"]) for ref in data["inputs"]}
                if (data["status"] == "verified_equivalent" and required <= checked_inputs
                        and _matches(validator, data["request"], "formal_equivalence criterion.request_schema")):
                    witnesses.append(key)
            if witnesses:
                status, reason = "fulfilled", "A linked Lean receipt verifies the commissioned equivalence request and exact scientific inputs"
        elif check["kind"] == "causal_dependence":
            from .support_grading import grade_support
            from .dependence import dependence_result
            target = check["inference"]
            for key, row in linked:
                if row["version"] != sheet.rows[key]["version"]:
                    continue
                if row["kind"] == "support_assessment":
                    examined(key, row)
                    grade = grade_support(sheet, key)
                    if grade["inference"] == target and grade["hypothesis"]["status"] == "demonstrated":
                        witnesses.append(key)
                elif row["kind"] == "importance":
                    examined(key, row)
                    application = sheet.rows[row["data"]["application"]]
                    if (application["data"]["inference"] == target["key"]
                            and application["data"]["inference_version"] == target["version"]):
                        grade = dependence_result(sheet, key)
                        if grade["dependence"]["status"] == "demonstrated":
                            witnesses.append(key)
            if witnesses:
                status, reason = "fulfilled", "Executed intervention comparisons demonstrate dependence on the commissioned G"
        elif check["kind"] == "distinct_theory":
            excluded = {sheet.canonical_theory(ref["key"]) for ref in check["excluded_theories"]}
            for key, row in linked:
                if row["kind"] != "comparison" or row["version"] != sheet.rows[key]["version"]:
                    continue
                examined(key, row)
                applications = [sheet.rows[row["data"][side]] for side in ("left", "right")]
                theories = {sheet.canonical_theory(app["data"]["theory"]) for app in applications}
                # Canonical identity alone does not prove a scientific difference.
                # Require an executed dependence assessment of the linked application.
                candidate_applications = {row["data"][side] for side in ("left", "right")
                    if sheet.canonical_theory(sheet.rows[row["data"][side]]["data"]["theory"]) not in excluded}
                protocols = [(pkey, prow) for pkey, prow in linked if prow["kind"] == "importance"
                             and prow["version"] == sheet.rows[pkey]["version"]
                             and prow["data"]["application"] in candidate_applications]
                from .dependence import dependence_result
                for pkey, prow in protocols:
                    examined(pkey, prow)
                if len(theories) > 1 and theories & excluded and candidate_applications and any(
                        dependence_result(sheet, pkey)["dependence"]["status"] == "demonstrated"
                        for pkey, _ in protocols):
                    witnesses.append(key)
            if witnesses:
                status, reason = "fulfilled", "A linked scientific comparison distinguishes canonical accounts with executed G dependence"
        results.append({"criterion": criterion["id"], "requirement": deepcopy(requirement),
                        "verified_scope": "native_fields_only" if requirement["kind"] == "native" else "scientific_requirement",
                        "status": status, "structures": structures,
                        "witnesses": sorted(set(witnesses)), "reason": reason,
                        "inputs": [inputs[key] for key in sorted(inputs)]})
    return {"goal": finding["goal"], "finding": finding_key, "results": results}


def create_discovery_goal(source, proposal):
    from .causal_sheet import CausalSheet
    sheet = CausalSheet(source.ledger.causal_sheet.history)
    data = deepcopy(proposal["goal"])
    row = sheet.put(proposal["key"], "discovery_goal", data)
    return ResearchResult(causal_sheet_updates=(row,))


def submit_finding(source, proposal):
    from .causal_sheet import CausalSheet
    sheet = CausalSheet(source.ledger.causal_sheet.history)
    goal_key = source.commission.discovery_goal
    data = {"goal": goal_key, "goal_version": sheet.rows[goal_key]["version"],
            **{name: deepcopy(proposal[name]) for name in
               ("answer", "source_claims", "source_observations", "scientific_links")}}
    row = sheet.put(proposal["key"], "discovery_finding", data)
    return ResearchResult(causal_sheet_updates=(row, *recheck_findings(sheet, [row["key"]])))


def recheck_findings(sheet, finding_keys=None):
    """Retain changed criterion outcomes on the caller's staged scientific sheet.

    Scientific results can resolve an earlier finding. The original completed
    action then includes the host receipt before any credit is projected. An
    unchanged source/result set produces no new receipt.
    """
    keys = [key for key, _ in sheet.of_kind("discovery_finding")] if finding_keys is None else list(finding_keys)
    updates = []
    for key in keys:
        sheet._require_kind(key, "discovery_finding")
        result = evaluate_finding(sheet, key)
        receipt_key = "discovery-check:" + identity([sheet.rows[key]["version"], result])
        if receipt_key not in sheet.rows:
            updates.append(sheet.put(receipt_key, "discovery_check", result))
    return tuple(updates)


def _source_identity(sheet, finding):
    return sorted(["claim:" + ident for ident, _, _, _ in _native_claims(sheet, finding)]
                  + ["observation:" + ident for ident, _, _ in _native_observations(sheet, finding)])


def _inference_identity(sheet, reference):
    from ...credit_assignment.causal_progress import _inference_identity as canonical_g
    return canonical_g(sheet.snapshot(reference))


def _application_identity(sheet, key):
    row = sheet.rows[key]
    data = row["data"]
    return [_inference_identity(sheet, {"key": data["inference"], "version": data["inference_version"]}),
            sheet.canonical_theory(data["theory"]), data["context_key"]]


def _protocol_identity(sheet, key):
    from .importance import protocol_identity
    protocol = deepcopy(sheet.rows[key]["data"])
    protocol["application"] = "application:" + identity(_application_identity(sheet, protocol["application"]))
    return protocol_identity(protocol)


def _lean_identity(receipt):
    """Ignore proof scripts, tool paths, record aliases and parameter spelling."""
    parameters = [binding for binding in receipt["result"]["bindings"] if "type" in binding]
    parameters.sort(key=lambda binding: identity([binding["type"], binding["scientific_value"]]))
    names = {binding["name"]: f"parameter_{index}" for index, binding in enumerate(parameters)}
    assumptions = [binding for binding in receipt["result"]["bindings"] if "proposition" in binding]
    assumptions.sort(key=lambda binding: identity(binding["scientific_value"]))
    names.update((binding["name"], f"assumption_{index}") for index, binding in enumerate(assumptions))
    def term(value):
        renamed = re.sub(r"[^\W\d]\w*", lambda match: names.get(match.group(), match.group()), value)
        return "".join(renamed.split())
    request = receipt["request"]
    return {"relation": request["relation"], "terms": sorted([term(request["left"]), term(request["right"])]),
        "parameters": [[term(binding["type"]), binding["scientific_value"]] for binding in parameters],
        "assumptions": [[term(binding["proposition"]), binding["scientific_value"]] for binding in assumptions]}


def _semantic_witness_identity(sheet, finding, criterion, witness, verified_inputs):
    predicate = criterion["requirement"]["predicate"]
    kind = predicate["kind"]
    if kind == "theory_structure":
        # contribution_identities already binds canonical scientific structure.
        # Fragment aliases, added citations, finding names and criterion labels
        # are provenance, not additional represented scientific contributions.
        return identity([kind, witness])
    reference = next(ref for ref in verified_inputs if ref["key"] == witness)
    row = sheet.snapshot(reference)
    if kind == "formal_equivalence":
        result = _lean_identity(row["data"])
    elif kind == "causal_dependence":
        protocol = row["data"]["importance"] if row["kind"] == "support_assessment" else witness
        result = {"G": _inference_identity(sheet, predicate["inference"]),
                  "protocol": _protocol_identity(sheet, protocol)}
    else:
        result = sorted([_application_identity(sheet, row["data"][side]) for side in ("left", "right")], key=identity)
    return identity([_source_identity(sheet, finding), kind, result])


def compile_discovery_points(sheet, goal_key=None):
    """Global new-information credit from saved, verified criterion witnesses.

    A distinct retained native source claim/observation supplies new information
    once globally. Reusing it for another goal or a narrower criterion does not
    create another information unit. Formal criteria use stable candidate source
    operands and scientific result content, never a new check/tool record name
    or requested schema. Eligibility remains request-specific; global novelty
    is different from a Discovery episode's one-slot-per-criterion yield.
    """
    points, dependencies, outcomes = {}, {}, []
    native_points, scientific_points = {}, {}
    receipts = {}
    for key, row in sheet.of_kind("discovery_check"):
        receipts.setdefault(row["data"]["finding"], []).append((key, row))
    for key, row in sheet.of_kind("discovery_finding"):
        finding = row["data"]
        if goal_key is not None and finding["goal"] != goal_key:
            continue
        goal = sheet.snapshot({"key": finding["goal"], "version": finding["goal_version"]})["data"]
        checks = receipts.get(key, ())
        credited, native_matched, scientific_fulfilled = set(), set(), set()
        criteria = {item["id"]: item for item in goal["criteria"]}
        for receipt_key, receipt in checks:
            for check in receipt["data"]["results"]:
                if check["status"] not in {"fulfilled", "native_match"}:
                    continue
                criterion = criteria[check["criterion"]]
                requirement = criterion["requirement"]
                predicate = requirement["predicate"]
                native = requirement["kind"] == "native"
                credited.add(check["criterion"])
                (native_matched if native else scientific_fulfilled).add(check["criterion"])
                for witness in check["witnesses"]:
                    witness_identity = witness if native else (
                        _semantic_witness_identity(sheet, finding, criterion, witness, check["inputs"]))
                    point = "discovery:" + identity([predicate["kind"], witness_identity])
                    points[point] = "1"
                    (native_points if native else scientific_points)[point] = "1"
                    dependencies.setdefault(point, set()).update([key, finding["goal"], receipt_key,
                        *(ref["key"] for ref in check["inputs"])])
        if checks:
            receipt_key, receipt = checks[-1]
            outcomes.append({**deepcopy(receipt["data"]),
                "check_receipt": {"key": receipt_key, "version": receipt["version"]},
                "credited_criteria": sorted(credited), "native_matched_criteria": sorted(native_matched),
                "fulfilled_scientific_criteria": sorted(scientific_fulfilled)})
        else:
            outcomes.append({"goal": finding["goal"], "finding": key, "results": [],
                "check_receipt": None, "credited_criteria": [], "native_matched_criteria": [],
                "fulfilled_scientific_criteria": [],
                "status": "awaiting_host_verification"})
    return {"points": points, "point_dependencies": {key: sorted(value) for key, value in dependencies.items()},
            "outcomes": outcomes, "total": str(sum(map(Fraction, points.values()), Fraction())),
            "native_information_points": native_points,
            "scientific_fulfillment_points": scientific_points,
            "native_information_total": str(sum(map(Fraction, native_points.values()), Fraction())),
            "scientific_fulfillment_total": str(sum(map(Fraction, scientific_points.values()), Fraction()))}


def discovery_credit(result, baseline, goal_key, column):
    """Local slots distinguish native information from scientific fulfillment.

    Root observes globally new information separately. This local episode does
    not keep earning yield merely by collecting further same-criterion facts.
    """
    def slots(value):
        sheet = Ledger(value).causal_sheet
        return {identity([goal_key, result["verified_scope"], result["criterion"]])
                for _, row in sheet.of_kind("discovery_check") if row["data"]["goal"] == goal_key
                for result in row["data"]["results"] if result["status"] in {"fulfilled", "native_match"}}
    points, before = slots(result), slots(baseline)
    return Credit(evaluated=result.evaluated, quality_resolution=1.0,
                  qualities=tuple((column, key, 1.0) for key in sorted(points - before)))


def source_credit(result, baseline, goal_key, column, *, selected_sources):
    """Source-level information yield; this is not Discovery fulfillment credit.

    Each requested slot earns at most one local information unit. A semantic
    criterion may expose useful source operands before Scientific can determine
    equivalence, dependence or account distinction; those operands remain local
    search information and do not manufacture a fulfilled Discovery criterion.
    Only operands from this Search's selected sources are eligible. Remove
    stable native witnesses already known at episode start before projecting
    into criterion slots: an old unrelated operand must not fill a fresh goal,
    including when a selected source later supplies genuinely new information.
    """
    if not isinstance(selected_sources, (set, frozenset)) or any(
            not isinstance(source, str) or not source.strip() for source in selected_sources):
        raise ValueError("Search credit requires its explicit selected source set")
    def operands(sheet, *, selected_only):
        for _, row in sheet.of_kind("evidence"):
            data = row["data"]
            if selected_only and data["source"] not in selected_sources:
                continue
            native = data["measurements"].get("native", {})
            for kind, collection, identify in (("claim", "claims", claim_identity),
                                                ("observation", "observations", observation_identity)):
                for selector in native.get(collection, {}):
                    ident, detail = identify(data, selector)
                    yield (kind, ident), detail
    sheet = Ledger(result).causal_sheet
    inherited = {witness for witness, _ in operands(Ledger(baseline).causal_sheet, selected_only=False)}
    eligible = [(witness[0], detail) for witness, detail in operands(sheet, selected_only=True)
                if witness not in inherited]
    goal = sheet.rows[goal_key]["data"]
    found = set()
    for criterion in goal["criteria"]:
        requirement = criterion["requirement"]
        check = requirement["predicate"] if requirement["kind"] == "native" else requirement["candidate"]
        for kind, detail in eligible:
            matches = check is None
            if check is not None and kind == "claim" and check["kind"] == "source_claim":
                operand = {**detail["structure"], "statement_id": detail["statement_id"],
                           "locator": detail["native_locator"]}
                matches = _matches(Draft202012Validator(check["claim_schema"]), operand,
                                   "source_claim criterion.claim_schema")
            elif check is not None and kind == "observation" and check["kind"] == "source_value":
                try:
                    measured = _field(detail["measurement"], check["field_path"])
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
                matches = _matches(Draft202012Validator(check["value_schema"]), measured,
                                   "source_value criterion.value_schema")
            if matches:
                found.add(identity([goal["inputs"], requirement["kind"], requirement["predicate"], check]))
    return Credit(evaluated=result.evaluated, quality_resolution=1.0,
                  qualities=tuple((column, key, 1.0) for key in sorted(found)))


def discovery_context(source):
    sheet = source.ledger.causal_sheet
    key = source.commission.discovery_goal
    goal = sheet.rows[key]
    known = {(row["key"], row["version"]): row for row in sheet.history}
    selected = {}
    embedded = set()
    def select(ref):
        pair = (ref["key"], ref["version"])
        if pair in known:
            selected[pair] = known[pair]
    def collect(value):
        if isinstance(value, dict):
            if isinstance(value.get("key"), str) and isinstance(value.get("version"), str):
                select(value)
                if "data" in value:
                    embedded.add((value["key"], value["version"]))
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
    for ref in goal["data"]["inputs"]:
        select(ref)
    for rkey in (*source.commission.inputs, *source.selected_keys):
        row = sheet.rows[rkey]
        select({"key": rkey, "version": row["version"]})
    for ref in source.scoped_snapshots():
        select(ref)
    collect(source.working_assessment.get("record_refs", ()))
    collect(source.unreviewed_results)
    collect(source.unreviewed_children)
    evidence = []
    scientific = []
    for pair, row in selected.items():
        if row["key"] == key or pair in embedded:
            continue
        target = evidence if row["kind"] == "evidence" else scientific
        target.append(sheet.snapshot_view([{"key": pair[0], "version": pair[1]}])[0])
    available = []
    for (rkey, version), row in known.items():
        if row["kind"] != "evidence":
            continue
        native = row["data"]["measurements"].get("native", {})
        available.append({"key": rkey, "version": version, "kind": "evidence",
            "source": row["data"]["source"], "locator": row["data"]["locator"],
            "claim_selectors": sorted(native.get("claims", {})),
            "observation_selectors": sorted(native.get("observations", {}))})
    return {"discovery_goal": {"key": key, "version": goal["version"], **deepcopy(goal["data"])},
            "criterion_results": compile_discovery_points(sheet, key)["outcomes"],
            "source_evidence": evidence,
            "available_source_evidence": available,
            "scientific_inputs": scientific, "historical_inputs": []}


def reader_extraction_context(reader, question):
    """Give processors the acquisition question and native operand definitions.

    Custom answer fields belong to Discovery's assembled answer. The leaf returns
    source-native claims/observations, with no reconstruction graph, scientific
    scores, parent history or independently selected research assignment.
    """
    goal = reader.ledger.causal_sheet.rows[reader.commission.discovery_goal]["data"]
    criteria = []
    for criterion in goal["criteria"]:
        requirement = criterion["requirement"]
        check = requirement["predicate"] if requirement["kind"] == "native" else requirement["candidate"]
        if check is not None:
            criteria.append({"id": criterion["id"],
                "purpose": "native_requirement" if requirement["kind"] == "native" else "scientific_candidate_filter",
                "native_constraint": deepcopy(check)})
    return {"question": question, "requested_source_criteria": criteria,
            "native_operand_definitions": native_operand_reference(),
            "instruction": "Extract the source's claims and measurements relevant to this reading question. "
                "Use the native operand fields below. The requested criteria describe the information need, "
                "not observations made by the source. Preserve source-native symbols, units, definitions and "
                "conditions. Observation source_id may be empty; readers supply host-validated occurrence "
                "anchors for unlabeled observations/claims. Locator prose is provenance, not identity. "
                "Never substitute a document URL or study ID for a missing native observation ID. "
                "Scientific reconstructs and compares theories from the returned evidence."}
