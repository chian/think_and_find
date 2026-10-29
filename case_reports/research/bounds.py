"""Executable tests of expected generalizability bounds, not confidence grades.

An expectation remains a revisable scientific proposal. This adapter makes one
finite/sampled realization explicit: membership of each declared circumstance
and the expected answer. It never equates a successful calculation with truth
of the model's transcription or empirical premises.
"""
import ast
import io
import tokenize
import json
from functools import lru_cache
from fractions import Fraction

from .importance import expression, rational, answer
from .dependence import paired_case_result, answer_matches, validate_target, target_schema, target_identity

FIELDS = ('expectation', 'assessment', 'scope', 'claim_relation', 'claim_intervention', 'target', 'grounds', 'rationale')


def expression_identity(source):
    """Whitespace-insensitive syntax identity preserving exact numeric lexemes."""
    ignored = {tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER, tokenize.COMMENT}
    return tuple((item.type, item.string) for item in tokenize.generate_tokens(io.StringIO(source).readline) if item.type not in ignored)


def references(data):
    return [data['assessment'], *data['grounds']]


def snapshot_references(data):
    return [data['expectation']]


def claim(sheet, expectation):
    row = sheet.snapshot(expectation)
    if row['kind'] != 'expectation':
        raise ValueError('Expected generalizability bounds must reference an expectation snapshot')
    key = row['data']['inference']
    return {'key': key, 'version': row['dependencies'][key]}


def context_pins(sheet, data):
    result = {}
    for ref in (data['expectation'], claim(sheet, data['expectation'])):
        result.setdefault(ref['key'], set()).add(ref['version'])
    return result


def merge_pins(*groups):
    result = {}
    for group in groups:
        for key, versions in group.items():
            result.setdefault(key, set()).update(versions)
    return result


def validate(sheet, key, data):
    if not isinstance(data, dict) or set(data) != set(FIELDS):
        raise ValueError('Expected-bounds test requires its exact declared fields')
    original = claim(sheet, data['expectation'])
    sheet._require_kind(data['assessment'], 'importance')
    protocol = sheet.rows[data['assessment']]['data']
    application = sheet.rows[protocol['application']]['data']
    if (application['inference'], application['inference_version']) != (original['key'], original['version']):
        raise ValueError('Expected-bounds test and protocol must address the same exact G version')
    validate_target(data['target'])
    for name in ('scope', 'rationale'):
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError('Expected-bounds membership and rationale must be explicit')
    if not isinstance(data['claim_relation'], str) or len(data['claim_relation']) > 2048:
        raise ValueError('Original G relation must be a bounded predicate, or empty when unformalized')
    if len(data['scope']) > 2048:
        raise ValueError('Membership expression exceeds the supported size')
    try:
        tree = ast.parse(data['scope'], mode='eval')
    except SyntaxError as exc:
        raise ValueError('Membership must be a mathematical Boolean expression') from exc
    if len(list(ast.walk(tree))) > 256:
        raise ValueError('Membership expression exceeds the supported operation count')
    names = set(protocol['domain']['cases'][0]['inputs']) | {'abs', 'min', 'max'}
    validate_intervention(data['claim_intervention'], protocol)
    if data['claim_relation'].strip():
        try:
            relation_tree = ast.parse(data['claim_relation'], mode='eval')
        except SyntaxError as exc:
            raise ValueError('Original G relation must be a mathematical Boolean predicate') from exc
        if {'baseline', 'altered', 'focal_baseline', 'focal_altered'} & set(protocol['domain']['cases'][0]['inputs']):
            raise ValueError('Claim-relation inputs reserve baseline and altered for paired answers')
        if len(list(ast.walk(relation_tree))) > 256 or any(isinstance(node, ast.Name) and node.id not in names | {'baseline', 'altered', 'focal_baseline', 'focal_altered'} for node in ast.walk(relation_tree)):
            raise ValueError('Original G relation may use only paired answers and circumstance inputs')
    if any(isinstance(node, ast.Name) and node.id not in names for node in ast.walk(tree)):
        raise ValueError('Membership may use only declared circumstance inputs')
    if not isinstance(data['grounds'], list) or len(data['grounds']) > 16 or any(
        not isinstance(ref, str) or ref not in sheet.rows for ref in data['grounds']):
        raise ValueError('Test grounds must be bounded explicit scientific keys')
    for other_key, other in sheet.of_kind('expectation_test'):
        old = other['data']
        if other_key != key and old['expectation'] == data['expectation'] and old['assessment'] == data['assessment']:
            if (expression_identity(old['scope']) != expression_identity(data['scope']) or
                    target_identity(old['target']) != target_identity(data['target']) or
                    expression_identity(old['claim_relation']) != expression_identity(data['claim_relation']) or
                    intervention_identity(old['claim_intervention']) != intervention_identity(data['claim_intervention'])):
                raise ValueError('A new test key cannot redefine the same B_G/protocol scope or target; revise the expectation explicitly')
    previous = sheet.rows.get(key)
    if previous and previous['data'] != data:
        raise ValueError('Expected-bounds tests are immutable; use a new explicit test with a new expectation when revising expectations')


def membership(data, inputs):
    try:
        value = expression(data['scope'], {name: rational(value) for name, value in inputs.items()})
        if type(value) is not bool:
            raise ValueError('Membership expression must produce a Boolean')
        return value, ''
    except (ValueError, KeyError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
        return None, str(exc)


def evaluate_claim_relation(predicate, receipt, inputs, focal=None):
    if not predicate.strip():
        return None, 'Original G has no declared executable relation'
    if receipt is None or receipt['baseline'] is None or receipt['altered'] is None:
        return None, 'Original G relation needs both executed paired answers'
    if receipt['baseline']['type'] != receipt['altered']['type']:
        return None, 'Paired answers have different types; no implicit Boolean/numeric conversion'
    env = {name: rational(value) for name, value in inputs.items()}
    for name in ('baseline', 'altered'):
        answer = receipt[name]
        env[name] = answer['value'] if answer['type'] == 'boolean' else rational(answer['value'])
    if focal is not None:
        if focal['baseline'] is None or focal['altered'] is None:
            return None, 'Original G relation needs both executed focal answers'
        if focal['baseline']['type'] != focal['altered']['type']:
            return None, 'Paired focal answers have different types'
        for name in ('baseline', 'altered'):
            answer = focal[name]
            env['focal_' + name] = answer['value'] if answer['type'] == 'boolean' else rational(answer['value'])
    try:
        result = expression(predicate, env)
        if type(result) is not bool:
            raise ValueError('Original G relation must produce a Boolean')
        return result, ''
    except (ValueError, KeyError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
        return None, str(exc)


def normalized_intervention(value):
    if not isinstance(value, dict):
        raise ValueError('An intervention must be an explicit object')
    if value.get('kind') in {'input', 'edge', 'node'}:
        return value
    # Old input maps remain readable. New schemas declare the operator type.
    return {'kind': 'input', 'replacements': value}


def intervention_identity(value):
    operator = normalized_intervention(value)
    if operator['kind'] == 'input':
        return ('input', tuple(sorted((name, expression_identity(source)) for name, source in operator['replacements'].items())))
    if operator['kind'] == 'edge':
        return ('edge', operator['parent'], operator['child'], expression_identity(operator['replacement']))
    return ('node', operator['node'], expression_identity(operator['replacement']))


def validate_intervention(value, protocol):
    operator = normalized_intervention(value)
    inputs = set(protocol['domain']['cases'][0]['inputs'])
    equations = set(protocol['model']['equations'])
    kind = operator['kind']
    if kind == 'input':
        if set(operator) != {'kind', 'replacements'} or not isinstance(operator['replacements'], dict):
            raise ValueError('Input intervention needs only kind and replacements')
        replacements = operator['replacements']
        if len(replacements) > 32 or not set(replacements) <= inputs:
            raise ValueError('Input intervention names existing exogenous circumstance inputs')
        expressions, allowed = list(replacements.values()), inputs | {'abs', 'min', 'max'}
    elif kind == 'edge':
        if set(operator) != {'kind', 'parent', 'child', 'replacement'}:
            raise ValueError('Edge intervention needs kind, parent, child and replacement')
        if operator['parent'] not in inputs | equations or operator['child'] not in equations:
            raise ValueError('Edge intervention requires declared parent and child nodes')
        expressions, allowed = [operator['replacement']], inputs | equations | {'abs', 'min', 'max'}
    else:
        if set(operator) != {'kind', 'node', 'replacement'} or operator['node'] not in equations:
            raise ValueError('Node intervention needs kind, declared node and replacement')
        expressions, allowed = [operator['replacement']], inputs | equations | {'abs', 'min', 'max'}
    for source in expressions:
        if not isinstance(source, str) or not source.strip() or len(source) > 2048:
            raise ValueError('Intervention replacement must be a bounded mathematical expression')
        try:
            tree = ast.parse(source, mode='eval')
        except SyntaxError as exc:
            raise ValueError('Intervention replacement is not an expression') from exc
        if len(list(ast.walk(tree))) > 256 or any(isinstance(node, ast.Name) and node.id not in allowed for node in ast.walk(tree)):
            raise ValueError('Intervention replacement uses undeclared variables or exceeds supported size')


def _substitute_name(source, name, replacement):
    """Replace one named edge input while retaining exact numeric tokens."""
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME and token.string == name:
            tokens.extend([(tokenize.OP, '('), *[(item.type, item.string) for item in
                tokenize.generate_tokens(io.StringIO(replacement).readline)
                if item.type not in {tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER}], (tokenize.OP, ')')])
        else:
            tokens.append((token.type, token.string))
    return tokenize.untokenize(tokens)


@lru_cache(maxsize=4096)
def _intervened_answer(protocol_json, inputs_json, intervention_json, focal=False):
    protocol, inputs, raw = json.loads(protocol_json), json.loads(inputs_json), json.loads(intervention_json)
    validate_intervention(raw, protocol)
    if focal:
        protocol['model']['answer'] = protocol['model']['focal']
    operator = normalized_intervention(raw)
    if operator['kind'] == 'input':
        env = {name: rational(value) for name, value in inputs.items()}
        overrides = {name: expression(source, env) for name, source in operator['replacements'].items()}
        if any(type(value) is bool for value in overrides.values()):
            raise ValueError('Input interventions must retain numeric input type')
        inputs = {**inputs, **{name: str(value) for name, value in overrides.items()}}
    elif operator['kind'] == 'node':
        protocol['model']['equations'][operator['node']] = operator['replacement']
    else:
        child = operator['child']
        protocol['model']['equations'][child] = _substitute_name(protocol['model']['equations'][child], operator['parent'], operator['replacement'])
    return answer(protocol, inputs, altered=False)


def checked_claim_relation(predicate, receipt, inputs, protocol, intervention):
    if not intervention or normalized_intervention(intervention) == {'kind': 'input', 'replacements': {}}:
        return None, 'Original G has no declared common intervention'
    if receipt is None or receipt['altered'] is None:
        return None, 'Common intervention requires an executed altered answer'
    from .dependence import focal_answers
    focal = focal_answers(protocol, inputs)
    try:
        expected = _intervened_answer(json.dumps(protocol, sort_keys=True), json.dumps(inputs, sort_keys=True),
                                     json.dumps(intervention, sort_keys=True))
        if expected != receipt['altered']:
            return None, 'Saved altered answer does not realize the declared common intervention'
        expected_focal = _intervened_answer(json.dumps(protocol, sort_keys=True), json.dumps(inputs, sort_keys=True),
                                           json.dumps(intervention, sort_keys=True), focal=True)
        if focal['altered'] is None:
            return None, 'Common intervention focal evaluation unresolved: ' + focal['error']
        if expected_focal != focal['altered']:
            return None, 'Focal altered value does not realize the declared common intervention, even if the final prediction masks the mismatch'
    except (ValueError, KeyError, ZeroDivisionError, OverflowError, TypeError, RecursionError) as exc:
        return None, 'Common intervention execution unresolved: ' + str(exc)
    return evaluate_claim_relation(predicate, receipt, inputs, focal)


def classify_case(sheet, key, case_id):
    sheet._require_kind(key, 'expectation_test')
    data = sheet.rows[key]['data']
    protocol = sheet.rows[data['assessment']]['data']
    cases = {case['id']: case for case in protocol['domain']['cases']}
    if case_id not in cases:
        raise ValueError('Select an existing circumstance from the test protocol')
    case = cases[case_id]
    pins = context_pins(sheet, data)
    current = sheet.usable(key, _pins=pins)
    in_scope, error = membership(data, case['inputs']) if current else (None, 'Expected-bounds test has stale inputs')
    receipt = paired_case_result(sheet, data['assessment'], case_id, _pins=pins) if current else None
    baseline = receipt['baseline'] if receipt else None
    altered = receipt['altered'] if receipt else None
    relation, relation_error = checked_claim_relation(data['claim_relation'], receipt, case['inputs'], protocol, data['claim_intervention'])
    return {'case_id': case_id, 'weight': str(rational(case['weight'])), 'in_scope': in_scope,
            'baseline': baseline, 'altered': altered, 'original_G_relation': relation, 'relation_error': relation_error,
            'baseline_fits': answer_matches(baseline, data['target']),
            'altered_fits': answer_matches(altered, data['target']),
            'error': '; '.join(value for value in (error, receipt['error'] if receipt else 'Missing/current receipt unavailable') if value)}


def grade_bounds(sheet, key):
    sheet._require_kind(key, 'expectation_test')
    data = sheet.rows[key]['data']
    protocol = sheet.rows[data['assessment']]['data']
    cases = [classify_case(sheet, key, case['id']) for case in protocol['domain']['cases']]
    masses = {name: Fraction() for name in ('total', 'in_scope', 'outside_scope', 'membership_unresolved',
                                            'expected', 'broken', 'outcome_unresolved')}
    for case in cases:
        weight = Fraction(case['weight'])
        masses['total'] += weight
        if case['in_scope'] is None:
            masses['membership_unresolved'] += weight
        elif not case['in_scope']:
            masses['outside_scope'] += weight
        else:
            masses['in_scope'] += weight
            state = 'outcome_unresolved' if case['baseline_fits'] is None else 'expected' if case['baseline_fits'] else 'broken'
            masses[state] += weight
    return {'kind': 'expected_generalizability_bounds_test', 'test': sheet.anchor(key),
            'expectation': data['expectation'], 'inference': claim(sheet, data['expectation']),
            'assessment': sheet.anchor(data['assessment']), 'current': sheet.usable(key, _pins=context_pins(sheet, data)),
            'scope': data['scope'], 'claim_relation': data['claim_relation'], 'claim_intervention': data['claim_intervention'], 'target': data['target'], 'domain_kind': protocol['domain']['kind'],
            'masses': {name: str(value) for name, value in masses.items()}, 'cases': cases,
            'meaning': 'Tests recorded expectations within the declared circumstance measure. Broken expected behavior is retained; no numerical confidence in B_G is inferred.',
            'qualification': 'Finite enumeration only' if protocol['domain']['kind'] == 'finite' else 'Sampled circumstances only; no continuous-domain or population proof'}


def record_schema(snapshot_schema):
    text = {'type': 'string', 'minLength': 1}
    properties = {'expectation': snapshot_schema, 'assessment': text,
                  'scope': {**text, 'maxLength': 2048}, 'claim_relation': {'type': 'string', 'maxLength': 2048, 'description': 'Original G predicate over baseline/altered prediction, focal_baseline/focal_altered relation outputs, and circumstance inputs; empty leaves correspondence unresolved'}, 'target': target_schema(),
                  'claim_intervention': intervention_schema(),
                  'grounds': {'type': 'array', 'maxItems': 16, 'uniqueItems': True, 'items': text}, 'rationale': text}
    return {'type': 'object', 'additionalProperties': False, 'required': list(FIELDS), 'properties': properties}


def intervention_schema():
    text = {'type': 'string', 'minLength': 1, 'maxLength': 2048}
    def obj(properties):
        return {'type': 'object', 'additionalProperties': False, 'required': list(properties), 'properties': properties}
    return {'oneOf': [
        obj({'kind': {'const': 'input'}, 'replacements': {'type': 'object', 'maxProperties': 32, 'additionalProperties': text}}),
        obj({'kind': {'const': 'edge'}, 'parent': text, 'child': text, 'replacement': text}),
        obj({'kind': {'const': 'node'}, 'node': text, 'replacement': text}),
        {'type': 'object', 'maxProperties': 0}],
        'description': 'Same executed input, edge or node operator in every compared model. Edge substitution affects only the chosen child equation; empty leaves correspondence unresolved.'}
