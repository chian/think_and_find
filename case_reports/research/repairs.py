"""Executed consequences of explicit repairs under preserved expectations.

The source expected-bounds test fixes the denominator and target. Proposed
complete models are evaluated through ordinary importance-protocol receipts;
this permits additions to equations/SCMs/simulation state. A changed wording or
repair label supplies no numerical operand. Necessity is scoped only to the
explicitly compared alternatives, never all conceivable theory repairs.
"""
import ast
from fractions import Fraction

from . import bounds
from .dependence import paired_case_result, answer_matches, focal_answers
from .importance import rational, expression

FIELDS = ('break', 'expectation_test', 'target_test', 'target', 'candidate', 'alternatives', 'claim_relation', 'construction_cases', 'test_cases', 'rationale', 'limitations')


def references(data):
    return [data['break'], data['expectation_test'], data['target_test'], data['candidate'], *data['alternatives']]


def snapshot_references(data):
    return [data['target']]


def context_pins(sheet, data):
    source = sheet.rows[data['expectation_test']]['data']
    target = sheet.rows[data['target_test']]['data']
    return bounds.merge_pins(bounds.context_pins(sheet, source), bounds.context_pins(sheet, target))


def domain_signature(protocol):
    return (protocol['domain']['kind'], tuple(sorted((case['id'], str(rational(case['weight'])),
        tuple(sorted((name, str(rational(value))) for name, value in case['inputs'].items())))
        for case in protocol['domain']['cases'])))


def expression_identity(text):
    return bounds.expression_identity(text)


def model_difference(original, proposed):
    left, right = original['model'], proposed['model']
    old, new = left['equations'], right['equations']
    changed = [name for name in old.keys() & new.keys()
               if expression_identity(old[name]) != expression_identity(new[name])]
    def parents(equations):
        return {(node.id, child) for child, source in equations.items() for node in ast.walk(ast.parse(source, mode='eval'))
                if isinstance(node, ast.Name) and node.id not in {'abs', 'min', 'max'}}
    def branches(equations):
        return {name: sum(isinstance(node, ast.IfExp) for node in ast.walk(ast.parse(source, mode='eval')))
                for name, source in equations.items()}
    def same_focal():
        return (left['focal'] == right['focal'] and
                expression_identity(old[left['focal']]) == expression_identity(new[right['focal']]))
    return {'preserves_focal': same_focal(), 'focal_expression_changed': not same_focal(),
            'added_equations': sorted(new.keys() - old.keys()), 'removed_equations': sorted(old.keys() - new.keys()),
            'changed_equations': sorted(changed), 'added_dependencies': [list(edge) for edge in sorted(parents(new) - parents(old))],
            'removed_dependencies': [list(edge) for edge in sorted(parents(old) - parents(new))],
            'conditional_branches_before': branches(old), 'conditional_branches_after': branches(new),
            'initial_state_changed': left['initial'] != right['initial'], 'horizon_changed': left['steps'] != right['steps'],
            'qualification': 'Syntactic structural comparison, not proof of semantic equivalence or a magic-rule classification'}


def validate(sheet, key, data):
    if not isinstance(data, dict) or set(data) != set(FIELDS):
        raise ValueError('Repair assessment requires its exact declared fields')
    sheet._require_kind(data['break'], 'expectation_break')
    for name in ('expectation_test', 'target_test'):
        sheet._require_kind(data[name], 'expectation_test')
    original = sheet.rows[data['expectation_test']]['data']
    target = sheet.rows[data['target_test']]['data']
    if sheet.rows[data['break']]['data']['expectation'] != original['expectation']:
        raise ValueError('Repair must retain the exact expectation challenged by its recorded break')
    if target['expectation'] != data['target'] or target['assessment'] != data['candidate']:
        raise ValueError('Target test must bind exactly the proposed expectation and candidate model')
    original_g, target_g = bounds.claim(sheet, original['expectation']), bounds.claim(sheet, data['target'])
    if original_g['key'] != target_g['key']:
        raise ValueError('Repair compares versions of the same causal inference identity')
    cursor = data['target']
    seen = set()
    while cursor != original['expectation']:
        if cursor['version'] in seen:
            raise ValueError('Cyclic expectation revision history')
        seen.add(cursor['version'])
        revision = sheet.snapshot(cursor)['data']['revision']
        if revision is None:
            raise ValueError('Target expectation must descend from the challenged expectation')
        cursor = revision['prior']
    if not isinstance(data['alternatives'], list) or len(data['alternatives']) > 16 or len(set(data['alternatives'])) != len(data['alternatives']):
        raise ValueError('Declare at most sixteen distinct alternative protocol keys')
    if data['candidate'] in data['alternatives']:
        raise ValueError('The selected candidate is not its own alternative')
    source_protocol = sheet.rows[original['assessment']]['data']
    case_ids = {case['id'] for case in source_protocol['domain']['cases']}
    if not isinstance(data['claim_relation'], str) or len(data['claim_relation']) > 2048:
        raise ValueError('Original G relation must be a bounded predicate, or empty when unformalized')
    source_relation = original['claim_relation']
    if data['claim_relation'].strip() != source_relation.strip():
        if not data['claim_relation'].strip() or not source_relation.strip() or expression_identity(data['claim_relation']) != expression_identity(source_relation):
            raise ValueError('Repair must test the same original G relation recorded before the break; it cannot redefine the claim')
    input_names = set(source_protocol['domain']['cases'][0]['inputs'])
    if input_names & {'baseline', 'altered', 'focal_baseline', 'focal_altered'}:
        raise ValueError('Claim-relation inputs reserve baseline and altered for the paired outputs')
    if data['claim_relation'].strip():
        try:
            tree = ast.parse(data['claim_relation'], mode='eval')
        except SyntaxError as exc:
            raise ValueError('Original G relation must be a mathematical Boolean expression') from exc
        allowed = input_names | {'baseline', 'altered', 'focal_baseline', 'focal_altered', 'abs', 'min', 'max'}
        if len(list(ast.walk(tree))) > 256 or any(isinstance(node, ast.Name) and node.id not in allowed for node in ast.walk(tree)):
            raise ValueError('Original G relation may use only paired answers and declared circumstance inputs')
    for name in ('construction_cases', 'test_cases'):
        if not isinstance(data[name], list) or len(set(data[name])) != len(data[name]) or not set(data[name]) <= case_ids:
            raise ValueError('Construction/test cases must be unique declared circumstance IDs')
    if set(data['construction_cases']) & set(data['test_cases']):
        raise ValueError('A circumstance used in construction cannot also be declared an independent test')
    trigger_refs = sheet.rows[data['break']]['data']['triggers']
    linked = False
    for ref in trigger_refs:
        trigger = sheet.snapshot(ref)
        if ref == sheet.anchor(data['expectation_test']) or ref == sheet.anchor(original['assessment']):
            linked = True
        if trigger['kind'] == 'importance_result' and trigger['data']['assessment'] == original['assessment']:
            linked = True
        if any(sheet._depends_on(ground, ref['key']) and sheet.rows[ref['key']]['version'] == ref['version']
               for ground in original['grounds']):
            linked = True
    if not linked:
        raise ValueError('The preserved break must link its actual trigger snapshot to the source test, its protocol/result, or its explicit scientific grounds')
    for assessment in [data['candidate'], *data['alternatives']]:
        sheet._require_kind(assessment, 'importance')
        protocol = sheet.rows[assessment]['data']
        application = sheet.rows[protocol['application']]['data']
        if application['inference'] != original_g['key'] or application['inference_version'] not in {original_g['version'], target_g['version']}:
            raise ValueError('Repair models must explicitly realize the source or target G version')
        if domain_signature(protocol) != domain_signature(source_protocol):
            raise ValueError('All repair models retain the original circumstance identities, inputs, weights and domain kind')
        if protocol['model']['focal'] != source_protocol['model']['focal']:
            raise ValueError('Repair models must retain the named focal G variable; variable renaming requires an explicit mapping outside this adapter')
        if expression_identity(protocol['model']['answer']) != expression_identity(source_protocol['model']['answer']):
            raise ValueError('Repair models must compare the same answer expression; record explicit mappings in the models')
        if protocol['method'] == 'simulation' and source_protocol['method'] == 'simulation' and protocol['model']['steps'] != source_protocol['model']['steps']:
            raise ValueError('Repair comparisons must retain the simulation horizon')
    for name in ('rationale', 'limitations'):
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError('Repair interpretation and limits must be explicit')
    prior = sheet.rows.get(key)
    if prior:
        if any(prior['data'][name] != data[name] for name in FIELDS if name not in {'alternatives', 'rationale', 'limitations'}):
            raise ValueError('Repair identity retains its source, target and candidate; create another comparison for a different candidate')
        if not set(prior['data']['alternatives']) <= set(data['alternatives']):
            raise ValueError('Previously compared alternatives cannot be removed from a repair assessment')


def grade_repair(sheet, key):
    sheet._require_kind(key, 'repair_assessment')
    data = sheet.rows[key]['data']
    source_test = sheet.rows[data['expectation_test']]['data']
    source_grade = bounds.grade_bounds(sheet, data['expectation_test'])
    target_grade = bounds.grade_bounds(sheet, data['target_test'])
    target_cases = {case['case_id']: case for case in target_grade['cases']}
    source_protocol = sheet.rows[source_test['assessment']]['data']
    candidate_protocol = sheet.rows[data['candidate']]['data']
    declared_cases = {case['id']: case for case in source_protocol['domain']['cases']}
    pins = context_pins(sheet, data)
    current = sheet.usable(key, _pins=pins)
    source_g, target_g = bounds.claim(sheet, source_test['expectation']), bounds.claim(sheet, data['target'])
    difference = model_difference(source_protocol, candidate_protocol)
    changed_g = source_g != target_g
    alternatives = []
    for assessment in data['alternatives']:
        protocol = sheet.rows[assessment]['data']
        application = sheet.rows[protocol['application']]['data']
        alternatives.append({'assessment': assessment,
            'declared_g_preserved': application['inference_version'] == source_g['version'],
            **model_difference(source_protocol, protocol)})
    masses = {name: Fraction() for name in ('total', 'original_in_scope', 'outside_scope', 'scope_unresolved',
        'retained', 'repaired', 'lost', 'still_broken', 'excluded_failure', 'unresolved', 'qualifying_restored', 'qualifying_unresolved')}
    rows = []
    provenance_masses = {role: {} for role in ('construction', 'declared_test', 'unclassified')}
    for source in source_grade['cases']:
        cid, weight = source['case_id'], Fraction(source['weight'])
        target_scope = target_cases[cid]['in_scope'] if current else None
        receipt = paired_case_result(sheet, data['candidate'], cid, _pins=pins) if current else None
        candidate_answer = receipt['baseline'] if receipt else None
        fits = answer_matches(candidate_answer, source_test['target'])
        source_receipt = paired_case_result(sheet, source_test['assessment'], cid, _pins=pins) if current else None
        original_relation, original_relation_error = bounds.checked_claim_relation(data['claim_relation'], source_receipt, declared_cases[cid]['inputs'], source_protocol, source_test['claim_intervention'])
        candidate_relation, candidate_relation_error = bounds.checked_claim_relation(data['claim_relation'], receipt, declared_cases[cid]['inputs'], candidate_protocol, source_test['claim_intervention'])
        original_focal = focal_answers(source_protocol, declared_cases[cid]['inputs']) if source_receipt else None
        candidate_focal = focal_answers(candidate_protocol, declared_cases[cid]['inputs']) if receipt else None
        if original_focal and candidate_focal and original_focal['baseline'] and candidate_focal['baseline'] and original_focal['baseline']['type'] != candidate_focal['baseline']['type']:
            candidate_relation, candidate_relation_error = None, 'Candidate focal variable changes type relative to the original G realization'
        if source_receipt and receipt and source_receipt['baseline'] and receipt['baseline'] and source_receipt['baseline']['type'] != receipt['baseline']['type']:
            candidate_relation, candidate_relation_error = None, 'Candidate answer changes type relative to the original observable'

        scope = source['in_scope'] if current else None
        original_fits = source['baseline_fits'] if current else None
        alt_results = []
        for alternative in alternatives:
            receipt_alt = paired_case_result(sheet, alternative['assessment'], cid, _pins=pins) if current else None
            relation, relation_error = bounds.checked_claim_relation(data['claim_relation'], receipt_alt, declared_cases[cid]['inputs'], sheet.rows[alternative['assessment']]['data'], source_test['claim_intervention'])
            alternative_focal = focal_answers(sheet.rows[alternative['assessment']]['data'], declared_cases[cid]['inputs']) if receipt_alt else None
            if original_focal and alternative_focal and original_focal['baseline'] and alternative_focal['baseline'] and original_focal['baseline']['type'] != alternative_focal['baseline']['type']:
                relation, relation_error = None, 'Alternative focal variable changes type relative to the original G realization'
            if source_receipt and receipt_alt and source_receipt['baseline'] and receipt_alt['baseline'] and source_receipt['baseline']['type'] != receipt_alt['baseline']['type']:
                relation, relation_error = None, 'Alternative answer changes type relative to the original observable'
            alt_results.append({'original_G_relation': relation, 'relation_error': relation_error, 'assessment': alternative['assessment'], 'preserves_focal': alternative['preserves_focal'],
                'declared_g_preserved': alternative['declared_g_preserved'],
                'fits': answer_matches(receipt_alt['baseline'] if receipt_alt else None, source_test['target'])})
        # Only the executed same original-G predicate establishes finite preservation.
        # AST or version differences alone never establish a reversal.
        preserving = [alt for alt in alt_results if alt['original_G_relation'] is True]
        unknown_preservation = [alt for alt in alt_results if alt['original_G_relation'] is None]
        necessity = None
        if any(alt['fits'] is True for alt in preserving):
            necessity = False
        elif preserving and all(alt['fits'] is False for alt in preserving) and not unknown_preservation:
            necessity = True
        masses['total'] += weight
        if scope is None:
            classification = 'scope_unresolved'
        elif not scope:
            classification = 'outside_scope'
        else:
            masses['original_in_scope'] += weight
            if target_scope is False:
                classification = 'excluded_failure' if original_fits is False else 'lost' if original_fits is True else 'unresolved'
            elif target_scope is None or original_fits is None or fits is None:
                classification = 'unresolved'
            elif original_fits:
                classification = 'retained' if fits else 'lost'
            else:
                classification = 'repaired' if fits else 'still_broken'
        masses[classification] += weight
        qualified = False
        relation_change = (original_relation is True and candidate_relation is False)
        relation_unknown = original_relation is None or candidate_relation is None
        if scope is True:
            if classification == 'repaired' and relation_change:
                qualified = necessity
            elif classification in {'repaired', 'unresolved'} and (relation_change or relation_unknown) and necessity is not False:
                qualified = None
        elif scope is None and (relation_change or relation_unknown):
            qualified = None
        if qualified is True:
            masses['qualifying_restored'] += weight
        elif qualified is None:
            masses['qualifying_unresolved'] += weight
        role = 'construction' if cid in data['construction_cases'] else 'declared_test' if cid in data['test_cases'] else 'unclassified'
        provenance_masses[role][classification] = provenance_masses[role].get(classification, Fraction()) + weight
        rows.append({'case_role': role, 'case_id': cid, 'weight': str(weight), 'original_in_scope': scope,
            'target_in_scope': target_scope, 'baseline_fits': original_fits, 'candidate_fits': fits,
            'candidate_answer': candidate_answer, 'classification': classification,
            'original_G_relation': original_relation, 'candidate_original_G_relation': candidate_relation,
            'relation_errors': [value for value in (original_relation_error, candidate_relation_error) if value],
            'scoped_necessity': necessity, 'qualifying_restored': qualified, 'alternatives': alt_results})
    denominator = masses['original_in_scope']
    membership_unknown = masses['scope_unresolved']
    def fraction(lower_mass, unresolved_mass):
        possible_denominator = denominator + membership_unknown
        if not possible_denominator:
            return {'lower': '0', 'upper': '0', 'value': None, 'reason': 'No declared circumstance is within the original expected scope'}
        lower = lower_mass / possible_denominator
        upper = min(Fraction(1), (lower_mass + unresolved_mass) / denominator) if denominator else Fraction(1)
        return {'lower': str(lower), 'upper': str(upper),
                'value': str(lower) if not membership_unknown and not unresolved_mass else None}
    return {'kind': 'repair_consequence_comparison', 'assessment': sheet.anchor(key), 'current': current,
        'break': data['break'], 'source': source_test['expectation'], 'target': data['target'],
        'source_inference': source_g, 'target_inference': target_g, 'g_changed': changed_g,
        'source_test': data['expectation_test'], 'target_test': data['target_test'],
        'source_protocol': source_test['assessment'], 'candidate': data['candidate'],
        'claim_relation': data['claim_relation'], 'claim_intervention': source_test['claim_intervention'], 'model_changes': difference, 'alternatives': alternatives,
        'masses': {name: str(value) for name, value in masses.items()},
        'repaired_fraction': fraction(masses['repaired'], masses['unresolved'] + membership_unknown),
        'qualifying_restored_fraction': fraction(masses['qualifying_restored'], masses['qualifying_unresolved']),
        'case_provenance_masses': {role: {state: str(mass) for state, mass in totals.items()}
                                   for role, totals in provenance_masses.items()},
        'corroboration_qualification': 'A declared test partition is not proof of independent empirical corroboration; construction fit and test consequences are reported separately',
        'conceptual_equivalence': 'Original G is tested only through the declared executable relation over paired outputs. Model-to-natural-language correspondence and equivalence beyond the declared cases remain scientific obligations; syntax/version differences confer no credit',
        'cases': rows, 'domain_kind': source_protocol['domain']['kind'],
        'necessity_scope': 'Only explicitly compared G-preserving alternatives; empty or unexecuted alternatives leave necessity unresolved',
        'qualification': 'Executed predictions conditional on supplied models and target interpretation. Syntactic G changes do not prove conceptual non-equivalence. Sample results do not establish continuous-domain coverage.',
        'meaning': 'The original expected-scope denominator and target remain fixed. Excluding a failed case from revised bounds does not repair it. Lost successes remain explicit; no model verdict or repair-kind label supplies credit.'}


def record_schema(snapshot_schema):
    text = {'type': 'string', 'minLength': 1}
    properties = {name: text for name in FIELDS if name not in {'target', 'alternatives', 'claim_relation', 'construction_cases', 'test_cases'}}
    properties['claim_relation'] = {'type': 'string', 'maxLength': 2048, 'description': 'Original G predicate over baseline/altered final predictions, focal_baseline/focal_altered values and circumstance inputs; empty leaves the change unresolved'}
    properties.update(target=snapshot_schema, alternatives={'type': 'array', 'maxItems': 16, 'uniqueItems': True, 'items': text})
    for name in ('construction_cases', 'test_cases'):
        properties[name] = {'type': 'array', 'maxItems': 256, 'uniqueItems': True, 'items': text}
    return {'type': 'object', 'additionalProperties': False, 'required': list(FIELDS), 'properties': properties}
