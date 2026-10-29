"""Offline executable B_G/repair illustration; synthetic models, no research job."""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('episodic_workflow', root / '__init__.py', submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
    from episodic_workflow.case_reports.research.importance import evaluate_cases
    from episodic_workflow.case_reports.research.bounds import grade_bounds
    from episodic_workflow.case_reports.research.repairs import grade_repair

    sheet = CausalSheet()
    sheet.put('G', 'inference', dict(statement='Input x contributes x to m', variables=['x', 'm', 'y'],
        intervention='Vary x', outcome='y', grain='scalar', domain='Synthetic declared models'))
    original_g = sheet.anchor('G')
    sheet.put('B', 'expectation', dict(inference='G', bounds='Expect y between 1 and 10 in the declared positive-input models',
        basis='provisional_judgment', rationale='Deliberately breakable synthetic expectation', grounds=[], revision=None))
    original_b = sheet.anchor('B')
    sheet.put('T', 'theory', dict(name='Stipulated response system', mechanism='An intermediate m contributes to y',
        predictions=['Expected interval for y'], admission='candidate', same_as='', distinction='Synthetic', compared_with=[]))

    def application(key, inference):
        sheet.put(key, 'application', dict(inference='G', inference_version=inference['version'], theory='T',
            context_key=key, context='Synthetic common two-point domain', context_distinction='Stipulated parameterization',
            direction='unresolved', inference_use='m enters y', consequence_without_inference='y may change'))

    cases = [dict(id='one', inputs={'x': 1}, weight=1), dict(id='two', inputs={'x': 2}, weight=1)]
    def protocol(key, app, equations, replacement='0'):
        sheet.put(key, 'importance', dict(application=app, method='equations', question='What is y?',
            contrast='Remove m dependence for the associated dependence calculation', comparator='exact_typed_answer',
            domain=dict(kind='finite', description='x=1 and x=2', measure='Uniform exact finite mass', cases=cases),
            model=dict(equations=equations, focal='m', replacement=replacement, answer='y', initial={}, steps=0, edge=[])))
    def expected_test(key, expectation, assessment, scope='True', relation='focal_baseline - focal_altered == x'):
        sheet.put(key, 'expectation_test', dict(expectation=expectation, assessment=assessment, scope=scope, claim_relation=relation, claim_intervention={'kind': 'edge', 'parent': 'x', 'child': 'm', 'replacement': '0'},
            target=dict(kind='interval', lower='1', upper='10'), grounds=[], rationale='Executable realization of the stated synthetic expected behavior'))
    def revised(key, prior, changes, scope='Both original circumstances'):
        sheet.put(key, 'expectation', dict(inference='G', bounds=scope, basis='provisional_judgment', rationale='Synthetic revision',
            grounds=[], revision=dict(prior=prior, triggers=[sheet.anchor('break')], changes=changes, change='Compare a proposed repair',
                repair_kind='unresolved', proposed_rule='Use the named complete candidate model', general_rules='Stipulated equations only',
                rule_reasoning='Execution establishes conditional consequences, not conceptual non-equivalence',
                retained=[], lost=[], new_predictions=[], limitations='Finite synthetic illustration')))
    def repair(key, target, target_test, candidate, alternatives):
        sheet.put(key, 'repair_assessment', {'break': 'break', 'expectation_test': 'old_test', 'target_test': target_test,
            'target': target, 'candidate': candidate, 'alternatives': alternatives,
            'claim_relation': 'focal_baseline - focal_altered == x', 'construction_cases': ['one'], 'test_cases': ['two'], 'rationale': 'Compare exact responses on preserved original circumstances',
            'limitations': 'Necessity only among declared alternatives; target is a stipulated theoretical requirement'} )

    application('old_app', original_g)
    protocol('old', 'old_app', {'m': 'x', 'y': 'm-1'})
    expected_test('old_test', original_b, 'old')
    sheet.merge(evaluate_cases(sheet, 'old', ['one', 'two']))
    before = grade_bounds(sheet, 'old_test')
    sheet.put('break', 'expectation_break', dict(expectation=original_b, triggers=[sheet.anchor('old_test')],
        expected='y at least 1', observed='y=0 at x=1', conflict='A case inside expected bounds fails', limitations='Synthetic formal target'))
    protocol('failed_preserving', 'old_app', {'m': 'x', 'y': 'm-2'})
    protocol('successful_preserving', 'old_app', {'m': 'x', 'z': '1', 'y': 'm-1+z'})
    sheet.merge(evaluate_cases(sheet, 'failed_preserving', ['one', 'two']))
    revised('same_G_bounds', original_b, ['repair_theory'])
    same_b = sheet.anchor('same_G_bounds')
    expected_test('same_test', same_b, 'successful_preserving')
    repair('same_G_repair', same_b, 'same_test', 'successful_preserving', ['failed_preserving'])

    sheet.put('G', 'inference', {**sheet.rows['G']['data'], 'statement': 'Input x contributes negative x to m'})
    changed_g = sheet.anchor('G')
    application('changed_app', changed_g)
    protocol('candidate', 'changed_app', {'m': '-x', 'y': 'm+3'}, replacement='0')
    revised('new_B', original_b, ['change_inference'])
    new_b = sheet.anchor('new_B')
    expected_test('new_test', new_b, 'candidate', relation='focal_baseline - focal_altered == -x')
    repair('repair', new_b, 'new_test', 'candidate', ['failed_preserving'])
    missing = grade_repair(sheet, 'repair')
    sheet.merge(evaluate_cases(sheet, 'candidate', ['one', 'two'], context='repair'))
    completed = grade_repair(sheet, 'repair')
    repair('unexamined_repair', new_b, 'new_test', 'candidate', [])
    unexamined = grade_repair(sheet, 'unexamined_repair')
    if unexamined['qualifying_restored_fraction']['value'] is not None:
        raise RuntimeError('An empty repair comparison was treated as proof of necessity')
    # An unexecuted preserving-G alternative must reopen necessity, not count as failed.
    sheet.put('repair', 'repair_assessment', {**sheet.rows['repair']['data'], 'alternatives': ['failed_preserving', 'successful_preserving']})
    pending_alternative = grade_repair(sheet, 'repair')
    sheet.merge(evaluate_cases(sheet, 'successful_preserving', ['one', 'two'], context='repair'))
    recovered_without_G_change = grade_repair(sheet, 'repair')
    same_g = grade_repair(sheet, 'same_G_repair')

    revised('shrunk_B', new_b, ['shrink_bounds'], scope='Only x=2 remains expected')
    shrunk_b = sheet.anchor('shrunk_B')
    expected_test('shrunk_test', shrunk_b, 'candidate', scope='x == 2', relation='focal_baseline - focal_altered == -x')
    repair('shrunk_repair', shrunk_b, 'shrunk_test', 'candidate', ['failed_preserving'])
    shrunk = grade_repair(sheet, 'shrunk_repair')

    # Rewording G and changing its focal expression does not establish a reversal.
    # A missing offset changes m=x to m=x+1, while the same edge intervention
    # removes x and retains the offset. The original causal relation still holds.
    sheet.put('G', 'inference', {**sheet.rows['G']['data'], 'statement': 'Input x contributes x to m with an explicit offset'})
    offset_g = sheet.anchor('G')
    application('offset_app', offset_g)
    protocol('offset', 'offset_app', {'m': 'x+1', 'y': 'm-1'}, replacement='1')
    revised('offset_B', original_b, ['change_inference', 'repair_theory'])
    offset_b = sheet.anchor('offset_B')
    expected_test('offset_test', offset_b, 'offset')
    repair('offset_repair', offset_b, 'offset_test', 'offset', ['failed_preserving'])
    sheet.merge(evaluate_cases(sheet, 'offset', ['one', 'two'], context='offset_repair'))
    offset = grade_repair(sheet, 'offset_repair')
    if offset['repaired_fraction']['value'] != '1/2' or offset['qualifying_restored_fraction']['value'] != '0':
        raise RuntimeError('A changed G version/focal expression manufactured reversal credit despite preserved original relation')

    # A computed but scientifically different alteration cannot masquerade as
    # the same original-G intervention simply because its output changes sign.
    protocol('wrong_intervention', 'offset_app', {'m': 'x+1', 'y': 'm-1'}, replacement='100')
    expected_test('wrong_test', offset_b, 'wrong_intervention')
    repair('wrong_repair', offset_b, 'wrong_test', 'wrong_intervention', ['failed_preserving'])
    sheet.merge(evaluate_cases(sheet, 'wrong_intervention', ['one', 'two'], context='wrong_repair'))
    wrong = grade_repair(sheet, 'wrong_repair')
    if wrong['qualifying_restored_fraction']['lower'] != '0' or wrong['qualifying_restored_fraction']['value'] is not None:
        raise RuntimeError('An unrelated intervention manufactured evidence of a changed causal relation')

    # An edge intervention holds another x pathway fixed; a global x override
    # is a different intervention and must not satisfy this correspondence.
    from episodic_workflow.case_reports.research.bounds import checked_claim_relation
    from episodic_workflow.case_reports.research.importance import answer
    edge_protocol = {**sheet.rows['old']['data'], 'model': {**sheet.rows['old']['data']['model'],
                    'equations': {'m': 'x', 'y': 'm+x'}}}
    receipt = {'baseline': answer(edge_protocol, {'x': 2}),
               'altered': answer(edge_protocol, {'x': 2}, altered=True), 'error': ''}
    edge_relation, _ = checked_claim_relation('baseline - altered == x', receipt, {'x': 2}, edge_protocol,
                                              {'kind': 'edge', 'parent': 'x', 'child': 'm', 'replacement': '0'})
    global_relation, _ = checked_claim_relation('baseline - altered == x', receipt, {'x': 2}, edge_protocol,
                                                {'kind': 'input', 'replacements': {'x': '0'}})
    if edge_relation is not True or global_relation is not None:
        raise RuntimeError('An edge intervention was conflated with changing every pathway from x')

    masked_protocol = {**edge_protocol, 'model': {**edge_protocol['model'],
        'equations': {'m': 'x', 'y': 'm-m'}, 'replacement': '-1'}}
    masked_receipt = {'baseline': answer(masked_protocol, {'x': 2}),
        'altered': answer(masked_protocol, {'x': 2}, altered=True), 'error': ''}
    masked_relation, masked_error = checked_claim_relation('focal_baseline > focal_altered', masked_receipt,
        {'x': 2}, masked_protocol, {'kind': 'edge', 'parent': 'x', 'child': 'm', 'replacement': '0'})
    if masked_relation is not None or 'Focal altered value' not in masked_error:
        raise RuntimeError('A matching final prediction masked a different focal intervention')

    if before['masses']['broken'] != '1' or completed['repaired_fraction']['value'] != '1/2':
        raise RuntimeError('Expected failure or repaired consequence mass was lost')
    if completed['qualifying_restored_fraction']['value'] != '1/2':
        raise RuntimeError('Explicit failed alternative did not resolve the scoped comparison')
    if pending_alternative['qualifying_restored_fraction']['value'] is not None:
        raise RuntimeError('Missing alternative computation was treated as a failed alternative')
    if recovered_without_G_change['qualifying_restored_fraction']['value'] != '0' or same_g['qualifying_restored_fraction']['value'] != '0':
        raise RuntimeError('A G-preserving repair manufactured G-change credit')
    if shrunk['masses']['excluded_failure'] != '1' or shrunk['repaired_fraction']['value'] != '0':
        raise RuntimeError('Shrinking revised bounds erased the original failure')
    rejected = {}
    try:
        sheet.put('retconned_test', 'expectation_test', {**sheet.rows['old_test']['data'], 'scope': 'x == 2'})
    except ValueError as exc:
        rejected['same_expectation_scope_rewrite'] = str(exc)
    else:
        raise RuntimeError('A new key silently narrowed the same expected bounds realization')
    unrelated = CausalSheet(sheet.history)
    unrelated.put('unrelated_premise', 'premise', dict(statement='Unrelated assertion', scope='Synthetic', basis='Stipulation'))
    unrelated.put('unrelated_break', 'expectation_break', dict(expectation=original_b,
        triggers=[unrelated.anchor('unrelated_premise')], expected='Other behavior', observed='Other finding',
        conflict='Unrelated to the executed cases', limitations='Synthetic'))
    try:
        unrelated.put('mislinked_repair', 'repair_assessment', {**sheet.rows['repair']['data'], 'break': 'unrelated_break'})
    except ValueError as exc:
        rejected['unrelated_break'] = str(exc)
    else:
        raise RuntimeError('An unrelated break borrowed executable failures from a different source')
    if sheet.snapshot(original_b)['data']['bounds'] != 'Expect y between 1 and 10 in the declared positive-input models':
        raise RuntimeError('Original expected generalizability bounds were overwritten')
    print(json.dumps({'qualification': 'SYNTHETIC executed consequences, not empirical or universal causal verification',
        'original_bounds_masses': before['masses'], 'before_candidate_execution': missing['qualifying_restored_fraction'],
        'unexamined_alternatives': unexamined['qualifying_restored_fraction'],
        'executed_repair': completed['repaired_fraction'], 'finite_scoped_change': completed['qualifying_restored_fraction'],
        'unexecuted_alternative': pending_alternative['qualifying_restored_fraction'],
        'G_preserving_alternative': recovered_without_G_change['qualifying_restored_fraction'],
        'same_G_candidate': same_g['qualifying_restored_fraction'], 'scope_shrink': shrunk['masses'],
        'structural_repair': same_g['model_changes'],
        'new_G_version_offset_preserves_relation': offset['qualifying_restored_fraction'],
        'mismatched_intervention': wrong['qualifying_restored_fraction'],
        'edge_vs_global_intervention': {'edge_relation': edge_relation, 'global_relation': global_relation},
        'rejected': rejected, 'masked_focal_intervention': {'relation': masked_relation, 'reason': masked_error}, 'case_provenance': completed['case_provenance_masses']}, indent=2))


if __name__ == '__main__':
    main()
