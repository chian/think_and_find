"""Offline executable dependence examples; synthetic models, no research or model calls."""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('episodic_workflow', root / '__init__.py',
                                                 submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
    from episodic_workflow.case_reports.research.importance import evaluate_cases, result_key
    from episodic_workflow.case_reports.research.dependence import (
        dependence_result, paired_case_result, focal_case_result, focal_answers, answer_matches, target_identity)

    sheet = CausalSheet()
    sheet.put('G', 'inference', dict(statement='Input X contributes positively to response Y', variables=['X', 'Y'],
        intervention='Remove the positive coupling', outcome='Y', grain='scalar', domain='Synthetic finite models'))
    original_g = sheet.anchor('G')
    sheet.put('T', 'theory', dict(name='Synthetic linear response', mechanism='Y=X', predictions=['Y follows X'],
        admission='candidate', same_as='', distinction='Illustrative formal model only', compared_with=[]))
    sheet.put('A', 'application', dict(inference='G', theory='T', context_key='toy', context='Synthetic finite circumstances',
        context_distinction='One model for an executable demonstration', direction='unresolved',
        inference_use='The equation term X realizes the proposed contribution',
        consequence_without_inference='Y may differ with the term removed'))
    cases = [dict(id='negative', inputs={'x': -1}, weight=1), dict(id='zero', inputs={'x': 0}, weight=2),
             dict(id='positive', inputs={'x': 1}, weight=1)]

    def protocol(method, model=None):
        equations = {'Y': 'x'}
        default = dict(equations=equations, focal='Y', replacement='0', answer='Y', initial={}, steps=0, edge=[])
        if method == 'scm':
            default.update(equations={'X': 'x', 'Y': 'X'}, edge=['X', 'Y'])
        elif method == 'simulation':
            default.update(equations={'Y': 'Y+x'}, replacement='Y', initial={'Y': '0'}, steps=3)
        return dict(application='A', method=method, question='Does removing G change Y?',
            contrast='Remove the sole coupling while keeping inputs and answer fixed',
            domain=dict(kind='finite', description='Three declared input points', measure='Weights 1,2,1; mass 4', cases=cases),
            model=model or default, comparator='exact_typed_answer')

    sheet.put('equations', 'importance', protocol('equations'))
    absent = dependence_result(sheet, 'equations')
    if absent['dependence']['status'] != 'unresolved' or paired_case_result(sheet, 'equations', 'positive') is not None:
        raise RuntimeError('Missing executions became measured no-effect results')
    sheet.merge(evaluate_cases(sheet, 'equations', ['negative']))
    partial = dependence_result(sheet, 'equations')
    if partial['calculation']['lower'] != '1/4' or partial['calculation']['upper'] != '1':
        raise RuntimeError('Partial measured dependence lost the unresolved denominator')
    sheet.merge(evaluate_cases(sheet, 'equations', ['zero', 'positive']))
    complete = {'equations': dependence_result(sheet, 'equations')}
    for method in ('scm', 'simulation'):
        sheet.put(method, 'importance', protocol(method))
        sheet.merge(evaluate_cases(sheet, method, ['negative', 'zero', 'positive']))
        complete[method] = dependence_result(sheet, method)
    if any(value['calculation']['value'] != '1/2' for value in complete.values()):
        raise RuntimeError('Representation-specific execution did not give measured 1/2 dependence')

    # A syntactically present link is insufficient when its effect cancels.
    cancellation = dict(equations={'X': 'x', 'Y': 'X-X'}, focal='Y', replacement='0', answer='Y', initial={}, steps=0,
                        edge=['X', 'Y'])
    sheet.put('cancelled', 'importance', protocol('scm', cancellation))
    sheet.merge(evaluate_cases(sheet, 'cancelled', ['negative', 'zero', 'positive']))
    cancelled = dependence_result(sheet, 'cancelled')
    if cancelled['dependence']['status'] != 'not_demonstrated' or cancelled['calculation']['value'] != '0':
        raise RuntimeError('A syntactic SCM link was mistaken for consequential dependence')

    bad_model = dict(equations={'Y': 'x/x'}, focal='Y', replacement='0', answer='Y', initial={}, steps=0, edge=[])
    sheet.put('singular', 'importance', protocol('equations', bad_model))
    sheet.merge(evaluate_cases(sheet, 'singular', ['negative', 'zero', 'positive']))
    singular = dependence_result(sheet, 'singular')
    if singular['calculation']['value'] is not None or singular['calculation']['unresolved_mass'] != '2':
        raise RuntimeError('A failed execution was scored as an ordinary circumstance')

    rejected = {}
    real = sheet.rows[result_key('equations', 'positive')]['data']
    for label, changed in (
        ('fabricated answer', {**real, 'baseline': {'type': 'rational', 'value': '999'}}),
        ('fabricated failure', {**real, 'baseline': None, 'error': 'baseline: fabricated error'})):
        try:
            sheet.put(result_key('equations', 'positive'), 'importance_result', changed)
        except ValueError as exc:
            rejected[label] = str(exc)
        else:
            raise RuntimeError('A forged host-shaped receipt was admitted')

    target = {'kind': 'interval', 'lower': '1/2', 'upper': '3/2'}
    matches = {
        'inside': answer_matches({'type': 'rational', 'value': '1'}, target),
        'outside': answer_matches({'type': 'rational', 'value': '0'}, target),
        'missing': answer_matches(None, target),
        'boolean_not_number': answer_matches({'type': 'boolean', 'value': True}, target),
        'equivalent_rational_target': target_identity({'kind': 'exact', 'answer': {'type': 'rational', 'value': '1.0'}})
            == target_identity({'kind': 'exact', 'answer': {'type': 'rational', 'value': '1'}}),
    }
    if matches != dict(inside=True, outside=False, missing=None, boolean_not_number=False, equivalent_rational_target=True):
        raise RuntimeError('Target matcher lost type, exact arithmetic or unresolved state')

    # A downstream cancellation can hide an incorrect focal intervention.
    masked_model = dict(equations={'Y': 'x', 'Z': '-Y', 'out': 'Y+Z'}, focal='Y', replacement='-x',
                        answer='out', initial={}, steps=0, edge=[])
    sheet.put('masked', 'importance', protocol('equations', masked_model))
    if focal_case_result(sheet, 'masked', 'positive') is not None:
        raise RuntimeError('An unexecuted case silently obtained a focal receipt')
    sheet.merge(evaluate_cases(sheet, 'masked', ['positive']))
    masked_final = paired_case_result(sheet, 'masked', 'positive')
    masked_focal = focal_case_result(sheet, 'masked', 'positive')
    zero_intervention = focal_answers(sheet.rows['masked']['data'], {'x': 0})
    if masked_final['baseline'] != masked_final['altered']:
        raise RuntimeError('Masked fixture did not cancel its downstream output')
    if masked_focal['baseline']['value'] != '1' or masked_focal['altered']['value'] != '-1':
        raise RuntimeError('Focal calculation lost the actual equation values')
    if masked_focal['altered'] == zero_intervention['baseline']:
        raise RuntimeError('Wrong focal alteration became equivalent to the declared zero-input intervention')
    if sheet.rows['masked']['data']['model']['answer'] != 'out':
        raise RuntimeError('Focal projection mutated the original prediction protocol')

    sheet.put('G', 'inference', {**sheet.rows['G']['data'], 'statement': 'Revised response claim'})
    stale = dependence_result(sheet, 'equations')
    historical = dependence_result(sheet, 'equations', _pins={'G': {original_g['version']}})
    if stale['current'] or not historical['current'] or historical['inference'] != original_g:
        raise RuntimeError('Original-G dependence did not preserve exact historical binding')
    if historical['calculation']['value'] != '1/2':
        raise RuntimeError('Historical paired results were not reusable under explicit original-G pins')
    print(json.dumps({
        'example': 'Synthetic executable-model dependence, not empirical validation or a continuum proof',
        'missing_status': absent['dependence']['status'],
        'partial': partial['calculation'],
        'completed': {method: value['calculation']['value'] for method, value in complete.items()},
        'cancelled_link': cancelled['dependence'],
        'execution_error': singular['calculation'],
        'forgery_rejections': rejected, 'target_matches': matches,
        'masked_output': {'final': masked_final, 'focal': masked_focal, 'expected_intervened_focal': zero_intervention['baseline']},
        'stale_current': stale['current'], 'original_g': historical['inference'],
        'historical_dependence': historical['calculation']['value'],
        'limitations': historical['limitations'],
    }, indent=2))


if __name__ == '__main__':
    main()
