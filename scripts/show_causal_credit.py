"""Offline executed G/B_G/support/repair example; synthetic models, no research calls."""
import importlib.util
import json
from fractions import Fraction
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
    from episodic_workflow.case_reports.research.support_grading import grade_support
    from episodic_workflow.credit_assignment.causal_support import compile_support, claim_id
    from episodic_workflow.case_reports.research.causal_commission import SheetCommission, read_sheet_result
    from episodic_workflow.case_reports.research.repairs import grade_repair

    sheet = CausalSheet()
    sheet.put('G', 'inference', dict(statement='Increasing input has a nonnegative response contribution', variables=['x','y'],
        intervention='Vary x relative to the zero-contribution ablation', outcome='y', grain='scalar', domain='Synthetic formal theories'))
    original = sheet.anchor('G')
    sheet.put('B', 'expectation', dict(inference='G', bounds='Provisionally expect G to remain usable across the declared formal theory targets',
        basis='provisional_judgment', rationale='A deliberately breakable expectation, not an asserted empirical law', grounds=[], revision=None))
    before_bounds = sheet.anchor('B')
    cases = [dict(id=str(x), inputs={'x':x}, weight=1) for x in range(3)]

    def theory_application(name, g=original, canonical_theory=None):
        theory, application = 'T_'+name, 'A_'+name
        if canonical_theory is not None:
            theory = canonical_theory
        else:
            sheet.put(theory, 'theory', dict(name='Synthetic '+name, mechanism='Explicit numerical relation for '+name,
            predictions=['The declared y response'], admission='candidate', same_as='', distinction='Formal model supplied below; no model distinctness grade', compared_with=[]))
        sheet.put(application, 'application', dict(inference='G', inference_version=g['version'], theory=theory,
            context_key=name, context=name, context_distinction='Explicit finite synthetic comparison', direction='unresolved',
            inference_use='The focal relation determines the computed response', consequence_without_inference='Calculated by the fixed ablation'))
        return application

    def protocol(name, application, equation, *, replacement='0'):
        sheet.put(name, 'importance', dict(application=application, method='equations', question='What is the response y?',
            contrast='Remove the declared contribution while preserving other terms', comparator='exact_typed_answer',
            domain=dict(kind='finite', description='x=0,1,2', measure='Uniform finite counting measure', cases=cases),
            model=dict(equations={'y':equation}, focal='y', replacement=replacement, answer='y', initial={},steps=0,edge=[])))

    def bound_test(name, assessment, expectation, lower, upper, claim_relation="baseline >= altered"):
        data=dict(expectation=expectation, assessment=assessment, scope='True',
            target=dict(kind='interval',lower=str(lower),upper=str(upper)), grounds=[],
            rationale='Synthetic declared target, fixed before comparison', claim_relation=claim_relation, claim_intervention={'x':'0'})
        sheet.put(name, 'expectation_test', data)

    def hypothesis(name, coefficient, *, equation=None, canonical_theory=None, observation=False, source_conditions=None, source_uncertainty="not reported"):
        application = theory_application(name, canonical_theory=canonical_theory)
        assessment = 'D_'+name
        protocol(assessment, application, equation or f'{coefficient}*x')
        evidence = 'E_'+name
        grounds=[application]
        facts=[]
        if observation:
            sheet.put(evidence,'evidence',dict(source='synthetic://study/'+name,locator='Table 1 row 1',study_id='study-'+name,
                finding='Synthetic observation of y at x=1',measurements={'native':{'observations':{'y':dict(
                    source_id='row1/y',locator='Table 1 row 1 y',quantity='y',conditions={'x':1} if source_conditions is None else source_conditions,value=coefficient,
                    units='toy',uncertainty=source_uncertainty)}}}))
            grounds.append(evidence)
            facts=[dict(evidence=evidence,observation='y',case_id='1')]
        argument='arg_'+name
        sheet.put(argument,'reasoning',dict(author='synthetic-author',premises=grounds,steps=['Formal operands are explicitly supplied'],
            conclusion='Calculate dependence and source agreement',limitations='Not empirical evidence'))
        support='S_'+name
        sheet.put(support,'expectation_support',dict(inference=original,expectation=before_bounds,grounds=grounds,
            argument='Assess the formal relation against G and B_G',limitations='Synthetic finite comparison'))
        test='Btest_'+name
        bound_test(test,assessment,before_bounds,0,100)
        check='check_'+name
        sheet.put(check,'support_assessment',dict(support=support,application=application,importance=assessment,
            bounds_test=test,prediction=dict(quantity='y',units='toy'),facts=facts,limitations='Source-model correspondence is explicitly stipulated'))
        sheet.put('C_'+name,'support_contribution',dict(support=support,applications=[application],
            facts=[{k:v for k,v in fact.items() if k!='case_id'} for fact in facts],argument=argument,checks=[check]))
        before = compile_support(sheet)
        sheet.merge(evaluate_cases(sheet,assessment,['0','1','2'],context=support))
        return check, before

    check_one, before_execution = hypothesis('one',1)
    hypothesis('two',2,equation='x+x*x')
    before_break=compile_support(sheet)
    hypothesis('same_theory_new_parameter',2,canonical_theory='T_one')
    if compile_support(sheet)['total']!=before_break['total']:
        raise RuntimeError('Same theory under a new parameterization minted hypothesis credit')
    # Renaming a formal hypothesis does not create another support unit.
    hypothesis('renamed_one',1)
    duplicate=compile_support(sheet)
    if duplicate['total'] != before_break['total']:
        raise RuntimeError('Renaming a hypothesis minted support')

    breaking = theory_application('breaking')
    protocol('D_break',breaking,'x')
    bound_test('Btest_break','D_break',before_bounds,-100,0)
    sheet.merge(evaluate_cases(sheet,'D_break',['0','1','2']))
    sheet.put('target_requirement','premise',dict(statement='This synthetic theory requires y<=0',scope='Declared finite x cases',basis='Formal target stipulated for this demonstration'))
    sheet.put('break','expectation_break',dict(expectation=before_bounds,triggers=[sheet.anchor('target_requirement'),sheet.anchor('Btest_break')],
        expected='Original G remains usable for the target requirement',observed='Positive response fails y<=0 for positive x',
        conflict='The formal target requires the opposing response on those cases',limitations='Not an empirical causal discovery'))
    sheet.put('G','inference',{**sheet.rows['G']['data'],'statement':'Negative input contribution in the affected synthetic model'})
    revised=sheet.anchor('G')
    sheet.put('B','expectation',dict(inference='G',bounds='Consider changed G for the failing target while retaining original support separately',basis='provisional_judgment',
        rationale='Do not erase the failed expectation',grounds=['target_requirement'],revision=dict(
            prior=before_bounds,triggers=[sheet.anchor('break')],changes=['change_inference'],change='Reverse the focal contribution',
            repair_kind='unresolved',proposed_rule='y=-x',general_rules='One explicit relation across declared cases',
            rule_reasoning='Executable comparison determines consequences',retained=[],lost=[],new_predictions=['Nonpositive response'],limitations='Declared finite alternatives only')))
    candidate_app=theory_application('candidate',revised)
    protocol('D_candidate',candidate_app,'-x')
    bound_test('Btest_candidate','D_candidate',sheet.anchor('B'),-100,0,claim_relation='baseline <= altered')
    alternative_app=theory_application('failed_preserving_alternative')
    protocol('D_alternative',alternative_app,'x+1',replacement='1')
    sheet.put('repair','repair_assessment',dict(**{'break':'break'},expectation_test='Btest_break',target_test='Btest_candidate',target=sheet.anchor('B'),
        candidate='D_candidate',alternatives=['D_alternative'],claim_relation='baseline >= altered',construction_cases=['1'],test_cases=['2'],rationale='Compare changed G against an explicit G-preserving alternative',limitations='Only this declared alternative, not all possible repairs'))
    sheet.put('change_argument','reasoning',dict(author='synthetic-author',premises=['break','target_requirement','Btest_break','D_break'],
        steps=['Execute the original, proposed and preserving alternative models on identical circumstances'],
        conclusion='Use computed repaired mass and scoped necessity',limitations='No model verdict establishes the result'))
    change=dict(**{'break':'break'},source=before_bounds,target=sheet.anchor('B'),importance='D_break',argument='change_argument',assessment='repair')
    sheet.put('change','causal_change',change)
    sheet.merge(evaluate_cases(sheet,'D_candidate',['0','1','2'],context='change'))
    pending_alternative=compile_support(sheet)
    sheet.merge(evaluate_cases(sheet,'D_alternative',['0','1','2'],context='change'))
    initial=compile_support(sheet)
    # A newly added formal hypothesis supports ORIGINAL G after the break.
    check_three, _ = hypothesis('three',3,equation='x+x*x*x')
    later=compile_support(sheet)
    # Additional source observation aligns at a declared source-matched circumstance.
    check_fact,_ = hypothesis('one_observed',1,observation=True)
    after_fact=compile_support(sheet)
    source_result=grade_support(sheet,check_fact)['facts'][0]

    # Archived model opinions remain inert.
    opinion_effects={}
    for verdict in ('valid','invalid','unresolved'):
        archived=CausalSheet(sheet.history)
        archived.put('legacy_review','verification',dict(reasoning='change_argument',reviewer='other-model',verdict=verdict,
            checks=['Archived unsupported opinion'],counterexample=''))
        if compile_support(archived)['points'] != after_fact['points']:
            raise RuntimeError('Model verdict changed executable credit')
        opinion_effects[verdict]='no effect'
    try:
        read_sheet_result(sheet,SheetCommission('Critique the argument',inputs=('change_argument',)),
            [dict(key='new_verdict',kind='verification',dependencies=[],data=dict(reasoning='change_argument',reviewer='model',verdict='valid',checks=['opinion'],counterexample=''))],actor='model')
    except ValueError as exc:
        model_verdict_rejection=str(exc)
    else:
        raise RuntimeError('Model-authored verification was admitted')

    # A real preserving repair defeats scoped necessity even though candidate still fits.
    successful_alternative=theory_application('successful_preserving_alternative')
    protocol('D_successful_alternative',successful_alternative,'x-3',replacement='-3')
    sheet.put('repair','repair_assessment',{**sheet.rows['repair']['data'],'alternatives':['D_alternative','D_successful_alternative']})
    # Rebind same change to the expanded explicit comparison, preserving its identity.
    sheet.put('change','causal_change',change)
    sheet.merge(evaluate_cases(sheet,'D_successful_alternative',['0','1','2'],context='change'))
    defeated=compile_support(sheet)
    actual=[before_execution['total'],before_break['total'],initial['total'],later['total'],after_fact['total'],defeated['total']]
    expected=['0','2','10/3','5','20/3','4']
    if actual!=expected:
        raise RuntimeError({'actual':actual,'expected':expected})
    if pending_alternative['changes']['change']['value'] is not None or pending_alternative['changes']['change']['lower']!='0':
        raise RuntimeError('Unexecuted preserving alternative certified necessity')
    if source_result['status']!='aligned' or source_result['comparison_basis']!='literal_reported_value_agreement_uncertainty_unknown':
        raise RuntimeError('Source comparison lost uncertainty qualification')
    # Execute source-binding edge cases without changing the reported earlier snapshots.
    check_broad,_ = hypothesis('broad_interval',1,observation=True,
        source_uncertainty={'kind':'interval','lower':'0','upper':'2'})
    broad = grade_support(sheet,check_broad)['facts'][0]
    if broad['status']!='aligned' or broad['discriminating'] is not False or broad['altered_matches'] is not True:
        raise RuntimeError('Aligned nondiscriminating evidence was incorrectly excluded')
    before_mismatch=compile_support(sheet)['total']
    check_wrong,_ = hypothesis('wrong_conditions',1,observation=True,source_conditions={'x':10})
    wrong=grade_support(sheet,check_wrong)['facts'][0]
    if wrong['status']!='source_conditions_mismatch' or compile_support(sheet)['total']!=before_mismatch:
        raise RuntimeError('Observation from a different circumstance received support credit')
    check_uncertain,_ = hypothesis('unstructured_uncertainty',1,observation=True,source_uncertainty='plus or minus an unspecified amount')
    uncertain=grade_support(sheet,check_uncertain)['facts'][0]
    if uncertain['status']!='unresolved_source_value_or_uncertainty' or compile_support(sheet)['total']!=before_mismatch:
        raise RuntimeError('An unsupported uncertainty statement was converted to a grading tolerance')
    legacy={name:value for name,value in sheet.rows['C_one']['data'].items() if name!='checks'}
    sheet.put('legacy_text_only','support_contribution',legacy)
    text_only=compile_support(sheet)['contributions']['legacy_text_only']
    if text_only['admitted_units']!=0 or text_only['status']!='awaiting_executable_operands':
        raise RuntimeError('Text-only support entered numerical credit')
    point='change:change'
    if 'E_one_observed' not in after_fact['point_dependencies'][point]:
        raise RuntimeError('Later observation revaluation lost its source attribution')
    print(json.dumps(dict(example='Synthetic executable models and observations; not empirical validation',
        policy=after_fact['policy_details'],support_before_break=before_break['claims'][claim_id(original)]['support'],
        pending_before_execution=before_execution['contributions']['C_one'],
        pending_alternative=pending_alternative['changes']['change'],
        initial_change=initial['changes']['change'],after_post_break_hypothesis=later['changes']['change'],
        after_aligned_observation=after_fact['changes']['change'],observation_alignment=source_result,
        after_preserving_repair=defeated['changes']['change'],totals=dict(zip(['before_execution','before_break','after_checked_repair','post_break_support','aligned_observation','preserving_repair'],actual)),
        source_edge_cases=dict(aligned_nondiscriminating=broad['status'],mismatched_conditions=wrong['status'],unstructured_uncertainty=uncertain['status'],legacy_text_only=text_only['status']),
        model_opinion_effects=opinion_effects,model_verdict_rejected=model_verdict_rejection),indent=2))
    return sheet

if __name__=='__main__':
    main()
