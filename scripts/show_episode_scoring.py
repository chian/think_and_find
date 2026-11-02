"""Offline executable sheets through real Episode commissions and source returns.

Uses the support demonstration fixture. No external provider, research job or
unit-test runner is invoked; scripted decisions only exercise the actual binding.
"""
from contextlib import redirect_stdout
import io
import json


def main():
    import show_causal_credit
    with redirect_stdout(io.StringIO()):
        sheet = show_causal_credit.main()
    from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
    from episodic_workflow.case_reports.research.engine import ResearchRun, ResearchLimits, ResearchProviders
    from episodic_workflow.case_reports.research.workflow import DiscoverySource
    from episodic_workflow.case_reports.research.acquisition import SearchStrategySource
    from episodic_workflow.case_reports.research.episode_contracts import SearchStrategyCommission
    from episodic_workflow.case_reports.research.records import ResearchResult
    from episodic_workflow.method_loop import EpisodeGoal, Context
    from episodic_workflow.credit_assignment.research import ResearchController
    from episodic_workflow.credit_assignment.causal_progress import compile_episode_points
    from episodic_workflow.credit_assignment.causal_support import compile_support
    from episodic_workflow.case_reports.research.grading import GRADING_FIELDS, module_for

    for kind in GRADING_FIELDS:
        if set(GRADING_FIELDS[kind]) != set(module_for(kind).FIELDS):
            raise RuntimeError('Dispatcher fields differ from actual subscoring contract: '+kind)

    # Remove only receipts for one commissioned protocol; retain scientific operands.
    incomplete = CausalSheet(row for row in sheet.history
        if row['kind'] != 'importance_result' or row['data']['assessment'] != 'D_two')
    initial = ResearchResult(causal_sheet_updates=tuple(incomplete.history))
    counts, requests, events = {}, [], []
    support = incomplete.rows['S_two']['data']
    def scripted(request):
        requests.append(request)
        operation = request['operation']
        counts[operation] = counts.get(operation, 0) + 1
        if operation == 'propose_discovery_attempt':
            return dict(question='Resolve the supplied synthetic support comparison', inputs=['check_two','Btest_two'],
                        snapshots=[support['inference'],support['expectation']], rationale='Offline scoring audit')
        if operation == 'propose_scientific_action':
            if counts[operation] == 1:
                return dict(action='reasoning', rationale='Fill missing calculations for this assigned support sheet',
                    commission=dict(question='Execute the fixed comparison needed by check_two; inspect its support result',
                        inputs=['check_two','Btest_two'], assignments=[['check_two','support']],
                        snapshots=[support['inference'],support['expectation']]))
            return dict(action='inspect', rationale='Inspect the returned support assessment', keys=['check_two'])
        if operation == 'propose_reasoning_action':
            if counts[operation] == 1:
                return dict(action='evaluate', rationale='Execute protocol reached through assigned support operands',
                    assessment='D_two', case_ids=['0','1','2'], evaluation_context='check_two')
            return dict(action='calculate', rationale='Read the executed support result', keys=['check_two'])
        raise RuntimeError('Unexpected offline operation '+operation)
    run = ResearchRun(ResearchProviders(scripted, lambda q,n: [], lambda row: '', trace_event=events.append),
        ResearchLimits(yield_window=2,root_yield_window=2,root_yield_threshold=None),
        'SYNTHETIC episode scoring demonstration', initial=initial)
    source = DiscoverySource(run)
    goal = EpisodeGoal.root(objective=run.research_request, result_contract={'result':'Synthetic sheet completion'}, task_context=run.context)
    episode = run.episode('discovery', source, goal=goal)
    episode.run(Context(tree=run.tree))
    diagnostics = run.global_goal.ledger.result().diagnostics
    if diagnostics:
        raise RuntimeError(list(diagnostics))
    live_sheet = run.global_goal.ledger.causal_sheet
    before_points = compile_episode_points(incomplete)
    after_points = compile_episode_points(live_sheet)
    from fractions import Fraction
    earned = {key: str(Fraction(value) - Fraction(before_points['points'].get(key, '0')))
              for key,value in after_points['points'].items()
              if Fraction(value) > Fraction(before_points['points'].get(key, '0'))}
    if (sum(map(Fraction, earned.values()), Fraction()) != Fraction(1, 10) or
            any(not key.startswith('support:') for key in earned)):
        raise RuntimeError({'unexpected_calculation_reward':earned,'operations':counts})
    scope_gains = {role: [event['gain'] for event in events
                         if event.get('event') == 'credit_verdict' and event['episode_type'] == role]
                   for role in ('ReasoningSource', 'ScientificSource', 'DiscoverySource')}
    if any(abs(sum(gains) - 0.1) > 1e-12 for gains in scope_gains.values()):
        raise RuntimeError({'unexpected_parent_or_child_credit': scope_gains})
    component_change = {name: str(Fraction(value) - Fraction(before_points['components'][name]))
                        for name, value in after_points['components'].items()}
    if any(Fraction(value) != (Fraction(1, 10) if name == 'support_for_G' else 0)
           for name, value in component_change.items()):
        raise RuntimeError({'unexpected_component_change': component_change})

    # Reading returns evidence only. Page selects the fixed comparison tool.
    from episodic_workflow.case_reports.research.observation_admission import compare_observations
    baseline = ResearchResult(causal_sheet_updates=tuple(sheet.history))
    evidence_sheet = CausalSheet(sheet.history)
    evidence = dict(source='synthetic://new-study',locator='Table 1 row 1',study_id='synthetic-new-study',
        finding='Synthetic source reports y=1 at x=1',measurements={
            'quote':'Synthetic y=1 at x=1', 'source_version':'synthetic-source-version',
            'relevance_basis':'Direct observation for the commissioned numerical comparison',
            'native':{'observations':{'y':dict(source_id='row1/y',locator='Table 1 row 1 y',
                quantity='y',conditions={'x':1},value=1,units='toy',uncertainty='not reported')}}})
    update = evidence_sheet.put('new_source_evidence','evidence',evidence)
    observation_run = ResearchRun(ResearchProviders(lambda q: {}, lambda q,n: [], lambda c: ''),
        ResearchLimits(), 'SYNTHETIC immediate observation grading', initial=baseline)
    reader = SearchStrategySource(observation_run,baseline,approach='Find an observation for the fixed support template')
    reader.commission = SearchStrategyCommission('Find y at a declared circumstance',('check_one',))
    reader.selected_keys = ['check_one']
    controller = ResearchController(reader.credit_columns, point_scores=True, initial=reader.credit(baseline))
    extraction = controller.observe('returned source observation', reader.project(ResearchResult(causal_sheet_updates=(update,))), is_root=False)
    if extraction.gain != 0:
        raise RuntimeError('Source extraction invoked a grader automatically')
    reader.ledger.merge(reader.pending)
    observation_run.global_goal.commit(reader.pending_global)
    compared = compare_observations(reader.ledger.causal_sheet, ['check_one'], ['new_source_evidence'])
    first = controller.observe('Page-selected source comparison', reader.project(compared), is_root=False)
    reader.ledger.merge(reader.pending)
    observation_run.global_goal.commit(reader.pending_global)
    repeat = controller.observe('same source observation repeated',reader.project(compare_observations(reader.ledger.causal_sheet, ['check_one'], ['new_source_evidence'])),is_root=False)
    if abs(first.gain - 0.1) > 1e-12 or repeat.gain != 0:
        raise RuntimeError('Immediate source credit or duplicate suppression failed')
    reader.ledger.merge(reader.pending)
    observation_run.global_goal.commit(reader.pending_global)
    from copy import deepcopy
    copied = deepcopy(evidence)
    copied['source'] = 'synthetic://copied-source'
    copied['measurements']['native']['observations'] = {'renamed_y': copied['measurements']['native']['observations']['y']}
    copy_sheet = CausalSheet(reader.ledger.causal_sheet.history)
    copy_update = copy_sheet.put('copied_source_evidence', 'evidence', copied)
    reader.ledger.merge(ResearchResult(causal_sheet_updates=(copy_update,)))
    observation_run.global_goal.commit(ResearchResult(causal_sheet_updates=(copy_update,)))
    alias = controller.observe('same observation under another extraction key',
        reader.project(compare_observations(reader.ledger.causal_sheet, ['check_one'], ['copied_source_evidence'])), is_root=False)
    if alias.gain != 0:
        raise RuntimeError('A copied observation or selector alias earned scientific credit')
    inherited = reader.ledger.result()
    if reader.assess(inherited) is not inherited:
        raise RuntimeError('Inherited evidence was re-admitted as newly returned source work')
    malformed_sheet = CausalSheet(reader.ledger.causal_sheet.history)
    malformed = malformed_sheet.put('malformed_native_source','evidence',
        {**evidence, 'measurements': {'native': {'observations': []}}})
    reader.ledger.merge(ResearchResult(causal_sheet_updates=(malformed,)))
    retained = compare_observations(reader.ledger.causal_sheet, ['check_one'], ['malformed_native_source'])
    if not retained.evaluated or retained.causal_sheet_updates or not retained.diagnostics:
        raise RuntimeError('An unscorable native observation lost otherwise valid source evidence')
    report = reader.parent_report(reader.pending)
    print(json.dumps(dict(example='SYNTHETIC real binding with scripted decisions and calculators',
        operations=counts,diagnostics=list(diagnostics),calculation_result_credit=earned,
        credit_gain_by_scope={role: sum(gains) for role,gains in scope_gains.items()},
        global_score_component_change=component_change,
        page_selected_observation_gain=first.gain,repeated_source_gain=repeat.gain,copied_observation_gain=alias.gain,inherited_evidence_rescored=False,
        malformed_operand_preserved_unscored=True,
        scoped_source_report_bytes=len(json.dumps(report).encode()),
        maximum_model_input_bytes=max(len(json.dumps(request).encode()) for request in requests),
        source_support_total=compile_support(observation_run.global_goal.ledger.causal_sheet)['total']),indent=2))


if __name__=='__main__':
    main()
