"""Offline executable sheets through real Episode commissions and source returns.

Uses the support demonstration fixture. No external provider, research job or
unit-test runner is invoked; scripted decisions only exercise the actual binding.
"""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path


def main():
    import show_causal_credit
    with redirect_stdout(io.StringIO()):
        sheet = show_causal_credit.main()
    from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
    from episodic_workflow.case_reports.research.engine import ResearchRun, ResearchLimits, ResearchProviders
    from episodic_workflow.case_reports.research.workflow import ScientificSource
    from episodic_workflow.case_reports.research.acquisition import SearchStrategySource
    from episodic_workflow.case_reports.research.episode_contracts import SearchStrategyCommission
    from episodic_workflow.case_reports.research.records import ResearchResult
    from episodic_workflow.method_loop import EpisodeGoal, Context
    from episodic_workflow.credit_assignment.research import ResearchController
    from episodic_workflow.credit_assignment.causal_sheet import investigation_credit
    from episodic_workflow.credit_assignment.causal_support import compile_support
    from episodic_workflow.case_reports.research.grading import GRADING_FIELDS, module_for

    for kind in GRADING_FIELDS:
        if set(GRADING_FIELDS[kind]) != set(module_for(kind).FIELDS):
            raise RuntimeError('Dispatcher fields differ from actual subscoring contract: '+kind)

    # Remove only receipts for one commissioned protocol; retain scientific operands.
    incomplete = CausalSheet(row for row in sheet.history
        if row['kind'] != 'importance_result' or row['data']['assessment'] != 'D_one')
    initial = ResearchResult(causal_sheet_updates=tuple(incomplete.history))
    counts, requests, events = {}, [], []
    support = incomplete.rows['S_one']['data']
    def scripted(request):
        requests.append(request)
        operation = request['operation']
        counts[operation] = counts.get(operation, 0) + 1
        if operation == 'propose_scientific_action':
            if counts[operation] == 1:
                return dict(action='reasoning', rationale='Fill missing calculations for this assigned support sheet',
                    commission=dict(question='Execute the fixed comparison needed by check_one; inspect its support result',
                        inputs=['check_one','Btest_one'], assignments=[['check_one','support']],
                        snapshots=[support['inference'],support['expectation']]))
            return dict(action='inspect', rationale='Inspect the returned support assessment', keys=['check_one'])
        if operation == 'propose_reasoning_action':
            if counts[operation] == 1:
                return dict(action='evaluate', rationale='Execute protocol reached through assigned support operands',
                    assessment='D_one', case_ids=['0','1','2'], evaluation_context='check_one')
            return dict(action='calculate', rationale='Read the executed support result', keys=['check_one'])
        raise RuntimeError('Unexpected offline operation '+operation)
    run = ResearchRun(ResearchProviders(scripted, lambda q,n: [], lambda row: '', trace_event=events.append),
        ResearchLimits(yield_window=2,root_yield_window=2), 'SYNTHETIC episode scoring demonstration', initial=initial)
    source = ScientificSource(run, initial)
    goal = EpisodeGoal.root(objective=run.research_request, result_contract={'result':'Synthetic sheet completion'}, task_context=run.context)
    episode = run.episode('scientific', source, goal=goal)
    episode.run(Context(tree=run.tree))
    live_sheet = run.global_goal.ledger.causal_sheet
    coverage = investigation_credit(live_sheet, [('check_one','support')])
    if sum(value for _,_,value in coverage.qualities) != 1:
        raise RuntimeError({'coverage':coverage.as_record(),'operations':counts,'diagnostics':list(run.global_goal.ledger.result().diagnostics),'trace_failures':[e for e in events if e.get('event')=='model_attempt' and e['attempt'].get('status')!='json_object_received_not_yet_admitted']})
    diagnostics = run.global_goal.ledger.result().diagnostics
    if diagnostics:
        raise RuntimeError(list(diagnostics))

    # A commissioned fixed template lets a source's actual observation be graded now.
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
    first = controller.observe('returned source observation',reader.project(ResearchResult(causal_sheet_updates=(update,))),is_root=False)
    reader.ledger.merge(reader.pending)
    observation_run.global_goal.commit(reader.pending_global)
    repeat = controller.observe('same source observation repeated',reader.project(ResearchResult(causal_sheet_updates=(update,))),is_root=False)
    if first.gain != 1 or repeat.gain != 0:
        raise RuntimeError('Immediate source credit or duplicate suppression failed')
    reader.ledger.merge(reader.pending)
    observation_run.global_goal.commit(reader.pending_global)
    from copy import deepcopy
    copied = deepcopy(evidence)
    copied['source'] = 'synthetic://copied-source'
    copied['measurements']['native']['observations'] = {'renamed_y': copied['measurements']['native']['observations']['y']}
    copy_sheet = CausalSheet(reader.ledger.causal_sheet.history)
    copy_update = copy_sheet.put('copied_source_evidence', 'evidence', copied)
    alias = controller.observe('same observation under another extraction key',
        reader.project(ResearchResult(causal_sheet_updates=(copy_update,))), is_root=False)
    if alias.gain != 0:
        raise RuntimeError('A copied observation or selector alias earned scientific credit')
    inherited = reader.ledger.result()
    if reader.assess(inherited) is not inherited:
        raise RuntimeError('Inherited evidence was re-admitted as newly returned source work')
    malformed_sheet = CausalSheet(reader.ledger.causal_sheet.history)
    malformed = malformed_sheet.put('malformed_native_source','evidence',
        {**evidence, 'measurements': {'native': {'observations': []}}})
    retained = reader.assess(ResearchResult(causal_sheet_updates=(malformed,)))
    if not retained.evaluated or retained.causal_sheet_updates != (malformed,) or not retained.diagnostics:
        raise RuntimeError('An unscorable native observation lost otherwise valid source evidence')
    report = reader.parent_report(reader.pending)
    if len(json.dumps(report).encode()) > 12000:
        raise RuntimeError('Source handoff exceeded its bounded scientific return')
    # All four declared sheet calculations are executable from real records.
    coverage_values = {}
    for key, criterion in [('D_one','importance'),('Btest_one','expected_bounds'),('check_one','support'),('repair','repair')]:
        credit = investigation_credit(sheet,[(key,criterion)],context='check_one')
        coverage_values[criterion] = sum(value for _,_,value in credit.qualities)
    print(json.dumps(dict(example='SYNTHETIC real binding with scripted decisions and calculators',
        operations=counts,diagnostics=list(diagnostics),assigned_support_coverage=sum(v for _,_,v in coverage.qualities),
        immediate_source_scientific_gain=first.gain,repeated_source_gain=repeat.gain,copied_observation_gain=alias.gain,inherited_evidence_rescored=False,
        malformed_operand_preserved_unscored=True,
        scoped_source_report_bytes=len(json.dumps(report).encode()),
        maximum_model_input_bytes=max(len(json.dumps(request).encode()) for request in requests),
        calculation_coverage=coverage_values,
        source_support_total=compile_support(observation_run.global_goal.ledger.causal_sheet)['total']),indent=2))


if __name__=='__main__':
    main()
