"""Verify failed-unit yield and parent handoff without external provider calls.

Run through this checkout's tools/one_workflow.sh with --run-dir pointing at
that launch's fresh run directory. These deterministic fixtures use the real
Episode loop, research sources, credit projection and numerical controller.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys


def _load_checkout():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "episodic_workflow", root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)


def _controller_checks():
    from episodic_workflow.case_reports.research.records import Credit
    from episodic_workflow.credit_assignment.research import ResearchController

    column = "page_extraction_gain"
    controller = ResearchController((column,), point_scores=True)
    for label in ("failed-1", "failed-2", "failed-3", "failed-4"):
        step = controller.observe(label, Credit(evaluated=False), is_root=False)
        assert not step.evaluated and step.gain == 0
    estimator = controller._incidence._estimators["scientific_credit"]
    assert estimator.incidence_samples() == (frozenset(),) * 4
    assert controller.state().stop and controller.state().score == 0
    assert controller.state().failures == 4

    mixed = ResearchController((column,), point_scores=True)
    mixed.observe("good", Credit(qualities=((column, "accepted", 2.0),)), is_root=False)
    failed = mixed.observe("failed", Credit(evaluated=False,
        identities=((column, "unadmitted"),),
        qualities=((column, "unadmitted", 100.0),)), is_root=False)
    assert not failed.stop and failed.score == 2.0 and failed.gain == 0
    assert "unadmitted" not in mixed.seen[column]
    recovered = mixed.observe("recovered", Credit(qualities=(
        (column, "accepted", 2.0), (column, "new", 1.0))), is_root=False)
    assert recovered.score == 3.0 and recovered.gain == 1.0
    assert mixed.state().failures == 1

    standing = ResearchController((column,), point_scores=True, yield_threshold=0)
    for label in ("failed-1", "failed-2", "failed-3", "failed-4"):
        assert not standing.observe(label, Credit(evaluated=False), is_root=True).stop
    assert standing._incidence._estimators["scientific_credit"].units == 4
    assert standing.state().failures == 4
    return {"failure_only_yields": True, "accepted_credit_preserved": True,
            "recovery_can_earn_credit": True, "standing_root_preserved": True}


def _fixture_run(agent, retrieve, emit):
    from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
    from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun
    from episodic_workflow.case_reports.research.records import ResearchResult

    key, question = "fixture:discovery-goal", "Verify delivery of failed source outcomes."
    sheet = CausalSheet()
    initial = ResearchResult(causal_sheet_updates=(
        sheet.put(key, "discovery_goal", {
            "question": question, "inputs": [],
            "answer_schema": {"type": "object", "additionalProperties": False,
                              "properties": {}, "required": []},
            "criteria": [{"id": "fixture_source_claim", "question": "Obtain a source-grounded scientific claim.",
                "check": {"kind": "source_claim", "claim_schema": {
                    "type": "object", "required": ["kind"], "properties": {
                        "kind": {"enum": ["equation", "causal_relation", "boundary_condition"]}}}}}]}),))
    providers = ResearchProviders(agent=agent, retrieve=retrieve, trace_event=emit,
        search=lambda query, limit: [{"url": "https://example.invalid/source",
            "title": "Deterministic failed source", "snippet": "Fixture source lead."}])
    run = ResearchRun(providers, ResearchLimits(root_yield_threshold=None,
        root_yield_window=4), question, initial=initial)
    return run, question, key


def _episode(run, grain, source, commission):
    from episodic_workflow.case_reports.research.engine import TREE
    from episodic_workflow.method_loop import EpisodeGoal, EpisodeTree

    reachable, pending = set(), [grain]
    while pending:
        name = pending.pop()
        if name not in reachable:
            reachable.add(name)
            pending.extend(TREE[name])
    run.tree = EpisodeTree(run.grains[grain], {
        run.grains[parent]: tuple(run.grains[child] for child in TREE[parent])
        for parent in reachable if TREE[parent]
    })
    contract = commission.report_contract()
    goal = EpisodeGoal.root(objective=commission.question,
        result_contract={"information": contract.as_record()}, task_context=run.context)
    return run.episode(grain, source, goal=goal, contract=contract)


def _assessment(source_type):
    return {**{key: "Deterministic regression fixture." for key in source_type.assessment_fields},
            "record_refs": [], "receipt_refs": [], "detail_refs": []}


def _parent_handoff(emit):
    from episodic_workflow.case_reports.research.acquisition import SearchSource
    from episodic_workflow.case_reports.research.episode_contracts import SearchCommission
    from episodic_workflow.method_loop import Context
    from episodic_workflow.task_executors._http import ProviderError

    requests, retrievals, events = [], [], []

    def agent(request):
        assert request["operation"] == "choose_research_source"
        requests.append(deepcopy(request))
        if len(requests) == 1:
            return {"action": "read_source", "option_id": request["candidates"][0]["option_id"],
                "objective": "Read the fixture source.", "rationale": "Exercise failed Page return.",
                "working_assessment": _assessment(SearchSource)}
        return {"action": "recall", "references": [],
            "working_assessment": _assessment(SearchSource)}

    def retrieve(candidate):
        retrievals.append(candidate)
        raise ProviderError("fixture: source unavailable")

    def trace(event):
        events.append(event)
        emit(event)

    run, question, goal_key = _fixture_run(agent, retrieve, trace)
    source = SearchSource(run, run.initial, query="fixture source")
    episode = _episode(run, "search", source,
        SearchCommission(question, (), discovery_goal=goal_key))
    record = episode.run(Context(tree=run.tree))
    child = record.unit_records[0].child
    assert child.ended_by == "yield_stop" and child.units_consumed == run.limits.yield_window
    assert child.controller_state.failures == len(retrievals) == child.units_consumed
    assert not record.unit_records[0].controller_input.evaluated
    assert record.controller_state.failures == 1
    assert record.ended_by == "yield_stop" and record.units_consumed == 4
    assert "fixture: source unavailable" in json.dumps(requests[1]["child_reports"])
    verdicts = [event for event in events if event["event"] == "credit_verdict"]
    failures = [event for event in verdicts if event["unit"] == "retrieval_failure"]
    assert len(failures) == child.units_consumed
    assert all(event["yield_observed"] and not event["evaluated"] for event in failures)
    assert all(event["operation_status"] == "failed" and event["gain"] == 0 for event in failures)
    assert all(event["diagnostics"][0]["origin"]["responsibility"] == "PageSource" for event in failures)
    assert sum(event["episode_type"] == "SearchSource" for event in verdicts) == record.units_consumed
    assert run.global_goal.preview()["total_score"] == 0
    return {"page_failed_units": child.units_consumed,
            "parent_observations_for_child": 1, "diagnostic_reached_next_parent_decision": True,
            "page_and_parent_yielded": True, "scientific_credit": 0}


def _transient_recovery(emit):
    from episodic_workflow.case_reports.research.acquisition import PageSource
    from episodic_workflow.case_reports.research.episode_contracts import PageCommission
    from episodic_workflow.method_loop import Context
    from episodic_workflow.task_executors._http import ProviderError

    retrievals = []

    def retrieve(candidate):
        retrievals.append(candidate)
        if len(retrievals) < 3:
            raise ProviderError("fixture: transient retrieval failure")
        return "The fixture source is available again."

    def agent(request):
        assert request["operation"] == "propose_page_child"
        return {"action": "recall", "references": [],
                "working_assessment": _assessment(PageSource)}

    run, question, goal_key = _fixture_run(agent, retrieve, emit)
    source = PageSource(run, run.initial, candidate={"url": "https://example.invalid/source"})
    episode = _episode(run, "page", source,
        PageCommission(question, (), discovery_goal=goal_key))
    record = episode.run(Context(tree=run.tree))
    assert len(retrievals) == 3 and source.document is not None and not source.failure
    assert record.unit_records[0].controller_step.stop is False
    assert record.unit_records[1].controller_step.stop is False
    assert record.controller_state.failures == 2 and source.evaluated_units == 2
    assert record.ended_by == "yield_stop" and record.units_consumed == 4
    return {"transient_failures": 2, "retrieval_recovered": True,
            "model_decisions_after_recovery": run.model_calls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    _load_checkout()
    with (args.run_dir / "trace.jsonl").open("x", encoding="utf-8") as trace:
        def emit(event):
            trace.write(json.dumps(event) + "\n")
            trace.flush()

        summary = {"controller": _controller_checks(),
                   "parent_handoff": _parent_handoff(emit),
                   "transient_recovery": _transient_recovery(emit)}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
