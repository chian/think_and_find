"""Run one live theory-reconstruction Episode through the production machinery.

Invoke through the single-workflow launcher, with --run-dir set to that launch's
run directory. The launcher supplies its temporary/cache paths and machine
resource bounds. This experiment supplies only a question and an empty theory
declaration; the actual episodes choose their work and their controllers yield.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import importlib.util
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


def _focused_tree(run, production_tree):
    from episodic_workflow.method_loop import EpisodeTree

    reachable = set()
    pending = ["theory_reconstruction"]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(production_tree[name])
    edges = {name: children for name, children in production_tree.items()
             if name in reachable}
    tree = EpisodeTree(run.grains["theory_reconstruction"], {
        run.grains[parent]: tuple(run.grains[child] for child in children)
        for parent, children in edges.items() if children
    }, self_nesting=tuple(name for name in run.tree.self_nesting if name in reachable),
        recursive_edges=tuple((parent, child) for parent, child in run.tree.recursive_edges
                              if parent in reachable and child in reachable))
    return tree, edges


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Theory reconstruction to commission")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="This launcher's run directory; all experiment output goes here")
    args = parser.parse_args(argv)
    if not args.question.strip():
        parser.error("--question must contain the reconstruction question")
    sys.dont_write_bytecode = True
    _load_checkout()

    from episodic_workflow.checkpoint import CheckpointJournal
    from episodic_workflow.cli import _write_result
    from episodic_workflow.configured_providers import (
        firecrawl_argo_config_from_env, firecrawl_argo_research_providers,
    )
    from episodic_workflow.case_reports.research.engine import TREE, ResearchLimits, ResearchRun
    from episodic_workflow.case_reports.research.episode_contracts import TheoryReconstructionCommission
    from episodic_workflow.case_reports.research.theory_episodes import TheoryReconstructionSource
    from episodic_workflow.case_reports.research.theory_language import empty_fragment
    from episodic_workflow.case_reports.research.theory_runtime import COMPONENT_KINDS
    from episodic_workflow.method_loop import Context, EpisodeCompletion, EpisodeGoal
    from episodic_workflow.method_loop.archive import iter_json, write_json

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    key = "theory:live-reconstruction"
    commission = TheoryReconstructionCommission(
        question=args.question, inputs=(), snapshots=(),
        theory_request={"reconstruction": key, "focus": [],
                        "requested_kinds": list(COMPONENT_KINDS)},
        seed=empty_fragment(key, args.question),
    )
    limits = ResearchLimits(root_yield_threshold=None)
    inputs = {"experiment": "theory_reconstruction", "commission": commission.as_record(),
              "limits": asdict(limits),
              "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with ExitStack() as streams:
        checkpoint = streams.enter_context(CheckpointJournal(run_dir / "checkpoint", inputs=inputs))
        trace_path = run_dir / "trace.jsonl"
        trace = streams.enter_context(trace_path.open("x", encoding="utf-8"))

        def trace_event(item):
            write_json(trace, item)
            trace.write("\n")
            trace.flush()

        print(f"Checkpoint: {checkpoint.path}\nTrace: {trace_path}", flush=True)
        providers = firecrawl_argo_research_providers(firecrawl_argo_config_from_env(),
            progress=True, trace_event=trace_event, checkpoint=checkpoint)
        run = ResearchRun(providers, limits, args.question)
        run.tree, focused_edges = _focused_tree(run, TREE)
        source = TheoryReconstructionSource(run, run.initial)
        contract = commission.report_contract()
        goal = EpisodeGoal.root(objective=commission.question,
            result_contract={"information": contract.as_record()}, task_context=run.context)
        episode = run.episode("theory_reconstruction", source, goal=goal, contract=contract)
        run.emit_trace("focused_episode_started", episode_tree=focused_edges,
                       commission=commission.as_record())
        record = episode.run(Context(tree=run.tree))
        result = episode.build_result(record)
        completion = EpisodeCompletion.from_record(record)
        values = source.parent_report(result, completion, episode.request)
        identifiers = tuple(dict.fromkeys((record.episode_id, record.goal.goal_id,
            *episode.request.message.audit_identifiers(), *result.audit_identifiers())))
        report = contract.admit(values, audit_identifiers=identifiers).as_record()
        artifacts = {
            "parent-report.json": report,
            "theory-graph.json": report["theory_fragment"],
            "research-state.json": result.as_record(),
            "controller.json": {"completion": completion.as_record(),
                                "state": record.controller_state,
                                "credit": source.credit(result)},
            "audit.json": record.as_record(),
        }
        summary = {"question": commission.question, "completion": completion.as_record(),
                   "model_calls": run.model_calls, "completed_units": run.total_units,
                   "checkpoint": checkpoint.provenance(),
                   "artifacts": {name: str(run_dir / name) for name in artifacts},
                   "trace": str(trace_path)}
        checkpoint.event({"event": "focused_episode_completed", "summary": summary}, trace_event)
        for name, value in artifacts.items():
            _write_result(run_dir / name, iter_json(value, indent=2), resume=False)
        _write_result(run_dir / "summary.json", iter_json(summary, indent=2), resume=False)
        checkpoint.complete()
        write_json(sys.stdout, summary, indent=2)
        sys.stdout.write("\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
