"""Offline Search Strategy -> Search -> Page -> Report with real boundaries.

Scripted choices exercise retrieval and final-child returns under the ordinary
statistical controllers. No external provider or research job is invoked.
"""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("episodic_workflow", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from episodic_workflow.case_reports.research.acquisition import SearchStrategySource
    from episodic_workflow.case_reports.research.engine import ResearchRun, ResearchProviders, ResearchLimits
    from episodic_workflow.case_reports.research.episode_contracts import SearchStrategyCommission
    from episodic_workflow.case_reports.research.records import ResearchResult
    from episodic_workflow.method_loop import Context, EpisodeGoal, EpisodeTree

    requests, events, counts = [], [], {}
    delivered, recalls = set(), set()

    def walk(value):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                yield from walk(child)

    def scripted(request):
        requests.append(request)
        op = request["operation"]
        counts[op] = counts.get(op, 0) + 1
        if len(requests) > 100:
            raise RuntimeError("Offline fixture failed to yield")
        if "own_results" in request:
            raise RuntimeError("An accumulated ledger report entered the prompt")
        for report in request.get("child_reports", []):
            identity = (op, request["objective"], report["details_ref"])
            if identity in delivered:
                raise RuntimeError("An incorporated child result was delivered twice")
            delivered.add(identity)
        if op == "extract_causal_evidence":
            span = next(iter(request["source_spans"]))
            return dict(findings=[dict(span=span, study_id=request["source"],
                finding="Synthetic source relation y=x", measurements={"observations": {}, "claims": {}},
                relevance_basis="A source-native relation to formalize", reasoning_question="What is its domain?",
                grading_needs=[dict(kind="G_registration", question="Formalize the contribution of x?",
                    qualifications="Synthetic statement only", sheet_keys=[])])],
                reading_memory=dict(summary="Synthetic source states y=x; no empirical generalization claimed.",
                    unresolved_links=["Formalize before comparing"], anchors=[dict(span=span)], retained_evidence=[]))
        assessment = dict(request["working_assessment"])
        for key in assessment.keys() - {"record_refs", "receipt_refs", "detail_refs"}:
            assessment[key] = "Source relation remains provisional; formal comparison is unresolved."
        supplied = list(walk({key: request.get(key) for key in
            ("new_results", "child_reports", "selected_sheet_inputs", "scientific_inputs")}))
        refs = [{"key": item["key"], "version": item["version"]} for item in supplied
                if isinstance(item.get("key"), str) and isinstance(item.get("version"), str)]
        assessment["record_refs"] = list({(r["key"], r["version"]): r
            for r in [*assessment["record_refs"], *refs]}.values())
        detail = next((item["details_ref"] for item in supplied if "details_ref" in item), None)
        receipt = next((item["id"] for item in supplied
                        if str(item.get("id", "")).startswith("grading-action:")), None)
        # Retrieval is selected by the owning episode and consumes one own unit.
        kind_key = next(((kind, key) for kind, key in
                         (("detail", detail), ("receipt", receipt)) if key and (op, kind) not in recalls), None)
        if kind_key:
            kind, key = kind_key
            recalls.add((op, kind))
            action = dict(action="recall", references=[dict(kind=kind, key=key)])
            if refs:
                action["references"].append(dict(kind="record", **refs[0]))
                recalls.add((op, "record"))
            if receipt and kind != "receipt":
                action["references"].append(dict(kind="receipt", key=receipt))
                recalls.add((op, "receipt"))
        elif refs and (op, "record") not in recalls:
            recalls.add((op, "record"))
            action = dict(action="recall", references=[dict(kind="record", **refs[0])])
        elif op == "propose_strategy_query":
            action = dict(query="synthetic relation", objective="Read the synthetic relation")
        elif op == "choose_research_source":
            option = request["candidates"][0]
            action = dict(option_id=option["option_id"], objective="Read " + option["url"],
                          rationale="Inspect the supplied source")
        elif op == "propose_page_child":
            if not request["source_summary"]["prior_reading"]:
                action = dict(action="read", option="report", reading_scope="unexamined", question="Read the synthetic relation")
            else:
                action = dict(action="inspect", keys=[], rationale="Inspect current scientific operands")
        else:
            raise RuntimeError("Unexpected operation " + op)
        return dict(action, working_assessment=assessment)

    def search(query, limit):
        return [{"url": "synthetic://handoff/" + str(n), "title": "Synthetic relation", "snippet": "y=x"}
                for n in range(2)]

    run = ResearchRun(ResearchProviders(scripted, search, lambda _: "Synthetic relation y=x.", trace_event=events.append),
        ResearchLimits(yield_window=2, root_yield_window=2, root_yield_threshold=None), "SYNTHETIC handoff audit")
    source = SearchStrategySource(run, ResearchResult(), approach="Inspect supplied synthetic sources")
    commission = SearchStrategyCommission("Find the synthetic relation", ())
    goal = EpisodeGoal.root(objective=commission.question, result_contract={"result": "Synthetic job return"}, task_context=run.context)
    episode = run.episode("search_strategy", source, goal=goal, contract=commission.report_contract())
    tree = EpisodeTree(run.grains["search_strategy"], {run.grains[parent]: tuple(run.grains[name] for name in children)
        for parent, children in {"search_strategy": ["search"], "search": ["page"],
                               "page": ["report", "lexical_probe", "source_table"]}.items()})
    record = episode.run(Context(tree=tree))
    result = source.build_result(record)
    if result.diagnostics or source.assessment_error:
        raise RuntimeError({"diagnostics": result.diagnostics, "assessment_error": source.assessment_error})
    report = source.parent_report(result, record)
    commission.report_contract().admit(report, audit_identifiers=(record.episode_id, record.goal.goal_id))
    if not report["unreviewed_results"] or any("unreviewed_results" in item for item in report["unreviewed_results"]):
        raise RuntimeError("Final unit missing or recursive handoff")
    required = {("choose_research_source", "detail"), ("choose_research_source", "record"),
                ("propose_strategy_query", "detail"), ("propose_page_child", "detail")}
    if not required <= recalls:
        raise RuntimeError({"missing_retrievals": sorted(required - recalls), "counts": counts, "failures": [r for r in walk(report) if r.get("ended_by") == "source_failed" or r.get("status") == "failed"]})
    if any(event.get("gain") for event in events if event.get("event") == "credit_verdict"):
        raise RuntimeError("Extraction or retrieval earned credit")
    print(json.dumps(dict(operations=counts, retrievals=sorted(recalls),
        delivered_once=len(delivered), final_result_preserved=True, credit=0,
        return_bytes=len(json.dumps(report).encode()),
        max_request_bytes=max(len(json.dumps(request).encode()) for request in requests)), indent=2))


if __name__ == "__main__":
    main()
