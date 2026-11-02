"""Offline Page loop from an empty sheet, using scripted source/agent responses.

This exercises reading -> Page decisions -> executable graders -> Search returns.
The synthetic relation is supplied explicitly; it is not a research conclusion.
"""
import importlib.util
import json
from pathlib import Path
import sys
from fractions import Fraction


def main(mode="report"):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("episodic_workflow", root / "__init__.py",
                                                submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from episodic_workflow.case_reports.research.engine import ResearchRun, ResearchProviders, ResearchLimits
    from episodic_workflow.case_reports.research.acquisition import PageSource, SearchSource, SearchStrategySource
    from episodic_workflow.case_reports.research.episode_contracts import PageCommission, SearchCommission, SearchStrategyCommission
    from episodic_workflow.case_reports.research.records import ResearchResult, Ledger
    from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
    from episodic_workflow.credit_assignment.causal_progress import compile_episode_points
    from episodic_workflow.method_loop import Context, EpisodeGoal, EpisodeTree

    url = "synthetic://page-grading/equation"
    text = "Synthetic source: y=x for x in {0,1,2}. Removing the x contribution gives y=0."
    if mode == "source_table":
        text += "\n\n| Input | Response |\n| --- | --- |\n| x | y=x; ablated y=0 |\n"
    calls, events, requests = {}, [], []
    pending_records = []

    def record(key, kind, data, dependencies=()):
        return dict(key=key, kind=kind, data=data, dependencies=list(dependencies))

    def scripted(request):
        requests.append(request)
        op = request["operation"]
        calls[op] = calls.get(op, 0) + 1
        if op == "propose_table_query":
            return dict(rows=[next(iter(request["available_rows"]))], question="Read the synthetic relation")
        if op == "extract_causal_evidence":
            span = next(iter(request["source_spans"]))
            return dict(findings=[dict(span=span, study_id="synthetic-equation", finding=text,
                measurements={"observations": {}, "claims": {}},
                relevance_basis="Explicit relation can be formalized and compared with a candidate G",
                reasoning_question="Does removing the declared contribution change the answer?",
                grading_needs=[dict(kind="G_registration", question="Register the supplied causal contribution?",
                    qualifications="Synthetic model only", sheet_keys=[]),
                    dict(kind="support", question="Does this source relation support G under the declared alteration?",
                    qualifications="Requires executable source and G formalisms", sheet_keys=[])])],
                reading_memory=dict(summary=text, unresolved_links=["No empirical generalization claimed"], anchors=[dict(span=span)]))
        if op != "propose_page_child":
            raise RuntimeError("Unexpected operation " + op)
        n = calls[op]
        if n == 1:
            choice = dict(action="read", option=mode, question="Read the supplied synthetic relation", reading_scope="unexamined")
            if mode == "lexical_probe":
                choice["query"] = "y=x"
            elif mode == "source_table":
                choice["option"] = next(key for key, value in request["available_children"].items() if value.get("episode") == "source_table")
                del choice["reading_scope"]
            return choice
        needs = request["own_results"]["grading_needs"]
        if not needs:
            raise RuntimeError("Reader flags did not reach Page before the next decision")
        evidence = needs[0]["evidence"]
        if n == 2:
            return dict(action="record", rationale="Register the source-grounded candidate and its application",
                grading_needs=[need["id"] for need in needs], records=[
                    record("G", "inference", dict(statement="Increasing x contributes nonnegatively to y",
                        variables=["x", "y"], intervention="Remove x contribution", outcome="y", grain="scalar", domain="Synthetic equations"), [evidence]),
                    record("B", "expectation", dict(inference="G", bounds="Expect the contribution for x=0,1,2",
                        basis="provisional_judgment", rationale="Provisional expectation grounded in this source",
                        grounds=[evidence], revision=None)),
                    record("T", "theory", dict(name="Synthetic linear response", mechanism="y=x", predictions=["y increases with x"],
                        admission="candidate", same_as="", distinction="Explicit source relation", compared_with=[]), [evidence]),
                    record("A", "application", dict(inference="G", inference_version="", theory="T", context_key="synthetic",
                        context="Synthetic finite inputs", context_distinction="Declared source context", direction="unresolved",
                        inference_use="x contributes to y", consequence_without_inference="Remove the contribution"), [evidence])])
        if n == 3:
            sheet = page.ledger.causal_sheet
            g, b = sheet.anchor("G"), sheet.anchor("B")
            model = dict(equations={"y": "x"}, focal="y", replacement="0", answer="y", initial={}, steps=0, edge=[])
            pending_records.extend([
                record("D", "importance", dict(application="A", method="equations", question="What is y?", contrast="Remove contribution",
                    comparator="exact_typed_answer", domain=dict(kind="finite", description="x=0,1,2", measure="Uniform finite count",
                    cases=[dict(id=str(x), inputs={"x": x}, weight=1) for x in range(3)]), model=model)),
                record("F", "source_formalization", dict(evidence=evidence, assessment="D", method="equations", model=model,
                    quantity="y", units="toy", derivation="Source explicitly supplies y=x and its alteration", limitations="Synthetic only")),
                record("BT", "expectation_test", dict(expectation=b, assessment="D", scope="True", target=dict(kind="interval", lower="0", upper="2"),
                    grounds=[evidence], rationale="Declared finite target", claim_relation="baseline >= altered", claim_intervention={"x": "0"})),
                record("S", "expectation_support", dict(inference=g, expectation=b, grounds=["A", "F", evidence],
                    argument="Compare formal consequences", limitations="Synthetic only")),
                record("arg", "reasoning", dict(author="scripted", premises=["A", "F", evidence], steps=["Source supplies y=x"],
                    conclusion="Compute source and G alignment", limitations="Synthetic only")),
                record("check", "support_assessment", dict(support="S", application="A", importance="D", bounds_test="BT", source_formalization="F",
                    prediction=dict(quantity="y", units="toy"), facts=[], limitations="Stipulated source correspondence")),
                record("C", "support_contribution", dict(support="S", applications=["A"], facts=[], argument="arg", checks=["check"]))])
            return dict(action="record", rationale="Supply explicit comparison operands", records=pending_records)
        if n == 4:
            return dict(action="evaluate", rationale="Execute fixed circumstances", assessment="D", case_ids=["0", "1", "2"], evaluation_context="check")
        return dict(action="calculate", rationale="Inspect actual support outcome", keys=["check", "C"])

    run = ResearchRun(ResearchProviders(scripted, lambda *_: [], lambda _: text, trace_event=events.append),
        ResearchLimits(yield_window=2, root_yield_window=2, root_yield_threshold=None), "SYNTHETIC Page grading demonstration")
    page = PageSource(run, ResearchResult(), candidate={"url": url, "title": "Synthetic source"})
    commission = PageCommission("Assess the explicit synthetic source relation", ())
    goal = EpisodeGoal.root(objective=commission.question, result_contract={"result": "Synthetic Page results"}, task_context=run.context)
    episode = run.episode("page", page, goal=goal, contract=commission.report_contract())
    page_tree = EpisodeTree(run.grains["page"], {
        run.grains["page"]: tuple(run.grains[name] for name in ("report", "lexical_probe", "source_table"))})
    episode.run(Context(tree=page_tree))
    result = run.global_goal.ledger.result()
    if result.diagnostics:
        raise RuntimeError(result.diagnostics)
    scores = compile_episode_points(run.global_goal.ledger.causal_sheet)
    expected = dict(G_registration="1", expected_generalizability_bounds_registration="1",
                    theory_context_application_registration="1", support_for_G="1/10", causal_change="0")
    if scores["components"] != expected:
        raise RuntimeError(scores["components"])
    gains = [e["gain"] for e in events if e.get("event") == "credit_verdict" and e["episode_type"] == "PageSource"]
    if len(gains) < 4 or any(abs(a-b) > 1e-12 for a, b in zip(gains[:4], [0, 3, 0, 0.1])) or any(gains[4:]):
        raise RuntimeError({"page_gains": gains})
    reader_role = {"report": "ReportSource", "lexical_probe": "LexicalProbeSource", "source_table": "SourceTableSource"}[mode]
    readers = [e for e in events if e.get("event") == "credit_verdict" and e["episode_type"] == reader_role]
    if any(e["gain"] for e in readers):
        raise RuntimeError("Reader extraction earned points")

    # Exact producer returns consumed by both Search and Search Strategy.
    search = SearchSource(run, ResearchResult(), query="synthetic equation")
    search.commission = SearchCommission("Find a useful source", ())
    search.selected_sources.add(url)
    strategy = SearchStrategySource(run, ResearchResult(), approach="Synthetic")
    strategy.commission = SearchStrategyCommission("Find a useful source", ())
    reports = [owner.parent_report(result) for owner in (page, search, strategy)]
    for owner, report in zip((page, search, strategy), reports):
        owner.commission.report_contract().admit(report, audit_identifiers=result.audit_identifiers())
        if not report["grading_needs"] or not report["grading_outcomes"]:
            raise RuntimeError("Upward return lost grading information")
        if not any(Fraction(outcome["score_component_changes"]["support_for_G"]) == Fraction(1, 10)
                   for outcome in report["grading_outcomes"]):
            raise RuntimeError("Upward return lost the numerical grader outcome")
        if not any(outcome["calculation"] for outcome in report["grading_outcomes"]):
            raise RuntimeError("Upward return lost computed scientific results")
    for owner in (search, strategy):
        if abs(sum(value for _, _, value in owner.credit(result).qualities) - 3.1) > 1e-10:
            raise RuntimeError("Source branch lost registration provenance")
    # Ensure an actionable flag after eight unrelated findings remains visible.
    overflow = CausalSheet()
    evidence = next(row for row in result.causal_sheet_updates if row["kind"] == "evidence")
    for i in range(9):
        overflow.put("earlier:" + str(i), "evidence", dict(source=url, locator=str(i), study_id="earlier",
            finding="Earlier bibliography finding", measurements={}))
    overflow.merge(result.causal_sheet_updates)
    large_result = ResearchResult(causal_sheet_updates=tuple(overflow.history), grading_actions=result.grading_actions)
    returned = search.parent_report(large_result)
    if not returned["grading_needs"] or not returned["grading_outcomes"]:
        raise RuntimeError("First-eight preview swallowed grading handoff")
    restored = Ledger(result).result()
    if restored.grading_actions != result.grading_actions:
        raise RuntimeError("Ledger round trip lost grader outcomes")
    print(json.dumps(dict(example="SYNTHETIC empty-sheet Page loop", reader=reader_role, page_unit_gains=gains,
        score_components=scores["components"], global_total=scores["total"], reader_gain=0,
        upward_returns=[type(owner).__name__ for owner in (page, search, strategy)],
        later_flag_preserved=True, grader_results_preserved=True, diagnostics=[]), indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "report")
