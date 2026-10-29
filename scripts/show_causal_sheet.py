"""Offline arithmetic demonstration; synthetic definitions are not scientific evidence.

No agents, provider calls, unit-test runner, or live research episode is started.
"""
import importlib.util
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "episodic_workflow", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
    from episodic_workflow.case_reports.research.causal_commission import SheetCommission, read_sheet_result
    from episodic_workflow.case_reports.research.importance import evaluate_cases
    from episodic_workflow.case_reports.research.records import Ledger, ResearchResult
    from episodic_workflow.case_reports.research.goal_state import ScientificGoalState
    from episodic_workflow.credit_assignment.causal_sheet import investigation_credit
    from episodic_workflow.credit_assignment.research import ResearchController

    sheet = CausalSheet()
    sheet.put("G", "inference", dict(statement="X contributes to the response Y through coupling k",
        variables=["X", "Y", "k", "u"], intervention="Remove the coupling contribution to Y",
        outcome="The exact response Y", grain="Scalar variables",
        domain="Expected generalizability bounds: stipulated additive coupling models only; no empirical claim"))
    sheet.put("P", "premise", dict(statement="Consider additive deterministic response equations",
        scope="Synthetic mathematical models only", basis="Stipulated definitions, not scientific evidence"))
    sheet.put("T", "theory", dict(name="Synthetic additive coupling model",
        mechanism="A coupling contribution changes a response", predictions=["Removing coupling can change Y"],
        admission="candidate", same_as="", distinction="An illustrative mathematical definition", compared_with=[]),
        dependencies=["P"])
    sheet.put("A", "application", dict(inference="G", theory="T", context_key="toy",
        context="Three stipulated input circumstances", context_distinction="One application in this demonstration",
        direction="unresolved", inference_use="Calculate the response with and without coupling",
        consequence_without_inference="The paired response may change"))
    sheet.put("derive", "reasoning", dict(author="toy-author", premises=["P", "A"],
        steps=["With coupling, Y=k*x+u", "Without coupling, Y=u", "Subtracting yields k*x"],
        conclusion="At k=1, the exact response changes exactly when x is nonzero",
        limitations="Conditional finite arithmetic only; no claim about empirical validity or continuous-domain measure"))
    cases = [dict(id="negative", inputs=dict(x=-1, k=1, u=1), weight=1),
             dict(id="zero", inputs=dict(x=0, k=1, u=1), weight=2),
             dict(id="positive", inputs=dict(x=1, k=1, u=1), weight=1)]

    def protocol(method, *, kind="finite", broken=False):
        model = dict(equations={"Y": "k*x+u"}, focal="Y", replacement="u", answer="Y", initial={}, steps=0, edge=[])
        if method == "scm":
            model.update(equations={"X": "x", "Y": "k*X+u"}, edge=["X", "Y"])
        elif method == "simulation":
            model.update(equations={"Y": "Y+k*x"}, replacement="Y", initial={"Y": "u"}, steps=3)
        if broken:
            model.update(equations={"Y": "k*x/x+u"})
        return dict(application="A", method=method, question="Does removing G change exact Y?",
            contrast="Remove only the coupling contribution; hold inputs and comparison fixed",
            domain=dict(kind=kind, description="Declared three-point circumstance measure, separate from G's expected generalizability bounds",
                        measure="Weights 1, 2, 1 on x=-1,0,1; total mass 4", cases=cases),
            model=model, comparator="exact_typed_answer")

    for method in ("equations", "scm", "simulation"):
        sheet.put(method, "importance", protocol(method), dependencies=["derive"])
    sheet.put("sample", "importance", protocol("equations", kind="sample"), dependencies=["derive"])
    sheet.put("error", "importance", protocol("equations", broken=True), dependencies=["derive"])
    commission = SheetCommission("Calculate the explicitly assigned equation comparison", inputs=("A",),
                                 assignments=(("equations", "importance"),))
    request = commission.request(sheet)
    assignments = commission.assignments
    controller = ResearchController(("causal_assessment",), point_scores=True,
                                    initial=investigation_credit(sheet, assignments))
    partial_updates = evaluate_cases(sheet, "equations", ["negative"])
    sheet.merge(partial_updates)
    partial = sheet.calculation_view(["equations"])
    partial_step = controller.observe("paired negative input", investigation_credit(sheet, assignments), is_root=False)
    sheet.merge(evaluate_cases(sheet, "equations", ["zero", "positive"]))
    completed_step = controller.observe("remaining inputs", investigation_credit(sheet, assignments), is_root=False)
    count = len(sheet.history)
    repeated = evaluate_cases(sheet, "equations", ["negative", "zero", "positive"])
    sheet.merge(repeated)
    duplicate_step = controller.observe("repeat same paired answers", investigation_credit(sheet, assignments), is_root=False)
    no_op = {"new_rows": len(sheet.history) - count, "controller_gain": duplicate_step.gain}
    for key in ("scm", "simulation", "sample", "error"):
        sheet.merge(evaluate_cases(sheet, key, ["negative", "zero", "positive"]))
    completed = sheet.calculation_view(["equations", "scm", "simulation", "sample", "error"])

    duplicate_protocol = protocol("equations")
    duplicate_protocol["question"] = "Reworded label for the same calculation"
    duplicate_protocol["domain"] = {**duplicate_protocol["domain"], "cases": [
        {**c, "id": "renamed-" + c["id"], "weight": c["weight"] * 3} for c in cases]}
    try:
        sheet.put("duplicate", "importance", duplicate_protocol)
    except ValueError as exc:
        duplicate_rejection = str(exc)
    try:
        read_sheet_result(sheet, commission, [{"key": repeated[0]["key"], "kind": "importance_result",
            "dependencies": [], "data": repeated[0]["data"]}], actor="toy-author")
    except ValueError as exc:
        fabricated_rejection = str(exc)
    try:
        changed = protocol("equations")
        changed["domain"] = {**changed["domain"], "cases": cases[1:]}
        sheet.put("equations", "importance", changed)
    except ValueError as exc:
        denominator_rejection = str(exc)

    critique = SheetCommission("Examine limitations of the public additive derivation", inputs=("derive",))
    checked = read_sheet_result(sheet, critique, [{"key": "critique", "kind": "reasoning", "dependencies": [],
        "data": dict(author="host-overwrites-this", premises=["derive"],
                     steps=["The derivation assumes the stipulated additive relation"],
                     conclusion="Its conclusion is conditional on that relation",
                     limitations="This critique is not a verification receipt or a numerical grade")}],
        actor="toy-critic")
    ledger = Ledger(ResearchResult(causal_sheet_updates=tuple(sheet.history)))
    ledger.merge(checked)
    sheet = ledger.causal_sheet
    critique_return = sheet.return_page(checked.causal_sheet_updates)
    calculation_return = sheet.return_page(partial_updates)
    sheet.put("P", "premise", {**sheet.rows["P"]["data"], "statement": "The coupling definition now needs reconsideration"})
    stale = sheet.calculation_view(["equations"])
    stale_step = controller.observe("upstream definition revised", investigation_credit(sheet, assignments), is_root=False)
    global_state = ScientificGoalState(ResearchResult(causal_sheet_updates=tuple(sheet.history))).preview()
    print(json.dumps({
        "example": "SYNTHETIC arithmetic, not scientific evidence or a continuous-space proof",
        "commissioned_input_keys": [r["key"] for r in request["inputs"]],
        "partial": partial, "completed": completed,
        "local_coverage": {"partial_gain": partial_step.gain, "completion_gain": completed_step.gain,
                           "duplicate": no_op, "stale_score": stale_step.score},
        "duplicate_protocol_rejected": duplicate_rejection,
        "fabricated_receipt_rejected": fabricated_rejection,
        "denominator_revision_rejected": denominator_rejection,
        "scoped_calculation_return": calculation_return, "scoped_critique_return": critique_return,
        "after_dependency_revision": stale,
        "root_progress_binding": global_state["root_progress_binding"],
        "root_limitation": global_state["meaning"]}, indent=2))


if __name__ == "__main__":
    main()
