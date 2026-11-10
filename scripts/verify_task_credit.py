"""Replay the retained enzyme task comparison through real scientific tools."""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from types import SimpleNamespace

from verify_episode_organization import ROOT, fixture, _result, _run, _view
from verify_scientific_tools import call
from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
from episodic_workflow.case_reports.research.episode_contracts import ScientificCommission
from episodic_workflow.case_reports.research.workflow import ScientificSource
from episodic_workflow.case_reports.research.task_comparisons import grade
from episodic_workflow.credit_assignment.causal_progress import compile_episode_points


def build():
    retained = {row["key"]: row["data"] for row in json.loads(
        (Path(__file__).parent / "fixtures/enzyme_task_transfer.json").read_text())}
    sheet = fixture()
    sheet.put("enzyme_model_definition", "premise", {
        "statement": "E+S reversibly binds C, which converts irreversibly to E+P. "
            "S'=-k1*(E0-C)*S+km1*C; C'=k1*(E0-C)*S-(km1+k2)*C; P'=k2*C; "
            "T=S+C; Km=(km1+k2)/k1. Standard Cs=E0*S/(Km+S) uses S'=-k2*Cs; "
            "quadratic-total C=2*E0*T/(E0+Km+T+sqrt((E0+Km+T)^2-4*E0*T)) uses T'=-k2*C.",
        "scope": "Exact model and investigator-selected settings retained from normal_live_20261109_02; not new empirical evidence.",
        "basis": "Retained public model definitions and endpoint tasks, including the actual initial C=P=0."})
    sheet.put("enzyme_G", "inference", retained["borghans_standard_closure_transfer_G"])
    expected = {**deepcopy(retained["borghans_transfer_provisional_bounds"]),
        "inference": "enzyme_G", "grounds": ["enzyme_model_definition"]}
    sheet.put("enzyme_bounds", "expectation", expected)
    sheet.put("enzyme_task_argument", "reasoning", {"author": "retained_replay",
        "premises": ["enzyme_G", "enzyme_bounds", "enzyme_model_definition"],
        "steps": [retained["borghans_endpoint_tasks_selected"]["statement"],
            "Compare product formation separately from free-substrate disappearance. The original expectation already cautions about leaving the low-enzyme regime.",
            "Restoring complex storage changes slow dynamics and its initialization; it can improve product timing while omitting an early binding event."],
        "conclusion": "Test task-specific usefulness with original successes, failures and nonrepresentations retained.",
        "limitations": "Numerical model comparison; source fidelity, empirical validation and universal necessity are not established."})
    binding = sheet.anchor("enzyme_model_definition")
    km = "((km1+k2)/k1)"
    a = f"(E0+{km}+T)"
    q = f"(2*E0*T/({a}+sqrt({a}**2-4*E0*T)))"
    cases = [{"id": name, "inputs": {"S0": "0.1", "target": "0.05", "k1": 10, "km1": 1,
        "k2": k2, "E0": enzyme}, "weight": 1}
        for name, k2, enzyme in (("A", "0.1", "0.01"), ("B", "0.1", 1), ("C", 10, 1), ("D", 10, 100))]
    def ode(equations, initial, event, direction):
        return {"kind": "ode", "source": binding, "equations": equations, "initial": initial,
            "event": {"expression": event, "direction": direction}, "time": {"start": "0", "end": "1000"},
            "solver": {"rtol": "1e-10", "atol": "1e-12"}}
    data = {"inference": sheet.anchor("enzyme_G"), "expectation": sheet.anchor("enzyme_bounds"),
        "original": {"kind": "equations", "source": binding, "equations": {"Km": km},
            "answer": "(S0-target+Km*log(S0/target))/(k2*E0)"},
        "candidate": ode({"T": f"-k2*{q}"}, {"T": "S0"}, "S0-T-target", 1),
        "reference": ode({"S": "-k1*(E0-C)*S+km1*C", "C": "k1*(E0-C)*S-(km1+k2)*C", "P": "k2*C"},
            {"S": "S0", "C": "0", "P": "0"}, "P-target", 1),
        "task": {"quantity": "time_to_product_target", "units": "source_model_time",
            "absolute_tolerance": "0", "relative_tolerance": "0.05"},
        "domain": {"kind": "finite", "description": "The four previously selected enzyme circumstances",
            "measure": "One distinct comparison per circumstance; not continuous parameter-space coverage", "cases": cases},
        "scope": "", "argument": "enzyme_task_argument",
        "limitations": "Original source << estimates have no invented numerical cutoff. Retain all physical units and source-numerics gaps."}
    return sheet, data, q


def execute(source, key, ids):
    sheet = source.ledger.causal_sheet
    if key not in source.selected_keys: source.selected_keys.append(key)
    receipt = call(source, "compare_tasks", {"assessment": key, "case_ids": ids}, [sheet.anchor(key)])
    assert receipt["data"]["status"] in {"computed", "partial"}, receipt["data"]
    return grade(source.ledger.causal_sheet, key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    if not directory.is_relative_to(ROOT / "runs"): raise ValueError("Output belongs in runs/")
    sheet, product, q = build()
    run, root, root_episode, _, _ = _run(sheet)
    root_view, _ = _view(run, root_episode)
    source = ScientificSource(run, _result(sheet))
    commission = ScientificCommission("Assess the retained enzyme task-specific alteration", tuple(sheet.rows), ())
    run.child(root_view, source, "scientific", commission.question,
        contract=commission.report_contract(), proposal={})
    recorded = source.record_results([{"key": "enzyme_product_change", "kind": "causal_change",
        "data": product, "dependencies": []}])
    assert not recorded.diagnostics, recorded.diagnostics
    source.ledger.merge(recorded)
    sheet = source.ledger.causal_sheet
    initial = compile_episode_points(sheet)
    assert initial["components"]["causal_change"] == "0"
    partial = execute(source, "enzyme_product_change", ["B"])
    first = compile_episode_points(source.ledger.causal_sheet)
    assert Fraction(first["components"]["applicability_difference"]) == 1
    assert Fraction(first["components"]["causal_change"]) > Fraction(9, 10)
    assert source.credit(source.ledger.result()).qualities
    assert root.credit(source.ledger.result()).qualities
    execute(source, "enzyme_product_change", ["B"])
    assert compile_episode_points(source.ledger.causal_sheet)["total"] == first["total"]
    product_result = execute(source, "enzyme_product_change", ["A", "C", "D"])
    outcomes = {row["case_id"]: row for row in product_result["cases"]}
    assert [outcomes[name]["original"]["passes"] for name in "ABCD"] == [True, False, False, False], outcomes
    assert [outcomes[name]["candidate"]["passes"] for name in "ABCD"] == [True, True, False, True], outcomes
    complete = compile_episode_points(source.ledger.causal_sheet)
    assert complete["components"]["support_for_G"] == "0"
    assert Fraction(complete["components"]["causal_change"]) > Fraction(first["components"]["causal_change"])
    sheet = source.ledger.causal_sheet
    cloned = deepcopy(product)
    cloned["candidate"]["equations"]["T"] = "  " + cloned["candidate"]["equations"]["T"] + " "
    for case in cloned["domain"]["cases"]: case["id"] = "alias_" + case["id"]
    sheet.put("enzyme_product_alias", "causal_change", cloned)
    execute(source, "enzyme_product_alias", ["alias_" + name for name in "ABCD"])
    assert compile_episode_points(source.ledger.causal_sheet)["total"] == complete["total"]
    refined = deepcopy(product)
    refined["candidate"]["solver"] = {"rtol": "1e-11", "atol": "1e-13"}
    refined["reference"]["solver"] = {"rtol": "1e-11", "atol": "1e-13"}
    refined["candidate"]["time"]["end"] = "2000"
    refined["reference"]["time"]["end"] = "2000"
    sheet.put("enzyme_precision_refinement", "causal_change", refined)
    execute(source, "enzyme_precision_refinement", list("ABCD"))
    assert compile_episode_points(source.ledger.causal_sheet)["total"] == complete["total"]
    # Explicitly outside-scope membership must not exclude the demonstrated improvement.
    outside = deepcopy(product); outside["scope"] = "False"
    sheet.put("enzyme_product_outside", "causal_change", outside)
    result = execute(source, "enzyme_product_outside", ["B"])
    assert result["cases"][1]["original_expected_scope"] is False
    assert Fraction(result["cases"][1]["improvement_credit"]) > 0
    assert compile_episode_points(source.ledger.causal_sheet)["total"] == complete["total"]
    # Further rationalization is a genuinely distinct, less accurate candidate, not another reward for the same repair.
    rational = deepcopy(product)
    rational["candidate"]["equations"]["T"] = "-k2*E0*T/(E0+(km1+k2)/k1+T)"
    sheet.put("enzyme_rational", "causal_change", rational)
    rational_result = execute(source, "enzyme_rational", ["B"])
    assert rational_result["cases"][1]["candidate"]["passes"] is False
    assert compile_episode_points(source.ledger.causal_sheet)["total"] == complete["total"]
    # The other task remains separately represented, including its omitted early events.
    free = deepcopy(product)
    free["task"]["quantity"] = "time_to_free_substrate_target"
    free["candidate"]["event"] = {"expression": f"T-{q}-target", "direction": -1}
    free["reference"]["event"] = {"expression": "S-target", "direction": -1}
    sheet.put("enzyme_free_substrate", "causal_change", free)
    free_result = execute(source, "enzyme_free_substrate", list("ABCD"))
    free_outcomes = {row["case_id"]: row for row in free_result["cases"]}
    assert free_outcomes["B"]["representation"]["candidate"] == "not_represented"
    assert free_outcomes["D"]["representation"]["candidate"] == "not_represented"
    assert free_outcomes["B"]["improvement_credit"] == "0" and free_outcomes["D"]["improvement_credit"] == "0"
    # Fake host receipts cannot turn a failed task into a success.
    actual = next(row for _, row in sheet.of_kind("task_result") if row["data"]["assessment"] == "enzyme_product_change")
    grounded = deepcopy(sheet.rows["enzyme_bounds"]["data"])
    grounded["grounds"] = [actual["key"]]
    sheet.put("enzyme_bounds_with_executed_ground", "expectation", grounded)
    forged = deepcopy(actual["data"]); forged["candidate"]["value"] = 0
    try: sheet.put("forged_task_result", "task_result", forged)
    except ValueError: pass
    else: raise AssertionError("A model-authored task result was accepted")
    final = compile_episode_points(sheet)
    summary = {"passed": True, "product": product_result, "free_substrate": free_result,
        "partial_product_credit": first["components"], "complete_product_credit": complete["components"],
        "final_credit": final["components"], "checks": ["real_ODE_event_comparison", "partial_then_broader_credit",
            "zero_original_support_does_not_zero_change", "repeat_and_case_alias_dedup", "outside_scope_extension",
            "rationalization_failure_retained", "early_event_nonrepresentation", "forged_result_rejected",
            "scientific_record_producer_and_root_credit"]}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": summary["checks"], "product_credit": complete["components"]}))


if __name__ == "__main__": main()
