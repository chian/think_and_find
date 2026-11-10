"""Focused production-contract checks for Root's Discovery/Scientific siblings.

Run via tools/one_workflow.sh and --run-dir beneath runs/. The fixtures exercise
real schemas, producers, consumers, criterion receipts and calculation methods;
scripted routing choices do not demonstrate autonomous scientific behavior.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import traceback


def _load_checkout():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("episodic_workflow", root / "__init__.py",
        submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return root


ROOT = _load_checkout()

from jsonschema import Draft202012Validator
from episodic_workflow.case_reports.research.acquisition import DiscoverySource
from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
from episodic_workflow.case_reports.research.computation_yield import RootVOCController
from episodic_workflow.case_reports.research.discovery_goals import (
    create_discovery_goal, submit_finding, evaluate_finding, recheck_findings,
    compile_discovery_points, discovery_credit)
from episodic_workflow.case_reports.research.engine import TREE, ResearchLimits, ResearchProviders, ResearchRun
from episodic_workflow.case_reports.research.episode_contracts import (
    ScientificCommission, TheoryReconstructionCommission, TheoryRepairCommission,
    TheoryComparisonCommission)
from episodic_workflow.case_reports.research.importance import evaluate_cases
from episodic_workflow.case_reports.research.records import Ledger, ResearchResult
from episodic_workflow.case_reports.research.scientific_tools import execute_tool
from episodic_workflow.case_reports.research.theory_episodes import (
    TheoryReconstructionSource, TheoryRepairSource, TheoryComparisonSource)
from episodic_workflow.case_reports.research.theory_language import empty_fragment
from episodic_workflow.case_reports.research.workflow import RootSource, ScientificSource
from episodic_workflow.credit_assignment.causal_progress import compile_progress
from episodic_workflow.credit_assignment.research import ResearchController
from episodic_workflow.method_loop import EpisodeGoal
from episodic_workflow.method_loop.episode import EpisodeView
from episodic_workflow.method_loop.identities import EpisodeRef
from source_anchor_fixtures import (anchor, claim_binding, observation_binding,
    stored_sidecar, retained_document, attach_response_anchors, result_anchor, unit_passages)


# Authored fixture text, not a purported quotation from a research publication.
SOURCE_TEXT = (
    "In this declared mathematical model, increasing x increases y when y=a*x "
    "and a is positive. Proportional transfer fails when a changes with x."
)


def fixture():
    from episodic_workflow.case_reports.research.documents import Document, Span
    sheet = CausalSheet()
    claims = {
        "positive_relation": {"kind": "causal_relation", "statement_id": "",
            "locator": "First sentence of the authored mathematical fixture",
            "left": "x", "relation": "increases", "right": "y",
            "conditions": "y=a*x with positive a held fixed", "symbols": {}},
        "qualified_failure": {"kind": "boundary_condition", "statement_id": "",
            "locator": "Second sentence of the authored mathematical fixture",
            "left": "proportional transfer", "relation": "fails_when", "right": "a changes with x",
            "conditions": "The fixed-a representation is transferred beyond its stated condition", "symbols": {}}}
    source = "https://example.invalid/verification/mathematical-source"
    document = Document(source, "Authored fixture", SOURCE_TEXT, 12000)
    passages = {"fixture": Span("fixture", 0, len(SOURCE_TEXT), SOURCE_TEXT)}
    sidecar = stored_sidecar(document, passages, claims={
        "positive_relation": claim_binding(document, passages, "increasing x increases y", ["increasing x increases y"]),
        "qualified_failure": claim_binding(document, passages, "Proportional transfer fails", ["Proportional transfer"], ["when a changes with x"])})
    evidence = sheet.put("fixture:evidence", "evidence", {
        "source": "https://example.invalid/verification/mathematical-source",
        "locator": f"characters 0:{len(SOURCE_TEXT)}", "study_id": "authored-verification-fixture",
        "finding": "The declared fixture states conditional dependence and a transfer boundary.",
        "measurements": {"quote": SOURCE_TEXT, "source_version": sha256(SOURCE_TEXT.encode()).hexdigest(),
            "native": {"claims": claims, "observations": {}}, "source_anchors": sidecar}})
    sheet.put("fixture:G", "inference", {
        "statement": "At fixed positive a, x contributes positively to y through y=a*x.",
        "variables": ["x", "y", "a"], "intervention": "Change x while holding a fixed.",
        "outcome": "y", "grain": "Declared deterministic mathematical model",
        "domain": "Explicit positive-a circumstances; this is a verifier fixture"})
    sheet.put("fixture:theory", "theory", {"name": "Declared linear fixture account",
        "mechanism": "A deterministic x-to-y relation", "predictions": ["y=a*x"],
        "admission": "candidate", "same_as": "", "distinction": "Authored mathematical fixture, not empirical corroboration",
        "compared_with": []}, dependencies=(evidence["key"],))
    sheet.put("fixture:fragment", "theory_fragment",
        empty_fragment("fixture:fragment", "Reconstruct the declared mathematical account"))
    return sheet


def _result(sheet):
    return ResearchResult(causal_sheet_updates=tuple(sheet.history))


def _assessment(source_class):
    return {**{name: "Contract fixture; no autonomous conclusion asserted." for name in source_class.assessment_fields},
            "record_refs": [], "receipt_refs": [], "detail_refs": []}


class FixtureAgent:
    def __init__(self):
        self.requests = []
        self.root_action = None

    def __call__(self, request):
        self.requests.append(deepcopy(request))
        if request["operation"] == "propose_root_action":
            response = {**deepcopy(self.root_action), "working_assessment": _assessment(RootSource)}
        else:
            classes = {"propose_scientific_action": ScientificSource,
                       "propose_theory_reconstruction": TheoryReconstructionSource,
                       "propose_theory_repair": TheoryRepairSource,
                       "propose_theory_comparison": TheoryComparisonSource}
            response = {"action": "inspect", "keys": [], "rationale": "Exercise an actual decision contract",
                        "working_assessment": _assessment(classes[request["operation"]])}
        Draft202012Validator(request["response_contract"]).validate(response)
        return response


def _run(sheet):
    agent = FixtureAgent()
    events = []
    providers = ResearchProviders(agent=agent, search=lambda *_: [], retrieve=lambda *_: SOURCE_TEXT,
                                  trace_event=events.append)
    run = ResearchRun(providers, ResearchLimits(), "Verify scientific episode organization", initial=_result(sheet))
    root = RootSource(run)
    episode = run.episode("root", root, goal=EpisodeGoal.root(objective=run.research_request,
        result_contract={"verification": "Focused production-contract fixtures"}))
    return run, root, episode, agent, events


def _view(run, episode, parent_path=()):
    path = (*parent_path, (episode.grain.name, episode.key))
    controller = run.controller(path)
    return EpisodeView(grain=episode.grain, key=episode.key, path=path, units_consumed=0,
        updates=(), controller_state=controller.state(),
        episode_ref=EpisodeRef(run_id="episode-organization-fixture", path=path), request=episode.request), controller


def _goal(sheet, check, question):
    requirement = ({"kind": "native", "predicate": check}
                   if check["kind"] in {"source_claim", "source_value"} else
                   {"kind": "scientific", "question": question, "candidate": None, "predicate": check})
    return {"question": question, "inputs": [sheet.anchor("fixture:G")],
        "answer_schema": {"type": "object", "additionalProperties": False,
            "required": ["account"], "properties": {"account": {"type": "string", "minLength": 1}}},
        "criteria": [{"id": "requested_result", "requirement": requirement}]}


def _source_goal(kind, relation):
    return {"kind": "source_claim", "claim_schema": {"type": "object",
        "required": ["kind", "relation"], "properties": {"kind": {"const": kind}, "relation": {"const": relation}}}}


def _observation_predicate(quantity, conditions, units, *, kind="measurement", value=None):
    return {"kind": "source_value", "field_path": [], "value_schema": {
        "type": "object", "required": ["kind", "quantity", "conditions", "value", "units"],
        "properties": {"kind": {"const": kind}, "quantity": quantity,
            "conditions": conditions, "value": value or {"type": "number"}, "units": units}}}


def _admit_goal(sheet, key, check):
    data = _goal(sheet, check, "Find source-grounded information for this explicit fixture criterion")
    sheet.put(key, "discovery_goal", data)
    return data


def _submit(sheet, goal, key, selector):
    from types import SimpleNamespace
    source = SimpleNamespace(ledger=Ledger(_result(sheet)), commission=SimpleNamespace(discovery_goal=goal))
    result = submit_finding(source, {"action": "submit_finding", "key": key,
        "answer": {"account": "The selected source assertion addresses the requested condition."},
        "source_claims": [{"evidence": sheet.anchor("fixture:evidence"), "claim": selector}],
        "source_observations": [], "scientific_links": [], "rationale": "Source-anchored contract fixture"})
    sheet.merge(result.causal_sheet_updates)
    return result


def hierarchy_and_controllers():
    assert TREE["root"] == ("discovery", "scientific")
    assert TREE["discovery"] == ("search",)
    assert TREE["scientific"] == ("theory_reconstruction", "theory_repair", "theory_comparison")
    assert TREE["theory_reconstruction"] == TREE["theory_repair"] == ("theory_format_repair",)
    assert TREE["theory_comparison"] == ()
    assert not {"reasoning", "search_strategy", "theory_gap"} & TREE.keys()
    sheet = fixture()
    _admit_goal(sheet, "fixture:goal", _source_goal("causal_relation", "increases"))
    run, root, episode, agent, _ = _run(sheet)
    root_view, controller = _view(run, episode)
    assert type(controller) is RootVOCController
    agent.root_action = {"action": "discovery", "goal": "fixture:goal", "rationale": "Check sibling routing"}
    discovery = root.propose(root_view).child
    _, discovery_controller = _view(run, discovery, root_view.path)
    assert isinstance(discovery.source, DiscoverySource) and type(discovery_controller) is ResearchController
    agent.root_action = {"action": "scientific", "commission": {
        "question": "Assess the source-grounded fixture", "inputs": ["fixture:goal", "fixture:evidence"], "snapshots": []},
        "rationale": "Check the other sibling"}
    scientific = root.propose(root_view).child
    _, scientific_controller = _view(run, scientific, root_view.path)
    assert isinstance(scientific.source, ScientificSource) and type(scientific_controller) is ResearchController
    return {"tree": TREE, "root": type(controller).__name__,
            "children": [type(discovery_controller).__name__, type(scientific_controller).__name__]}


def scientific_tool_decision_contracts():
    from episodic_workflow.method_loop.communication import ParentReport
    sheet = fixture()
    run, root, episode, agent, _ = _run(sheet)
    root_view, _ = _view(run, episode)
    inputs = ("fixture:G", "fixture:theory", "fixture:evidence", "fixture:fragment")
    scientific = ScientificSource(run, _result(sheet))
    commission = ScientificCommission("Inspect the supplied scientific account", inputs, ())
    scientific_episode = run.child(root_view, scientific, "scientific", commission.question,
        contract=commission.report_contract(), proposal={})
    scientific_view, _ = _view(run, scientific_episode, root_view.path)
    scientific.propose(scientific_view)
    requests = [(ScientificSource, agent.requests[-1])]
    theory_request = {"reconstruction": "fixture:fragment", "focus": [],
                      "requested_kinds": ["entity", "quantity", "equation", "assumption", "gap"]}
    for source_class, commission_class, grain in (
            (TheoryReconstructionSource, TheoryReconstructionCommission, "theory_reconstruction"),
            (TheoryRepairSource, TheoryRepairCommission, "theory_repair"),
            (TheoryComparisonSource, TheoryComparisonCommission, "theory_comparison")):
        if source_class is TheoryComparisonSource:
            child_commission = commission_class("Compare the supplied accounts", inputs, (), (), "")
        else:
            child_commission = commission_class(f"Exercise {grain} on the supplied account", inputs, (),
                theory_request=theory_request, seed=None)
        child = source_class(run, _result(sheet))
        child_episode = run.child(scientific_view, child, grain, child_commission.question,
            contract=child_commission.report_contract(), proposal={})
        child_view, child_controller = _view(run, child_episode, scientific_view.path)
        assert type(child_controller) is ResearchController
        child.propose(child_view)
        ParentReport(child_commission.report_contract(), child.parent_report(child.ledger.result()))
        requests.append((source_class, agent.requests[-1]))
        if source_class is not TheoryComparisonSource:
            from episodic_workflow.case_reports.research.theory_episodes import repair_theory_unit
            child.theory_repair = {"additions": {}, "errors": ["Authored malformed-output fixture"]}
            correction = repair_theory_unit(child, child_view)
            assert correction.child.grain.name == "theory_format_repair"
            assert correction.child.source.commission.seed is None
            repaired = correction.child.source
            ParentReport(repaired.commission.report_contract(), repaired.parent_report(repaired.ledger.result()))
    tools = {
        "lookup": {"keys": [sheet.anchor("fixture:G")], "kinds": ["inference"], "query": ""},
        "lean_equivalence": {"relation": "equal", "parameters": [], "assumptions": [],
                             "left": "0", "right": "0", "proof": "rfl"},
        "causal_calculation": {"assessment": "fixture:protocol", "case_ids": ["positive"], "evaluation_context": ""}}
    tools["compare_tasks"] = {"assessment": "fixture:task_comparison", "case_ids": ["positive"]}
    verified = {}
    for source_class, request in requests:
        validator = Draft202012Validator(request["response_contract"])
        names = []
        for name, arguments in tools.items():
            validator.validate({"action": "scientific_tool", "name": name,
                "question": "Check an explicit fixture question", "inputs": [sheet.anchor("fixture:G")],
                "arguments": arguments, "rationale": "Verify the real producer's action contract",
                "working_assessment": _assessment(source_class)})
            names.append(name)
        verified[source_class.__name__] = names
    return verified


def scientific_discovery_communication():
    sheet = fixture()
    run, root, episode, agent, _ = _run(sheet)
    root_view, _ = _view(run, episode)
    scientific = ScientificSource(run, _result(sheet))
    commission = ScientificCommission("Commission an explicit missing-information search", tuple(sheet.rows), ())
    scientific_episode = run.child(root_view, scientific, "scientific", commission.question,
        contract=commission.report_contract(), proposal={})
    _view(run, scientific_episode, root_view.path)
    goal = _goal(sheet, _source_goal("causal_relation", "increases"), "Find the declared x-to-y source relation")
    result = create_discovery_goal(scientific, {"action": "create_discovery_goal", "key": "fixture:communicated_goal",
        "goal": goal, "rationale": "Exercise the production Scientific goal writer"})
    scientific.ledger.merge(result)
    report = scientific.parent_report(scientific.ledger.result())
    assert report["discovery_goals"] == [{"key": "fixture:communicated_goal",
        "version": scientific.ledger.causal_sheet.rows["fixture:communicated_goal"]["version"]}]
    root.ledger.merge(result)
    run.global_goal.commit(result)
    agent.root_action = {"action": "discovery", "goal": "fixture:communicated_goal",
                         "rationale": "Forward Scientific's exact goal"}
    discovery = root.propose(root_view).child.source
    forwarded = discovery.commission.request(discovery.ledger.causal_sheet)
    assert discovery.ledger.causal_sheet.rows["fixture:communicated_goal"]["data"] == goal
    returned = submit_finding(discovery, {"action": "submit_finding", "key": "fixture:communicated_finding",
        "answer": {"account": "The source states the conditional positive relation."},
        "source_claims": [{"evidence": sheet.anchor("fixture:evidence"), "claim": "positive_relation"}],
        "source_observations": [], "scientific_links": [], "rationale": "Answer the exact commissioned criterion"})
    root.ledger.merge(returned)
    run.global_goal.commit(returned)
    agent.root_action = {"action": "scientific", "commission": {
        "question": "Interpret the returned finding", "inputs": ["fixture:communicated_finding"], "snapshots": []},
        "rationale": "Return Discovery's qualified information to its scientific consumer"}
    consumer = root.propose(root_view).child.source
    assert {"fixture:communicated_finding", "fixture:communicated_goal", "fixture:G", "fixture:evidence"} <= set(consumer.commission.inputs)
    supplied = consumer.commission.request(consumer.ledger.causal_sheet)["inputs"]
    actual = next(row for row in supplied if row["key"] == "fixture:evidence")
    assert actual["data"]["measurements"]["quote"] == SOURCE_TEXT
    assert actual["data"]["measurements"]["source_version"] == sha256(SOURCE_TEXT.encode()).hexdigest()
    return {"goal_forwarded_unchanged": True, "source_quote_preserved": True,
            "scientific_inputs": list(consumer.commission.inputs), "discovery_scope": forwarded}


def source_criteria_and_dedup():
    results = {}
    for label, kind, relation, selector in (
            ("positive", "causal_relation", "increases", "positive_relation"),
            ("supported_negative", "boundary_condition", "fails_when", "qualified_failure")):
        sheet = fixture()
        goal_key = "fixture:" + label + "_goal"
        _admit_goal(sheet, goal_key, _source_goal(kind, relation))
        _submit(sheet, goal_key, "fixture:first_finding", selector)
        checked = evaluate_finding(sheet, "fixture:first_finding")
        assert checked["results"][0]["status"] == "native_match"
        assert checked["results"][0]["verified_scope"] == "native_fields_only"
        assert compile_discovery_points(sheet, goal_key)["total"] == "1"
        _submit(sheet, goal_key, "fixture:duplicate_finding", selector)
        assert compile_discovery_points(sheet, goal_key)["total"] == "1"
        assert recheck_findings(sheet) == ()
        results[label] = {"criterion_status": checked["results"][0]["status"], "credit": "1", "duplicate_credit": "0"}
        if label == "positive":
            _admit_goal(sheet, "fixture:broader_goal", {"kind": "source_claim", "claim_schema": {
                "type": "object", "required": ["kind"], "properties": {"kind": {"const": "causal_relation"}}}})
            local_baseline = _result(sheet)
            _submit(sheet, "fixture:broader_goal", "fixture:same_claim_broader_goal", selector)
            assert compile_discovery_points(sheet)["total"] == "1"
            local = discovery_credit(_result(sheet), local_baseline, "fixture:broader_goal", "fixture_local")
            assert sum(value for _, _, value in local.qualities) == 1.0
            _admit_goal(sheet, "fixture:additional_negative_goal", _source_goal("boundary_condition", "fails_when"))
            _submit(sheet, "fixture:additional_negative_goal", "fixture:new_native_claim", "qualified_failure")
            assert compile_discovery_points(sheet)["total"] == "2"
            results["broad_narrow_identity"] = {"same_claim_global_credit": "1",
                "new_local_criterion_credit": "1", "different_native_claim_total": "2"}
    sheet = fixture()
    _admit_goal(sheet, "fixture:negative_goal", _source_goal("boundary_condition", "fails_when"))
    _submit(sheet, "fixture:negative_goal", "fixture:wrong_source_claim", "positive_relation")
    assert evaluate_finding(sheet, "fixture:wrong_source_claim")["results"][0]["status"] == "native_no_match"
    assert compile_discovery_points(sheet)["total"] == "0"
    results["wrong_relation_rejected"] = True
    return results


def scientific_structure_requirements():
    """Replay the actual V=J false fulfillment, then test typed source linkage."""
    from episodic_workflow.case_reports.research.documents import Document, Span
    live = json.loads((Path(__file__).parent / "fixtures" / "discovery_semantic_native_mismatch.json").read_text())
    sheet = fixture()
    for row in live["evidence"]:
        sheet.put(row["key"], "evidence", row["data"])
        assert sheet.rows[row["key"]]["version"] == row["version"]
    native = _source_goal("equation", "=")
    structure = {"kind": "theory_structure",
        "component_schema": {"type": "object", "required": ["kind", "lhs", "rhs"], "properties": {
            "kind": {"const": "equation"},
            "lhs": {"type": "object", "required": ["op", "expression", "coordinate"], "properties": {
                "op": {"const": "derivative"}, "coordinate": {"const": "time"},
                "expression": {"type": "object", "required": ["op", "ref"],
                    "properties": {"op": {"const": "quantity"}, "ref": {"const": "grazer_biomass"}}}}},
            "rhs": {"type": "object", "required": ["op", "args"], "properties": {
                "op": {"const": "multiply"}, "args": {"type": "array", "contains": {
                    "type": "object", "required": ["op", "ref"],
                    "properties": {"op": {"const": "quantity"}, "ref": {"const": "food_quality"}}}}}}}},
        "structure_schema": {"type": "object", "required": ["quantities"], "properties": {
            "quantities": {"type": "array", "contains": {"type": "object",
                "required": ["id", "entity", "property", "role"], "properties": {
                    "id": {"const": "food_quality"}, "entity": {"const": "grazer"},
                    "property": {"const": "food quality"}, "role": {"const": "state"}}}}}}}
    goal = _goal(sheet, native, "Acquire source information and reconstruct the food-quality relationship")
    goal["answer_schema"] = {"type": "object"}
    goal["criteria"] = [
        {"id": "equation_candidate", "requirement": {"kind": "native", "predicate": native}},
        {"id": "food_quality_model_equation", "requirement": {"kind": "scientific",
            "question": "Represent grazer biomass growth as depending on defined food quality.",
            "candidate": native, "predicate": structure}}]
    goal_key = "fixture:typed_structure_goal"
    sheet.put(goal_key, "discovery_goal", goal)
    def submit(key, selections, links=()):
        from types import SimpleNamespace
        source = SimpleNamespace(ledger=Ledger(_result(sheet)), commission=SimpleNamespace(discovery_goal=goal_key))
        result = submit_finding(source, {"action": "submit_finding", "key": key, "answer": {},
            "source_claims": selections, "source_observations": [], "scientific_links": list(links),
            "rationale": "Verify typed representation and exact source scope, not scientific truth."})
        sheet.merge(result.causal_sheet_updates)
        return evaluate_finding(sheet, key)
    actual = submit("fixture:actual_original_finding", live["finding"]["source_claims"])
    actual_status = {item["criterion"]: item["status"] for item in actual["results"]}
    assert actual_status == {"equation_candidate": "native_match",
                             "food_quality_model_equation": "needs_scientific_assessment"}
    assert actual["results"][0]["verified_scope"] == "native_fields_only"
    assert actual["results"][1]["verified_scope"] == "scientific_requirement"
    from episodic_workflow.credit_assignment.causal_progress import compile_episode_points
    progress = compile_episode_points(sheet)
    assert progress["components"]["discovery_progress"] == "1"
    assert "discovery_fulfillment" not in progress["components"]
    assert progress["discovery"]["native_information_total"] == "1"
    assert progress["discovery"]["scientific_fulfillment_total"] == "0"

    text = "Grazer biomass growth equals food quality times grazer biomass."
    document = Document("https://example.invalid/typed-food-quality-fixture", "Authored formal fixture", text, 12000)
    passages = {"fixture": Span("fixture", 0, len(text), text)}
    evidence = sheet.put("fixture:food_quality_source", "evidence", {
        "source": document.source, "locator": f"characters 0:{len(text)}", "study_id": "Authored typed verification fixture",
        "finding": text, "measurements": {"quote": text, "source_version": document.sha256,
            "native": {"claims": {"growth": {"kind": "equation", "statement_id": "", "locator": "Authored sentence",
                "left": "grazer biomass growth", "relation": "=", "right": "food quality times grazer biomass",
                "conditions": "Authored mathematical fixture, not an empirical result",
                "symbols": {"grazer biomass growth": "Time derivative of grazer biomass",
                    "food quality": "The declared coefficient of biomass production",
                    "grazer biomass": "The declared grazer biomass state"}}}, "observations": {}},
            "source_anchors": stored_sidecar(document, passages, claims={"growth": claim_binding(document, passages,
                text, ["Grazer biomass growth", "food quality times grazer biomass"])})}})
    source_ref = {"key": evidence["key"], "version": evidence["version"]}
    selections = [{"evidence": source_ref, "claim": "growth"}]
    pending = submit("fixture:food_quality_pending", selections)
    assert pending["results"][1]["status"] == "needs_scientific_assessment"

    fragment = empty_fragment("fixture:food_quality_fragment", "Authored grazer-growth reconstruction")
    common = {"definition": "Authored verification definition", "native_id": "", "evidence": [], "supersedes": []}
    fragment["entities"]["grazer"] = {**deepcopy(common), "kind": "grazer"}
    for name, role, prop in (("time", "coordinate", "time"), ("grazer_biomass", "state", "biomass"),
                             ("food_quality", "state", "food quality")):
        fragment["quantities"][name] = {**deepcopy(common), "role": role, "entity": "grazer", "property": prop,
            "value_type": "scalar", "units": "authored", "bindings": {}}
    quantity = lambda key: {"op": "quantity", "ref": key, "bindings": {}}
    fragment["statements"]["growth"] = {**deepcopy(common), "kind": "equation", "evidence": [source_ref],
        "scope": {"entities": ["grazer"], "conditions": [], "description": "Authored model only"},
        "domain": [], "references": [],
        "lhs": {"op": "derivative", "expression": quantity("grazer_biomass"), "coordinate": "time", "order": 1},
        "rhs": {"op": "multiply", "args": [quantity("food_quality"), quantity("grazer_biomass")]}}
    wrong = deepcopy(fragment)
    wrong["reconstruction"] = "fixture:unrelated_component_fragment"
    wrong["statements"]["growth"]["evidence"] = [sheet.anchor("fixture:evidence")]
    wrong["entities"]["unrelated"] = {**deepcopy(common), "kind": "unrelated", "evidence": [source_ref]}
    wrong_row = sheet.put(wrong["reconstruction"], "theory_fragment", wrong)
    unrelated = submit("fixture:wrong_component_evidence", selections, [sheet.anchor(wrong_row["key"])])
    assert unrelated["results"][1]["status"] == "needs_scientific_assessment"

    row = sheet.put(fragment["reconstruction"], "theory_fragment", fragment,
        dependencies=("fixture:food_quality_pending",))
    assert recheck_findings(sheet, ["fixture:food_quality_pending"])
    resolved = evaluate_finding(sheet, "fixture:food_quality_pending")["results"][1]
    assert resolved["status"] == "fulfilled" and resolved["structures"]
    assert resolved["structures"][0]["component"] == {"kind": "statement", "id": "growth"}
    # Original-only finding cannot borrow the authored food-quality source.
    assert evaluate_finding(sheet, "fixture:actual_original_finding")["results"][1]["status"] == "needs_scientific_assessment"
    credited = compile_discovery_points(sheet)["total"]
    alias = deepcopy(fragment)
    alias["reconstruction"] = "fixture:aliased_food_quality_fragment"
    alias_row = sheet.put(alias["reconstruction"], "theory_fragment", alias)
    duplicate = submit("fixture:repeated_food_quality", selections, [sheet.anchor(alias_row["key"])])
    assert duplicate["results"][1]["status"] == "fulfilled"
    assert compile_discovery_points(sheet)["total"] == credited
    assert recheck_findings(sheet) == ()
    return {"actual_call179_native_equation": actual_status, "pending_before_reconstruction": True,
        "unrelated_component_evidence_cannot_resolve": True, "source_grounded_structure_resolves": True,
        "fragment_alias_adds_no_credit": True,
        "qualification": "Checks declared typed structure and source linkage, not source fidelity or empirical truth."}


def native_operand_type_contracts():
    """Reproduce the actual first live goal's custom/native operand mismatch."""
    live = json.loads((Path(__file__).parent / "fixtures" / "discovery_native_operand_mismatch.json").read_text())
    Draft202012Validator.check_schema(live["answer_schema"])
    sheet = fixture()
    rejected = {}
    def reject(name, goal, expected):
        key = "fixture:invalid_" + name
        try:
            sheet.put(key, "discovery_goal", goal)
        except ValueError as error:
            assert expected in str(error), str(error)
            rejected[name] = str(error)
        else:
            raise AssertionError(f"Invalid native operand goal {name} was admitted")
        assert key not in sheet.rows
    reject("old_unscoped_goal", live, "criteria")
    bad_claim = deepcopy(live["criteria"][0]["check"]["claim_schema"])
    reject("live_custom_fields", _goal(sheet, {"kind": "source_claim", "claim_schema": bad_claim},
        "Retained custom/native operand mismatch"), "claim_type")
    nested = {"$defs": {"candidate": bad_claim},
              "allOf": [{"oneOf": [{"$ref": "#/$defs/candidate"}]}]}
    reject("nested_custom_fields", _goal(sheet, {"kind": "source_claim", "claim_schema": nested},
        "A local reference cannot change the native claim vocabulary"), "claim_type")
    for name, path, schema, expected in (
            ("custom_observation_field", ["native_quantity"], {"type": "string"}, "field_path[0]"),
            ("unknown_uncertainty", ["uncertainty", "sd"], {"type": "number"}, "field_path"),
            ("object_scalar_value", ["value"], {"type": "object"}, "field_path")):
        reject(name, _goal(sheet, {"kind": "source_value", "field_path": path, "value_schema": schema},
            "Observation selectors use actual native fields"), expected)
    reject("kind_relation_pair", _goal(sheet, _source_goal("equation", "increases"),
        "Equation source claims require an equality relation"), "relation.const")
    missing_answer = _goal(sheet, _source_goal("causal_relation", "increases"), "Missing answer custom type")
    missing_answer["answer_schema"] = {"$ref": "#/$defs/not_present"}
    reject("dangling_answer_type", missing_answer, "answer_schema.$ref")
    reject("dangling_claim_type", _goal(sheet, {"kind": "source_claim", "claim_schema": {
        "$ref": "#/$defs/not_present"}}, "Missing native constraint custom type"), "claim_schema.$ref")
    recursive = _goal(sheet, _source_goal("causal_relation", "increases"), "Valid recursive answer type")
    recursive["answer_schema"] = {"type": "object", "properties": {
        "next": {"anyOf": [{"type": "null"}, {"$ref": "#"}]}}}
    sheet.put("fixture:recursive_answer", "discovery_goal", recursive)
    symbols = {"$defs": {"symbol_map": {"type": "object", "properties": {"x": {"type": "string"}}}},
        "properties": {"kind": {"const": "equation"}, "symbols": {"$ref": "#/$defs/symbol_map"}}}
    _admit_goal(sheet, "fixture:native_symbol_map", {"kind": "source_claim", "claim_schema": symbols})
    return {"rejected_before_admission": rejected, "custom_answer_schema_preserved": True,
            "recursive_answer_type_admitted": True, "symbol_definition_names_remain_open": True}


def production_reader_request_contract():
    """Exercise actual Lexical extraction and its repair with the new commission."""
    from types import SimpleNamespace
    from episodic_workflow.case_reports.research.acquisition import LexicalProbeSource
    from episodic_workflow.case_reports.research.documents import Document
    from episodic_workflow.case_reports.research.episode_contracts import LexicalCommission, ExtractionRepairCommission
    from episodic_workflow.case_reports.research.extraction_repair import ExtractionRepairSource
    sheet = fixture()
    goal_key = "fixture:reader_goal"
    _admit_goal(sheet, goal_key, _source_goal("causal_relation", "increases"))
    requests = []
    memory = {"summary": "Source-only extraction fixture.", "unresolved_links": [],
              "anchors": [], "retained_evidence": []}
    def agent(request):
        requests.append(deepcopy(request))
        assert request["operation"] == "extract_causal_evidence"
        assert "extraction_request" in request and "theory_extraction" not in request
        context = request["extraction_request"]
        assert set(context) == {"question", "requested_source_criteria", "native_operand_definitions", "instruction"}
        assert "scores" not in context and "theory_fragment" not in context and "scientific_inputs" not in context
        response = ({"corrections": [{"issue": 0, "replacement": "Corrected source-only memory."}]}
                    if "repair" in request else {"findings": [], "reading_memory": deepcopy(memory)})
        Draft202012Validator(request["response_contract"]).validate(response)
        return response
    run = ResearchRun(ResearchProviders(agent=agent, search=lambda *_: [], retrieve=lambda *_: SOURCE_TEXT),
        ResearchLimits(), "Exercise actual acquisition-only extraction contracts", initial=_result(sheet))
    document = Document("https://example.invalid/reader", "Declared reader fixture", SOURCE_TEXT, 12000)
    question = "Read the declared source relationship."
    reader = LexicalProbeSource(run, run.initial, document=document, query="increasing x")
    reader.commission = LexicalCommission(question, (), discovery_goal=goal_key)
    reader.goal = SimpleNamespace(goal_id="fixture:lexical")
    first = reader.read((document.spans[0],), question)
    assert first.evaluated and first.reading_attempts[0]["status"] == "completed"
    assert reader.extraction_repair is None and len(requests) == 1
    repair = {"spans": (document.spans[0],), "question": question,
        "response": {"findings": [], "reading_memory": deepcopy(memory)},
        "errors": [{"path_segments": ["reading_memory", "summary"], "why": "Fixture correction target"}]}
    child = ExtractionRepairSource(run, run.initial, document=document, repair=repair,
        memory=memory, seen_spans={})
    child.commission = ExtractionRepairCommission(question, (), discovery_goal=goal_key)
    child.goal = SimpleNamespace(goal_id="fixture:extraction_repair")
    corrected = child.read((document.spans[0],), question)
    assert corrected.evaluated and child.extraction_repair is None and len(requests) == 2
    assert child.pending_memory["summary"] == "Corrected source-only memory."
    return {"lexical_reached_real_provider_contract": True,
            "extraction_repair_reached_real_provider_contract": True,
            "no_theory_request_attribute_required": True, "source_only_context": True}


def observation_identity_and_repair():
    """Replay the actual document-ID extraction through ordinary reader repair."""
    from types import SimpleNamespace
    from episodic_workflow.case_reports.research.acquisition import LexicalProbeSource
    from episodic_workflow.case_reports.research.documents import Document, Span
    from episodic_workflow.case_reports.research.episode_contracts import LexicalCommission, ExtractionRepairCommission
    from episodic_workflow.case_reports.research.extraction_repair import ExtractionRepairSource
    from episodic_workflow.case_reports.research.support_observations import observation_identity
    live = json.loads((Path(__file__).parent / "fixtures" / "discovery_document_id_observations.json").read_text())
    raw = live["response"]
    bad = sum(len(finding["measurements"]["observations"]) for finding in raw["findings"])
    assert bad == 6
    native = next(finding for finding in raw["findings"] if finding["measurements"]["observations"])
    evidence = {"source": live["source"], "study_id": native["study_id"], "locator": "Authored identity-check context",
        "measurements": {"native": deepcopy(native["measurements"])}}
    for selector in evidence["measurements"]["native"]["observations"]:
        try:
            observation_identity(evidence, selector)
        except ValueError as error:
            assert "not the document URL or study ID" in str(error)
        else:
            raise AssertionError("A document ID became an observation identity")
    first = deepcopy(next(iter(evidence["measurements"]["native"]["observations"].values())))
    study_id = deepcopy(evidence)
    study_id["study_id"] = "Explicit study identity"
    study_id["measurements"]["native"]["observations"] = {"first": {**first, "source_id": "Explicit study identity"}}
    try:
        observation_identity(study_id, "first")
    except ValueError as error:
        assert "not the document URL or study ID" in str(error)
    else:
        raise AssertionError("A study identity became an observation identity")
    first["source_id"] = "Complete native observation A"
    copied = deepcopy(first)
    copied.update(value="another transcription of the same outcome", units="another unit rendering", uncertainty="not reported")
    duplicated = deepcopy(evidence)
    duplicated["measurements"]["native"]["observations"] = {"first": first, "copy": copied}
    assert observation_identity(duplicated, "first")[0] == observation_identity(duplicated, "copy")[0]
    for field, changed in (("quantity", "a different measured quantity"), ("conditions", "a different experimental condition")):
        collision = deepcopy(duplicated)
        collision["measurements"]["native"]["observations"]["copy"][field] = changed
        for selector in ("first", "copy"):
            try:
                observation_identity(collision, selector)
            except ValueError as error:
                assert "different quantity or condition" in str(error)
            else:
                raise AssertionError("Different observations silently shared one native identity")
    spans = tuple(Span(**value) for value in live["passages"].values())
    characters = [" "] * max(span.end for span in spans)
    for span in spans:
        characters[span.start:span.end] = span.text
    document = Document(live["source"], "Retained source-span regression fixture", "".join(characters), 12000)
    passages = {span.key: span for span in spans}
    # Adapt two model paraphrases to the new source-literal value contract;
    # the retained original response fixture itself remains unchanged.
    raw = deepcopy(raw)
    for finding in raw["findings"]:
        observations = finding["measurements"]["observations"]
        if "oxidative_phosphorylation_flux" in observations:
            observations["oxidative_phosphorylation_flux"]["value"] = "compensatory decrease"
        if "fructose_2_6_bisphosphate" in observations:
            observations["fructose_2_6_bisphosphate"]["value"] = "decrease"
    raw = attach_response_anchors(document, passages, raw, {
        "anaerobic_flux": ("Enzyme overexpression had no effect", "glycolytic flux under anaerobic conditions", "no effect", 1, "under anaerobic conditions"),
        "aerobic_flux": ("increased glycolytic flux up to the anaerobic level", "glycolytic flux up to the anaerobic level", "increased glycolytic flux up to the anaerobic level", 1, "under aerobic conditions"),
        "oxidative_phosphorylation_flux": ("compensatory decrease in flux in oxidative phosphorylation", "flux in oxidative phosphorylation", "compensatory decrease"),
        "enzyme_substrates": ("The concentrations of the enzyme substrates showed only small or insignificant changes", "concentrations of the enzyme substrates", "only small or insignificant changes"),
        "phosphofructokinase_2_activity": ("compensatory decrease in 6-phosphofructo-2-kinase activity", "6-phosphofructo-2-kinase activity", "decrease", 1, "in cells overexpressing the enzyme"),
        "fructose_2_6_bisphosphate": ("corresponding decrease in fructose 2,6-bisphosphate concentration", "fructose 2,6-bisphosphate concentration", "decrease", 2, "in cells overexpressing the enzyme")}, {
        "anaerobic_effect": ("Enzyme overexpression had no effect", ["Enzyme overexpression"], ["under anaerobic conditions"]),
        "aerobic_effect": ("increased glycolytic flux up to the anaerobic level", ["glycolytic flux up to the anaerobic level"], ["under aerobic conditions"]),
        "effector_decrease": ("this could compensate for the increased enzyme concentration in the overproducer", ["the increased enzyme concentration in the overproducer"], ["Measurements in vitro"])},
        {0: ["on glycolytic flux in the yeast Saccharomyces cerevisiae"], 4: ["## Abstract"]})
    precise = {
        "anaerobic_flux": "Abstract, sentence 2, anaerobic glycolytic-flux clause",
        "aerobic_flux": "Abstract, sentence 2, aerobic glycolytic-flux clause",
        "oxidative_phosphorylation_flux": "Abstract, sentence 4, oxidative-phosphorylation flux",
        "enzyme_substrates": "Abstract, sentence 5, enzyme-substrate concentrations",
        "phosphofructokinase_2_activity": "Abstract, sentence 7, phosphofructokinase-2 activity clause",
        "fructose_2_6_bisphosphate": "Abstract, sentence 7, fructose-2,6-bisphosphate concentration clause"}
    requests = []
    initial_response = raw
    def agent(request):
        requests.append(deepcopy(request))
        if "repair" not in request:
            return deepcopy(initial_response)
        corrections = []
        for issue in request["repair"]["issues"]:
            path = issue["path_segments"]
            assert len(path) == 5 and path[0] == "findings" and path[3] == "observations"
            observation = deepcopy(request["repair"]["base_response"]["findings"][path[1]]["measurements"]["observations"][path[4]])
            observation.update(source_id="", locator=precise[path[4]])
            corrections.append({"issue": issue["issue"], "replacement": observation})
        result = {"corrections": corrections}
        Draft202012Validator(request["response_contract"]).validate(result)
        return result
    sheet = fixture()
    goal_key = "fixture:observation_identity_goal"
    _admit_goal(sheet, goal_key, _source_goal("causal_relation", "increases"))
    run = ResearchRun(ResearchProviders(agent=agent, search=lambda *_: [], retrieve=lambda *_: document.text),
        ResearchLimits(), "Exercise retained observation-ID failure and normal repair", initial=_result(sheet))
    initial_response = deepcopy(raw)
    for finding in initial_response["findings"]:
        for observation in finding["measurements"]["observations"].values():
            observation["source_id"] = "Conflicting native observation identity"
    collision_reader = LexicalProbeSource(run, run.initial, document=document, query="flux")
    collision_reader.commission = LexicalCommission(live["question"], (), discovery_goal=goal_key)
    collision_reader.goal = SimpleNamespace(goal_id="fixture:colliding_observation_reader")
    collided = collision_reader.read(spans, live["question"])
    assert collision_reader.extraction_repair is not None
    assert len(collision_reader.extraction_repair["errors"]) == bad
    assert all(not row["data"]["measurements"]["native"]["observations"]
               for row in collided.causal_sheet_updates if row["kind"] == "evidence")
    initial_response = raw
    reader = LexicalProbeSource(run, run.initial, document=document, query="flux")
    reader.commission = LexicalCommission(live["question"], (), discovery_goal=goal_key)
    reader.goal = SimpleNamespace(goal_id="fixture:observation_identity_reader")
    first_result = reader.read(spans, live["question"])
    assert reader.extraction_repair is not None
    rejected = reader.extraction_repair["errors"]
    assert len([error for error in rejected if error.get("path_segments", [None])[0] == "findings"]) == bad
    assert all(not row["data"]["measurements"]["native"]["observations"]
               for row in first_result.causal_sheet_updates if row["kind"] == "evidence")
    retained = Ledger(run.initial)
    retained.merge(first_result)
    repair = ExtractionRepairSource(run, retained.result(), document=document,
        repair=reader.extraction_repair, memory=reader.pending_memory or reader.memory,
        seen_spans=reader.seen_spans)
    repair.commission = ExtractionRepairCommission(live["question"], (), discovery_goal=goal_key)
    repair.goal = SimpleNamespace(goal_id="fixture:ordinary_extraction_repair")
    repaired = repair.read(spans, live["question"])
    assert repaired.evaluated and repair.extraction_repair is None
    retained.merge(repaired)
    identities = [observation_identity(row["data"], selector)[0]
        for _, row in retained.causal_sheet.of_kind("evidence") if row["data"]["source"] == live["source"]
        for selector in row["data"]["measurements"]["native"]["observations"]]
    assert len(identities) == len(set(identities)) == bad
    return {"actual_live_observations_rejected": bad, "ordinary_extraction_repair": True,
            "repaired_distinct_observations": len(identities), "same_observation_copies_deduplicated": True}


def source_occurrence_replay():
    """Actual call8/27 repeats stay duplicates; actual call28 stays new."""
    from types import SimpleNamespace
    from episodic_workflow.case_reports.research.acquisition import LexicalProbeSource
    from episodic_workflow.case_reports.research.documents import Span
    from episodic_workflow.case_reports.research.episode_contracts import LexicalCommission
    from episodic_workflow.case_reports.research.episode_credit import lexical_credit, _extracted_items
    live = json.loads((Path(__file__).parent / "fixtures" / "source_occurrence_reread.json").read_text())
    assert [entry["call"] for entry in live] == [8, 27, 28]
    document = retained_document(live[0]["request"]["source"], [entry["request"]["passages"] for entry in live])
    observation_quotes = {
        "daphnia_biomass": ("_Daphnia_ biomass increased", "_Daphnia_ biomass", "increased", 1, "that is, with nutrient enrichment"),
        "algal_biomass": ("algal biomass often remained constant", "algal biomass", "often remained constant", 1, "that is, with nutrient enrichment"),
        "persistent_oscillations": ("persistent oscillations have only rarely been observed", "persistent oscillations", "only rarely been observed", 1, "that is, with nutrient enrichment"),
        "daphnia_establishment": ("_Daphnia_ could establish significant populations", "_Daphnia_ could establish significant populations", "could establish significant populations only at relatively high dilution rates", 1, "at relatively high dilution rates")}
    for key in ("daphnia_biomass", "algal_biomass", "persistent_oscillations"):
        observation_quotes["nutrient_enrichment_" + key] = observation_quotes[key]
    claim_quotes = {
        "numerical_energy_enrichment_boundary": ("the paradox of energy enrichment should be restricted", ["the paradox of energy enrichment should be restricted"], ["<0.15 μM realized TP"]),
        "stoichiometric_explanation_limit": ("stoichiometric constraints cannot explain the lack of population cycles", ["stoichiometric constraints"], ["high nutrient supply"]),
        "inedibility_proposal": ("mechanically inedible algae with nutrient enrichment", ["mechanically inedible algae"], ["inedible algae with nutrient enrichment"]),
        "sinking_quality": ("Sinking has a positive influence on algal quality", ["Sinking has a positive influence"], ["sinking losses reduce algal density"]),
        "sinking_invasion": ("At sufficiently high sinking losses", ["allow grazers to invade the algae‐only equilibrium"], ["sufficiently high sinking losses"]),
        "dilution_density": ("dilution losses prevented the buildup", ["dilution losses prevented"], ["at relatively high dilution rates"])}
    sheet = fixture()
    goal_key = "fixture:source_occurrence_goal"
    _admit_goal(sheet, goal_key, _source_goal("causal_relation", "increases"))
    pending = None
    def agent(request):
        Draft202012Validator(request["response_contract"]).validate(pending)
        # Unit text is supplied once, labeled; no model character counting.
        assert all("text" not in passage and passage["source_units"] for passage in request["passages"].values())
        return deepcopy(pending)
    run = ResearchRun(ResearchProviders(agent=agent, search=lambda *_: [], retrieve=lambda *_: document.text),
        ResearchLimits(), "Retained exact source-occurrence replay", initial=_result(sheet))
    retained = Ledger(run.initial)
    gained, observation_sets = {}, {}
    for entry in live:
        spans = {key: Span(**value) for key, value in entry["request"]["passages"].items()}
        source_literal_response = deepcopy(entry["response"])
        for finding in source_literal_response["findings"]:
            for selector, observation in finding["measurements"]["observations"].items():
                if selector.endswith("persistent_oscillations"):
                    observation["value"] = "only rarely been observed"
        pending = attach_response_anchors(document, spans, source_literal_response, observation_quotes, claim_quotes,
            {1: ["stoichiometric constraints cannot explain the lack of population cycles"]} if entry["call"] == 8 else {})
        pending["reading_memory"]["retained_evidence"] = []
        pending["reading_memory"]["anchors"] = []
        before = retained.result()
        reader = LexicalProbeSource(run, before, document=document, query="source occurrence replay")
        reader.commission = LexicalCommission(entry["request"]["question"], (), discovery_goal=goal_key)
        reader.goal = SimpleNamespace(goal_id="fixture:source_occurrence:" + str(entry["call"]))
        outcome = reader.read(tuple(spans.values()), entry["request"]["question"])
        assert outcome.evaluated and reader.extraction_repair is None, reader.extraction_repair
        items = _extracted_items(outcome, lambda data: data["source"] == document.source)
        observation_sets[entry["call"]] = {item for item in items if item.startswith("observation:")}
        gain = lexical_credit(outcome, before, document, {"all": tuple(spans.values())})
        gained[entry["call"]] = sum(value for _, _, value in gain.qualities)
        retained.merge(outcome)
    assert observation_sets[8] == observation_sets[27] and len(observation_sets[8]) == 3
    assert gained[27] == 2, "Only first typed extraction of the two explanatory claims is new"
    assert gained[28] == 4, "New sinking/Sommer source information must remain credited"
    before = retained.result()
    reader = LexicalProbeSource(run, before, document=document, query="repeat final exact extraction")
    reader.commission = LexicalCommission(live[-1]["request"]["question"], (), discovery_goal=goal_key)
    reader.goal = SimpleNamespace(goal_id="fixture:repeat_final_extraction")
    repeated = reader.read(tuple(spans.values()), live[-1]["request"]["question"])
    assert reader.extraction_repair is None
    assert not lexical_credit(repeated, before, document, {"all": tuple(spans.values())}).qualities
    return {"actual_calls": [8, 27, 28], "repeated_observation_credit": 0,
            "new_typed_claims_call27": gained[27], "new_information_call28": gained[28]}


def source_result_condition_drift_replay():
    """Actual calls274/396 repeat nine results under changed condition wording."""
    from types import SimpleNamespace
    from episodic_workflow.case_reports.research import source_anchors
    from episodic_workflow.case_reports.research.acquisition import LexicalProbeSource, PageSource, SearchSource
    from episodic_workflow.case_reports.research.documents import Document, Span
    from episodic_workflow.case_reports.research.episode_contracts import LexicalCommission, DiscoveryCommission, SearchCommission
    from episodic_workflow.case_reports.research.episode_credit import lexical_credit, _extracted_items
    from episodic_workflow.case_reports.research.source_claims import claim_identity
    from episodic_workflow.case_reports.research.support_observations import observation_identity
    live = json.loads((Path(__file__).parent / "fixtures" / "source_result_condition_drift.json").read_text())
    assert [entry["call"] for entry in live] == [274, 396]
    span_groups = [unit_passages(entry["request"]["passages"]) for entry in live]
    document = retained_document(live[0]["request"]["source"],
        [{key: vars(span) for key, span in spans.items()} for spans in span_groups])
    units = {(unit["start"], unit["end"]): unit for unit in source_anchors.source_units(document)}
    literals = {"ki": "21", "a_control_vmax": "169", "a_no_vmax": "169",
        "a_control_km": "0.73", "a_no_km": "0.73", "a_phloretin_vmax": "169", "a_with_vmax": "169",
        "a_phloretin_km": "2.99", "a_with_km": "2.99", "b_control_vmax": "198", "b_no_vmax": "198",
        "b_control_km": "1.85", "b_no_km": "1.85", "b_phloretin_vmax": "193", "b_with_vmax": "193",
        "b_phloretin_km": "6.35", "b_with_km": "6.35"}
    sheet = fixture()
    goal_key = "fixture:reported_result_values"
    observations = [observation for entry in live for finding in entry["response"]["findings"]
                    for observation in finding["measurements"]["observations"].values()]
    _admit_goal(sheet, goal_key, _observation_predicate(
        {"enum": sorted({item["quantity"] for item in observations})},
        {"type": "string", "pattern": ".+"},
        {"enum": sorted({item["units"] for item in observations})}, kind="fit_parameter"))
    pending = None
    def agent(request):
        Draft202012Validator(request["response_contract"]).validate(pending)
        return deepcopy(pending)
    run = ResearchRun(ResearchProviders(agent=agent, search=lambda *_: [], retrieve=lambda *_: document.text),
        ResearchLimits(), "Replay retained numerical source results with condition drift", initial=_result(sheet))
    retained = Ledger(run.initial)
    observed, all_conditions, gains = {}, {}, {}
    for entry, spans in zip(live, span_groups):
        original_units = {unit["id"]: unit for unit in entry["request"]["units"]}
        def rebind(value):
            old_unit = original_units[value["unit"]]
            unit = units[(old_unit["start"], old_unit["end"])]
            return {**deepcopy(value), "unit": unit["id"]}
        pending = deepcopy(entry["response"])
        # Isolate the nine reported observations. Original claims/prose and all
        # original responses remain saved in the retained JSON fixture.
        pending["findings"] = [finding for finding in pending["findings"] if finding["measurements"]["observations"]]
        for finding in pending["findings"]:
            finding["measurements"]["claims"] = {}
            anchored = {}
            for selector, old in finding["source_anchors"]["observations"].items():
                statement = rebind(old["statement"])
                anchored[selector] = {"statement": statement, "quantity": rebind(old["quantity"]),
                    "conditions": [rebind(condition) for condition in old["conditions"]],
                    "result": result_anchor(document, spans, literals[selector], 1, unit_id=statement["unit"])}
            finding["source_anchors"] = {"observations": anchored, "claims": {}, "finding": []}
        pending["reading_memory"]["retained_evidence"] = []
        pending["reading_memory"]["anchors"] = []
        before = retained.result()
        page = PageSource(run, before, candidate={"url": document.source})
        page.document = document
        search = SearchSource(run, before, query="reported transport measurements")
        search.commission = SearchCommission(entry["request"]["question"], (), discovery_goal=goal_key)
        search.selected_sources.add(document.source)
        discovery = DiscoverySource(run, before)
        discovery.commission = DiscoveryCommission(entry["request"]["question"], (), discovery_goal=goal_key)
        reader = LexicalProbeSource(run, before, document=document, query="reported numerical results")
        reader.commission = LexicalCommission(entry["request"]["question"], (), discovery_goal=goal_key)
        reader.goal = SimpleNamespace(goal_id="fixture:condition_drift:" + str(entry["call"]))
        outcome = reader.read(tuple(spans.values()), entry["request"]["question"])
        assert outcome.evaluated and reader.extraction_repair is None, reader.extraction_repair
        observed[entry["call"]] = _extracted_items(outcome, lambda data: data["source"] == document.source)
        assert len(observed[entry["call"]]) == 9
        all_conditions[entry["call"]] = {observation_identity(row["data"], selector)[0]: {
            "native": observation["conditions"],
            "source_occurrences": [(condition["start"], condition["end"]) for condition in
                row["data"]["measurements"]["source_anchors"]["observations"][selector]["conditions"]]}
            for row in outcome.causal_sheet_updates if row["kind"] == "evidence"
            for selector, observation in row["data"]["measurements"]["native"]["observations"].items()}
        child_gain = sum(value for _, _, value in lexical_credit(outcome, before, document, {"all": tuple(spans.values())}).qualities)
        page_gain = sum(value for _, _, value in page.credit(outcome).qualities)
        retained.merge(outcome)
        search_gain = sum(value for _, _, value in search.credit(retained.result()).qualities)
        selections = [{"evidence": {"key": row["key"], "version": row["version"]}, "observation": selector}
            for row in outcome.causal_sheet_updates if row["kind"] == "evidence"
            for selector in row["data"]["measurements"]["native"]["observations"]]
        discovery.ledger.merge(retained.result())
        found = submit_finding(discovery, {"action": "submit_finding", "key": "fixture:result_read:" + str(entry["call"]),
            "answer": {"account": "The source reports the selected numerical transport results with qualifications."},
            "source_claims": [], "source_observations": selections, "scientific_links": [],
            "rationale": "Return actual source values without treating a changed condition description as new data"})
        retained.merge(found)
        discovery_gain = sum(value for _, _, value in discovery.credit(retained.result()).qualities)
        global_total = compile_discovery_points(retained.causal_sheet)["total"]
        gains[entry["call"]] = {"child": child_gain, "page": page_gain, "search": search_gain,
            "discovery": discovery_gain, "global_total": global_total}
    assert observed[274] == observed[396]
    assert all_conditions[274] != all_conditions[396], "The actual changed condition sets must remain in this regression"
    assert gains[274] == {"child": 9, "page": 9, "search": 1, "discovery": 1, "global_total": "9"}
    assert gains[396] == {"child": 0, "page": 0, "search": 0, "discovery": 0, "global_total": "9"}

    authored = fixture()
    original = authored.rows["fixture:evidence"]["data"]
    source_document = Document(original["source"], "Authored conditional claim", SOURCE_TEXT, 12000)
    source_spans = {"claim": Span("claim", 0, len(SOURCE_TEXT), SOURCE_TEXT)}
    changed = deepcopy(original)
    changed["measurements"]["native"]["claims"]["positive_relation"]["conditions"] = "Positive a; condition wording has been expanded"
    expanded = stored_sidecar(source_document, source_spans, claims={"positive_relation": claim_binding(
        source_document, source_spans, "increasing x increases y", ["increasing x increases y"], ["a is positive"])},
        ledger=Ledger(_result(authored)))
    changed["measurements"]["source_anchors"]["claims"]["positive_relation"] = expanded["claims"]["positive_relation"]
    changed["measurements"]["source_anchors"]["source_spans"] = expanded["source_spans"]
    assert claim_identity(original, "positive_relation")[0] == claim_identity(changed, "positive_relation")[0]
    return {"actual_calls": [274, 396], "first_extraction": gains[274], "condition_drift_reread": gains[396],
            "actual_nine_results_unchanged": True, "claim_condition_drift_adds_no_identity": True}


def source_result_literal_and_repair():
    """Literal mismatch returns to ordinary repair; source headings stay usable."""
    from types import SimpleNamespace
    from episodic_workflow.case_reports.research import source_anchors
    from episodic_workflow.case_reports.research.acquisition import LexicalProbeSource
    from episodic_workflow.case_reports.research.documents import Document, Span
    from episodic_workflow.case_reports.research.episode_contracts import LexicalCommission, ExtractionRepairCommission
    from episodic_workflow.case_reports.research.extraction_repair import ExtractionRepairSource
    from episodic_workflow.case_reports.research.support_observations import observation_identity
    text = "Quantity heading: biomass. Reported value was −2e0 units. Flag was true. Reported text was often  increased."
    document = Document("https://example.invalid/literal-source-results", "Authored literal results", text, 12000)
    spans = {"literal": Span("literal", 0, len(text), text)}
    heading_unit = anchor(document, spans, "biomass")["unit"]
    binding = observation_binding(document, spans, "Reported value was", "biomass", "−2e0",
        quantity_unit_id=heading_unit)
    assert binding["quantity"]["unit"] != binding["statement"]["unit"]
    bad_observation = {"source_id": "", "kind": "measurement", "locator": "Authored reported value", "quantity": "biomass",
        "conditions": "Authored fixture", "value": 99, "units": "units", "uncertainty": "not reported"}
    response = {"findings": [{"span": "literal", "study_id": "Authored literal-value fixture",
        "finding": "The source reports a value under its quantity heading.",
        "measurements": {"observations": {"reported": bad_observation}, "claims": {}},
        "source_anchors": {"observations": {"reported": binding}, "claims": {}, "finding": []}}],
        "reading_memory": {"summary": "Authored literal-value fixture", "unresolved_links": [], "anchors": [], "retained_evidence": []}}
    requests = []
    def agent(request):
        requests.append(deepcopy(request))
        if "repair" not in request:
            return deepcopy(response)
        assert request["repair"]["base_response"]["findings"][0]["source_anchors"] == response["findings"][0]["source_anchors"]
        issues = request["repair"]["issues"]
        assert len(issues) == 1 and issues[0]["path_segments"] == ["findings", 0, "measurements", "observations", "reported"]
        corrected = {"corrections": [{"issue": issues[0]["issue"], "replacement": {**bad_observation, "value": -2}}]}
        Draft202012Validator(request["response_contract"]).validate(corrected)
        return corrected
    sheet = fixture()
    goal_key = "fixture:literal_value_goal"
    _admit_goal(sheet, goal_key, _observation_predicate(
        {"const": "biomass"}, {"const": "Authored fixture"}, {"const": "units"}))
    run = ResearchRun(ResearchProviders(agent=agent, search=lambda *_: [], retrieve=lambda *_: text),
        ResearchLimits(), "Exercise source-result literal repair", initial=_result(sheet))
    reader = LexicalProbeSource(run, run.initial, document=document, query="reported source value")
    reader.commission = LexicalCommission("Extract the reported value", (), discovery_goal=goal_key)
    reader.goal = SimpleNamespace(goal_id="fixture:literal_mismatch")
    first = reader.read(tuple(spans.values()), "Extract the reported value")
    assert reader.extraction_repair is not None
    assert reader.extraction_repair["errors"][0]["path_segments"] == ["findings", 0, "measurements", "observations", "reported"]
    assert all(not row["data"]["measurements"]["native"]["observations"] for row in first.causal_sheet_updates)
    retained = Ledger(run.initial)
    retained.merge(first)
    repair = ExtractionRepairSource(run, retained.result(), document=document, repair=reader.extraction_repair,
        memory=reader.pending_memory or reader.memory, seen_spans=reader.seen_spans)
    repair.commission = ExtractionRepairCommission("Extract the reported value", (), discovery_goal=goal_key)
    repair.goal = SimpleNamespace(goal_id="fixture:literal_value_repair")
    repaired = repair.read(tuple(spans.values()), "Extract the reported value")
    assert repaired.evaluated and repair.extraction_repair is None, repair.extraction_repair
    admitted = next(row["data"] for row in repaired.causal_sheet_updates if row["kind"] == "evidence")
    assert admitted["measurements"]["native"]["observations"]["reported"] == {**bad_observation, "value": -2}
    assert observation_identity(admitted, "reported")[0]
    typed = deepcopy(admitted)
    typed["measurements"]["native"]["observations"] = {
        "boolean": {**bad_observation, "quantity": "flag", "value": True},
        "text": {**bad_observation, "quantity": "reported text", "value": "often increased"}}
    typed["measurements"]["source_anchors"] = stored_sidecar(document, spans, observations={
        "boolean": observation_binding(document, spans, "Flag was true", "Flag", "true"),
        "text": observation_binding(document, spans, "Reported text was", "Reported text", "often  increased")})
    assert observation_identity(typed, "boolean")[0] and observation_identity(typed, "text")[0]
    try:
        source_anchors.resolve_entry(SimpleNamespace(document=document), spans, "observations", "partial",
            observation_binding(document, spans, "Reported value was", "biomass", "2", quantity_unit_id=heading_unit), {})
    except source_anchors.SourceAnchorError as error:
        assert error.path[:3] == ["observations", "partial", "result"]
    else:
        raise AssertionError("A substring of the signed exponent value became a new result")
    return {"ordinary_literal_mismatch_repair": True, "only_unaccepted_value_changed": True,
        "unicode_minus_exponent_numeric_match": True, "external_quantity_heading_retained": True,
        "literal_boolean_and_normalized_text": True, "partial_numeric_token_rejected": True}


def source_anchor_boundaries_and_repair():
    """Boundary drift, distinct source roles, and sidecar-only ordinary repair."""
    from types import SimpleNamespace
    from episodic_workflow.case_reports.research import source_anchors
    from episodic_workflow.case_reports.research.acquisition import LexicalProbeSource
    from episodic_workflow.case_reports.research.documents import Document, Span
    from episodic_workflow.case_reports.research.episode_contracts import LexicalCommission, ExtractionRepairCommission
    from episodic_workflow.case_reports.research.extraction_repair import ExtractionRepairSource
    from episodic_workflow.case_reports.research.support_observations import observation_identity
    live = json.loads((Path(__file__).parent / "fixtures" / "source_unit_quote_resolution.json").read_text())
    supplied = live["passage"]
    text_parts = [" "] * (supplied["end"] - supplied["start"])
    for part in supplied["source_units"]:
        text_parts[part["start"] - supplied["start"]:part["end"] - supplied["start"]] = part["text"]
    retained_span = Span(supplied["key"], supplied["start"], supplied["end"], "".join(text_parts))
    duplicate_document = retained_document(live["source"], [{retained_span.key: vars(retained_span)}])
    duplicate_reader = SimpleNamespace(document=duplicate_document)
    duplicate_passages = {retained_span.key: retained_span}
    quote = live["anchor"]["quote"]
    assert retained_span.text.count(quote) == 2
    matching_units = [unit for unit in source_anchors.source_units(duplicate_document)
                      if (unit["start"], unit["end"]) in {(37157, 37189), (37223, 37255)}]
    assert len(matching_units) == 2
    resolved = []
    for unit in matching_units:
        selection = {**live["anchor"], "unit": unit["id"]}
        exact = source_anchors.resolve_finding(duplicate_reader, duplicate_passages, [selection], {})[0]
        assert exact["start"] == unit["start"] and exact["end"] == unit["end"]
        resolved.append(exact["unit"]["id"])
    assert len(set(resolved)) == 2, "The selected host unit disambiguates identical clauses in one supplied passage"
    ambiguous_text = "growth rose, growth rose at dawn. Next source statement."
    ambiguous_document = Document("https://example.invalid/repeated-quote", "Quote ambiguity", ambiguous_text, 12000)
    ambiguous_spans = {"ambiguous": Span("ambiguous", 0, len(ambiguous_text), ambiguous_text)}
    first_unit = source_anchors.source_units(ambiguous_document)[0]
    for short_quote in ("growth rose", "at dawn. Next source"):
        try:
            source_anchors.resolve_finding(SimpleNamespace(document=ambiguous_document), ambiguous_spans,
                [{"span": "ambiguous", "unit": first_unit["id"], "quote": short_quote}], {})
        except source_anchors.SourceAnchorError as error:
            assert error.path[:2] == ["finding", 0]
        else:
            raise AssertionError("A repeated quote within one unit or a quote crossing units was accepted")
    overlapping_document = Document("https://example.invalid/overlapping-quote", "Overlapping quote", "aaaa", 12000)
    try:
        source_anchors.resolve_finding(SimpleNamespace(document=overlapping_document),
            {"overlap": Span("overlap", 0, 4, "aaaa")}, [{"span": "overlap",
                "unit": source_anchors.source_units(overlapping_document)[0]["id"], "quote": "aaa"}], {})
    except source_anchors.SourceAnchorError as error:
        assert error.path == ["finding", 0, "quote"]
    else:
        raise AssertionError("Overlapping repeated quote occurrences were accepted")
    text = "Daphnia biomass was 2 at low nutrient supply, 2 at high nutrient supply."
    document = Document("https://example.invalid/source-occurrence", "Authored occurrences", text, 12000)
    passages = {"roles": Span("roles", 0, len(text), text)}
    sheet = fixture()
    def data(binding, *, value=2, conditions="low nutrient supply"):
        return {"source": document.source, "study_id": "Authored occurrence fixture", "locator": f"characters 0:{len(text)}",
            "finding": "Authored fixture measurement", "measurements": {"quote": text, "source_version": document.sha256,
                "native": {"observations": {"one": {"source_id": "", "kind": "measurement", "locator": "Authored clause",
                    "quantity": "Daphnia biomass", "conditions": conditions, "value": value, "units": "authored", "uncertainty": "exact"}}, "claims": {}},
                "source_anchors": stored_sidecar(document, passages, observations={"one": binding}, ledger=Ledger(_result(sheet)))}}
    ids = []
    for index, (statement, quantity) in enumerate((
            ("Daphnia biomass was 2", "Daphnia"),
            ("Daphnia biomass was 2 at low nutrient supply", "Daphnia biomass"),
            ("biomass was 2", "biomass"))):
        current = data(observation_binding(document, passages, statement, quantity, "2", 1, "low nutrient supply"))
        ids.append(observation_identity(current, "one")[0])
        sheet.put("fixture:drift:" + str(index), "evidence", current)
    assert len(set(ids)) == 1, "Narrow→expanded→other narrow quote retains the canonical occurrence"
    changed_condition = data(observation_binding(document, passages, "biomass was 2", "biomass", "2", 1, "high nutrient supply"),
                conditions="high nutrient supply")
    assert observation_identity(changed_condition, "one")[0] == ids[0], "Condition description drift cannot mint an observation"
    high = data(observation_binding(document, passages, "biomass was 2", "biomass", "2", 2, "high nutrient supply"),
                conditions="high nutrient supply")
    assert observation_identity(high, "one")[0] != ids[0], "Equal values at distinct actual result positions remain distinct"

    multi = "Daphnia biomass, algal biomass rose at high nutrient supply."
    second = Document("https://example.invalid/two-source-quantities", "Two quantities", multi, 12000)
    second_spans = {"two": Span("two", 0, len(multi), multi)}
    def binding(quantity):
        return observation_binding(second, second_spans, "biomass rose", quantity, "rose", 1, "high nutrient supply")
    known = {}
    a = source_anchors.resolve_entry(SimpleNamespace(document=second), second_spans, "observations", "a", binding("Daphnia biomass"), known)
    b = source_anchors.resolve_entry(SimpleNamespace(document=second), second_spans, "observations", "b", binding("algal biomass"), known)
    assert a["quantity"]["identity"]["id"] != b["quantity"]["identity"]["id"]
    try:
        source_anchors.resolve_entry(SimpleNamespace(document=second), second_spans, "observations", "broad", binding("Daphnia biomass, algal biomass"), known)
    except source_anchors.SourceAnchorError as error:
        assert error.path == ["observations", "broad", "quantity"]
    else:
        raise AssertionError("A broad quote merged two established quantity occurrences")

    base_observation = {"source_id": "", "kind": "measurement", "locator": "Authored sentence", "quantity": "Daphnia biomass",
        "conditions": "high nutrient supply", "value": "rose", "units": "authored", "uncertainty": "not reported"}
    response = {"findings": [{"span": "two", "study_id": "Authored two-quantity fixture", "finding": "Two quantities rose.",
        "measurements": {"observations": {"a": base_observation, "b": {**base_observation, "quantity": "algal biomass"}}, "claims": {}},
        "source_anchors": {"observations": {"a": binding("Daphnia biomass, algal biomass"),
            "b": binding("algal biomass")}, "claims": {}, "finding": []}}],
        "reading_memory": {"summary": "Authored two-quantity fixture", "unresolved_links": [], "anchors": [], "retained_evidence": []}}
    goal_key = "fixture:anchor_repair_goal"
    _admit_goal(sheet, goal_key, _source_goal("causal_relation", "increases"))
    requests = []
    def agent(request):
        requests.append(deepcopy(request))
        if "repair" not in request:
            return deepcopy(response)
        corrections = []
        for issue in request["repair"]["issues"]:
            path = issue["path_segments"]
            assert path[:4] == ["findings", 0, "source_anchors", "observations"] and len(path) == 5
            replacement = binding("Daphnia biomass" if path[4] == "a" else "algal biomass")
            corrections.append({"issue": issue["issue"], "replacement": replacement})
        corrected = {"corrections": corrections}
        Draft202012Validator(request["response_contract"]).validate(corrected)
        return corrected
    run = ResearchRun(ResearchProviders(agent=agent, search=lambda *_: [], retrieve=lambda *_: multi),
        ResearchLimits(), "Exercise targeted anchor repair", initial=_result(sheet))
    reader = LexicalProbeSource(run, run.initial, document=second, query="two quantities")
    reader.commission = LexicalCommission("Extract both quantities", (), discovery_goal=goal_key)
    reader.goal = SimpleNamespace(goal_id="fixture:ambiguous_first_anchor")
    first = reader.read(tuple(second_spans.values()), "Extract both quantities")
    assert reader.extraction_repair is not None
    assert all(not row["data"]["measurements"]["native"]["observations"] for row in first.causal_sheet_updates)
    assert len(reader.extraction_repair["errors"]) == 2
    retained = Ledger(run.initial)
    retained.merge(first)
    repair = ExtractionRepairSource(run, retained.result(), document=second, repair=reader.extraction_repair,
        memory=reader.pending_memory or reader.memory, seen_spans=reader.seen_spans)
    repair.commission = ExtractionRepairCommission("Extract both quantities", (), discovery_goal=goal_key)
    repair.goal = SimpleNamespace(goal_id="fixture:source_anchor_repair")
    fixed = repair.read(tuple(second_spans.values()), "Extract both quantities")
    assert fixed.evaluated and repair.extraction_repair is None, repair.extraction_repair
    admitted = next(row["data"] for row in fixed.causal_sheet_updates if row["kind"] == "evidence")
    assert admitted["measurements"]["native"] == response["findings"][0]["measurements"]
    assert observation_identity(admitted, "a")[0] != observation_identity(admitted, "b")[0]
    return {"actual_call34_identical_clauses_disambiguated_by_unit": True,
            "within_unit_ambiguity_rejected": True, "cross_unit_quote_rejected": True,
            "overlapping_quote_ambiguity_rejected": True,
            "quote_drift_steps": 3, "condition_description_drift_adds_no_credit": True,
            "equal_values_at_distinct_source_positions": True, "separate_quantities": True,
            "ambiguous_anchor_targeted_repair": True, "accepted_native_fields_unchanged": True}


def observation_request_credit():
    """Retained call457's loose goal and call516's excluded fit cannot earn flux credit."""
    from types import SimpleNamespace
    from episodic_workflow.case_reports.research.discovery_goals import _native_matches, source_credit
    retained = json.loads((Path(__file__).parent / "fixtures" / "discovery_flux_credit_mismatch.json").read_text())
    assert [entry["call"] for entry in retained] == [457, 516]
    sheet = fixture()
    old = deepcopy(retained[0]["response"]["goal"])
    old["inputs"] = [sheet.anchor("fixture:G")]
    try:
        sheet.put("fixture:loose_numeric_goal", "discovery_goal", old)
    except ValueError as error:
        assert "field_path=[]" in str(error), str(error)
    else:
        raise AssertionError("The actual loose numeric goal was admitted")
    goal = _goal(sheet, _observation_predicate({"const": "stationary_product_flux_per_enzyme"},
        {"type": "object", "required": ["substrate_M"],
         "properties": {"substrate_M": {"const": 0.00002}}}, {"const": "s^-1"}),
        "Measure stationary per-enzyme product flux at 20 micromolar substrate; exclude fitted saturation parameters")
    for label, condition in (("unspecified", {"type": "object"}),
                             ("optional", {"type": "object", "properties": {"substrate_M": {"const": 0.00002}}})):
        malformed = deepcopy(goal)
        malformed["criteria"][0]["requirement"]["predicate"]["value_schema"]["properties"]["conditions"] = condition
        try:
            sheet.put("fixture:unconstrained_" + label, "discovery_goal", malformed)
        except ValueError as error:
            assert "constrain the requested circumstances" in str(error), str(error)
        else:
            raise AssertionError("Optional or unspecified conditions admitted")
    named_type = deepcopy(goal)
    predicate = named_type["criteria"][0]["requirement"]["predicate"]
    predicate["value_schema"] = {"$defs": {"Endpoint": predicate["value_schema"]}, "$ref": "#/$defs/Endpoint"}
    sheet.put("fixture:named_flux_type", "discovery_goal", named_type)
    sheet.put("fixture:flux_goal", "discovery_goal", goal)
    check = goal["criteria"][0]["requirement"]["predicate"]
    observation = {"source_id": "Authored endpoint observation", "locator": "Authored endpoint",
        "kind": "measurement", "quantity": "stationary_product_flux_per_enzyme",
        "conditions": {"substrate_M": 0.00002}, "value": 42, "units": "s^-1", "uncertainty": "not reported"}
    # Same numerical fit as retained call7/516, explicitly excluded by the commissioning question.
    excluded = {**observation, "kind": "fit_parameter", "quantity": "w", "value": 730,
        "conditions": "single-molecule Michaelis-Menten equation (6) fit in Fig. 2b", "units": "s−1"}
    def matches(*values):
        return _native_matches(check, [], [(str(index), {"measurement": value}, sheet.anchor("fixture:evidence"))
                                         for index, value in enumerate(values)])[0]
    assert matches(observation)
    for change in ({"kind": "fit_parameter"}, {"kind": "simulation_output"}, {"kind": "model_prediction"},
                   {"kind": "unknown"}, {"quantity": "unrelated_numeric_quantity"},
                   {"conditions": {"substrate_M": 0.0001}}, {"units": "min^-1"}):
        assert not matches({**observation, **change}), change
    assert not matches(excluded)
    assert not matches({**observation, "quantity": "other"}, {**observation, "units": "other"})
    initial = _result(sheet)
    original = deepcopy(sheet.rows["fixture:evidence"]["data"])
    original["measurements"]["native"]["observations"] = {"single_rate_fit": excluded}
    sheet.put("fixture:fit_evidence", "evidence", original)
    source = SimpleNamespace(ledger=Ledger(_result(sheet)), commission=SimpleNamespace(discovery_goal="fixture:flux_goal"))
    def submit(key, selections):
        result = submit_finding(source, {"action": "submit_finding", "key": key,
            "answer": {"account": "The endpoints remain unresolved; a saturation fit is not an endpoint observation."},
            "source_claims": [], "source_observations": selections, "scientific_links": [], "rationale": "Retained failure check"})
        sheet.merge(result.causal_sheet_updates)
        source.ledger.merge(result)
        assert evaluate_finding(sheet, key)["results"][0]["status"] == "native_no_match"
    submit("fixture:excluded_fit_return", [{"evidence": sheet.anchor("fixture:fit_evidence"), "observation": "single_rate_fit"}])
    submit("fixture:unresolved_return", [])
    assert not discovery_credit(_result(sheet), initial, "fixture:flux_goal", "fixture").qualities
    assert compile_discovery_points(sheet)["total"] == "0"
    assert not source_credit(_result(sheet), initial, "fixture:flux_goal", "fixture",
        selected_sources={original["source"]}).qualities
    positive = deepcopy(original)
    positive["measurements"]["native"]["observations"] = {"endpoint": observation}
    sheet.put("fixture:endpoint", "evidence", positive)
    assert source_credit(_result(sheet), initial, "fixture:flux_goal", "fixture",
        selected_sources={original["source"]}).qualities
    return {"actual_loose_goal_rejected": True, "excluded_fit_credit": 0, "unresolved_return_credit": 0,
            "same_observation_required": True, "measured_endpoint_matches": True,
            "scope": "Typed native correspondence; scientific and source-fidelity judgments remain separate."}


def search_request_relative_credit():
    """Inherited evidence cannot pre-fill this Search's information slots."""
    from dataclasses import replace
    from episodic_workflow.case_reports.research.acquisition import SearchSource
    from episodic_workflow.case_reports.research.documents import Document, Span
    from episodic_workflow.case_reports.research.episode_contracts import SearchCommission
    sheet = fixture()  # Unrelated source claims already exist before the new goals.
    def put_source(target, key, url, *, selectors=("first",), observation=False):
        text = ("The source states that x increases y. A separate source claim states that z increases y. "
                "The observed y was 2 under condition A.")
        document = Document(url, "Authored Search-return fixture", text, 12000)
        passages = {"page": Span("page", 0, len(text), text)}
        native_claims = {
            "first": {"kind": "causal_relation", "statement_id": "", "locator": "First authored source assertion",
                "left": "x", "relation": "increases", "right": "y", "conditions": "Authored fixture", "symbols": {}},
            "second": {"kind": "causal_relation", "statement_id": "", "locator": "Second authored source assertion",
                "left": "z", "relation": "increases", "right": "y", "conditions": "Authored fixture", "symbols": {}}}
        bindings = {
            "first": claim_binding(document, passages, "x increases y", ["x increases y"]),
            "second": claim_binding(document, passages, "z increases y", ["z increases y"])}
        observations = {"reported": {"source_id": "", "kind": "measurement", "locator": "Third authored source assertion",
            "quantity": "y", "conditions": "condition A", "value": 2, "units": "authored", "uncertainty": "exact"}} if observation else {}
        sidecar = stored_sidecar(document, passages,
            claims={selector: bindings[selector] for selector in selectors},
            observations={"reported": observation_binding(document, passages,
                "The observed y was 2", "observed y", "2", 1, "under condition A")} if observation else {})
        return target.put(key, "evidence", {"source": url, "locator": f"characters 0:{len(text)}",
            "study_id": "Authored Search source: " + url, "finding": "Authored source assertion returned by Page.",
            "measurements": {"quote": text, "source_version": document.sha256, "source_anchors": sidecar,
                "native": {"claims": {selector: native_claims[selector] for selector in selectors}, "observations": observations}}})
    old_url = "https://example.invalid/search/already-seen"
    new_url = "https://example.invalid/search/newly-selected"
    unselected_url = "https://example.invalid/search/not-selected"
    put_source(sheet, "fixture:seen_source", old_url)
    checks = {
        "causal_dependence": {"kind": "causal_dependence", "inference": sheet.anchor("fixture:G")},
        "formal_equivalence": {"kind": "formal_equivalence", "required_inputs": [sheet.anchor("fixture:G")],
            "request_schema": {"type": "object", "required": ["relation"], "properties": {"relation": {"const": "equal"}}}},
        "distinct_theory": {"kind": "distinct_theory", "excluded_theories": [sheet.anchor("fixture:theory")]},
        "wrong_claim": _source_goal("causal_relation", "decreases"),
        "wrong_value": _observation_predicate({"const": "y"}, {"const": "condition A"}, {"const": "authored"}, value={"const": 999}),
        "matching_value": _observation_predicate({"const": "y"}, {"const": "condition A"}, {"const": "authored"}, value={"const": 2})}
    for label, check in checks.items():
        _admit_goal(sheet, "fixture:search_goal:" + label, check)
    run, _, _, _, _ = _run(sheet)
    initial = _result(sheet)
    def search(label):
        source = SearchSource(run, initial, query="authored source return")
        source.commission = SearchCommission("Find the commissioned source information", (),
            discovery_goal="fixture:search_goal:" + label)
        return source
    returned = CausalSheet(sheet.history)
    put_source(returned, "fixture:unselected_page", unselected_url, observation=True)
    before_selected_return = _result(returned)
    selected_record = put_source(returned, "fixture:selected_page", new_url, observation=True)
    selected_return = _result(returned)
    duplicate = CausalSheet(returned.history)
    put_source(duplicate, "fixture:renamed_return_same_information", new_url, observation=True)
    extra = CausalSheet(duplicate.history)
    put_source(extra, "fixture:additional_same_criterion_information", new_url, selectors=("second",))
    semantic_results = {}
    for label in ("causal_dependence", "formal_equivalence", "distinct_theory"):
        source = search(label)
        assert not source.credit(initial).qualities
        source.selected_sources.add(new_url)
        assert not source.credit(initial).qualities
        assert not source.credit(before_selected_return).qualities, "Unselected Page evidence cannot fill this Search's slot"
        first = source.credit(selected_return)
        assert sum(value for _, _, value in first.qualities) == 1
        assert source.credit(_result(duplicate)).qualities == first.qualities
        assert source.credit(_result(extra)).qualities == first.qualities
        controller = ResearchController(source.credit_columns, initial=source.credit(initial), point_scores=True)
        assert controller.observe("first_selected_page", first, is_root=False).gain == 1
        assert controller.observe("duplicate_selected_page", source.credit(_result(duplicate)), is_root=False).gain == 0
        assert controller.observe("extra_same_criterion_page", source.credit(_result(extra)), is_root=False).gain == 0
        assert not source.credit(replace(initial, evaluated=False)).qualities
        assert source.credit(replace(initial, evaluated=False)).evaluated is False
        seen = search(label)
        seen.selected_sources.add(old_url)
        assert not seen.credit(initial).qualities, "Selecting an already-known source adds no information"
        new_at_seen_url = CausalSheet(sheet.history)
        put_source(new_at_seen_url, "fixture:new_information_same_source", old_url, selectors=("second",))
        assert sum(value for _, _, value in seen.credit(_result(new_at_seen_url)).qualities) == 1
        source.ledger.merge(selected_return)
        proposal = {"action": "submit_finding", "key": "fixture:pending_search:" + label,
            "answer": {"account": "The selected source is a candidate; scientific correspondence remains unresolved."},
            "source_claims": [{"evidence": {"key": selected_record["key"], "version": selected_record["version"]}, "claim": "first"}],
            "source_observations": [], "scientific_links": [], "rationale": "Return the exact source assertion for Scientific"}
        pending = submit_finding(source, proposal)
        pending_sheet = CausalSheet(returned.history)
        pending_sheet.merge(pending.causal_sheet_updates)
        assert evaluate_finding(pending_sheet, proposal["key"])["results"][0]["status"] == "needs_scientific_assessment"
        assert not discovery_credit(_result(pending_sheet), initial, source.commission.discovery_goal, "fulfillment").qualities
        assert compile_discovery_points(pending_sheet)["total"] == "0"
        if label == "causal_dependence":
            _protocol(pending_sheet, depends_on=(proposal["key"],))
            pending_sheet.merge(evaluate_cases(pending_sheet, "fixture:protocol", ["positive"]))
            receipts = recheck_findings(pending_sheet)
            assert receipts
            assert evaluate_finding(pending_sheet, proposal["key"])["results"][0]["status"] == "fulfilled"
            assert sum(value for _, _, value in discovery_credit(_result(pending_sheet), initial,
                source.commission.discovery_goal, "fulfillment").qualities) == 1
            assert compile_discovery_points(pending_sheet)["total"] == "1"
        semantic_results[label] = {"selected_new_information": 1, "duplicate_extra_information_gain": 0,
            "unrelated_unselected_information": 0, "known_source_new_operand": 1, "pending_scientific_assessment": True}
    for label, expected in (("wrong_claim", 0), ("wrong_value", 0), ("matching_value", 1)):
        source = search(label)
        source.selected_sources.add(new_url)
        assert sum(value for _, _, value in source.credit(selected_return).qualities) == expected
    return {"semantic_criteria": semantic_results, "source_type_mismatches_rejected": True,
            "empty_invalid_outcomes_gain_zero": True, "discovery_waits_for_executed_scientific_result": True}


def _protocol(sheet, *, depends_on=()):
    sheet.put("fixture:application", "application", {"inference": "fixture:G", "inference_version": "",
        "theory": "fixture:theory", "context_key": "fixture:finite",
        "context": "Declared positive-a model", "context_distinction": "One verification context",
        "direction": "unresolved", "inference_use": "y depends on x through a*x",
        "consequence_without_inference": "Removing the x-to-y contribution changes y"}, dependencies=depends_on)
    return sheet.put("fixture:protocol", "importance", {"application": "fixture:application", "method": "scm",
        "question": "Does the declared contribution affect y?", "contrast": "y=a*x versus y=0",
        "domain": {"kind": "finite", "description": "One mathematical circumstance",
            "measure": "Unit counting measure", "cases": [{"id": "positive", "inputs": {"x": 3, "a": 2}, "weight": 1}]},
        "model": {"equations": {"y": "a*x"}, "focal": "y", "replacement": "0", "answer": "y",
            "initial": {}, "steps": 0, "edge": ["x", "y"]}, "comparator": "exact_typed_answer"})


def scientific_feedback_receipts():
    sheet = fixture()
    _admit_goal(sheet, "fixture:dependence_goal", {"kind": "causal_dependence", "inference": sheet.anchor("fixture:G")})
    _submit(sheet, "fixture:dependence_goal", "fixture:pending_finding", "positive_relation")
    before = evaluate_finding(sheet, "fixture:pending_finding")
    assert before["results"][0]["status"] == "needs_scientific_assessment"
    assert compile_discovery_points(sheet)["total"] == "0"
    _protocol(sheet, depends_on=("fixture:pending_finding",))
    sheet.merge(evaluate_cases(sheet, "fixture:protocol", ["positive"]))
    assert not any(key.startswith("progress:first_examination:") for key in compile_progress(sheet)["points"])
    produced = recheck_findings(sheet)
    assert produced
    after = evaluate_finding(sheet, "fixture:pending_finding")
    assert after["results"][0]["status"] == "fulfilled"
    assert compile_discovery_points(sheet)["total"] == "1"
    assert recheck_findings(sheet) == ()
    return {"before": before, "after": after, "new_host_receipts": len(produced), "credit": "1"}


def source_grounded_examination():
    sheet = fixture()
    _protocol(sheet)
    run, root, episode, _, _ = _run(sheet)
    root_view, _ = _view(run, episode)
    source = ScientificSource(run, _result(sheet))
    commission = ScientificCommission("Examine the declared model", tuple(sheet.rows), ())
    child = run.child(root_view, source, "scientific", commission.question,
        contract=commission.report_contract(), proposal={})
    _view(run, child, root_view.path)
    receipt = execute_tool(source, {"action": "scientific_tool", "name": "causal_calculation",
        "question": "Check the declared intervention", "inputs": [sheet.anchor("fixture:protocol")],
        "arguments": {"assessment": "fixture:protocol", "case_ids": ["positive"], "evaluation_context": ""},
        "rationale": "A completed tool call is not automatically a scientific examination"})
    tool_record = next(row for row in receipt.causal_sheet_updates if row["kind"] == "tool_receipt")
    assert tool_record["data"]["status"] == "computed"
    sheet.merge(receipt.causal_sheet_updates)
    assert not any(key.startswith("progress:first_examination:") for key in compile_progress(sheet)["points"])
    sheet.put("fixture:argument", "reasoning", {"author": "organization-verifier",
        "premises": ["fixture:G", "fixture:theory", "fixture:evidence"],
        "steps": ["The authored source explicitly holds a fixed and states that increasing x increases y.",
                  "The declared causal account represents that contribution as y=a*x; transfer beyond fixed a remains conditional."],
        "conclusion": "The source supports examining this bounded G in the declared account; its applicability outside the fixed-a condition remains unresolved.",
        "limitations": "This is an authored mathematical fixture, not empirical corroboration."})
    data = {"inference": sheet.anchor("fixture:G"), "theories": [sheet.anchor("fixture:theory")],
        "question": "What does the supplied source establish about this bounded G?", "argument": "fixture:argument",
        "grounds": ["fixture:evidence"]}
    sheet.put("fixture:examination", "examination", data)
    first = [key for key in compile_progress(sheet)["points"] if key.startswith("progress:first_examination:")]
    assert len(first) == 1
    sheet.put("fixture:repeated_examination", "examination", data)
    assert [key for key in compile_progress(sheet)["points"] if key.startswith("progress:first_examination:")] == first
    sheet.put("fixture:source_free_theory", "theory", {"name": "Unsourced placeholder",
        "mechanism": "Unresolved", "predictions": ["Unresolved"], "admission": "candidate", "same_as": "",
        "distinction": "Negative validation fixture", "compared_with": []})
    sheet.put("fixture:source_free_argument", "reasoning", {"author": "organization-verifier",
        "premises": ["fixture:G", "fixture:source_free_theory"], "steps": ["A source-free assertion is only a proposed task."],
        "conclusion": "The account remains source-free.", "limitations": "No source evidence is supplied."})
    try:
        sheet.put("fixture:source_free_examination", "examination", {**data,
            "theories": [sheet.anchor("fixture:source_free_theory")], "argument": "fixture:source_free_argument",
            "grounds": ["fixture:source_free_theory"]})
    except ValueError as error:
        rejected = str(error)
    else:
        raise AssertionError("A source-free argument received first-examination registration")
    return {"raw_tool_examination_credit": "0", "source_grounded_first_credit": "1",
            "repeated_credit": "0", "source_free_rejected": rejected}


def another_theory_using_G():
    """Check the pending→Scientific feedback protocol, not general theory identity."""
    from types import SimpleNamespace
    from episodic_workflow.case_reports.research.documents import Document, Span
    cases = {}
    for case, admission, equation in (
            ("different_declared_account", "distinct", "a*x*x"),
            ("same_theory_alias_and_context", "same_theory", "a*x")):
        sheet = fixture()
        shared = sheet.put("fixture:shared_G", "inference", {
            "statement": "At fixed positive a, increasing nonnegative x increases y in the declared account.",
            "variables": ["x: nonnegative input", "y: output", "a: positive fixed parameter"],
            "intervention": "Change x while keeping a fixed", "outcome": "y",
            "grain": "Declared mathematical accounts", "domain": "Positive a and nonnegative x"})
        text = ("The second declared account is y=a*x*x for nonnegative x and positive a. "
                "It shares the positive x-to-y contribution but differs from the linear account y=a*x."
                if admission == "distinct" else
                "This is the same declared linear account y=a*x, expressed under an alias in another context. "
                "It retains the same positive x-to-y contribution for positive a.")
        candidate_source = "https://example.invalid/verification/" + case
        candidate_document = Document(candidate_source, "Authored candidate", text, 12000)
        candidate_passages = {"candidate": Span("candidate", 0, len(text), text)}
        candidate_sidecar = stored_sidecar(candidate_document, candidate_passages, claims={
            "candidate_relation": claim_binding(candidate_document, candidate_passages,
                "positive x-to-y contribution", ["positive x-to-y contribution"])})
        evidence = sheet.put("fixture:candidate_evidence", "evidence", {
            "source": candidate_source,
            "locator": f"characters 0:{len(text)}", "study_id": "authored-fixture:" + case,
            "finding": "Authored candidate-account text for testing the scientific feedback protocol.",
            "measurements": {"quote": text, "source_version": sha256(text.encode()).hexdigest(),
                "source_anchors": candidate_sidecar,
                "native": {"observations": {}, "claims": {"candidate_relation": {"kind": "causal_relation",
                    "statement_id": "", "locator": "The complete authored candidate-account passage",
                    "left": "x", "relation": "increases", "right": "y",
                    "conditions": "Nonnegative x and positive a held fixed; y=" + equation, "symbols": {}}}}}})
        goal = {"question": "Find another theoretical account that meaningfully depends on the specified bounded G",
            "inputs": [sheet.anchor("fixture:shared_G"), sheet.anchor("fixture:theory")],
            "answer_schema": {"type": "object", "additionalProperties": False,
                "required": ["candidate"], "properties": {"candidate": {"type": "string", "minLength": 1}}},
            "criteria": [
                {"id": "another_theory", "requirement": {"kind": "scientific",
                 "question": "Does the supplied account differ from the known theory?", "candidate": None,
                 "predicate": {"kind": "distinct_theory", "excluded_theories": [sheet.anchor("fixture:theory")]}}},
                {"id": "uses_G", "requirement": {"kind": "scientific",
                 "question": "Does a represented prediction depend on the specified G?", "candidate": None,
                 "predicate": {"kind": "causal_dependence", "inference": sheet.anchor("fixture:shared_G")}}}]}
        sheet.put("fixture:another_theory_goal", "discovery_goal", goal)
        discovery = SimpleNamespace(ledger=Ledger(_result(sheet)),
            commission=SimpleNamespace(discovery_goal="fixture:another_theory_goal"))
        found = submit_finding(discovery, {"action": "submit_finding", "key": "fixture:candidate_finding",
            "answer": {"candidate": case}, "source_claims": [{"evidence": {
                "key": evidence["key"], "version": evidence["version"]}, "claim": "candidate_relation"}],
            "source_observations": [], "scientific_links": [], "rationale": "Return a candidate without inventing a scientific verdict"})
        sheet.merge(found.causal_sheet_updates)
        before = evaluate_finding(sheet, "fixture:candidate_finding")
        assert {item["status"] for item in before["results"]} == {"needs_scientific_assessment"}
        assert compile_discovery_points(sheet, "fixture:another_theory_goal")["total"] == "0"

        # These are declared test models, not LLM-certified discoveries. The
        # production verifier consumes their actual linked scientific records
        # and executed receipts, rather than a supplied pass/fail decision.
        candidate = sheet.put("fixture:candidate_theory", "theory", {
            "name": "Declared quadratic account" if admission == "distinct" else "Alias of declared linear account",
            "mechanism": "y=" + equation, "predictions": ["y=" + equation],
            "admission": admission, "same_as": "" if admission == "distinct" else "fixture:theory",
            "distinction": "Different constitutive equation" if admission == "distinct" else "A renamed parameter context of the same equation",
            "compared_with": ["fixture:theory"]}, dependencies=(evidence["key"], "fixture:candidate_finding"))
        protocols, predictions = [], {}
        for label, theory_key, law in (("original", "fixture:theory", "a*x"),
                                       ("candidate", candidate["key"], equation)):
            application = "fixture:" + label + "_G_application"
            sheet.put(application, "application", {"inference": shared["key"], "inference_version": shared["version"],
                "theory": theory_key, "context_key": "fixture:" + label + "_account_context",
                "context": "Positive fixed a with nonnegative x", "context_distinction": "Declared " + label + " account",
                "direction": "unresolved", "inference_use": "The represented x-to-y contribution in y=" + law,
                "consequence_without_inference": "Removing the x-to-y contribution changes y"},
                dependencies=("fixture:candidate_finding",) if label == "candidate" else ())
            protocol = "fixture:" + label + "_G_protocol"
            sheet.put(protocol, "importance", {"application": application, "method": "scm",
                "question": "Does the bounded x-to-y contribution affect the represented outcome?",
                "contrast": "y=" + law + " versus removing the focal x-to-y contribution",
                "domain": {"kind": "finite", "description": "One explicitly declared mathematical circumstance",
                    "measure": "Unit counting weight", "cases": [{"id": "witness", "inputs": {"x": 3, "a": 2}, "weight": 1}]},
                "model": {"equations": {"y": law}, "focal": "y", "replacement": "0", "answer": "y",
                    "initial": {}, "steps": 0, "edge": ["x", "y"]}, "comparator": "exact_typed_answer"})
            executed = evaluate_cases(sheet, protocol, ["witness"])
            sheet.merge(executed)
            receipt = executed[0]["data"]
            assert not receipt["error"] and receipt["altered"] == {"type": "rational", "value": "0"}
            predictions[label] = receipt["baseline"]
            protocols.append(protocol)
        assert predictions["original"] == {"type": "rational", "value": "6"}
        assert predictions["candidate"] == {"type": "rational", "value": "18" if admission == "distinct" else "6"}
        sheet.put("fixture:account_comparison", "comparison", {
            "left": "fixture:original_G_application", "right": "fixture:candidate_G_application",
            "mapping": "The same positive-input x, outcome y and fixed positive a instantiate the declared G in both accounts.",
            "conflict": "The quadratic account predicts 18 versus 6 at x=3,a=2; an alias predicts the same 6." if admission == "distinct" else "The account is a same_theory alias, so no distinct theory is established.",
            "repair_search_bounds": "Two authored mathematical fixture accounts only; no general scientific distinctness verdict."},
            dependencies=("fixture:candidate_finding", *protocols))
        receipts = recheck_findings(sheet, ["fixture:candidate_finding"])
        after = evaluate_finding(sheet, "fixture:candidate_finding")
        statuses = {item["criterion"]: item["status"] for item in after["results"]}
        assert statuses["uses_G"] == "fulfilled"
        if admission == "distinct":
            assert statuses["another_theory"] == "fulfilled"
            assert compile_discovery_points(sheet, "fixture:another_theory_goal")["total"] == "2"
        else:
            assert statuses["another_theory"] == "needs_scientific_assessment"
            assert compile_discovery_points(sheet, "fixture:another_theory_goal")["total"] == "1"
            assert sheet.canonical_theory(candidate["key"]) == sheet.canonical_theory("fixture:theory")
        assert receipts and recheck_findings(sheet, ["fixture:candidate_finding"]) == ()
        cases[case] = {"before": before, "after": after,
            "credit": compile_discovery_points(sheet, "fixture:another_theory_goal")["total"],
            "host_receipts": len(receipts), "executed_predictions": predictions}
    return {"cases": cases,
        "qualification": "This verifies pending-to-assessed communication and executable dependence/identity handling for explicitly declared fixture accounts. It does not prove general scientific theory distinctness or autonomous interpretation of arbitrary literature."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    if not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("Verification output belongs under this checkout's runs/")
    directory.mkdir(parents=True, exist_ok=True)
    checks = {}
    for name, check in (
            ("hierarchy_and_controllers", hierarchy_and_controllers),
            ("scientific_tool_decision_contracts", scientific_tool_decision_contracts),
            ("scientific_discovery_communication", scientific_discovery_communication),
            ("source_criteria_and_dedup", source_criteria_and_dedup),
            ("scientific_structure_requirements", scientific_structure_requirements),
            ("native_operand_type_contracts", native_operand_type_contracts),
            ("production_reader_request_contract", production_reader_request_contract),
            ("observation_identity_and_repair", observation_identity_and_repair),
            ("source_occurrence_replay", source_occurrence_replay),
            ("source_result_condition_drift_replay", source_result_condition_drift_replay),
            ("source_result_literal_and_repair", source_result_literal_and_repair),
            ("source_anchor_boundaries_and_repair", source_anchor_boundaries_and_repair),
            ("search_request_relative_credit", search_request_relative_credit),
            ("observation_request_credit", observation_request_credit),
            ("scientific_feedback_receipts", scientific_feedback_receipts),
            ("source_grounded_examination", source_grounded_examination),
            ("another_theory_using_G", another_theory_using_G)):
        try:
            checks[name] = {"passed": True, "details": check()}
        except Exception as error:
            checks[name] = {"passed": False, "error": f"{type(error).__name__}: {error}",
                            "traceback": traceback.format_exc()}
    report = {"passed": all(check["passed"] for check in checks.values()),
        "scope": "Focused production schema, routing, source, verification and credit fixtures; scripted choices are not autonomy evidence.",
        "checks": checks}
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
