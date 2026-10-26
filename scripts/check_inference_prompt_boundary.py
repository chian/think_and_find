"""Inspect next decision payloads against a real run's child identity boundary.

No provider is constructed: payload capture aborts before an agent response,
episode action, or scientific judgment can occur.
"""

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "episodic_workflow", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    from episodic_workflow.case_reports.research.records import (
        Fact, Ledger, ResearchResult, assumption_model_view)
    from episodic_workflow.case_reports.research.goal_state import ScientificGoalState
    from episodic_workflow.case_reports.research.engine import ResearchRun
    from episodic_workflow.case_reports.research.tools import TOOLS, input_options
    from episodic_workflow.case_reports.research.workflow import GuidelineSource, InvestigationSource
    from episodic_workflow.method_loop.communication import ReportContract
    from episodic_workflow.method_loop.episode import EpisodeGoal, EpisodeView
    from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS, UNITS, CONDITIONS

    audit = json.loads(args.audit.read_text())
    investigation_unit = next(unit for unit in audit["units"]
                              if unit.get("child") and unit["child"]["scope_level"] == "investigate")
    strategy_unit = next(unit for unit in investigation_unit["child"]["units"] if unit.get("child"))

    def result(record):
        return ResearchResult(**{**record, "facts": tuple(
            Fact(**{**fact, "parents": tuple(fact.get("parents", ()))}) for fact in record["facts"])})

    state = result(investigation_unit["child_result"])
    ledger = Ledger(state)
    before = copy.deepcopy(ledger.result().as_record())
    best_guesses = [row for row in state.assumptions if row.get("entry_type") == "best_guess"]
    assert any(row["derivation"]["input_fact_keys"] for row in best_guesses)
    candidates = [{key: row[key] for key in (
        "subject", "field", "value", "unit", "statement", "basis", "entry_type", "derivation")}
        for row in best_guesses]

    def protected_view(unit):
        child = unit["child"]
        identifiers = (child["episode_id"], child["goal"]["goal_id"],
                       *result(unit["child_result"]).audit_identifiers())
        contract = ReportContract.from_record(child["request"]["report_contract"])
        report = contract.admit(unit["episode_update"]["report"], audit_identifiers=identifiers)
        view = object.__new__(EpisodeView)
        object.__setattr__(view, "updates", (SimpleNamespace(report=report),))
        return view, report

    views = {"investigation": protected_view(strategy_unit), "root": protected_view(investigation_unit)}
    for view, _ in views.values():
        try:
            view.model_inputs({"candidates": candidates}, reports=view.child_reports)
        except ValueError as exc:
            assert str(exc) == "parent report contains a child audit identity"
        else:
            raise AssertionError("Recorded unprojected provenance did not reproduce the protected-identity error")

    class CapturedPayload(Exception):
        pass

    captured = {}
    def capture(name):
        def stop(request):
            captured[name] = request
            raise CapturedPayload
        return stop

    shared = ScientificGoalState(DEFAULT_METRICS, state)
    score_before = shared.preview()["weighted_score"]
    for name, source_type in (("investigation", InvestigationSource), ("root", GuidelineSource)):
        node = investigation_unit["child"] if name == "investigation" else audit
        run = SimpleNamespace(fixed_scenarios=True, metrics=DEFAULT_METRICS,
            initial=state, global_goal=shared, context=node["goal"]["task_context"],
            tools={key: value for key, value in TOOLS.items() if key != "compare_predictions_with_observations"},
            agent=capture(name), diagnostic_sequence=0)
        run.identify_diagnostic = MethodType(ResearchRun.identify_diagnostic, run)
        source = source_type(run, state) if name == "investigation" else source_type(run)
        source.focus = tuple(sorted(set(UNITS) | set(CONDITIONS)))
        source.scientific = True
        source.goal = EpisodeGoal.from_record(node["goal"])
        source.parent_contract = (ReportContract.from_record(node["request"]["report_contract"])
                                  if node["request"]["report_contract"] is not None else None)
        source.requested_subjects = tuple(node["request"]["message"]["subjects"])
        source.candidates = {InvestigationSource.candidate_key(row): row for row in candidates}
        source.communicated = {"findings": set(), "issues": set(), "diagnostics": set()}
        view = views[name][0]
        state_call = ast.parse(node["controller_state"], mode="eval").body
        assert isinstance(state_call, ast.Call) and isinstance(state_call.func, ast.Name)
        assert state_call.func.id == "ResearchState"
        controller = {item.arg: ast.literal_eval(item.value) for item in state_call.keywords}
        completed = 2 if name == "investigation" else 1
        object.__setattr__(view, "request", SimpleNamespace(goal=source.goal))
        object.__setattr__(view, "units_consumed", completed)
        object.__setattr__(view, "controller_state", SimpleNamespace(
            history=controller["history"][:completed + 1], reason="continue"))
        try:
            source.propose(view)
        except CapturedPayload:
            pass
        else:
            raise AssertionError("Decision inspection did not stop at the agent boundary")
        assert name in captured
        views[name][1].validate_model_inputs(captured[name])
        assert {"context", "global_scientific_goal", "attempt_diagnostics", "numeric_yield"} <= set(captured[name])
        assert captured[name]["global_scientific_goal"]["weighted_score"] == score_before

    numeric = input_options(ledger)
    evidence = ledger.evidence_options()
    for candidate in candidates:
        projected = assumption_model_view(candidate, evidence_options=evidence, numeric_options=numeric)
        if not candidate["derivation"]["input_fact_keys"] and not candidate["derivation"]["source_evidence_ids"]:
            assert projected == candidate, "Valid empty-lineage historical prompts must remain unchanged"
            continue
        assert projected["basis"] == candidate["basis"]
        assert projected["derivation"]["added_assumptions"] == candidate["derivation"]["added_assumptions"]
        assert not projected["derivation"]["unavailable_input_count"]
        for row in projected["derivation"]["inputs"]:
            assert {key: value for key, value in row.items() if key != "reference"} == numeric[row["reference"]].semantic_record()
        assert all(ref in evidence for ref in projected["derivation"]["source_refs"])
    for summary in captured["root"]["assumption_options"].values():
        derivation = summary.get("derivation", {})
        assert all(ref in summary["evidence"] for ref in derivation.get("source_refs", ()))
        for row in derivation.get("inputs", ()):
            assert row["reference"] in captured["root"]["calculation_inputs"]["numeric_options"]
            assert {key: value for key, value in row.items() if key != "reference"} == numeric[row["reference"]].semantic_record()
    assert ledger.result().as_record() == before, "Prompt projection changed persisted scientific records"
    assert shared.preview()["weighted_score"] == score_before, "Prompt projection changed scientific credit"
    print(json.dumps({"audit": str(args.audit), "checks": [
        "Actual recorded internal provenance reproduces the failure for both parent boundaries",
        "Full actual ResearchSource.inputs and investigation proposal payload pass EpisodeView.model_inputs",
        "Full actual ResearchSource.inputs and root proposal payload pass EpisodeView.model_inputs",
        "Empty-lineage historical candidate payloads remain identical",
        "Projected references resolve to unchanged native source and input semantics",
        "Stored source facts, best-guess lineage, review records and shared credit remain unchanged"],
        "scope": "Pure saved-state prompt/communication inspection; no providers, episode actions, or new scientific judgments."}, indent=2))


if __name__ == "__main__":
    main()
