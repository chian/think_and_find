"""Check inference contracts against saved real evidence, without running a workflow."""

import argparse
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "episodic_workflow", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    from episodic_workflow.case_reports.research.assumptions import (
        assess_assumption, assumption_candidate, unassessed_assumption)
    from episodic_workflow.case_reports.research.engine import ResearchSource
    from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, identity
    from episodic_workflow.case_reports.research.tools import TOOLS, input_options
    from episodic_workflow.case_reports.research.workflow import InvestigationSource

    report = json.loads(args.report.read_text())
    attempt = next(row for row in reversed(report["model_attempts"])
                   if row["operation"] == "assess_assumption_reasonableness"
                   and row.get("response", {}).get("verdict") in {"supported", "provisional"}
                   and row["request"]["candidate"].get("value") is not None)
    request, response = attempt["request"], attempt["response"]
    records = list(request["evidence"].values())
    facts = tuple(Fact(**{**row, "parents": tuple(row.get("parents", ()))})
                  for row in records if row.get("kind") in {"source_observation", "source_model_parameter", "request_constraint"})
    qualitative = tuple(row for row in report["qualitative_evidence"] if row.get("kind") == "request_constraint") + tuple(
        row for row in records if row.get("kind") == "source_evidence")
    scenarios = tuple(row["definition"] for row in report["scenarios"].values())
    ledger = Ledger(ResearchResult(facts=facts, evidence=qualitative, scenarios=scenarios))
    assert {key: ledger.evidence_options()[key] for key in request["evidence"]} == request["evidence"], "Recorded evidence order was not reconstructed exactly"
    before = ledger.result().as_record()
    source_refs = list(dict.fromkeys([
        *response["supporting_evidence"],
        *(ref for criterion in response["criteria"].values() for ref in criterion["evidence"]),
    ]))
    numeric_ref = next(key for key, fact in input_options(ledger).items()
                       if identity(fact.semantic_record()) in {identity(request["evidence"][ref]) for ref in source_refs})
    candidate = {**request["candidate"], "entry_type": "best_guess", "source_refs": source_refs,
                 "input_refs": [numeric_ref], "added_assumptions": []}
    normalized = assumption_candidate(candidate, ledger)
    checks = []

    def passed(name):
        checks.append(name)

    def rejects(action, state=ledger):
        try:
            assumption_candidate(action, state)
        except ValueError:
            return
        raise AssertionError("Invalid candidate was accepted")

    assert normalized["entry_type"] == "best_guess"
    assert normalized["derivation"]["source_evidence_ids"] == [identity(request["evidence"][ref]) for ref in source_refs]
    assert normalized["derivation"]["input_fact_keys"] == [input_options(ledger)[numeric_ref].key]
    passed("typed candidate preserves exact source identities and numeric lineage")

    shifted = Ledger(ledger.result())
    extra = next(Fact(**{**row, "parents": tuple(row.get("parents", ()))})
                 for row in report["evidence_and_assumptions"]
                 if row["kind"] in {"source_observation", "source_model_parameter"}
                 and Fact(**{**row, "parents": tuple(row.get("parents", ())) }).key not in ledger.facts)
    old_qualitative = next(ref for ref in source_refs if request["evidence"][ref].get("kind") == "source_evidence")
    shifted.merge(ResearchResult(facts=(extra,)))
    assert identity(shifted.evidence_options()[old_qualitative]) != identity(request["evidence"][old_qualitative])
    assert assumption_candidate(normalized, shifted) == normalized
    passed("stable references survive a real additional source fact shifting evidence indices")

    rejects({**candidate, "source_refs": ["absent:evidence"]})
    rejects({**candidate, "input_refs": ["absent:input"]})
    rejects({**candidate, "value": float("nan")})
    rejects({**candidate, "unit": "not-a-canonical-unit"})
    rejects({**candidate, "subject": "not-a-reviewed-case"})
    fixed_field, fixed = next(iter(ledger.scenarios[candidate["subject"]]["fixed_parameters"].items()))
    rejects({**candidate, "field": fixed_field, "value": fixed["value"], "unit": fixed["unit"]})
    passed("invented references, nonfinite values, wrong units, unknown scope and fixed-input replacement reject")

    empty = {**candidate, "source_refs": [], "input_refs": []}
    unassessed = unassessed_assumption(empty, ledger)
    guess_ledger = Ledger(ledger.result())
    guess_ledger.merge(ResearchResult(assumptions=(unassessed,)))
    assert not unassessed["review"]["completed"] and unassessed["reasonableness"] == 0
    assert any(f.parents == (unassessed["key"],) for f in input_options(guess_ledger).values())
    self_ref = next(key for key, fact in input_options(guess_ledger).items() if fact.parents == (unassessed["key"],))
    rejects({**candidate, "input_refs": [self_ref]}, guess_ledger)
    passed("unsupported guesses remain numeric inputs without fabricated support; self-dependency rejects")

    alternate_value = input_options(ledger)[numeric_ref].value
    assert alternate_value != candidate["value"], "This recorded case does not exercise indirect numeric inference"
    dependent = assumption_candidate({**candidate, "value": alternate_value, "input_refs": [self_ref]}, guess_ledger)
    assert dependent["derivation"]["input_assumption_versions"] == {unassessed["key"]: unassessed["version"]}
    guess_ledger.assumptions[unassessed["key"]] = {**unassessed, "version": identity([unassessed["version"], "review-only"])}
    assert assumption_candidate(dependent, guess_ledger) == dependent
    guess_ledger.assumptions[unassessed["key"]] = {**unassessed, "statement": unassessed["statement"] + " Changed premise."}
    rejects(dependent, guess_ledger)
    passed("input claim changes reject stale lineage while evidence-only review versions preserve it")

    reviewed_requests = []
    def recorded_review(new_request):
        assert {key: new_request["candidate"][key] for key in request["candidate"]} == request["candidate"]
        assert new_request["evidence"] == request["evidence"]
        assert new_request["proposed_source_refs"] == [ref for ref in request["evidence"] if ref in source_refs]
        reviewed_requests.append(new_request)
        return copy.deepcopy(response)

    reviewed = assess_assumption(recorded_review,
        {**normalized, "evidence": list(request["evidence"])}, ledger, request["context"])
    assert len(reviewed_requests) == 1
    assert reviewed.assumptions[0]["derivation"] == normalized["derivation"]
    assert all(f.kind == "model_assumption" for f in reviewed.facts)
    assert ledger.result().as_record() == before
    passed("unchanged recorded review admits separate inferred record and leaves all source facts/spans unchanged")

    # Construct only the schema owner. There is no ResearchRun or provider instance.
    owner = object.__new__(InvestigationSource)
    owner.ledger, owner.focus = ledger, ()
    owner.run = SimpleNamespace(fixed_scenarios=True, tools={k: v for k, v in TOOLS.items() if k != "compare_predictions_with_observations"})
    owner.requested_subjects = (candidate["subject"],)
    owner.parameter_requirements = frozenset((candidate["field"],))
    available_evidence, available_numeric = owner.inference_options()
    assert available_evidence == ledger.evidence_options()
    assert available_numeric == input_options(ledger)
    passed("inference decision retains inherited and cross-field native evidence and numeric inputs")
    schema = owner.proposal_schema()
    assert "action" not in schema.get("required", [])
    assert any(row.get("required") == ["candidates"] and row["properties"]["candidates"]["minItems"] == 1 for row in schema["anyOf"])
    for row in schema["allOf"]:
        if "action" in row["if"].get("properties", {}):
            assert "action" in row["if"].get("required", [])
    passed("candidate-only schema permits nonempty inference without action; action conditions require action presence")

    propose = ast.parse(textwrap.dedent(inspect.getsource(InvestigationSource.propose)))
    branch = next(node for node in ast.walk(propose) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "action is None"
                  and any(isinstance(child, ast.Return) for child in node.body))
    assert len(branch.body) == 1 and "self.leaf" in ast.unparse(branch.body[0])
    child_line = min(node.lineno for node in ast.walk(propose) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "StrategySource")
    assert branch.lineno < child_line
    assert "self.project" in inspect.getsource(ResearchSource.leaf)
    assert "self.assess(result)" in inspect.getsource(ResearchSource.project)
    assert "relevant = options" in inspect.getsource(InvestigationSource.assess)
    passed("inference-only branch returns before any strategy child; leaf projects through mandatory review using inherited evidence")
    print(json.dumps({"report": str(args.report), "recorded_assessment_call": attempt["call"],
        "checks_passed": len(checks), "checks": checks,
        "scope": "Pure contracts plus replay of one unchanged recorded assessment. No workflow/provider run; no new scientific judgment or live behavior validation."}, indent=2))


if __name__ == "__main__":
    main()
