"""Focused real-kernel/evaluator checks; execute through the one-workflow launcher."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_checkout():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("episodic_workflow", root / "__init__.py",
        submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)


_load_checkout()

from episodic_workflow.case_reports.research.causal_sheet import CausalSheet
from episodic_workflow.case_reports.research.records import Ledger, ResearchResult
from episodic_workflow.case_reports.research.scientific_tools import execute_tool, _lean_config


def fixture():
    sheet = CausalSheet()
    sheet.put("tool_check_G", "inference", {"statement": "Changing x can change y through y=a*x.",
        "variables": ["x", "y", "a"], "intervention": "Change x while holding a fixed.",
        "outcome": "y", "grain": "Declared mathematical test model", "domain": "Explicit finite cases"})
    sheet.put("tool_check_theory", "theory", {"name": "Declared linear test model",
        "mechanism": "A deterministic relation from x to y", "predictions": ["y=a*x"],
        "admission": "candidate", "same_as": "", "distinction": "Verifier fixture, not empirical evidence",
        "compared_with": []})
    sheet.put("tool_check_application", "application", {"inference": "tool_check_G", "inference_version": "",
        "theory": "tool_check_theory", "context_key": "tool_check_finite_domain",
        "context": "Declared finite arithmetic circumstances", "context_distinction": "One fixture context",
        "direction": "unresolved", "inference_use": "Structural x-to-y dependence",
        "consequence_without_inference": "Removing the focal contribution changes y"})
    sheet.put("tool_check_protocol", "importance", {"application": "tool_check_application", "method": "scm",
        "question": "What changes when the x-to-y contribution is removed?", "contrast": "y=a*x versus y=0",
        "domain": {"kind": "finite", "description": "One positive and one zero circumstance",
            "measure": "Unit counting weights", "cases": [
                {"id": "positive", "inputs": {"x": 3, "a": 2}, "weight": 1},
                {"id": "zero", "inputs": {"x": 0, "a": 2}, "weight": 1}]},
        "model": {"equations": {"y": "a*x"}, "focal": "y", "replacement": "0", "answer": "y",
            "initial": {}, "steps": 0, "edge": ["x", "y"]}, "comparator": "exact_typed_answer"})
    return sheet


class ToolSource:
    def __init__(self, sheet):
        self.ledger = Ledger(ResearchResult(causal_sheet_updates=tuple(sheet.history)))
        self.selected_keys = list(sheet.rows)
        self.selected_snapshots = []
        self.commission = SimpleNamespace(inputs=tuple(sheet.rows))
        self.run = SimpleNamespace(providers=SimpleNamespace(checkpoint=None))
        self.last_proposal = None

    def scoped_snapshots(self):
        return list(self.selected_snapshots)

    def authorize_evaluation(self, assessment):
        if assessment not in self.selected_keys:
            raise ValueError("Unselected test protocol")


def call(source, name, arguments, inputs):
    request = {"action": "scientific_tool", "name": name, "question": "Focused scientific tool verification",
        "inputs": inputs, "arguments": arguments, "rationale": "Check production tool execution and receipts"}
    source.last_proposal = request
    result = execute_tool(source, request)
    source.ledger.merge(result)
    receipt = next(row for row in reversed(result.causal_sheet_updates) if row["kind"] == "tool_receipt")
    assert receipt["key"] in source.selected_keys, "The invoking episode must receive its tool result next"
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    directory = Path(args.run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = ToolSource(fixture())
    sheet = source.ledger.causal_sheet
    ref = {"key": "tool_check_G", "version": sheet.rows["tool_check_G"]["version"]}
    binding = lambda name, index: {"name": name, "type": "Nat", "source": ref,
                                   "selector": ["variables", str(index)]}
    base = {"relation": "equal", "parameters": [binding("x", 0), binding("y", 1)],
            "assumptions": [], "left": "x + y", "right": "y + x", "proof": "exact Nat.add_comm x y"}
    results = {}
    good = call(source, "lean_equivalence", base, [ref])
    results["real_lean_equivalence"] = good["data"]
    assert good["data"]["status"] == "verified_equivalent", good["data"]
    bad = call(source, "lean_equivalence", {**base, "left": "x + 1", "right": "x", "proof": "rfl"}, [ref])
    results["failed_proof_stays_unresolved"] = bad["data"]
    assert bad["data"]["status"] == "unresolved", bad["data"]
    rejected = call(source, "lean_equivalence", {**base, "proof": "exact sorry"}, [ref])
    results["admitted_placeholder_rejected"] = rejected["data"]
    assert rejected["data"]["status"] == "error", rejected["data"]
    protocol = {"key": "tool_check_protocol", "version": sheet.rows["tool_check_protocol"]["version"]}
    scm = call(source, "causal_calculation", {"assessment": "tool_check_protocol",
        "case_ids": ["positive", "zero"], "evaluation_context": ""}, [protocol])
    results["scm_intervention_receipts"] = scm["data"]
    assert scm["data"]["status"] == "computed", scm["data"]
    cases = scm["data"]["result"]["cases"]
    assert cases[0]["baseline"] != cases[0]["altered"]
    assert cases[1]["baseline"] == cases[1]["altered"]
    lookup = call(source, "lookup", {"keys": [ref], "kinds": [], "query": ""}, [])
    results["scientific_lookup"] = lookup["data"]
    assert lookup["data"]["status"] == "found" and ref["key"] in source.selected_keys
    no_text = call(source, "lookup", {"keys": [], "kinds": ["evidence"], "query": ""}, [])
    results["lookup_source_text_rejected"] = no_text["data"]
    assert no_text["data"]["status"] == "error"
    if "Mathlib.Tactic.Ring" in (_lean_config() or {}).get("imports", []):
        real_base = {**base, "parameters": [{**binding("x", 0), "type": "Real"},
                                            {**binding("y", 1), "type": "Real"}],
                     "left": "2 * x + y", "right": "y + x * 2", "proof": "ring"}
        real = call(source, "lean_equivalence", real_base, [ref])
        results["mathlib_real_equivalence"] = real["data"]
        assert real["data"]["status"] == "verified_equivalent", real["data"]
    summary = {"passed": True, "checks": list(results), "count": len(results)}
    (directory / "tool-results.json").write_text(json.dumps(results, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
