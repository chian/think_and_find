"""Reconstruct an explicitly admitted fork with network and live activation denied."""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parent", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--through-record", type=int, required=True)
    parser.add_argument("--allow-changed", action="append", default=[])
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "episodic_workflow", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    from episodic_workflow.checkpoint import CheckpointError, CheckpointJournal, _atomic
    from episodic_workflow.case_reports.research import ResearchLimits
    from episodic_workflow.configured_providers import run_configured_safety_guideline_request

    def deny_external(event, arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "subprocess.Popen"}:
            raise CheckpointError("Offline reconstruction attempted an external effect: " + event)

    # Cache stdlib platform discovery (which may invoke local uname) before
    # prohibiting every external effect during scientific reconstruction.
    platform.platform()
    sys.addaudithook(deny_external)
    with CheckpointJournal(args.destination, fork_from=args.parent,
            through_record=args.through_record, allowed_changes=args.allow_changed,
            fork_reason=args.reason) as journal:
        journal.authorize = lambda: {"authorized": False, "mechanism": "offline_prefix_verification"}
        inputs = journal.inputs
        try:
            run_configured_safety_guideline_request(
                disease=inputs["disease"], research_request=inputs["research_request"],
                scenario=inputs["scenario"], search_limit=inputs["search_limit"], metrics=inputs["metric"],
                limits=ResearchLimits(**{name: inputs[name] for name in ("text_window_chars", "yield_window")}),
                checkpoint=journal)
        except CheckpointError as exc:
            if str(exc) != "Host denied continuation" or not journal.cursor.prefix_verified:
                raise
        else:
            raise CheckpointError("Fork ended without reaching the explicit denied activation boundary")
        if journal.activated or journal.position != journal.replay_count:
            raise CheckpointError("Offline verification unexpectedly changed execution state")
        operations = {}
        for row in journal.records:
            if row["kind"] == "call":
                operations[row["operation"]] = operations.get(row["operation"], 0) + 1
        receipt = {"verified": True, "live_authorized": False, "external_effects": "denied_by_python_audit_hook",
            "through_record": args.through_record, "matched_records": journal.position,
            "reused_external_operations": operations, "reconstruction": journal.cursor.report(),
            "provenance": journal.provenance()}
        _atomic(journal.path / "fork-verification.json", receipt)
        print(json.dumps({key: value for key, value in receipt.items() if key != "provenance"}, indent=2))


if __name__ == "__main__":
    main()
