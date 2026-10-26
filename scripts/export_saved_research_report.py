"""Render saved, committed research values without resuming their workflow.

This is a persistence adapter into ResearchReport, not a research runner. It
never opens a checkpoint for writing, dispatches a provider, or executes a
scientific calculation. Supply the ResearchResult archive to export explicitly.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from episodic_workflow.case_reports.research.engine import ResearchLimits
from episodic_workflow.case_reports.research.goal_state import ScientificGoalState
from episodic_workflow.case_reports.research.records import Fact, ResearchResult
from episodic_workflow.case_reports.research.report import ResearchReport


def export_report(checkpoint, result_archive, output):
    """Project one saved result through the normal report implementation."""
    checkpoint = Path(checkpoint).resolve()
    result_archive = Path(result_archive).resolve()
    output = Path(output).resolve()
    for path in (output.with_suffix(".md"), output.with_suffix(".json")):
        if path.exists():
            raise FileExistsError(f"Report export already exists: {path}")
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    saved = json.loads(result_archive.read_text())

    # A child's completion label is not part of shared scientific state.
    shared = {**saved, "evaluated": True, "stop_reason": ""}
    state_id = hashlib.sha256(json.dumps(
        shared, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("ascii")).hexdigest()
    del shared

    root_start = None
    root_end = None
    matches = []
    model_calls = total_units = 0
    for path in sorted((checkpoint / "records").glob("*.json")):
        record = json.loads(path.read_text())
        body = record.get("body", {})
        payload = body.get("payload", {})
        if record.get("frame_type") == "model_request":
            model_calls += 1
        if body.get("event_kind") == "unit_completed":
            total_units += 1
        if (body.get("event_kind") == "episode_started"
                and payload.get("grain") == "evaluate_airborne_safety_guideline"):
            root_start = {"episode_id": body["episode_id"], **payload}
        if (root_start is not None
                and body.get("episode_id") == root_start["episode_id"]
                and body.get("event_kind") == "episode_completed"):
            root_end = payload
        if payload.get("goal_state_id") == state_id:
            matches.append({"record": path.name, "sha256": record["sha256"],
                            "event": body.get("event_kind")})
    if root_start is None:
        raise ValueError("Saved checkpoint has no root episode definition")
    if not matches:
        raise ValueError("The supplied result does not match any committed shared-state boundary")

    # These fields are the persisted ResearchResult contract. No reviewer,
    # estimator, controller, or numerical executor runs during reconstruction.
    saved["facts"] = tuple(Fact(**{**row, "parents": tuple(row.get("parents", ()))})
                           for row in saved["facts"])
    result = ResearchResult(**{
        field.name: tuple(saved[field.name]) if isinstance(saved[field.name], list)
        else saved[field.name]
        for field in fields(ResearchResult) if field.name in saved})
    del saved
    goal = root_start["goal_view"]
    metrics = tuple(goal["result_contract"]["metrics"])
    global_goal = ScientificGoalState(metrics, result)
    limits = ResearchLimits(**{name: manifest["inputs"][name]
                              for name in ResearchLimits.__dataclass_fields__
                              if name in manifest["inputs"]})
    ended_by = root_end["ended_by"] if root_end else "not_recorded"
    controller_state = (root_end or {}).get("controller_state", {})
    report_record = SimpleNamespace(
        goal=SimpleNamespace(as_record=lambda: goal), ended_by=ended_by,
        credit_history=controller_state.get("history", ()),
        rarefaction=controller_state.get("rarefaction"),
        controller_state=SimpleNamespace(
            history=controller_state.get("history", ()),
            rarefaction=controller_state.get("rarefaction")))
    run = SimpleNamespace(
        metrics=metrics, global_goal=global_goal,
        root_credit_projection=lambda status=None: (
            global_goal.preview() if status is None else status)["weighted_score"],
        research_request=goal.get("task_context", {}).get("research_request") or goal["objective"],
        model_attempts=(), implementation=manifest["implementation"]["files"],
        implementation_sha256_at_start=manifest["implementation"]["files"],
        documents={}, provider_events=(), model_calls=model_calls,
        total_units=total_units, limits=limits,
        fixed_scenarios=bool(manifest["inputs"].get("scenario_definitions")),
        tree=SimpleNamespace(children={}))
    report = ResearchReport.from_result(result, report_record, run)
    saved_at = datetime.fromtimestamp(
        (checkpoint / "records" / matches[-1]["record"]).stat().st_mtime,
        timezone.utc).isoformat()
    report.data["saved_snapshot"] = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "saved_state_timestamp": saved_at,
        "timestamp_basis": "Modification time of the matching committed checkpoint boundary file.",
        "checkpoint_created_at": manifest.get("created_at"),
        "run_state": f"{'root has no recorded termination' if root_end is None else 'root episode ended: ' + ended_by}; committed state saved {saved_at}",
        "disease": manifest["inputs"].get("disease"),
        "checkpoint": str(checkpoint), "result_archive": str(result_archive),
        "shared_state_id": state_id, "matching_committed_boundaries": matches,
        "root_ended_by": ended_by,
        "meaning": "Saved numerical results; rendering does not resume or terminate the run.",
        "omitted_audit_fields": ["model_attempts", "model_response_failures",
                                "source_documents", "provider_events", "episode_tree"],
    }
    for name in report.data["saved_snapshot"]["omitted_audit_fields"]:
        report.data.pop(name, None)
    if root_end is None:
        report.data["control"].pop("credit_history", None)
        report.data["control"].pop("rarefaction", None)
    report.data["control"]["reason"] = (
        "Root has no recorded termination; this report contains the saved results."
        if root_end is None else result.stop_reason or root_end["ended_by"])

    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".md").write_text(report.markdown(), encoding="utf-8")
    output.with_suffix(".json").write_text(
        json.dumps(report.as_record(), ensure_ascii=True, allow_nan=False,
                   separators=(",", ":")) + "\n", encoding="utf-8")
    return {"markdown": str(output.with_suffix(".md")),
            "json": str(output.with_suffix(".json")),
            "status": report.data["status"],
            "computed": report.data["computational_progress"],
            "committed_snapshot_matches": len(matches),
            "root_ended_by": ended_by}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--result-archive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Output stem for the rendered .md and .json files")
    args = parser.parse_args()
    print(json.dumps(export_report(args.checkpoint, args.result_archive, args.output)))


if __name__ == "__main__":
    main()
