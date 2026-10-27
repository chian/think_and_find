"""Command-line entry point for configured episodic workflow runs."""

from __future__ import annotations

import argparse
import filecmp
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
import uuid
from pathlib import Path
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from typing import Any, TextIO

from .configured_providers import (
    ProviderConfigurationError,
    run_configured_safety_guideline_request,
)
from .checkpoint import CheckpointError, CheckpointJournal
from .method_loop.reconstruction import ReconstructionError
from .method_loop.archive import iter_json, write_json


PRAGMATIC_CAUSAL_PLURALISM_NAME = 'pragmatic causal pluralism'
DEFAULT_CAUSAL_REPORTING_MILESTONE = 10
# V11.8 narrows the standing target: each reversal must stay within one focal
# organism or stable interacting biological system. Different organisms may
# still supply separate cases. The literal-zero stop and reporting milestone
# remain unchanged. A version label alone never changes the target.
# Prompt wording is intentionally outside this fingerprint so it can improve;
# prompt edits still need a semantic comparison against this fixed task intent.
_PRAGMATIC_CAUSAL_PLURALISM_INTENT_DIGEST = (
    '8e4019a21431783e0d06808458898773268a42c958f77aff0e32d52cc87264de'
)


def _verify_pragmatic_causal_pluralism_intent() -> str:
    from .reversal_task import (QUESTION, RUBRIC, THEORY_RUBRIC, QUALITY_RUBRIC,
                                ROOT_STOP_YIELD, digest)

    observed = digest({
        'task_name': PRAGMATIC_CAUSAL_PLURALISM_NAME,
        'question': QUESTION,
        'candidate_fit_rubric': RUBRIC,
        'theory_fit_rubric': THEORY_RUBRIC,
        'quality_rubric': QUALITY_RUBRIC,
        'default_reporting_milestone': DEFAULT_CAUSAL_REPORTING_MILESTONE,
        'root_stop_threshold': ROOT_STOP_YIELD,
    })
    if observed != _PRAGMATIC_CAUSAL_PLURALISM_INTENT_DIGEST:
        raise ProviderConfigurationError(
            'Pragmatic causal pluralism intent changed. Compare the proposed '
            'task text, rubrics, milestone, and root stop rule with the v8 '
            'registration before deliberately changing this fingerprint.'
        )
    return QUESTION


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run an env-configured disease workflow."""

    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    args = None
    try:
        args = _parser().parse_args(argv)
        if args.task == 'causal-reversals':
            from .reversal_task import run_discovery
            from .environment import workflow_environment
            if args.resume or args.fork_checkpoint or args.disease or args.report_path:
                raise ProviderConfigurationError('Causal discovery accepts a question request-file and target count; disease and legacy checkpoint/report options belong to the retained domain workflow.')
            question=_verify_pragmatic_causal_pluralism_intent()
            if args.request_file:
                question+='\n\nAdditional operator instructions:\n'+Path(args.request_file).read_text()
            from .configured_providers import firecrawl_argo_config_from_env
            config = firecrawl_argo_config_from_env(workflow_environment(env_file=args.provider_env_file))
            result = run_discovery(config=config,output=args.experiment_dir,progress=args.progress,
                                   question=question,target=args.target_count)
            print(json.dumps(result, indent=args.indent), file=out)
            return 0 if result['run_complete'] else 1
        if args.provider_env_file:
            raise ProviderConfigurationError('--provider-env-file currently belongs to --task causal-reversals')
        if args.target_scenario is not None or args.scenarios_file is not None or args.source_url or args.legacy_workflow:
            raise ProviderConfigurationError("Legacy input bypasses and seeded sources are disabled. Use --scenario for caller-defined cases; answers and evidence cannot be supplied.")
        from .case_reports.research import ResearchLimits
        from .case_reports.research.scenarios import load_scenarios
        with ExitStack() as streams:
            if args.fork_checkpoint and (args.resume or not args.checkpoint_path):
                raise CheckpointError("A compatible fork requires a new --checkpoint-path and cannot use --resume")
            if not args.fork_checkpoint and (args.fork_through_record is not None or args.fork_reason or args.fork_allow_changed):
                raise CheckpointError("Fork admission options require --fork-checkpoint")
            if args.resume and args.checkpoint_path:
                raise CheckpointError("Use --resume for an existing checkpoint, not --checkpoint-path")
            checkpoint = streams.enter_context(CheckpointJournal(args.resume, resume=True)) if args.resume else None
            if args.fork_checkpoint:
                checkpoint = streams.enter_context(CheckpointJournal(args.checkpoint_path,
                    fork_from=args.fork_checkpoint, through_record=args.fork_through_record,
                    allowed_changes=args.fork_allow_changed, fork_reason=args.fork_reason))
            saved = checkpoint.inputs if checkpoint is not None else {}
            legacy_root_window = checkpoint is not None and "root_yield_window" not in saved
            root_window_default = (saved.get("yield_window", ResearchLimits().yield_window)
                                   if legacy_root_window else ResearchLimits().root_yield_window)
            for name, default in (("disease", None), ("scenario", "bazant-bush"), ("search_limit", 10),
                                  ("text_window_chars", ResearchLimits().text_window_chars),
                                  ("yield_window", ResearchLimits().yield_window),
                                  ("root_yield_window", root_window_default), ("metric", None)):
                if getattr(args, name) is None:
                    setattr(args, name, saved.get(name, default))
            if legacy_root_window and args.root_yield_window != root_window_default:
                raise CheckpointError("This checkpoint used the same yield window for root and children; changing the root window requires a new run")
            if not args.disease:
                raise ProviderConfigurationError("Supply the disease of interest; no disease is built in")
            request = Path(args.request_file).read_text() if args.request_file else saved.get("research_request", "")
            if Path(args.scenario).is_file():
                args.scenario = str(Path(args.scenario).resolve())
            inputs = {name: getattr(args, name) for name in (
                "disease", "scenario", "search_limit", "text_window_chars", "yield_window", "root_yield_window", "metric")}
            if legacy_root_window:
                # Preserve old saved inputs and their original root-window semantics.
                del inputs["root_yield_window"]
            inputs.update(research_request=request,
                          scenario_definitions=[case.as_record() for case in load_scenarios(args.scenario)])
            if checkpoint is None:
                path = args.checkpoint_path or str(Path(".artifacts/checkpoints") /
                    (datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]))
                checkpoint = streams.enter_context(CheckpointJournal(path, inputs=inputs))
            elif inputs != checkpoint.inputs:
                raise CheckpointError("Checkpoint runtime inputs or scenario definitions differ from this invocation")
            checkpoint.authorize = lambda: {"authorized": True, "mechanism": "explicit_cli_fork" if args.fork_checkpoint else "explicit_cli_resume",
                "checkpoint": str(checkpoint.path), "exclusive_writer_lock": True}
            print(f"Checkpoint: {checkpoint.path}" + (f"; reconstructing {checkpoint.replay_count} saved frames" if args.resume or args.fork_checkpoint else ""),
                  file=err, flush=True)
            limits = ResearchLimits(**{name: getattr(args, name) for name in (
                "text_window_chars", "yield_window", "root_yield_window")})
            trace_path = Path(args.trace_path) if args.trace_path else checkpoint.path / "trace.jsonl"
            audit_path = Path(args.audit_path) if args.audit_path else checkpoint.path / "audit.json"
            report_path = Path(args.report_path) if args.report_path else checkpoint.path / "report.json"
            output_paths = [path.resolve() for path in (trace_path, audit_path, report_path)]
            if len(set(output_paths)) != len(output_paths):
                raise CheckpointError("Trace, audit and report require distinct output paths")
            for path in output_paths:
                if (path in {checkpoint.path / name for name in ("manifest.json", "pending.json", "lock")}
                        or any(directory in path.parents for directory in
                               (checkpoint.path / "records", checkpoint.path / "attempts", checkpoint.path / "archive"))):
                    raise CheckpointError("Output path overlaps checkpoint continuation records")
                if not path.parent.is_dir():
                    raise CheckpointError(f"Output directory does not exist: {path.parent}")
                if path.exists() and (not args.resume or (path != trace_path.resolve()
                        and checkpoint.manifest["status"] != "complete")):
                    raise CheckpointError(f"Output already exists: {path}")
            trace_stream = streams.enter_context(trace_path.open("a" if args.resume else "x"))

            def trace_event(event):
                write_json(trace_stream, event)
                trace_stream.write("\n")
                trace_stream.flush()

            record, report = run_configured_safety_guideline_request(
                disease=args.disease, research_request=request, scenario=args.scenario,
                transmission_model=args.transmission_model, search_limit=args.search_limit,
                require_transmissibility_evidence=args.require_transmissibility_evidence,
                progress=args.progress, trace_event=trace_event if trace_stream is not None else None,
                metrics=args.metric, limits=limits, checkpoint=checkpoint,
            )
            report_record = checkpoint.archive.store({
                **report.as_record(), "checkpoint": checkpoint.provenance()})
            checkpoint.event({"event": "workflow_completed", "audit": record.as_record(),
                              "report": report_record})
            checkpoint.complete()
            _write_result(audit_path, iter_json(record.as_record(), indent=2), resume=bool(args.resume))
            content = (_markdown(report) + "\nCheckpoint: " + str(checkpoint.path) + "\n" +
                       "```json\n" + json.dumps(checkpoint.provenance(), indent=2) + "\n```"
                       if args.format == "markdown" else None)
            _write_result(report_path, (content,) if content is not None else
                          iter_json(report_record, indent=args.indent), resume=bool(args.resume))
    except (ProviderConfigurationError, CheckpointError, ReconstructionError, ValueError, OSError) as exc:
        checkpoint_failure = isinstance(exc, (CheckpointError, ReconstructionError))
        print(
            json.dumps(
                {
                    "passed": False,
                    "errors": [
                        {
                            "child": "checkpoint" if checkpoint_failure else "configured_providers",
                            "field": "continuation" if checkpoint_failure else "environment",
                            "code": "checkpoint_error" if checkpoint_failure else "configuration_error",
                            "message": str(exc),
                        }
                    ],
                },
                indent=args.indent if args else 2,
                sort_keys=True,
            ),
            file=err,
        )
        return 2

    if args.format == "markdown":
        print(content, file=out)
        return 0 if report.passed else 1
    write_json(out, report_record, canonical=True, indent=args.indent)
    out.write("\n")
    return 0 if report.passed else 1


def _write_result(path, chunks, *, resume):
    """Publish streamed output atomically without loading an existing report."""
    fd, temporary = tempfile.mkstemp(prefix=".result-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            for chunk in chunks:
                stream.write(chunk)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if resume and path.exists() and filecmp.cmp(temporary, path, shallow=False):
            return
        # Atomic publication without overwriting another result or a prior run.
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)


def _json_record(value: Any) -> Any:
    """Preserve non-finite diagnostics and unbounded results in strict JSON."""

    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {key: _json_record(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_record(item) for item in value]
    return value


def _markdown(report):
    if hasattr(report, "markdown"):
        return report.markdown()
    data = report.as_record()
    lines = ["# Safety Guideline", "", f"Status: {'conditional result' if report.passed else 'incomplete'}", ""]
    for child, outputs in data.get("main_outputs", {}).items():
        lines.extend([f"## {child}", "", "```json", json.dumps(_json_record(outputs), indent=2, allow_nan=False), "```", ""])
    for name, assumption in data.get("admitted_assumptions", {}).items():
        lines.extend([f"## Assumption: {name}", "", "```json", json.dumps(_json_record(assumption), indent=2, allow_nan=False), "```", ""])
    for error in data.get("errors", []):
        lines.append(f"- {error['child']}: {error['message']}")
    lines.extend(["", "Conditional model calculation, not a guarantee of safety or independent validation."])
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m episodic_workflow",
        description=(
            "Run an env-configured Firecrawl/Jev episodic workflow for a "
            "requested disease or pathogen."
        ),
    )
    parser.add_argument(
        "disease", nargs="?",
        help="Disease of interest; there is no default pathogen.",
    )
    parser.add_argument('--task', choices=('airborne', 'causal-reversals'), default='airborne',
                        help='Select the existing domain workflow or pragmatic causal pluralism research.')
    parser.add_argument('--target-count',type=int,default=DEFAULT_CAUSAL_REPORTING_MILESTONE,
                        help='Requested reporting milestone for causal discovery (default: 10); neither a credit capacity nor a stop rule.')
    parser.add_argument('--provider-env-file', help='Explicit provider configuration file for causal discovery; credentials are read in memory and not copied.')
    parser.add_argument('--experiment-dir', help='New output directory for --task causal-reversals; defaults outside the source workspace.')
    parser.add_argument("--request-file", help="Optional additional natural-language instructions, without supplied scientific answers.")
    parser.add_argument("--scenario",
                        help="Caller-selected cases: bazant-bush (default), classroom, nursing-home, a built-in case name, or a scenario JSON file.")
    parser.add_argument(
        "--search-limit",
        type=int,
        default=None,
        help="Maximum Firecrawl results requested from each scoped query.",
    )
    parser.add_argument("--target-scenario", help=argparse.SUPPRESS)
    parser.add_argument("--scenarios-file", help=argparse.SUPPRESS)
    parser.add_argument("--legacy-workflow", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--transmission-model", help=argparse.SUPPRESS)
    parser.add_argument("--require-transmissibility-evidence", action="store_true", help=argparse.SUPPRESS)
    from .task_executors.scenario_metrics.evaluate import DEFAULT_METRICS
    from .case_reports.research import ResearchLimits
    parser.add_argument("--metric", action="append", choices=DEFAULT_METRICS, help=argparse.SUPPRESS)
    parser.add_argument("--text-window-chars", type=int)
    parser.add_argument("--yield-window", type=int,
                        help="Marginal-yield window for child episodes (default: 4).")
    parser.add_argument("--root-yield-window", type=int,
                        help="Marginal-yield window for the root episode (default: 12).")
    parser.add_argument("--checkpoint-path", help="New durable checkpoint directory (default: a unique directory under .artifacts/checkpoints).")
    parser.add_argument("--resume", metavar="CHECKPOINT", help="Continue this exact interrupted run by verified local reconstruction; saved provider responses are not requested again.")
    parser.add_argument("--fork-checkpoint", metavar="CHECKPOINT", help="Create an explicitly admitted changed-code experiment from an immutable saved unit prefix; not ordinary resume.")
    parser.add_argument("--fork-through-record", type=int, help="Exact committed unit-completion record ending the reused prefix; not a run limit.")
    parser.add_argument("--fork-allow-changed", action="append", default=[], metavar="PATH", help="Explicitly admit each changed implementation file; must exactly match the detected change set.")
    parser.add_argument("--fork-reason", help="Compatibility rationale recorded with the immutable parent and old/new implementation identities.")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--audit-path", help="Write the complete recursive episode trace, contracts, and numerical verdicts.")
    parser.add_argument("--trace-path", help="Stream model/provider attempts and credit verdicts as JSON Lines to a new file.")
    parser.add_argument("--report-path", help="Also save the report to a new file (never overwrite an existing report).")
    parser.add_argument("--progress", action="store_true", help="Log model operation names to stderr without source content or credentials.")
    parser.add_argument("--source-url", action="append", default=[], help=argparse.SUPPRESS)
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation for the emitted credit report.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
