"""One launcher for the causal-inference research Episodes."""
from __future__ import annotations
import argparse
import filecmp
import json
import os
import sys
import tempfile
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import uuid

from .checkpoint import CheckpointError, CheckpointJournal
from .configured_providers import ProviderConfigurationError, run_configured_research
from .case_reports.research import ResearchLimits
from .case_reports.research.task import RESEARCH_OBJECTIVE
from .method_loop.archive import iter_json, write_json
from .method_loop.reconstruction import ReconstructionError


def main(argv=None, *, stdout=None, stderr=None):
    out, err = stdout or sys.stdout, stderr or sys.stderr
    parser = argparse.ArgumentParser(description="Investigate generalized causal inferences using iterative Episodes")
    parser.add_argument("request", nargs="?", help="Optional scientific scope; defaults to generic causal-example discovery")
    parser.add_argument("--request-file")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--resume")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--root-yield-threshold", type=float,
                        help="Scientific parent marginal-gain cutoff; zero makes numerical yielding unreachable")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--trace-path")
    parser.add_argument("--report-path")
    parser.add_argument("--audit-path")
    for name in ("search-limit", "text-window-chars", "yield-window", "root-yield-window"):
        parser.add_argument("--" + name, type=int)
    args = parser.parse_args(argv)
    try:
        if args.resume and args.checkpoint_path:
            raise CheckpointError("Choose a new checkpoint or resume an existing one")
        if args.request and args.request_file:
            raise ValueError("Supply request text or --request-file, not both")
        with ExitStack() as streams:
            checkpoint = streams.enter_context(CheckpointJournal(args.resume, resume=True)) if args.resume else None
            saved = checkpoint.inputs if checkpoint else {}
            request = Path(args.request_file).read_text() if args.request_file else args.request or saved.get("research_request") or RESEARCH_OBJECTIVE
            if not request:
                raise ValueError("Supply the scientific question as request text or --request-file")
            defaults = ResearchLimits()
            fields = ("search_limit", "text_window_chars", "yield_window", "root_yield_window", "root_yield_threshold")
            limits_data = {name: getattr(args, name) if getattr(args, name) is not None else
                           saved.get("limits", {}).get(name, getattr(defaults, name)) for name in fields}
            limits = ResearchLimits(**limits_data)
            inputs = {"research_request": request, "limits": limits_data}
            if checkpoint is None:
                path = args.checkpoint_path or str(Path(".artifacts/checkpoints") /
                    (datetime.now(timezone.utc).strftime("causal-%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]))
                checkpoint = streams.enter_context(CheckpointJournal(path, inputs=inputs))
            elif checkpoint.inputs != inputs:
                raise CheckpointError("Checkpoint request or episode limits differ")
            checkpoint.authorize = lambda: {"authorized": True, "mechanism": "explicit_cli_resume",
                                            "checkpoint": str(checkpoint.path), "exclusive_writer_lock": True}
            trace_path = Path(args.trace_path) if args.trace_path else checkpoint.path / "trace.jsonl"
            report_path = Path(args.report_path) if args.report_path else checkpoint.path / "report.json"
            audit_path = Path(args.audit_path) if args.audit_path else checkpoint.path / "audit.json"
            paths = [p.resolve() for p in (trace_path, report_path, audit_path)]
            if len(set(paths)) != len(paths):
                raise CheckpointError("Trace, audit and report require different paths")
            for path in paths:
                if path in {checkpoint.path / n for n in ("manifest.json", "pending.json", "lock")} or any(
                        folder in path.parents for folder in (checkpoint.path / "records", checkpoint.path / "attempts", checkpoint.path / "archive")):
                    raise CheckpointError("Output overlaps durable checkpoint records")
                if path.exists() and not args.resume:
                    raise CheckpointError(f"Output already exists: {path}")
            trace = streams.enter_context(trace_path.open("a" if args.resume else "x"))
            def event(item):
                write_json(trace, item)
                trace.write("\n")
                trace.flush()
            print(f"Checkpoint: {checkpoint.path}", file=err, flush=True)
            record, report = run_configured_research(research_request=request, limits=limits,
                progress=args.progress, trace_event=event, checkpoint=checkpoint)
            payload = {**report.as_record(), "checkpoint": checkpoint.provenance()}
            checkpoint.event({"event": "workflow_completed", "audit": record.as_record(), "report": payload})
            checkpoint.complete()
            _write_result(audit_path, iter_json(record.as_record(), indent=2), resume=bool(args.resume))
            _write_result(report_path, iter_json(payload, indent=2), resume=bool(args.resume))
            if args.format == "markdown":
                print(report.markdown(), file=out)
            else:
                write_json(out, payload, indent=2)
                out.write("\n")
            return 0 if report.passed else 1
    except (ProviderConfigurationError, CheckpointError, ReconstructionError, ValueError, OSError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}), file=err)
        return 2


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
