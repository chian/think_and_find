"""Durable external-response journal for exact, checked workflow continuation.

Replay rebuilds the ordinary nested method loop, including its source objects
and controller histories. It does not deserialize Python stacks or substitute
previous controller verdicts for re-executing deterministic code.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
import fcntl
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Mapping

from .method_loop import Episode
from .method_loop.archive import ArchivedRecord, ArchivedSequence, JsonArchive, iter_json, write_json
from .method_loop.identities import EpisodeRef
from .task_executors._http import ProviderError


class CheckpointError(RuntimeError):
    """A checkpoint cannot safely continue; never treat this as source failure."""


def _record(value):
    if isinstance(value, (ArchivedRecord, ArchivedSequence)):
        return value
    if hasattr(value, "as_record"):
        return _record(value.as_record())
    if is_dataclass(value):
        return {field.name: _record(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _record(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_record(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def _json(value, *, canonical=False):
    return json.dumps(value, ensure_ascii=True, allow_nan=False,
                      sort_keys=canonical, separators=(",", ":"))


def _digest(value):
    digest = hashlib.sha256()
    for chunk in iter_json(value, canonical=True, normalize_nonfinite=False):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def implementation_identity():
    root = Path(__file__).resolve().parent
    paths = list(root.glob("*.py"))
    for directory in ("case_reports", "credit_assignment", "method_loop", "task_executors"):
        paths.extend((root / directory).rglob("*.py"))
    return {"python": sys.version, "platform": platform.platform(),
            "files": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in sorted(paths)}}


def _load(path):
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"Unreadable checkpoint file {path}: {exc}") from exc


def _atomic(path, value):
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            write_json(stream, value, normalize_nonfinite=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise CheckpointError(f"Cannot commit checkpoint file {path}: {exc}") from exc
    finally:
        if temporary is not None and os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except OSError:
                pass


class CheckpointJournal:
    SCHEMA = "episodic-workflow-checkpoint-v1"

    def __init__(self, path, *, inputs=None, resume=False, fork_from=None,
                 through_record=None, allowed_changes=(), fork_reason=""):
        if resume and fork_from is not None:
            raise CheckpointError("An explicit fork is not an ordinary resume")
        fork = self._prepare_fork(fork_from, through_record, allowed_changes, fork_reason) if fork_from is not None else None
        self.path = Path(path).expanduser().resolve()
        self.position = 0
        self.records = []
        self.secrets = ()
        self._lock = None
        self.pending = None
        self.current_episode = None
        self.cursor = None
        self.activated = not (resume or fork is not None)
        self.authorize = None
        self.attempt = None
        self._historical_effects = False
        self._inflight = False
        if not resume:
            self.path.mkdir(parents=True, mode=0o700, exist_ok=False)
        elif not self.path.is_dir():
            raise CheckpointError(f"Checkpoint directory does not exist: {self.path}")
        lock_fd = os.open(self.path / "lock", os.O_CREAT | os.O_RDWR, 0o600)
        self._lock = os.fdopen(lock_fd, "r+")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.close()
            raise CheckpointError(f"Checkpoint is already in use: {self.path}") from exc
        try:
            if resume:
                self.manifest = _load(self.path / "manifest.json")
                self._validate_manifest()
                if inputs is not None and _digest(inputs) != _digest(self.inputs):
                    raise CheckpointError("Checkpoint runtime inputs differ from this invocation")
                if self.manifest["implementation"] != implementation_identity():
                    raise CheckpointError("Checkpoint implementation/Python differs; unchanged code is required for exact replay")
                self._read_records()
            elif fork is not None:
                self.manifest, self.records = fork
                if inputs is not None and inputs != self.inputs:
                    raise CheckpointError("Fork runtime inputs cannot change")
                (self.path / "records").mkdir(mode=0o700)
                for record in self.records:
                    _atomic(self.path / "records" / f"{record['index']:012d}.json", record)
                _atomic(self.path / "manifest.json", self.manifest)
            else:
                if not isinstance(inputs, dict):
                    raise CheckpointError("A new checkpoint requires runtime inputs")
                (self.path / "records").mkdir(mode=0o700)
                self.manifest = {"schema": self.SCHEMA, "inputs": inputs,
                    "implementation": implementation_identity(), "status": "active",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "provider_configuration": None, "logical_run_id": "run_" + uuid.uuid4().hex}
                _atomic(self.path / "manifest.json", self.manifest)
            self.replay_count = len(self.records)
            self.archive = JsonArchive(self.path / "archive")
            self._begin_attempt(resume)
            if resume or fork is not None:
                self._reconstruction_cursor()
        except BaseException:
            self.close()
            raise

    @property
    def inputs(self):
        return self.manifest["inputs"]

    @property
    def replaying(self):
        return not self.activated or self._historical_effects

    def _validate_manifest(self):
        m = self.manifest
        if (not isinstance(m, dict) or m.get("schema") != self.SCHEMA
                or not isinstance(m.get("inputs"), dict)
                or not isinstance(m.get("implementation"), dict)
                or m.get("status") not in {"active", "complete"}
                or not isinstance(m.get("logical_run_id"), str)
                or "provider_configuration" not in m):
            raise CheckpointError("Invalid or unsupported checkpoint manifest")
        fork = m.get("fork")
        if fork is not None:
            if (not isinstance(fork, dict) or fork.get("kind") != "verified_prefix_experiment"
                    or not isinstance(fork.get("fork_id"), str)
                    or fork.get("new_implementation") != m["implementation"]
                    or fork.get("parent_logical_run_id") != m["logical_run_id"]
                    or not isinstance(fork.get("historical_run_implementation"), dict)
                    or type(fork.get("through_record")) is not int):
                raise CheckpointError("Invalid compatible-fork provenance")

    @classmethod
    def _prepare_fork(cls, source, through, allowed_changes, reason):
        """Read an immutable parent; changed-code experiments require explicit admission."""
        if type(through) is not int or through < 0 or not isinstance(reason, str) or not reason.strip():
            raise CheckpointError("A fork requires an exact unit boundary and explicit compatibility reason")
        parent = object.__new__(cls)
        parent.path = Path(source).expanduser().resolve()
        parent.records, parent.pending = [], None
        try:
            lock = (parent.path / "lock").open("rb")
        except OSError as exc:
            raise CheckpointError(f"Cannot read parent checkpoint: {exc}") from exc
        with lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CheckpointError("Cannot fork a checkpoint owned by a running executor") from exc
            parent.manifest = _load(parent.path / "manifest.json")
            parent._validate_manifest()
            parent._read_records()
            _, previous = parent._validated_attempts()
            if previous is None or previous["status"] == "running":
                raise CheckpointError("Fork parent needs a committed terminal attempt")
            parent._reconstruction_cursor()
            if through >= len(parent.records):
                raise CheckpointError("Fork boundary is outside committed history")
            boundary = parent.records[through]
            if (boundary["frame_type"] != "run_event" or boundary["body"].get("event_kind") != "unit_completed"
                    or boundary["body"]["payload"].get("controller_step", {}).get("stop") is not False):
                raise CheckpointError("Fork requires a completed, continuing unit boundary")
            current, old = implementation_identity(), parent.manifest["implementation"]
            if any(current[name] != old.get(name) for name in ("python", "platform")):
                raise CheckpointError("Compatible forks cannot change Python or platform")
            changes = {name: {"old": old["files"].get(name), "new": current["files"].get(name)}
                       for name in old["files"].keys() | current["files"].keys()
                       if old["files"].get(name) != current["files"].get(name)}
            if set(allowed_changes) != set(changes):
                raise CheckpointError(f"Fork changed-file admission must exactly name {sorted(changes)!r}")
            prefix = parent.records[:through + 1]
            source_terminal = parent._terminal_ref()
            historical = parent.manifest.get("fork", {}).get("historical_run_implementation", old["files"])
            provenance = {"kind": "verified_prefix_experiment", "fork_id": "fork_" + uuid.uuid4().hex,
                "reason": reason, "parent_path": str(parent.path),
                "parent_logical_run_id": parent.manifest["logical_run_id"],
                "parent_manifest_sha256": _digest(parent.manifest),
                "parent_registration_hash": previous["registration_hash"],
                "parent_attempt_sha256": _digest(previous), "parent_terminal": source_terminal,
                "through_record": through, "prefix_sha256": boundary["sha256"],
                "old_implementation": old, "new_implementation": current,
                "admitted_file_changes": dict(sorted(changes.items())),
                "historical_run_implementation": historical,
                "identity_semantics": "Distinct experiment fork; original episode namespace retained only for exact reconstruction."}
            manifest = {"schema": cls.SCHEMA, "inputs": parent.inputs,
                "implementation": current, "status": "active",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "provider_configuration": parent.manifest["provider_configuration"],
                "logical_run_id": parent.manifest["logical_run_id"], "fork": provenance}
        return manifest, prefix

    def run_start_implementation(self, current):
        """Only the historical start event retains its original code identity."""
        return self.manifest.get("fork", {}).get("historical_run_implementation", current)

    def provenance(self):
        return {"checkpoint_path": str(self.path), "logical_run_id": self.manifest["logical_run_id"],
                "current_implementation": self.manifest["implementation"],
                "fork": self.manifest.get("fork")}

    def _read_records(self):
        previous = ""
        for index, path in enumerate(sorted((self.path / "records").glob("*.json"))):
            if path.name != f"{index:012d}.json":
                raise CheckpointError("Checkpoint record sequence has a gap or invalid filename")
            record = _load(path)
            if not isinstance(record, dict):
                raise CheckpointError(f"Invalid checkpoint record {index}")
            body = {key: value for key, value in record.items() if key != "sha256"}
            if (record.get("index") != index or record.get("previous") != previous
                    or record.get("kind") not in {"call", "event"}
                    or record.get("sha256") != _digest(body)):
                raise CheckpointError(f"Checkpoint integrity failure at record {index}")
            previous = record["sha256"]
            if (record.get("frame_type") not in {"trace_event", "run_event", "model_request", "http_request"}
                    or not isinstance(record.get("body"), dict)
                    or not isinstance(record.get("request_sha256"), str)
                    or (record["kind"] == "call" and not {"value", "error"} <= set(record))):
                raise CheckpointError(f"Invalid checkpoint frame at record {index}")
            self.records.append(record)
        pending_path = self.path / "pending.json"
        if pending_path.exists():
            pending = _load(pending_path)
            if (not isinstance(pending, dict) or type(pending.get("index")) is not int
                    or pending.get("kind") != "call"
                    or pending.get("frame_type") not in {"model_request", "http_request"}
                    or not isinstance(pending.get("body"), dict)):
                raise CheckpointError("Invalid pending checkpoint request")
            index = pending["index"]
            if index == len(self.records):
                self.pending = pending
            elif (index != len(self.records) - 1 or index < 0
                  or any(pending.get(key) != self.records[index].get(key)
                         for key in ("kind", "operation", "request_sha256", "frame_type", "body"))):
                raise CheckpointError("Pending checkpoint request does not match committed history")
            # A committed response followed by a crash before unlink is complete.
        if self.manifest["status"] == "complete" and (
                self.manifest.get("records") != len(self.records) or self.pending is not None):
            raise CheckpointError("Completed checkpoint is missing committed history")
        fork = self.manifest.get("fork")
        if fork is not None:
            through = fork["through_record"]
            if through >= len(self.records) or self.records[through]["sha256"] != fork["prefix_sha256"]:
                raise CheckpointError("Fork no longer contains its admitted immutable prefix")

    def _registration_hash(self):
        return _digest({key: self.manifest[key] for key in
            ("schema", "inputs", "implementation", "logical_run_id", "fork") if key in self.manifest})

    def _validated_attempts(self):
        directory = self.path / "attempts"
        paths = sorted(directory.glob("*.json"))
        registration_hash = self._registration_hash()
        previous = None
        for index, path in enumerate(paths):
            attempt = _load(path)
            reference = None if previous is None else {"attempt": previous["attempt"],
                "sha256": _digest(previous), "terminal": previous.get("terminal")}
            if (path.name != f"{index:012d}.json" or not isinstance(attempt, dict)
                    or attempt.get("attempt") != index
                    or attempt.get("logical_run_id") != self.manifest["logical_run_id"]
                    or attempt.get("registration_hash") != registration_hash
                    or attempt.get("resume_from") != reference
                    or attempt.get("status") not in {"running", "interrupted", "complete"}
                    or (index < len(paths) - 1 and attempt["status"] == "running")):
                raise CheckpointError("Checkpoint physical-attempt lineage is inconsistent")
            terminal = attempt.get("terminal")
            if attempt["status"] != "running":
                if not isinstance(terminal, dict) or type(terminal.get("through_record")) is not int:
                    raise CheckpointError("Checkpoint attempt lacks terminal evidence")
                through = terminal["through_record"]
                if (through < -1 or through >= len(self.records)
                        or terminal.get("through_hash") != (self.records[through]["sha256"] if through >= 0 else "")):
                    raise CheckpointError("Checkpoint terminal reference differs from committed history")
                if terminal.get("pending_hash") is not None:
                    pending = self.pending
                    if through + 1 < len(self.records):
                        record = self.records[through + 1]
                        pending = {key: record[key] for key in (
                            "index", "kind", "operation", "request_sha256", "frame_type", "body")}
                    if pending is None or _digest(pending) != terminal["pending_hash"]:
                        raise CheckpointError("Checkpoint terminal pending request differs from continuation")
            previous = attempt
        return paths, previous

    def _begin_attempt(self, resume):
        directory = self.path / "attempts"
        directory.mkdir(mode=0o700, exist_ok=True)
        paths, previous = self._validated_attempts()
        registration_hash = self._registration_hash()
        if previous and previous.get("status") == "running":
            # Holding the exclusive lock proves no prior executor still owns this run.
            previous.update(status="interrupted", reason="exclusive_lock_reacquired_after_process_exit",
                            terminal=self._terminal_ref())
            _atomic(paths[-1], previous)
        reference = None if previous is None else {"attempt": previous["attempt"],
            "sha256": _digest(previous), "terminal": previous.get("terminal")}
        self.attempt = {"attempt": len(paths), "logical_run_id": self.manifest["logical_run_id"],
            "status": "running", "resume_from": reference,
            "registration_hash": registration_hash,
            "started_at": datetime.now(timezone.utc).isoformat()}
        _atomic(directory / f"{len(paths):012d}.json", self.attempt)

    def _terminal_ref(self):
        return {"through_record": len(self.records) - 1,
                "through_hash": self.records[-1]["sha256"] if self.records else "",
                "pending_hash": _digest(self.pending) if self.pending is not None else None}

    def _frame(self, kind, operation, request):
        if kind == "event":
            return SimpleNamespace(frame_type="trace_event", body={"sha256": _digest(request),
                                                                  "operation": operation})
        episode = self.current_episode
        return SimpleNamespace(frame_type="model_request" if operation.startswith("model.") else "http_request",
            body={"episode_id": episode.episode_id if episode else None,
                  "episode_path": [{"grain": grain, "key": key} for grain, key in episode.path] if episode else [],
                  **({"preflight": True} if episode is None else {}),
                  "request_id": f"request_{self.position}",
                  "request": {"operation": operation, "sha256": _digest(request)}})

    def _reconstruction_cursor(self):
        from .method_loop.reconstruction import ExpectedFrame, RecordedReply, ReconstructionCursor
        frames = []
        for record in self.records:
            reply = None
            if record["kind"] == "call":
                reply = RecordedReply(record["frame_type"].replace("_request", "_response"),
                    {"request_id": record["body"]["request_id"],
                     "response": {"value": record["value"], "error": record["error"]}},
                    {"index": record["index"], "sha256": record["sha256"]})
            frames.append(ExpectedFrame(record["frame_type"], record["body"],
                {"index": record["index"], "sha256": record["sha256"]}, reply))
        if self.pending is not None:
            frames.append(ExpectedFrame(self.pending["frame_type"], self.pending["body"],
                {"index": self.pending["index"], "sha256": _digest(self.pending)}))
        self.cursor = ReconstructionCursor(frames, self.manifest["logical_run_id"], allow_preflight=True)

    def _activate(self):
        if self.cursor is None or not self.cursor.prefix_verified:
            raise CheckpointError("Live continuation requires a fully verified reconstruction prefix")
        if not callable(self.authorize):
            raise CheckpointError("Saved prefix matched; live work has not been authorized by the host")
        if self.manifest["implementation"] != implementation_identity():
            raise CheckpointError("Implementation changed during reconstruction; live continuation denied")
        admission = self.authorize()
        if not isinstance(admission, dict) or admission.get("authorized") is not True:
            raise CheckpointError("Host denied continuation")
        self.attempt["run_reconstructed"] = {"resume_from": self.attempt["resume_from"],
            "reconstruction": self.cursor.report(), "activation_admission": admission,
            "fork": self.manifest.get("fork"), "implementation": self.manifest["implementation"]}
        _atomic(self.path / "attempts" / f"{self.attempt['attempt']:012d}.json", self.attempt)
        self.activated = True

    def configure(self, identity, *, secrets=()):
        self.secrets = tuple(value for value in secrets if value and value != "no-auth")
        signature = _digest(identity)
        previous = self.manifest["provider_configuration"]
        if previous is not None and previous != signature:
            raise CheckpointError("Checkpoint provider/model configuration differs from this invocation")
        if previous is None:
            self.manifest["provider_configuration"] = signature
            _atomic(self.path / "manifest.json", self.manifest)

    def _clean(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[redacted]")
            return value
        if isinstance(value, dict):
            return {self._clean(key): self._clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._clean(item) for item in value]
        return value

    def _match(self, kind, operation, request):
        return self._match_frame(self._frame(kind, operation, request))

    def _match_frame(self, frame):
        if not self.activated:
            if self.cursor.prefix_verified:
                self._activate()
            else:
                reply = self.cursor.accept(frame)
                historical = self.position < self.replay_count
                record = self.records[self.position] if historical else None
                if historical:
                    self.position += 1
                    self._historical_effects = True
                if self.cursor.prefix_verified:
                    self._activate()
                if historical:
                    return record
                if reply is not None:
                    raise CheckpointError("Pending request unexpectedly has a committed reply")
        if self.manifest["status"] == "complete":
            raise CheckpointError("Completed checkpoint attempted additional work")
        if self.pending is not None:
            if self.pending["frame_type"] != frame.frame_type or _digest(self.pending["body"]) != _digest(frame.body):
                raise CheckpointError("Replay did not reach the interrupted request")
        return None

    def _append(self, kind, operation, request, *, frame=None, **result):
        self._historical_effects = False
        frame = frame or self._frame(kind, operation, request)
        if frame.frame_type == "trace_event":
            request_sha256 = frame.body["sha256"]
        elif frame.frame_type in {"model_request", "http_request"}:
            request_sha256 = frame.body["request"]["sha256"]
        else:
            request_sha256 = _digest(request)
        body = {"index": len(self.records), "previous": self.records[-1]["sha256"] if self.records else "",
                "kind": kind, "operation": operation, "request_sha256": request_sha256,
                "frame_type": frame.frame_type, "body": frame.body, **result}
        record = {**body, "sha256": _digest(body)}
        _atomic(self.path / "records" / f"{len(self.records):012d}.json", record)
        self.records.append(record)
        self.position += 1
        self.pending = None

    def _begin(self, operation, request):
        self._historical_effects = False
        self._inflight = True
        frame = self._frame("call", operation, request)
        self.pending = {"index": len(self.records), "kind": "call", "operation": operation,
                        "request_sha256": frame.body["request"]["sha256"],
                        "frame_type": frame.frame_type, "body": frame.body}
        _atomic(self.path / "pending.json", self.pending)

    def _finish(self, operation, request, value=None, error=None):
        value = json.loads(_json(self._clean(value)))
        self._append("call", operation, request, value=value, error=error)
        try:
            (self.path / "pending.json").unlink(missing_ok=True)
        except OSError as exc:
            raise CheckpointError(f"Cannot finalize committed checkpoint response: {exc}") from exc
        self._inflight = False
        if error is not None:
            self._raise(error)
        return value

    def _error(self, exc):
        allowed = {ProviderError: "ProviderError", ValueError: "ValueError", TypeError: "TypeError",
                   OSError: "OSError", TimeoutError: "TimeoutError", ConnectionError: "ConnectionError"}
        name = allowed.get(type(exc))
        if name is None:
            raise CheckpointError(f"Unrecordable provider exception {type(exc).__name__}; request remains pending") from exc
        return {"type": name, "args": self._clean(list(exc.args))}

    @staticmethod
    def _raise(error):
        types = {"ProviderError": ProviderError, "ValueError": ValueError, "TypeError": TypeError,
                 "OSError": OSError, "TimeoutError": TimeoutError, "ConnectionError": ConnectionError}
        if not isinstance(error, dict) or error.get("type") not in types or not isinstance(error.get("args"), list):
            raise CheckpointError("Invalid checkpoint exception record")
        raise types[error["type"]](*error["args"])

    def call(self, operation, request, invoke):
        if self._inflight:
            raise CheckpointError("Overlapping external calls require a different reconstruction contract")
        record = self._match("call", operation, request)
        if record is not None:
            if record.get("error") is not None:
                self._raise(record["error"])
            return record["value"]
        self._begin(operation, request)
        try:
            value = invoke()
        except Exception as exc:
            return self._finish(operation, request, error=self._error(exc))
        return self._finish(operation, request, value)

    async def call_async(self, operation, request, invoke):
        if self._inflight:
            raise CheckpointError("Overlapping external calls require a different reconstruction contract")
        record = self._match("call", operation, request)
        if record is not None:
            if record.get("error") is not None:
                self._raise(record["error"])
            return record["value"]
        self._begin(operation, request)
        try:
            value = await invoke()
        except Exception as exc:
            return self._finish(operation, request, error=self._error(exc))
        return self._finish(operation, request, value)

    def event(self, value, sink=None):
        operation = value.get("event", "event")
        frame = self._frame("event", operation, value)
        if self._match_frame(frame) is not None:
            return
        self._append("event", operation, value, frame=frame)
        if sink is not None:
            sink(value)

    def boundary(self, kind, episode, payload):
        frame = SimpleNamespace(frame_type="run_event", body={"episode_id": episode.episode_id,
            "event_kind": kind, "payload": _record(payload)})
        if self._match_frame(frame) is not None:
            return
        self._append("event", kind, frame.body, frame=frame)

    @contextmanager
    def episode_scope(self, episode):
        previous = self.current_episode
        self.current_episode = episode
        try:
            yield
        finally:
            self.current_episode = previous

    def instrument(self, episode, state):
        initial = _digest(_record(state()))
        prior_on_unit, prior_build_result = episode.on_unit, episode.build_result

        def on_unit(item, contribution, view):
            if prior_on_unit is not None:
                prior_on_unit(item, contribution, view)
            self.boundary("unit_completed", self.current_episode,
                {**_record(view), "goal_state_id": _digest(_record(state()))})

        def build_result(record):
            with self.episode_scope(record.episode_ref):
                return prior_build_result(record)

        return _InstrumentedEpisode(**{field.name: getattr(episode, field.name) for field in fields(Episode)
                                      if field.name not in {"on_unit", "build_result", "archive_record"}},
            on_unit=on_unit, build_result=build_result, archive_record=self.archive.store, checkpoint=self,
            runtime_boundary=lambda: {"goal_state_id": _digest(_record(state())),
                "initial_goal_state_id": initial, "goal_view": episode.goal.as_record()})

    def now(self):
        return self.call("document_timestamp", {}, lambda: datetime.now(timezone.utc).isoformat())

    def complete(self):
        if self.position != len(self.records) or self.pending is not None:
            raise CheckpointError("Workflow ended before consuming its checkpoint history")
        if self.cursor is not None and not self.cursor.prefix_verified:
            raise CheckpointError("Workflow ended before verifying its reconstruction prefix")
        self.manifest["status"] = "complete"
        self.manifest["records"] = len(self.records)
        _atomic(self.path / "manifest.json", self.manifest)
        self.attempt.update(status="complete", terminal=self._terminal_ref())
        _atomic(self.path / "attempts" / f"{self.attempt['attempt']:012d}.json", self.attempt)

    def close(self):
        if self._lock is not None:
            try:
                if self.attempt is not None and self.attempt["status"] == "running":
                    self.attempt.update(status="interrupted", terminal=self._terminal_ref())
                    _atomic(self.path / "attempts" / f"{self.attempt['attempt']:012d}.json", self.attempt)
            finally:
                self._lock.close()
                self._lock = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class _InstrumentedEpisode(Episode):
    """Adapted from OpenChia's linker: identical lifecycle boundary placement."""

    checkpoint: object = None
    runtime_boundary: object = None

    def run(self, ctx):
        if not ctx.has_run_id:
            ctx.bind_run_id(self.checkpoint.manifest["logical_run_id"])
        ref = EpisodeRef(ctx.require_run_id(), (*ctx.path, (self.grain.name, self.key)))
        with self.checkpoint.episode_scope(ref):
            self.checkpoint.boundary("episode_started", ref, {
                "grain": self.grain.name, "key": self.key,
                "episode_path": [{"grain": grain, "key": key} for grain, key in ref.path],
                "request": self.request.as_record(), **self.runtime_boundary()})
            record = super().run(ctx)
            self.checkpoint.boundary("episode_completed", ref, {
                "ended_by": record.ended_by, "end_reason": record.end_reason,
                "units_consumed": record.units_consumed,
                "controller_state": _record(record.controller_state)})
            if len(ref.path) == 1:
                record = record.archived(self.checkpoint.archive.store)
                ctx._child_updates.pop(record.path, None)
            return record

    async def run_async(self, ctx):
        raise CheckpointError("This workflow checkpoint binding supports the serial synchronous Episode entry point only")


class JournaledClient:
    """Keep provider internals/credentials outside the durable journal."""

    def __init__(self, client, journal, name):
        self.client, self.journal, self.name = client, journal, name

    def __getattr__(self, name):
        return getattr(self.client, name)

    def call(self, prompt, system_prompt=None):
        return self.journal.call(self.name + ".call", {"prompt": prompt, "system_prompt": system_prompt},
            lambda: self.client.call(prompt, system_prompt=system_prompt))

    async def noul(self, **kwargs):
        return await self.journal.call_async(self.name + ".noul", kwargs,
            lambda: self.client.noul(**kwargs))

    async def nouls(self, **kwargs):
        return await self.journal.call_async(self.name + ".nouls", kwargs,
            lambda: self.client.nouls(**kwargs))
