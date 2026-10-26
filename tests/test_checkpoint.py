"""Deterministic continuation infrastructure checks, never workflow evidence."""

import asyncio
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch

from episodic_workflow.checkpoint import CheckpointError, CheckpointJournal, _record, implementation_identity
from episodic_workflow.method_loop import (
    ChildEpisodeUnit, ClosedRecord, Context, Episode, EpisodeGoal, EpisodeRequest,
    EpisodeTree, Grain, leaves,
)
from episodic_workflow.method_loop.communication import ReportContract
from episodic_workflow.method_loop.reconstruction import ReconstructionError
from episodic_workflow.task_executors._http import ProviderError
from episodic_workflow.task_executors.evidence_extraction.argo import ModelOutputError, parse_json_object


class Interrupted(BaseException):
    pass


@dataclass(frozen=True)
class Record(ClosedRecord):
    value: int

    def as_record(self):
        return {"value": self.value}


@dataclass(frozen=True)
class Step:
    stop: bool = False
    requests_transition: bool = False


@dataclass(frozen=True)
class State:
    history: tuple
    stop: bool = False


class Controller:
    epoch = "one"

    def __init__(self, _):
        self.history = []

    def observe(self, label, value, *, is_root=False):
        self.history.append((label, value.value))
        return Step()

    def state(self):
        return State(tuple(self.history))

    def transitioned(self, epoch):
        result = Controller(None)
        result.epoch = epoch
        return result


def nested_run(journal, *, interrupt=None, operations=None):
    """Small ordinary method_loop graph, with no scientific/provider adapters."""
    operations = [] if operations is None else operations
    shared = {"values": [], "cursor": 0, "coverage": [], "contracts": [], "calls": 0}

    def operation(name, response):
        shared["calls"] += 1
        def invoke():
            operations.append(name)
            if name == interrupt:
                raise Interrupted()
            return response
        return journal.call("model." + name, {"state": shared}, invoke)

    root_grain, child_grain = (Grain(name, "unit", "result", Controller) for name in ("parent", "reader"))
    tree = EpisodeTree(root_grain, {root_grain: (child_grain,)})
    goal = EpisodeGoal.root(objective="deterministic continuation check", result_contract={"value": "integer"})
    contract = ReportContract("continue", {}, {"trend": "brief result"})

    class Parent:
        launched = False

        def next(self, view):
            if self.launched:
                return None
            operation("choose", {"child": "reader"})
            self.launched = True
            child_goal = EpisodeGoal.child(goal, objective="read two units", result_contract={"value": "integer"})
            shared["contracts"].append(contract.as_record())

            def extract(index):
                raw = operation(f"extract-{index}", '{"value": ' + str(index + 1) + '}')
                value = parse_json_object(raw)["value"]
                operation(f"review-{index}", {"accepted": True})
                shared["values"].append(value)
                shared["cursor"] = index + 1
                shared["coverage"].append([index * 10, (index + 1) * 10])
                return value

            def result(record):
                operation("child-result", {"total": sum(shared["values"])})
                return Record(sum(shared["values"]))

            child = journal.instrument(Episode(child_grain, "reader:1",
                leaves(iter([0, 1]), extract, lambda _, value: Record(value), label=str),
                EpisodeRequest(child_goal, Record(0), contract), result), lambda: shared)
            return ChildEpisodeUnit(child, lambda value, *_: value,
                                    lambda *_: {"trend": "two units retained"})

    episode = journal.instrument(Episode(root_grain, "parent:1", Parent(),
        EpisodeRequest(goal, Record(0)), lambda _: Record(sum(shared["values"]))), lambda: shared)
    record = episode.run(Context(tree=tree))
    return {"record": _record(record), "shared": shared}


def subprocess_case(path, mode):
    operations = []
    with CheckpointJournal(path, inputs={"purpose": "mechanism test"} if mode == "start" else None,
                           resume=mode != "start") as journal:
        journal.authorize = lambda: {"authorized": True, "mechanism": "test host"}
        try:
            result = nested_run(journal, interrupt="review-1" if mode == "start" else None,
                                operations=operations)
        except Interrupted:
            print(json.dumps({"operations": operations, "interrupted": True}))
            return
        journal.complete()
        print(json.dumps({"operations": operations, "result": result}))


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "checkpoint"

    def open(self, resume=False):
        result = CheckpointJournal(self.path, inputs=None if resume else {"purpose": "mechanism test"}, resume=resume)
        result.authorize = lambda: {"authorized": True, "mechanism": "test host"}
        return result

    def completed_parent(self):
        with self.open() as journal:
            nested_run(journal)
            journal.complete()
            through = next(row["index"] for row in journal.records
                           if row["operation"] == "unit_completed")
        return through

    def fork(self, through, **kwargs):
        return CheckpointJournal(self.path.parent / "fork", fork_from=self.path,
            through_record=through, fork_reason="Explicit mechanism-level compatibility check", **kwargs)

    def test_fork_reuses_exact_prefix_and_keeps_completed_parent_immutable(self):
        through = self.completed_parent()
        before = {str(p): p.read_bytes() for p in self.path.rglob("*") if p.is_file()}
        with self.fork(through) as fork:
            self.assertEqual(len(fork.cursor.report()["expected_active_stack"]), 2)
            self.assertEqual(fork.manifest["fork"]["through_record"], through)
            self.assertEqual(fork.manifest["fork"]["prefix_sha256"], fork.records[-1]["sha256"])
            fork.authorize = lambda: {"authorized": True, "mechanism": "explicit test fork"}
            operations = []
            result = nested_run(fork, operations=operations)
            self.assertEqual(operations, ["extract-1", "review-1", "child-result"])
            self.assertEqual(result["shared"]["values"], [1, 2])
            self.assertIn("fork", fork.attempt["run_reconstructed"])
            fork.complete()
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.path.rglob("*") if p.is_file()})

    def test_fork_offline_verification_then_resume_does_not_repeat_saved_calls(self):
        through = self.completed_parent()
        with self.fork(through) as fork:
            operations = []
            with self.assertRaisesRegex(CheckpointError, "authorized"):
                nested_run(fork, operations=operations)
            self.assertEqual(operations, [])
            self.assertTrue(fork.cursor.prefix_verified)
            path = fork.path
        with CheckpointJournal(path, resume=True) as fork:
            fork.authorize = lambda: {"authorized": True}
            operations = []
            nested_run(fork, operations=operations)
            self.assertEqual(operations, ["extract-1", "review-1", "child-result"])

    def test_fork_requires_exact_declared_code_changes_and_same_python(self):
        through = self.completed_parent()
        actual = implementation_identity()
        altered = {**actual, "files": {**actual["files"], "checkpoint.py": "changed-mechanism-fixture"}}
        with patch("episodic_workflow.checkpoint.implementation_identity", return_value=altered):
            with self.assertRaisesRegex(CheckpointError, "changed-file admission"):
                self.fork(through)
            with self.fork(through, allowed_changes=("checkpoint.py",)) as fork:
                self.assertEqual(fork.manifest["fork"]["old_implementation"], actual)
                self.assertEqual(fork.manifest["fork"]["new_implementation"], altered)
                self.assertEqual(fork.run_start_implementation(altered["files"]), actual["files"])
        with self.assertRaisesRegex(CheckpointError, "implementation"):
            CheckpointJournal(self.path.parent / "fork", resume=True)
        with patch("episodic_workflow.checkpoint.implementation_identity", return_value={**actual, "python": "different"}):
            with self.assertRaisesRegex(CheckpointError, "Python or platform"):
                self.fork(through)

    def test_fork_rejects_nonunit_boundary_and_live_parent(self):
        through = self.completed_parent()
        with self.assertRaisesRegex(CheckpointError, "unit boundary"):
            self.fork(0)
        with self.open(True):
            with self.assertRaisesRegex(CheckpointError, "running executor"):
                self.fork(through)

    def test_fork_divergence_never_falls_through_to_live(self):
        through = self.completed_parent()
        with self.fork(through) as fork:
            fork.authorize = lambda: {"authorized": True}
            calls = []
            with self.assertRaises(ReconstructionError):
                fork.call("model.wrong", {}, lambda: calls.append(True))
            self.assertFalse(calls)
            self.assertFalse(fork.activated)

    def test_fork_rejects_corrupt_parent_without_creating_destination(self):
        through = self.completed_parent()
        record = self.path / "records" / f"{through:012d}.json"
        record.write_text("{truncated")
        with self.assertRaises(CheckpointError):
            self.fork(through)
        self.assertFalse((self.path.parent / "fork").exists())

    def test_nested_partial_extraction_rebuilds_all_state_without_repeating_responses(self):
        first = []
        with self.open() as journal:
            with self.assertRaises(Interrupted):
                nested_run(journal, interrupt="review-1", operations=first)
        with self.open(True) as journal:
            self.assertEqual(len(journal.cursor.report()["expected_active_stack"]), 2)
            second = []
            resumed = nested_run(journal, operations=second)
            self.assertEqual(second, ["review-1", "child-result"])
            self.assertEqual(resumed["shared"]["cursor"], 2)
            self.assertEqual(resumed["shared"]["coverage"], [[0, 10], [10, 20]])
            self.assertEqual(resumed["shared"]["values"], [1, 2])
            self.assertEqual(resumed["shared"]["calls"], 6)
            self.assertEqual(resumed["record"]["units_consumed"], 1)
            self.assertEqual(resumed["record"]["units"][0]["child"]["units_consumed"], 2)
            journal.complete()
        with self.open(True) as journal:
            repeated = []
            self.assertEqual(nested_run(journal, operations=repeated), resumed)
            self.assertEqual(repeated, [])
            journal.complete()

    def test_fresh_process_different_hash_seed(self):
        command = [sys.executable, "-c", "from episodic_workflow.tests.test_checkpoint import subprocess_case; "
                   "import sys; subprocess_case(sys.argv[1], sys.argv[2])", str(self.path)]
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
        first = subprocess.run([*command, "start"], env={**env, "PYTHONHASHSEED": "1"},
                               text=True, capture_output=True, check=True)
        self.assertTrue(json.loads(first.stdout)["interrupted"])
        second = subprocess.run([*command, "resume"], env={**env, "PYTHONHASHSEED": "39"},
                                text=True, capture_output=True, check=True)
        data = json.loads(second.stdout)
        self.assertEqual(data["operations"], ["review-1", "child-result"])
        third = subprocess.run([*command, "resume"], env={**env, "PYTHONHASHSEED": "4"},
                               text=True, capture_output=True, check=True)
        self.assertEqual(json.loads(third.stdout)["result"], data["result"])
        self.assertEqual(json.loads(third.stdout)["operations"], [])

    def test_divergence_is_fatal_before_live_call(self):
        with self.open() as journal:
            journal.call("model.one", {"question": "original"}, lambda: "saved")
        with self.open(True) as journal:
            invoked = []
            with self.assertRaises(ReconstructionError):
                journal.call("model.one", {"question": "different"}, lambda: invoked.append(True))
            self.assertEqual(invoked, [])

    def test_saved_prefix_needs_separate_host_authorization(self):
        with self.open() as journal:
            journal.call("model.one", {}, lambda: "saved")
        with self.open(True) as journal:
            journal.authorize = None
            with self.assertRaisesRegex(CheckpointError, "authorized"):
                journal.call("model.one", {}, lambda: self.fail("repeated"))

    def test_failures_and_malformed_raw_output_replay_exactly(self):
        for resumed in (False, True):
            with self.open(resumed) as journal:
                def failure():
                    if resumed:
                        self.fail("repeated failure")
                    raise ProviderError("recorded failure")
                with self.assertRaisesRegex(ProviderError, "recorded failure"):
                    journal.call("model.failure", {}, failure)
                raw = journal.call("model.malformed", {}, lambda: self.fail("repeated") if resumed else "{bad")
                with self.assertRaises(ModelOutputError) as error:
                    parse_json_object(raw)
                self.assertEqual(error.exception.raw_response, "{bad")

    def test_async_assessment_response_is_replayed(self):
        for resumed in (False, True):
            with self.open(resumed) as journal:
                async def answer():
                    if resumed:
                        self.fail("repeated async operation")
                    return {"one": .7}
                self.assertEqual(asyncio.run(journal.call_async("jev.nouls", {}, answer)), {"one": .7})

    def test_locks_released_on_interruption(self):
        with self.open():
            with self.assertRaisesRegex(CheckpointError, "already in use"):
                self.open(True)
        with self.open(True) as journal:
            self.assertEqual(journal.attempt["resume_from"]["attempt"], 0)

    def test_corrupt_committed_record_rejected_but_abandoned_temp_ignored(self):
        with self.open() as journal:
            journal.call("model.one", {}, lambda: "saved")
        temporary = self.path / "records" / ".write-interrupted"
        temporary.touch()
        with self.open(True):
            pass
        record = self.path / "records" / "000000000000.json"
        record.write_text('{"truncated":')
        with self.assertRaises(CheckpointError):
            self.open(True)

    def test_changed_inputs_and_provider_configuration_rejected(self):
        with self.open() as journal:
            journal.configure({"model": "one"}, secrets=("private-value",))
            self.assertEqual(journal.call("model.one", {}, lambda: "private-value"), "[redacted]")
        with self.assertRaisesRegex(CheckpointError, "inputs"):
            CheckpointJournal(self.path, inputs={"different": True}, resume=True)
        with self.open(True) as journal:
            with self.assertRaisesRegex(CheckpointError, "configuration"):
                journal.configure({"model": "two"})
        self.assertNotIn("private-value", "".join(path.read_text() for path in self.path.rglob("*.json")))

    def test_replayed_diagnostics_not_emitted_twice(self):
        events = []
        with self.open() as journal:
            journal.event({"event": "one", "value": 1}, events.append)
        with self.open(True) as journal:
            journal.event({"event": "one", "value": 1}, events.append)
            journal.event({"event": "two", "value": 2}, events.append)
        self.assertEqual([item["event"] for item in events], ["one", "two"])

    def test_unclean_process_exit_preserves_pending_request_and_attempt_lineage(self):
        code = ("from episodic_workflow.checkpoint import CheckpointJournal; import os,sys; "
                "j=CheckpointJournal(sys.argv[1],inputs={'purpose':'mechanism test'}); "
                "j.call('model.one',{},lambda: os._exit(44))")
        result = subprocess.run([sys.executable, "-c", code, str(self.path)],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])})
        self.assertEqual(result.returncode, 44)
        with self.open(True) as journal:
            self.assertIsNotNone(journal.pending)
            self.assertEqual(journal.call("model.one", {}, lambda: "continued"), "continued")
            self.assertIn("run_reconstructed", journal.attempt)
        previous = json.loads((self.path / "attempts" / "000000000000.json").read_text())
        self.assertEqual(previous["status"], "interrupted")
        self.assertEqual(previous["reason"], "exclusive_lock_reacquired_after_process_exit")

    def test_clock_value_is_preserved(self):
        with self.open() as journal:
            timestamp = journal.now()
        with self.open(True) as journal:
            self.assertEqual(journal.now(), timestamp)

    def test_cli_rejects_input_change_before_provider_construction(self):
        from episodic_workflow.cli import main
        from episodic_workflow.case_reports.research.scenarios import load_scenarios
        inputs = {"disease": "caller-specified agent", "scenario": "classroom", "search_limit": 10,
            "text_window_chars": 12000, "yield_window": 4, "metric": None, "research_request": "",
            "scenario_definitions": [case.as_record() for case in load_scenarios("classroom")]}
        with CheckpointJournal(self.path, inputs=inputs):
            pass
        with patch("episodic_workflow.cli.run_configured_safety_guideline_request",
                   side_effect=AssertionError("provider construction must not occur")):
            errors = StringIO()
            status = main(["different input", "--resume", str(self.path)], stdout=StringIO(), stderr=errors)
        self.assertEqual(status, 2)
        self.assertIn("inputs or scenario definitions differ", errors.getvalue())

    def test_result_writes_are_idempotent_only_when_identical(self):
        from episodic_workflow.cli import _write_result
        path = Path(self.directory.name) / "output.json"
        _write_result(path, "one", resume=False)
        before = path.stat().st_mtime_ns
        _write_result(path, "one", resume=True)
        self.assertEqual(path.stat().st_mtime_ns, before)
        with self.assertRaises(FileExistsError):
            _write_result(path, "different", resume=True)

    def test_committed_response_survives_stale_pending_file(self):
        with self.open() as journal:
            journal.call("model.one", {}, lambda: "saved")
            record = journal.records[0]
        pending = {key: record[key] for key in (
            "index", "kind", "operation", "request_sha256", "frame_type", "body")}
        (self.path / "pending.json").write_text(json.dumps(pending))
        with self.open(True) as journal:
            self.assertIsNone(journal.pending)
            self.assertEqual(journal.call("model.one", {}, lambda: self.fail("repeated committed call")), "saved")

    def test_completed_checkpoint_with_lost_tail_is_rejected(self):
        with self.open() as journal:
            journal.event({"event": "one"})
            journal.complete()
        (self.path / "records" / "000000000000.json").unlink()
        with self.assertRaisesRegex(CheckpointError, "missing committed history"):
            self.open(True)

    def test_overlapping_external_requests_rejected_before_second_invocation(self):
        with self.open() as journal:
            async def exercise():
                started, release = asyncio.Event(), asyncio.Event()
                async def first():
                    started.set()
                    await release.wait()
                    return "one"
                task = asyncio.create_task(journal.call_async("jev.first", {}, first))
                await started.wait()
                try:
                    with self.assertRaisesRegex(CheckpointError, "Overlapping"):
                        await journal.call_async("jev.second", {}, lambda: self.fail("second call started"))
                finally:
                    release.set()
                self.assertEqual(await task, "one")
            asyncio.run(exercise())

    def test_failed_atomic_publication_leaves_no_partial_output(self):
        from episodic_workflow.cli import _write_result
        path = Path(self.directory.name) / "output.json"
        with patch("episodic_workflow.cli.os.link", side_effect=OSError("interrupted publication")):
            with self.assertRaises(OSError):
                _write_result(path, "complete body", resume=False)
        self.assertFalse(path.exists())
        self.assertFalse(list(Path(self.directory.name).glob(".result-*")))
        _write_result(path, "complete body", resume=True)
        self.assertEqual(path.read_text(), "complete body")


if __name__ == "__main__":
    unittest.main()
