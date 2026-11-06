"""Supervise one Think and Find workload inside verified CPU/RAM bounds.

Only the standard library is imported here. The shell launcher places this
supervisor inside the owned systemd service before application imports begin.
Episode continuation belongs to the application's numerical iterator.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


RESOURCES = {"cpu_percent": 100, "memory_bytes": 5 * 1024**3, "swap_bytes": 0}


def control(*arguments):
    return subprocess.run(
        ["systemctl", "--user", *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


def cgroup():
    return next(
        line.split(":", 2)[2]
        for line in Path("/proc/self/cgroup").read_text().splitlines()
        if line.startswith("0::")
    )


def members(path):
    found = set()
    for file in path.rglob("cgroup.procs"):
        try:
            found.update(int(pid) for pid in file.read_text().split())
        except FileNotFoundError:
            pass
    return found


def verify_resources(slice_name, service):
    group = control("show", slice_name, "--property=ControlGroup", "--value")
    if not group or cgroup() != group + "/" + service:
        raise ValueError("Think and Find supervisor is outside its owned service")
    path = Path("/sys/fs/cgroup") / group.lstrip("/")
    cpu = (path / "cpu.max").read_text().split()
    if (
        len(cpu) != 2
        or cpu[0] == "max"
        or int(cpu[0]) * 100 != RESOURCES["cpu_percent"] * int(cpu[1])
        or int((path / "memory.max").read_text()) != RESOURCES["memory_bytes"]
        or int((path / "memory.swap.max").read_text()) != RESOURCES["swap_bytes"]
        or (path / "pids.max").read_text().strip() != "max"
    ):
        raise ValueError("Think and Find workload resource boundary is unverified")
    if members(path) != {os.getpid()}:
        raise ValueError("Another workload process is still present in this window")
    return group, path


def supervise(directory, command):
    window = os.environ.get("CODEX_THREAD_ID", "")
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", window):
        raise ValueError("A stable CODEX_THREAD_ID is required")
    root = Path(__file__).resolve().parents[1]
    directory = Path(directory).resolve()
    if not command or not directory.is_relative_to(root / "runs"):
        raise ValueError("A command and a directory beneath think_and_find/runs/ are required")
    key = window.replace("-", "")
    service = f"think-and-find-workflow-{key}.service"
    slice_name = f"thinkandfindwindow{key}.slice"
    control(
        "set-property", "--runtime", slice_name, "CPUQuota=100%",
        f"MemoryMax={RESOURCES['memory_bytes']}", "MemorySwapMax=0", "TasksMax=infinity",
    )
    group, path = verify_resources(slice_name, service)
    receipt = {
        "application": "think_and_find", "window_id": window,
        "service": service, "slice": slice_name, "slice_cgroup": group,
        "supervisor_pid": os.getpid(), "run_directory": str(directory),
        "resources": RESOURCES, "command": command,
    }
    resource_file = directory / "workflow-resources.json"
    resource_file.write_text(json.dumps(receipt, indent=2) + "\n")
    env = dict(os.environ)
    env.update(
        THINK_AND_FIND_WORKFLOW_RESOURCES=str(resource_file),
        PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(directory / "tmp"),
        TMP=str(directory / "tmp"), TEMP=str(directory / "tmp"),
        XDG_CACHE_HOME=str(directory / "cache"),
    )
    for name in ("tmp", "cache"):
        (directory / name).mkdir(exist_ok=True)

    def interrupted(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = time.monotonic()
    process = None
    failure = None
    code = 1
    try:
        process = subprocess.Popen(command, cwd=root, env=env)
        code = process.wait()
        # Detached descendants remain in the same cgroup. Keep the service and
        # window lock until they exit, even after the application driver returns.
        while members(path) - {os.getpid()}:
            time.sleep(0.2)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        code = 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        result = {
            **receipt, "returncode": code, "failure": failure,
            "driver_returncode": process.returncode if process is not None else None,
            "elapsed_seconds": time.monotonic() - started,
            "descendants_exited": not (members(path) - {os.getpid()}),
        }
        (directory / "workflow-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return code


if __name__ == "__main__":
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("Usage: workflow_resources.py runs/<unique-run> -- command arguments...")
    raise SystemExit(supervise(sys.argv[1], sys.argv[3:]))
