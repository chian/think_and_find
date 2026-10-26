"""Verify imports and execution do not depend on adjacent repositories."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def runtime_files():
    yield from ROOT.glob("*.py")
    for folder in ("case_reports", "credit_assignment", "method_loop", "task_executors"):
        yield from (ROOT / folder).rglob("*.py")


class StandaloneTests(unittest.TestCase):
    def test_all_runtime_imports_are_local_or_standard_library(self):
        allowed = sys.stdlib_module_names | {"episodic_workflow"}
        for path in runtime_files():
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module]
                for name in names:
                    self.assertIn(name.split(".")[0], allowed, f"Nonlocal dependency in {path.relative_to(ROOT)}: {name}")

    def test_every_runtime_module_imports_without_changing_sys_path(self):
        before = list(sys.path)
        for path in runtime_files():
            parts = list(path.relative_to(ROOT).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            module_name = ".".join(["episodic_workflow", *parts])
            module = importlib.import_module(module_name)
            self.assertTrue(Path(module.__file__).resolve().is_relative_to(ROOT))
        self.assertEqual(sys.path, before)

    def test_checkout_runs_alone_without_site_packages(self):
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder) / "isolated-checkout"
            shutil.copytree(ROOT, checkout, ignore=shutil.ignore_patterns(".venv*", ".git", ".agents", ".codex", ".aws", ".artifacts", ".env", ".env.*", "__pycache__"))
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", str(checkout), "--help"],
                cwd=folder,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--request-file", result.stdout)
            self.assertNotIn("--target-scenario", result.stdout)

    def test_isolated_checkout_runs_offline_provider_suite(self):
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder) / "isolated-checkout"
            shutil.copytree(ROOT, checkout, ignore=shutil.ignore_patterns(".venv*", ".git", ".agents", ".codex", ".aws", ".artifacts", ".env", ".env.*", "__pycache__", "test_standalone.py"))
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-B", str(checkout / "scripts/check.py")],
                cwd=folder,
                capture_output=True,
                text=True,
                timeout=30,
                env={"EPISODIC_RUN_LIVE_TESTS": "0"},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("OK (skipped=1)", result.stderr)
