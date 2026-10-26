"""Real configured workflow run.

This test intentionally does not mock search, retrieval, LLM extraction, or
transport. Supply the original research request, not an externally filled
parameter set. Known-value calculator fixtures live in separate offline tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
from episodic_workflow.configured_providers import run_configured_safety_guideline_request
from episodic_workflow.cli import _json_record
from episodic_workflow.environment import workflow_environment
from episodic_workflow.case_reports.research.engine import TREE
from episodic_workflow.tests.report_assertions import assert_admitted_research_provenance


REQUIRED_ENV = (
    "EPISODIC_TEST_RESEARCH_REQUEST",
)


class RealConfiguredWorkflowTests(unittest.TestCase):
    def test_real_configured_safety_guideline_workflow(self) -> None:
        if not _bool_env("EPISODIC_RUN_LIVE_TESTS"):
            self.skipTest("Run scripts/check.py --live to enable live provider calls")
        env = workflow_environment()
        missing = tuple(name for name in REQUIRED_ENV if not env.get(name))
        if missing:
            self.fail(
                "Real configured workflow tests require environment values: "
                + ", ".join(missing)
            )

        record, report = run_configured_safety_guideline_request(research_request=env["EPISODIC_TEST_RESEARCH_REQUEST"], env=env)

        if env.get("EPISODIC_TEST_REPORT_PATH"):
            path = Path(env["EPISODIC_TEST_REPORT_PATH"])
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(_json_record(report.as_record()), stream, indent=2, sort_keys=True, allow_nan=False)
                stream.write("\n")

        data = report.as_record()
        self.assertTrue(report.passed, data["unresolved"][:12])
        self.assertEqual(record.scope_level, "evaluate_safety_guideline")
        self.assertEqual(data["episode_tree"], {k: list(v) for k, v in TREE.items() if v})
        self.assertTrue(data["calculations"])
        assert_admitted_research_provenance(self, data)


def _bool_env(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    unittest.main()
