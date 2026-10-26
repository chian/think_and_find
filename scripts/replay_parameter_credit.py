"""Shadow-score saved acquisition observations; never run or seed a workflow."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--episode-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "episodic_workflow", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)

    from episodic_workflow.case_reports.research.progress import (
        PARAMETER_EVIDENCE, parameter_requirements, parameter_support, progress_credit)
    from episodic_workflow.case_reports.research.records import Credit, Fact, ResearchResult
    from episodic_workflow.credit_assignment.research import ResearchController

    requirements = parameter_requirements(args.parameter)

    def projection(records):
        facts = tuple(Fact(**record) for record in records
                      if record.get("kind") in {"source_observation", "source_model_parameter"})
        result = ResearchResult(facts=facts)
        return parameter_support(result, requirements)

    controller = None
    assessment = None
    results = []
    failures = []
    with args.trace.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "model_attempt":
                attempt = event["attempt"]
                if attempt["operation"] == "assess_information_progress":
                    assessment = attempt
            if event["event"] != "credit_verdict" or event["episode_id"] != args.episode_id:
                continue
            if assessment is None:
                raise ValueError("No recorded assessment snapshot for this observation")
            request = assessment["request"]
            if request["parent_contract"] != event["contract"]:
                raise ValueError("Assessment snapshot does not belong to the selected episode")
            if request.get("assumption_assessments"):
                raise ValueError("This replay only projects admitted acquisition records, not old assumption reviews")
            records = request["evidence"]
            if controller is None:
                inherited = [record for key, record in records.items()
                             if key not in request["new_evidence"]]
                controller = ResearchController((PARAMETER_EVIDENCE,), point_scores=True,
                    initial=progress_credit(projection(inherited), evaluated=True))
            rows = projection(records.values())
            credit = progress_credit(rows, evaluated=True)
            # The removed whole-question assessor cannot suppress already admitted records.
            verdict = controller.observe(event["unit"], credit, is_root=False)
            results.append({"unit": event["unit"], "recorded_model_calls": event["model_calls"],
                "old_score": event["score"], "old_evaluated": event["evaluated"],
                "shadow_score": verdict.score, "shadow_gain": verdict.gain,
                "shadow_stop": verdict.stop, "fields": {row["field"]: row["points"] for row in rows}})
            if not event["evaluated"]:
                failures.append(event)
            if projection([*records.values(), *records.values()]) != rows:
                raise AssertionError("Repeated admitted records changed credit")
    if controller is None:
        raise ValueError("Selected episode has no recorded observations")

    repeated = ResearchController((PARAMETER_EVIDENCE,), point_scores=True, initial=credit)
    for index in range(repeated.window):
        verdict = repeated.observe("repeated-recorded-state", credit, is_root=False)
        assert verdict.gain == 0
        assert verdict.stop == (index == repeated.window - 1)
    failed_check = "not exercised: no failed observation in selected episode"
    if failures:
        failed = ResearchController((PARAMETER_EVIDENCE,), point_scores=True, initial=credit)
        for index in range(failed.window):
            verdict = failed.observe(failures[0]["unit"], Credit(evaluated=False), is_root=False)
            assert verdict.gain == 0
            assert verdict.stop == (index == failed.window - 1)
        assert failed.state().failures == failed.window
        failed_check = "passed: repeated recorded failure yields after four zero-gain observations"
    print(json.dumps({"trace": str(args.trace), "episode_id": args.episode_id,
        "parameter": args.parameter, "requirements": sorted(requirements),
        "scope": "Numerical shadow replay of recorded admitted facts, not new agent behavior or scientific validation. Assumption review and contradiction classification require a fresh run.",
        "observations": results, "rarefaction": controller.state().rarefaction,
        "checks": {"duplicate_records": "passed", "zero_yield": "passed", "failed_yield": failed_check}},
        indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
