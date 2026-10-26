"""Synthetic contract tests for the additive branch in the original workflow."""

import io
import json
import unittest
from unittest.mock import patch

from episodic_workflow.case_reports.assumption_investigation import AssumptionLimits, AssumptionServices
from episodic_workflow.case_reports.evidence_children import _run_admit_behavior_assumptions_episode
from episodic_workflow.case_reports.evidence_extraction import EvidenceSource
from episodic_workflow.case_reports.request_admission import interpret_guideline_request
from episodic_workflow.case_reports.safety_guideline import (
    GUIDELINE_CHILD_NAMES, GUIDELINE_OUTPUTS, run_evaluate_safety_guideline_episode,
)
from episodic_workflow.case_reports.research.assumptions import RUBRIC
from episodic_workflow.cli import main
from episodic_workflow.configured_providers import WorkflowEvidenceProviders
from episodic_workflow.tests.fixtures import FACTS, SCENARIO, SOURCE_TEXT, SOURCE_URL, PROVIDER_ENV, applicability_response


TEXT = "The participants were seated quietly and wore no face coverings."
SOURCE = EvidenceSource("https://example.test/qualitative", TEXT)
REQUEST = "Estimate cumulative exposure for synthetic-disease in a room, with risk tolerance 0.1 scalar."


class AssumptionAgent:
    def __init__(self, *, verdict="supported", invalid_action=False):
        self.requests = []
        self.verdict, self.invalid_action = verdict, invalid_action

    def __call__(self, request):
        self.requests.append(request)
        operation = request.get("operation")
        if operation == "interpret_guideline_request":
            return {"disease_name": "synthetic-disease", "setting": "room",
                    "transmission_model": "steady_state_cumulative_exposure_bound", "scope_error": ""}
        if operation == "propose_assumption_investigation":
            if self.invalid_action:
                return {"action": "stop", "objective": "Stop without numerical approval"}
            if not request["evidence_options"]:
                return {"action": "gather_evidence", "objective": "Find qualitative behavior evidence", "query": ""}
            fields = [a["field"] for a in request["assessments"].values()]
            field = next((f for f in request["fields"] if f not in fields), request["fields"][-1])
            return {"action": "assess_assumption", "objective": f"Justify {field}", "field": field,
                    "value": .0001 if field == "breathing_flow_m3_s" else 1,
                    "unit": "m3/s" if field == "breathing_flow_m3_s" else "fraction",
                    "statement": f"Conditional {field} for a seated unmasked target",
                    "basis": "Synthetic inference from qualitative activity evidence",
                    "evidence": list(request["evidence_options"])}
        if operation == "choose_assumption_evidence":
            option = next(iter(request["sources"]))
            return {"source": option, "window": 0, "question": "What is the observed activity and masking?"}
        if operation == "extract_research_evidence":
            if request["source"] == "request":
                return {"facts": [{"subject": "target", "field": "risk_tolerance", "value": .1,
                                   "unit": "scalar", "quote": "risk tolerance 0.1 scalar",
                                   "kind": "request_constraint", "basis": "Explicit test constraint", "context": "target"}]}
            return {"facts": [], "evidence": [{"subject": "synthetic observation", "claim": "Quiet unmasked activity",
                    "fields": ["breathing_flow_m3_s", "mask_penetration"], "quote": TEXT,
                    "context": "Seated adults", "basis": "Qualitative activity evidence, not a printed flow rate"}]}
        if operation == "review_research_evidence":
            return {"decisions": [{"option": c["option"], "admit": True, "reason": "Synthetic evidence admission"}
                                  for c in request["candidates"]]}
        if operation == "assess_assumption_reasonableness":
            verdict = self.verdict
            return {"verdict": verdict, "request_constraint": None,
                    "criteria": {name: {"grade": 0 if verdict == "unsupported" else 2,
                                         "reason": "Synthetic reviewer judgment", "evidence": list(request["evidence"])} for name in RUBRIC},
                    "limitations": ["Synthetic evidence is not scientific validation"], "alternatives": [],
                    "missing_evidence": [], "explanation": f"Synthetic {verdict} assessment"}
        if operation == "review_evidence_applicability":
            return applicability_response(request)
        if request.get("assumption_admission", {}).get("required"):
            return {"values": []}
        return {"values": [{"field": name, "value": FACTS[name][0], "unit": FACTS[name][1],
                            "normalized_value": FACTS[name][0], "normalized_unit": FACTS[name][1],
                            "source_ref": request["source"]["source_ref"], "evidence_span": FACTS[name][2],
                            "confidence": 1, "notes": []} for name in request["required_fields"] if name in FACTS]}


class AssumptionBranchTests(unittest.TestCase):
    def run_branch(self, agent, *, limits=None):
        return _run_admit_behavior_assumptions_episode(
            agent, (SOURCE,), admit_assumption=True,
            target_context={"setting": "seated adults without face coverings"},
            assumption_services=AssumptionServices(limits=limits),
        )

    def test_new_branch_is_a_real_child_of_the_old_admission_episode(self):
        agent = AssumptionAgent()
        record, report = self.run_branch(agent)
        self.assertTrue(report.passed, report.as_record())
        self.assertEqual(record.scope_level, "admit_behavior_assumptions")
        branch = record.unit_records[-1].child
        self.assertEqual(branch.scope_level, "assumption_investigation")
        self.assertEqual(branch.path[:-1], record.path)
        self.assertEqual([unit.child.scope_level for unit in branch.unit_records],
                         ["assumption_evidence", "assumption_assessment", "assumption_assessment"])
        self.assertEqual(branch.ended_by, "yield_stop")
        self.assertEqual(branch.controller_state.history, (0, 0, .5, 1))
        self.assertEqual(report.outputs["breathing_flow_m3_s"], .0001)
        self.assertNotIn("0.0001", TEXT)
        self.assertNotIn("extracted_values", report.outputs)
        self.assertEqual(report.outputs["evidence_basis"], "reviewed_conditional_inference_not_source_extraction")
        self.assertTrue(report.outputs["admitted_assumptions"]["breathing_flow_m3_s"]["review"]["limitations"])
        self.assertEqual(report.outputs["assumption_documents"][SOURCE.source_ref]["text"], TEXT)
        self.assertEqual(branch.goal.as_record()["result_contract"]["fields"], ["breathing_flow_m3_s", "mask_penetration"])
        json.dumps(report.as_record(), allow_nan=False)

    def test_unsupported_and_contradicted_assumptions_do_not_pass_admission(self):
        for verdict in ("unsupported", "contradicted"):
            with self.subTest(verdict=verdict):
                record, report = self.run_branch(AssumptionAgent(verdict=verdict))
                self.assertFalse(report.passed)
                self.assertNotIn("breathing_flow_m3_s", report.outputs)
                self.assertIn("unjustified_assumption", json.dumps(report.as_record()))
                self.assertEqual(record.unit_records[-1].child.ended_by, "yield_stop")

    def test_invalid_stop_action_hits_operational_limit_not_scientific_convergence(self):
        record, report = self.run_branch(AssumptionAgent(invalid_action=True), limits=AssumptionLimits())
        self.assertFalse(report.passed)
        branch = record.unit_records[-1].child
        self.assertEqual(branch.ended_by, "source_failed")
        self.assertEqual(branch.controller_state.history, (0,))
        self.assertEqual(branch.controller_state.failures, 3)

    def test_revisiting_an_older_candidate_withdraws_current_credit(self):
        base = AssumptionAgent()
        attempts = []
        def agent(request):
            answer = base(request)
            if request.get("operation") == "propose_assumption_investigation" and request["evidence_options"]:
                value = (.0001, .0002, .0001)[len(attempts)]
                attempts.append(value)
                answer.update(field="breathing_flow_m3_s", value=value, unit="m3/s")
            if request.get("operation") == "assess_assumption_reasonableness":
                verdict, grade = (("supported", 2), ("provisional", 1), ("unsupported", 0))[len(attempts) - 1]
                answer["verdict"] = verdict
                for criterion in answer["criteria"].values():
                    criterion["grade"] = grade
            return answer
        record, report = self.run_branch(agent, limits=AssumptionLimits())
        self.assertFalse(report.passed)
        branch = record.unit_records[-1].child
        self.assertEqual(branch.controller_state.history, (0, 0, .5, .25, 0))

    def test_investigation_can_change_its_evidence_approach_after_child_findings(self):
        base = AssumptionAgent()
        queries = []
        def search(query, limit):
            queries.append(query)
            return ({"url": SOURCE.source_ref, "text": TEXT},)
        def agent(request):
            if request.get("operation") == "propose_assumption_investigation" and not request["evidence_options"]:
                query = "occupant behavior" if request["findings"] else "equipment"
                return {"action": "gather_evidence", "objective": "Resolve activity assumptions", "query": query}
            if request.get("operation") == "extract_research_evidence" and queries[-1] == "equipment":
                return {"facts": [], "findings": ["This reading only describes equipment, not occupant activity."]}
            return base(request)
        record, report = _run_admit_behavior_assumptions_episode(
            agent, (SOURCE,), admit_assumption=True,
            target_context={"setting": "seated adults"},
            assumption_services=AssumptionServices(search=search),
        )
        self.assertTrue(report.passed, report.as_record())
        self.assertEqual(queries, ["equipment", "occupant behavior"])
        branch = record.unit_records[-1].child
        self.assertEqual([unit.child.scope_level for unit in branch.unit_records],
                         ["assumption_evidence", "assumption_evidence", "assumption_assessment", "assumption_assessment"])

    def test_successful_direct_admission_does_not_run_the_new_branch(self):
        agent = AssumptionAgent()
        _, report = _run_admit_behavior_assumptions_episode(agent, (EvidenceSource(SOURCE_URL, SOURCE_TEXT),))
        self.assertTrue(report.passed)
        self.assertFalse(any(r.get("operation") == "propose_assumption_investigation" for r in agent.requests))

    def test_branch_does_not_override_an_explicit_target_constraint(self):
        _, report = _run_admit_behavior_assumptions_episode(
            AssumptionAgent(), (SOURCE,), admit_assumption=True,
            target_context={"target_scenario": {"breathing_flow_m3_s": .0002}},
            assumption_services=AssumptionServices(limits=AssumptionLimits()),
        )
        self.assertFalse(report.passed)
        self.assertIn("cannot override an explicit target constraint", json.dumps(report.as_record()))

    def test_invalid_source_evidence_cannot_be_laundered_through_the_branch(self):
        agent = AssumptionAgent()
        def invalid(request):
            if not request.get("operation"):
                result = agent({**request, "assumption_admission": {"required": False}})
                result.update(similarity_basis="Synthetic similarity", similarity_evidence_span=TEXT)
                return result
            return agent(request)
        _, report = _run_admit_behavior_assumptions_episode(
            invalid, (SOURCE,), admit_assumption=True,
            target_context={"setting": "quiet seated"},
        )
        self.assertFalse(report.passed)
        self.assertFalse(any(r.get("operation") == "propose_assumption_investigation" for r in agent.requests))

    def test_original_calculation_receives_assumptions_without_replacing_its_children(self):
        agent = AssumptionAgent()
        def search(query, limit):
            if "assumptions" in query or "respiratory minute" in query or "face covering" in query:
                return ({"url": SOURCE.source_ref, "text": TEXT, "title": "Synthetic qualitative evidence"},)
            return ({"url": SOURCE_URL, "text": SOURCE_TEXT, "title": "Synthetic outbreak"},)
        scenario = {k: v for k, v in SCENARIO.items() if k not in {"breathing_flow_m3_s", "mask_penetration"}}
        record, report = run_evaluate_safety_guideline_episode(
            disease_name="synthetic-disease", target_scenario=scenario, search=search, llm=agent,
        )
        self.assertTrue(report.passed, report.as_record())
        self.assertEqual(record.scope_level, "evaluate_safety_guideline")
        self.assertEqual(tuple(report.child_reports), GUIDELINE_CHILD_NAMES)
        self.assertIn("cumulative_exposure_bound_person_s", report.main_outputs["compute_safety_bound"])
        self.assertIn("assumption_investigation", json.dumps(report.outputs["admit_target_exposure_scenario"]))


class RestoredRequestTests(unittest.TestCase):
    def test_prompt_interpretation_does_not_populate_unspecified_scientific_inputs(self):
        result = interpret_guideline_request(AssumptionAgent(), REQUEST, GUIDELINE_OUTPUTS)
        self.assertEqual(result["target_scenario"], {"risk_tolerance": .1, "description": REQUEST})
        self.assertNotIn("breathing_flow_m3_s", result["target_scenario"])

    def test_invented_request_number_fails_closed(self):
        agent = AssumptionAgent()
        def bad(request):
            answer = agent(request)
            if request.get("operation") == "extract_research_evidence":
                answer["facts"][0]["value"] = 999
            return answer
        with self.assertRaises(ValueError):
            interpret_guideline_request(bad, REQUEST, GUIDELINE_OUTPUTS)

    def test_cli_calls_approved_tree_not_legacy_pipeline(self):
        from episodic_workflow.case_reports.research import ResearchProviders
        providers = ResearchProviders(lambda _: {"action": "stop"}, lambda *_: (), lambda _: None)
        out, err = io.StringIO(), io.StringIO()
        with patch("episodic_workflow.configured_providers.workflow_environment", return_value=PROVIDER_ENV), \
             patch("episodic_workflow.configured_providers.firecrawl_argo_research_providers", return_value=providers), \
             patch("episodic_workflow.configured_providers.run_evaluate_safety_guideline_episode", side_effect=AssertionError("Legacy pipeline invoked")):
            code = main([REQUEST, "--max-units-per-episode", "2"], stdout=out, stderr=err)
        self.assertEqual(code, 1, err.getvalue())
        data = json.loads(out.getvalue())
        self.assertEqual(data["episode_tree"]["evaluate_safety_guideline"], ["investigate"])
        self.assertNotIn("child_reports", data)
        self.assertIn("scenarios", data)


if __name__ == "__main__":
    unittest.main()
