"""Open source investigations can estimate inputs without changing target scope."""

from copy import deepcopy
import unittest

from episodic_workflow.case_reports.research.assumptions import RUBRIC
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun
from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, numeric_references
from episodic_workflow.case_reports.research.workflow import GuidelineSource, InvestigationSource, source_calculation_view
from episodic_workflow.method_loop import EpisodeGoal
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS
from .test_source_value_handoff import initial_state


class InvestigationScopeTests(unittest.TestCase):
    @staticmethod
    def schema_allows_candidate(schema, candidate):
        """Inspect the actual subject/field constraints; runtime validates the claim."""
        branches = schema["properties"]["candidates"]["items"]["oneOf"]
        branch = branches[1 if candidate["value"] is None else 0]
        properties = branch["properties"]
        if (candidate["subject"] not in properties["subject"]["enum"]
                or candidate["field"] not in properties["field"]["enum"]):
            return False
        for condition in branch.get("allOf", ()):
            if condition["if"]["properties"]["subject"]["const"] == candidate["subject"]:
                allowed = condition["then"]["properties"]["field"]
                if allowed is False or candidate["field"] not in allowed["enum"]:
                    return False
        return True

    def setUp(self):
        self.initial = initial_state()
        self.subjects = [row["subject"] for row in self.initial.scenarios]
        self.proposal = None
        self.requests = []

    def agent(self, request):
        self.requests.append(request)
        operation = request["operation"]
        if operation == "propose_investigation_strategy":
            for candidate in self.proposal.get("candidates", ()):
                self.assertTrue(self.schema_allows_candidate(request["response_contract"], candidate))
            return self.proposal
        if operation == "assess_assumption_reasonableness":
            return {"verdict": "unsupported", "request_constraint": None,
                "criteria": {name: {"grade": 0, "reason": "Explicit conditional estimate only.",
                                    "evidence": []} for name in RUBRIC},
                "limitations": ["No supporting evidence yet."], "alternatives": [],
                "missing_evidence": ["Evidence for the estimate."],
                "explanation": "The estimate remains a usable unsupported hypothesis.",
                "supporting_evidence": [], "contradicting_evidence": [], "invalid_evidence": []}
        if operation in {"review_research_model", "review_scenario_scope"}:
            return {"admit": True, "reason": "The explicit conditional calculation preserves its source scope."}
        self.fail("Unexpected operation: " + operation)

    def investigation(self, *, subjects=(), initial=None):
        initial = initial or self.initial
        run = ResearchRun(ResearchProviders(self.agent, lambda *_: (), lambda _: None),
            ResearchLimits(), "Estimate the fixed target cases", DEFAULT_METRICS, initial=initial)
        source = InvestigationSource(run, initial)
        contract = source.contract("Find a source that informs concentration", subjects=subjects,
            parameter="quanta_concentration_exhaled_m3", deliverable="source_findings",
            target_scenarios={row["subject"]: row for row in initial.scenarios
                              if row.get("role", "target") == "target"})
        run.episode("investigate", source,
            goal=EpisodeGoal.root(objective="Find concentration evidence", result_contract={}),
            focus=(), contract=contract)
        # Exercise the real proposal schema/runtime, assessment and deterministic
        # tool. These adapters omit only the generic episode scheduler.
        source.inputs = lambda view, **request: request
        source.leaf = lambda callback, label: callback()
        return source

    def candidate(self, subject, *, field="quanta_concentration_exhaled_m3", value=29., unit="quanta/m3"):
        return {"subject": subject, "field": field, "value": value, "unit": unit,
            "statement": "A bounded conditional estimate for this case.",
            "basis": "Explicit estimate to examine numerical consequences.",
            "entry_type": "best_guess", "source_refs": [], "input_refs": [],
            "added_assumptions": ["Applicability remains uncertain."]}

    def record_candidate(self, source, candidate, **proposal):
        self.proposal = {"candidates": [candidate], **proposal}
        staged = source.propose(None)
        source.project(staged)
        assessed = source.pending
        source.ledger.merge(assessed)
        return assessed

    def test_root_and_investigation_schema_have_no_target_numeric_selector(self):
        ledger = Ledger(self.initial)
        event = "observed-source"
        native = Fact(event, "breathing_flow_m3_s", .0002, "m3/s", "source_model_parameter",
                      "Native source input", source="https://example.test/native", quote="0.0002 m3/s", context="Native source event")
        ledger.merge(ResearchResult(scenarios=({"subject": event, "role": "calibration",
            "description": "Registered native source context"},), facts=(native,)))
        child = self.investigation(initial=ledger.result())
        root = GuidelineSource(child.run)
        root.focus = ()
        native_reference = next(key for key, fact in numeric_references(child.ledger).items()
                                if fact.key == native.key)
        for source in (root, child):
            with self.subTest(source=type(source).__name__):
                schema = source.proposal_schema()
                self.assertNotIn("facts", schema["properties"])
                rule = next(row for row in schema["allOf"]
                    if row.get("then", {}).get("properties", {}).get("source_inputs") is False)
                self.assertEqual(set(rule["if"]["properties"]["subject"]["enum"]), set(self.subjects))
                self.assertEqual(rule["else"]["then"]["required"], ["source_inputs"])
                self.assertEqual(schema["properties"]["source_inputs"]["items"]["enum"], [native_reference])
        view = source_calculation_view(child.ledger)
        self.assertEqual([row["reference"] for row in view["native_input_records"]], [native_reference])
        self.assertTrue(all(row["subject"] == event for row in view["native_input_records"]))
        _, inference = child.inference_options()
        self.assertTrue(any(fact.subject in self.subjects for fact in inference.values()),
                        "Target provenance remains available for inference, without a calculation selector")

    def test_actual_empty_subject_contract_accepts_estimate_then_calculates(self):
        ledger = Ledger(self.initial)
        subject = self.subjects[0]
        ledger.scenarios[subject]["fixed_parameters"]["breathing_flow_m3_s"] = {
            "value": .00015, "unit": "m3/s"}
        ledger.merge(ResearchResult(facts=(Fact(subject, "breathing_flow_m3_s", .00015,
            "m3/s", "scenario_constraint", "Fixture-supplied flow for this calculation"),)))
        source = self.investigation(initial=ledger.result())
        candidate = self.candidate(subject)
        proposal = {"candidates": [candidate]}
        schema = source.proposal_schema()
        old_schema = deepcopy(schema)
        for branch in old_schema["properties"]["candidates"]["items"]["oneOf"]:
            branch["properties"]["subject"]["enum"] = []
        self.assertFalse(self.schema_allows_candidate(old_schema, candidate))
        self.assertTrue(self.schema_allows_candidate(schema, candidate))
        assessed = self.record_candidate(source, candidate)
        claim = next(a for a in assessed.assumptions if a["field"] == candidate["field"])
        self.assertEqual(claim["review"]["verdict"], "unsupported")
        status = scientific_goal_status(source.ledger.result(), DEFAULT_METRICS)
        self.assertEqual(status["recorded_target_inputs"]["recorded"], 1)
        self.proposal = {"action": "calculate_transmission_metrics", "subject": subject,
            "metrics": ["quanta_emission_rate_s"],
            "conditions": {"uniform_breathing_and_masks": "Uniform breathing and mask assumptions."},
            "parameterization": "conditional-estimate", "basis": "Calculate from the recorded estimate."}
        result = source.propose(None)
        output = next(f for f in result.facts if f.subject == subject and f.field == "quanta_emission_rate_s")
        self.assertAlmostEqual(output.value, 29 * .00015)
        self.assertIn(claim["key"], result.calculations[-1]["assumption_versions"])
        self.assertEqual(set(source.ledger.scenarios), set(self.subjects))
        self.assertFalse(any(request["operation"] == "propose_strategy_query" for request in self.requests))

    def test_registered_source_context_can_estimate_its_own_calculation_operand(self):
        ledger = Ledger(self.initial)
        event = "observed-source-event"
        ledger.merge(ResearchResult(scenarios=({"subject": event, "role": "calibration",
            "description": "A registered observed event", "scope_review": "Observed source identity"},),
            facts=(Fact(event, "quanta_emission_rate_s", .02, "quanta/s", "source_model_parameter",
                "Published source estimate", source="https://example.test/source",
                quote="Emission was 0.02 quanta/s", context="This event"),)))
        source = self.investigation(subjects=(self.subjects[0],), initial=ledger.result())
        candidate = self.candidate(event, field="breathing_flow_m3_s", value=.0002, unit="m3/s")
        self.record_candidate(source, candidate)
        self.proposal = {"action": "calibrate_infectiousness", "subject": event,
            "operator": "cq_from_emission", "source_inputs": [key for key, fact in numeric_references(source.ledger).items()
                if fact.subject == event], "basis": "Conditional source quotient, not independent inversion."}
        result = source.propose(None)
        output = next(f for f in result.facts if f.subject == event and f.field == "quanta_concentration_exhaled_m3")
        self.assertAlmostEqual(output.value, 100)
        self.assertEqual(result.calculations[-1]["operator"], "cq_from_emission")
        self.assertEqual({name for name, row in source.ledger.scenarios.items()
                          if row.get("role", "target") == "target"}, set(self.subjects))

    def test_explicit_target_scope_and_fixed_inputs_remain_enforced(self):
        source = self.investigation(subjects=(self.subjects[0],))
        candidate = self.candidate(self.subjects[1])
        self.assertFalse(self.schema_allows_candidate(source.proposal_schema(), candidate))
        # Bypass only provider schema validation to verify the runtime guard.
        source.run.agent = lambda request: {"candidates": [candidate]}
        with self.assertRaises(ValueError) as failure:
            source.propose(None)
        self.assertIn("assigned target or reviewed calibration", str(failure.exception))
        source = self.investigation()
        for field, value, unit in (("room_volume_m3", 100, "m3"),
                                   ("air_density_kg_m3", 1.204, "kg/m3")):
            candidate = self.candidate(self.subjects[0], field=field, value=value, unit=unit)
            with self.subTest(field=field):
                self.assertFalse(self.schema_allows_candidate(source.proposal_schema(), candidate))
            source.run.agent = lambda request: {"candidates": [candidate]}
            with self.assertRaises(ValueError):
                source.propose(None)

    def test_observed_source_needs_registration_before_estimation(self):
        ledger = Ledger(self.initial)
        event = "unregistered-observed-event"
        ledger.merge(ResearchResult(facts=(Fact(event, "index_case_count", 1, "person",
            "source_observation", "Observed index case", source="https://example.test/source",
            quote="One index case", context="Source event"),)))
        source = self.investigation(initial=ledger.result())
        candidate = self.candidate(event)
        self.proposal = {"candidates": [candidate]}
        with self.assertRaises(ValueError) as failure:
            source.propose(None)
        self.assertIn("no reviewed scenario", str(failure.exception))
        self.record_candidate(source, candidate, scenarios=[{"subject": event, "role": "calibration",
            "description": "The observed source event", "basis": "Estimate source operands."}])
        self.assertEqual(source.ledger.scenarios[event]["role"], "calibration")
        self.assertEqual(len([r for r in self.requests if r["operation"] == "review_scenario_scope"]), 1)


if __name__ == "__main__":
    unittest.main()
