"""Physical baselines precede sharing and never become research discoveries."""

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.model_defaults import (
    MODEL_DEFAULTS, initialize_model_defaults, model_default_request, retained_defaults,
)
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, identity
from episodic_workflow.case_reports.research.engine import research_parameter_requirements
from episodic_workflow.case_reports.research.progress import parameter_support
from episodic_workflow.case_reports.research.relevance import (
    source_profiles, target_comparison_context, target_relevance_credit,
)
from episodic_workflow.case_reports.research.scenario_sharing import (
    applicable_assumptions, independent_groups, initialize_parameter_scope, needed_fields, scenario_version,
)
from episodic_workflow.case_reports.research.scenarios import initial_scenarios, load_scenarios
from episodic_workflow.case_reports.research.tools import main_calculation_inputs
from episodic_workflow.case_reports.research.workflow import investigable_parameters, run_guideline_research
from episodic_workflow.task_executors.scenario_metrics.dependencies import CALCULATED_PARAMETER_FIELDS
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS, UNITS


def retain_all(item):
    return {"decisions": {subject: {field: {
        "decision": "retain", "reason": "No exceptional physical conditions are specified.",
        "scenario_evidence": []} for field in fields}
        for subject, fields in item["default_candidates"].items()}}


def shared_scope(item):
    return {"features": [], "dependencies": [{"parameter": field, "factors": [],
        "reason": "No difference in this test's stipulated physical property."}
        for field in item["parameters"]]}


class ModelDefaultsTests(unittest.TestCase):
    def test_builtin_targets_receive_baselines_without_found_values_or_discovery_credit(self):
        initial = initial_scenarios(load_scenarios())
        planned = initialize_model_defaults(initial, retain_all)
        scoped = initialize_parameter_scope(planned, shared_scope)
        ledger = Ledger(scoped)
        for scenario in scoped.scenarios:
            self.assertEqual(set(retained_defaults(scenario)), set(MODEL_DEFAULTS))
            self.assertFalse(set(MODEL_DEFAULTS) & needed_fields(scenario))
            self.assertFalse(CALCULATED_PARAMETER_FIELDS & needed_fields(scenario))
            values = {f.field: f for f in main_calculation_inputs(ledger, scenario["subject"])
                      if f.subject == scenario["subject"] and f.kind == "model_default"}
            for field, baseline in MODEL_DEFAULTS.items():
                self.assertEqual(values[field].value, baseline["value"])
                self.assertEqual(values[field].unit, UNITS[field])
                self.assertNotIn(field, scenario["fixed_parameters"])
        status = scientific_goal_status(scoped, DEFAULT_METRICS)
        self.assertEqual(status["recorded_target_inputs"]["recorded"], 0)
        self.assertEqual(status["weighted_score"], 0)
        counted = {row["parameter"] for row in status["independent_target_values"]}
        self.assertFalse(counted & (MODEL_DEFAULTS.keys() | CALCULATED_PARAMETER_FIELDS))

    def test_constants_judge_then_sharing_then_credit_and_research(self):
        calls = []
        def judge(item):
            if item.get("operation") == "assess_scenario_model_defaults":
                calls.append("constants")
                return retain_all(item)
            calls.append("sharing")
            self.assertEqual(calls, ["constants", "sharing"])
            self.assertFalse(MODEL_DEFAULTS.keys() & set(item["parameters"]))
            self.assertFalse(CALCULATED_PARAMETER_FIELDS & set(item["parameters"]))
            return shared_scope(item)
        def run(**kwargs):
            calls.append("research")
            self.assertEqual(calls, ["constants", "sharing", "research"])
            self.assertTrue(kwargs["initial"].scenario_parameter_relations)
            self.assertEqual(scientific_goal_status(kwargs["initial"], DEFAULT_METRICS)["weighted_score"], 0)
            return kwargs["initial"]
        with patch("episodic_workflow.case_reports.research.workflow._run_research", side_effect=run):
            run_guideline_research(disease="test pathogen", providers=SimpleNamespace(
                scenario_parameter_assessor=judge))
        self.assertEqual(calls, ["constants", "sharing", "research"])

    def test_explicit_exception_is_promoted_only_for_affected_case_before_sharing(self):
        initial = initial_scenarios(load_scenarios())
        ordinary = {**initial.scenarios[0], "subject": "ordinary-target", "description": "People in an ordinary room."}
        exceptional = {**initial.scenarios[1], "subject": "pressure-target",
                       "description": "People in a chamber maintained at twice atmospheric pressure."}
        initial = ResearchResult(scenarios=(ordinary, exceptional))
        def judge(item):
            result = retain_all(item)
            result["decisions"]["pressure-target"]["air_density_kg_m3"] = {
                "decision": "revisit", "reason": "The specified chamber pressure materially changes air density.",
                "scenario_evidence": ["twice atmospheric pressure"]}
            return result
        planned = initialize_model_defaults(initial, judge)
        self.assertNotIn("air_density_kg_m3", needed_fields(planned.scenarios[0]))
        self.assertIn("air_density_kg_m3", needed_fields(planned.scenarios[1]))
        observed = []
        scoped = initialize_parameter_scope(planned, lambda item: observed.append(item) or shared_scope(item))
        self.assertIn("air_density_kg_m3", observed[0]["parameters"])
        groups = [row for row in independent_groups(Ledger(scoped)) if row["parameter"] == "air_density_kg_m3"]
        self.assertEqual([row["subjects"] for row in groups], [["pressure-target"]])
        self.assertFalse(any(f.subject == "pressure-target" and f.field == "air_density_kg_m3" for f in scoped.facts))
        self.assertIn("air_density_kg_m3", investigable_parameters(Ledger(scoped), UNITS))

    def test_revisit_requires_actual_scenario_evidence(self):
        initial = initial_scenarios(load_scenarios()[:1])
        for evidence in ([], ["an invented vacuum chamber"]):
            def judge(item):
                result = retain_all(item)
                result["decisions"][initial.scenarios[0]["subject"]]["air_density_kg_m3"] = {
                    "decision": "revisit", "reason": "Would like more data.", "scenario_evidence": evidence}
                return result
            with self.subTest(evidence=evidence), self.assertRaises(ValueError):
                initialize_model_defaults(initial, judge)

    def test_retained_constant_source_evidence_has_no_local_or_target_credit(self):
        planned = initialize_model_defaults(initial_scenarios(load_scenarios()[:1]), retain_all)
        ordinary = {**planned.scenarios[0], "subject": "ordinary"}
        promoted = {**ordinary, "subject": "promoted", "model_defaults": {
            **ordinary["model_defaults"], "air_density_kg_m3": {
                **ordinary["model_defaults"]["air_density_kg_m3"], "decision": "revisit"}}}
        targets = {"ordinary": ordinary, "promoted": promoted}
        fact = Fact("observed-event", "air_density_kg_m3", 1.3, "kg/m3",
                    "source_observation", "Reported chamber density.",
                    source="https://example.org/density", quote="Air density was 1.3 kg/m3.",
                    context="Observed air in the reported chamber.")
        result = ResearchResult(facts=(fact,), evaluated=True)
        ordinary_requirements = research_parameter_requirements("settling_speed_m_s", targets, ("ordinary",))
        mixed_requirements = research_parameter_requirements("settling_speed_m_s", targets)
        self.assertEqual(parameter_support(result, ordinary_requirements), [])
        self.assertEqual([row["field"] for row in parameter_support(result, mixed_requirements)],
                         ["air_density_kg_m3"])
        groups = {"event-type": {"members": [(fact.source, fact.subject)],
            "description": "Observed chamber event", "target_relevance": {s: 1 for s in targets},
            "basis": "Applicable physical measurement."}}
        profile = source_profiles(result, mixed_requirements)[(fact.source, fact.subject)]
        relationships = tuple({"source": fact.source, "source_subject": fact.subject,
            "target_subject": subject, "parameter": "settling_speed_m_s", "score": 1,
            "known": 1, "considered": 1, "source_profile_version": identity(profile),
            "target_version": identity(target_comparison_context(target, ())) }
            for subject, target in targets.items())
        for relationships_or_empty in ((), relationships):
            with self.subTest(assessed_relationships=bool(relationships_or_empty)):
                rows, _ = target_relevance_credit(replace(result,
                    source_target_relationships=relationships_or_empty), mixed_requirements,
                    groups, targets, parameter="settling_speed_m_s")
                self.assertEqual({row["target"]: row["points"] for row in rows},
                                 {"ordinary": 0, "promoted": 1})

    def test_replanning_does_not_leave_old_default_usable(self):
        initial = initial_scenarios(load_scenarios()[:1])
        scenario = {**initial.scenarios[0], "description": "Chamber at twice atmospheric pressure."}
        planned = initialize_model_defaults(ResearchResult(scenarios=(scenario,)), retain_all)
        ledger = Ledger(planned)
        subject = scenario["subject"]
        self.assertTrue(any(a["field"] == "air_density_kg_m3" for a, _ in applicable_assumptions(ledger, subject)))
        def promote(item):
            result = retain_all(item)
            result["decisions"][subject]["air_density_kg_m3"] = {
                "decision": "revisit", "reason": "Specified pressure changes air density.",
                "scenario_evidence": ["twice atmospheric pressure"]}
            return result
        ledger.merge(initialize_model_defaults(planned, promote))
        self.assertTrue(any(a.get("role") == "model_default" and a["field"] == "air_density_kg_m3"
                            for a in ledger.assumptions.values()))
        self.assertFalse(any(a["field"] == "air_density_kg_m3" for a, _ in applicable_assumptions(ledger, subject)))
        self.assertFalse(any(f.subject == subject and f.field == "air_density_kg_m3"
                             for f in main_calculation_inputs(ledger, subject)))

    def test_caller_fixed_value_has_precedence_over_default(self):
        case = load_scenarios()[0]
        case = replace(case, fixed=(*case.fixed, ("gravity_m_s2", 8.0)))
        planned = initialize_model_defaults(initial_scenarios((case,)), retain_all)
        self.assertNotIn("gravity_m_s2", planned.scenarios[0]["model_defaults"])
        options = [f for f in planned.facts if f.field == "gravity_m_s2"]
        self.assertEqual([(f.value, f.kind) for f in options], [(8.0, "scenario_constraint")])
        self.assertNotIn("gravity_m_s2", needed_fields(planned.scenarios[0]))

    def test_persisted_groups_cannot_restore_retained_or_calculated_fields(self):
        planned = initialize_model_defaults(initial_scenarios(load_scenarios()[:1]), retain_all)
        scenario = planned.scenarios[0]
        excluded = set(MODEL_DEFAULTS) | CALCULATED_PARAMETER_FIELDS
        old_groups = tuple({"kind": "group", "group_id": field, "parameter": field,
            "subjects": [scenario["subject"]],
            "scenario_versions": {scenario["subject"]: scenario_version(scenario)}} for field in excluded)
        result = replace(planned, scenario_parameter_relations=old_groups)
        self.assertFalse(excluded & {row["parameter"] for row in independent_groups(Ledger(result))})
        self.assertFalse(excluded & set(scientific_goal_status(result, DEFAULT_METRICS)["target_inputs"][scenario["subject"]]))

    def test_default_request_is_a_separate_judge_with_exact_case_and_field_contract(self):
        initial = initial_scenarios(load_scenarios()[:1])
        seen = []
        initialize_model_defaults(initial, lambda item: seen.append(model_default_request(item)) or retain_all(item))
        request = seen[0]
        self.assertEqual(request["operation"], "assess_scenario_model_defaults")
        subject = initial.scenarios[0]["subject"]
        schema = request["response_contract"]["properties"]["decisions"]["properties"][subject]
        self.assertEqual(set(schema["required"]), set(MODEL_DEFAULTS))
        self.assertNotIn("features", request["response_contract"]["properties"])

    def test_configured_provider_records_two_distinct_reasoning_preflight_calls(self):
        from episodic_workflow.configured_providers import WorkflowProviderConfig, firecrawl_argo_research_providers
        requests, traces = [], []
        class Client:
            def call(self, text, **kwargs):
                request = json.loads(text)
                requests.append(request)
                response = (retain_all(request) if request["operation"] == "assess_scenario_model_defaults"
                    else shared_scope({"parameters": list(request["target_parameters"])}))
                return json.dumps(response)
        config = WorkflowProviderConfig(firecrawl_api_key="offline-fixture", typesafe_api_key="offline-fixture",
            llm_base_url="http://localhost:1", llm_model="reasoning-fixture", llm_fast_model="fast-fixture")
        with patch("episodic_workflow.configured_providers._llm_client", return_value=Client()), \
                patch("episodic_workflow.configured_providers.JevDecisionClient", return_value=SimpleNamespace()):
            providers = firecrawl_argo_research_providers(config, trace_event=traces.append)
            planned = initialize_model_defaults(initial_scenarios(load_scenarios()[:1]),
                providers.scenario_parameter_assessor)
            initialize_parameter_scope(planned, providers.scenario_parameter_assessor)
        operations = ["assess_scenario_model_defaults", "analyze_scenario_parameter_scope"]
        self.assertEqual([row["operation"] for row in requests], operations)
        self.assertEqual([row["operation"] for row in providers.preflight_attempts], operations)
        self.assertEqual([row["call"] for row in providers.preflight_attempts], [1, 2])
        self.assertEqual([row["model"] for row in providers.preflight_attempts], ["reasoning-fixture"] * 2)
        self.assertEqual([row["attempt"]["operation"] for row in traces], operations)


if __name__ == "__main__":
    unittest.main()
