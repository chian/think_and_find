"""Executed source calculations reach target selection and live global credit."""

from dataclasses import replace
from episodic_workflow.case_reports.research.records import numeric_references
import unittest
from unittest.mock import patch

from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun, ResearchSource
from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.model_defaults import initialize_model_defaults
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, assumption_model_view, current_premises, current_support
from episodic_workflow.case_reports.research.relevance import assess_numeric_target_relationships
from episodic_workflow.case_reports.research.scenario_sharing import initialize_parameter_scope
from episodic_workflow.case_reports.research.scenarios import initial_scenarios, load_scenarios
from episodic_workflow.case_reports.research.target_values import eligible_source_values, project_source_target_values
from episodic_workflow.case_reports.research.tools import execute_tool
from episodic_workflow.method_loop import Context, EpisodeGoal
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS, UNITS


FIELD = "quanta_concentration_exhaled_m3"
SOURCE = "Source event with conditional emission model"
REVIEW = {"admit": True, "reason": "Conditional arithmetic is compatible; source breathing is an assumption."}


def fixture():
    initial = initialize_model_defaults(initial_scenarios(load_scenarios()[:2]))
    initial = initialize_parameter_scope(initial, lambda request: {"features": [],
        "dependencies": [{"parameter": field, "factors": [], "reason": "Equal test targets"}
                         for field in request["parameters"]]})
    ledger = Ledger(initial)
    facts = tuple(Fact(SOURCE, field, value, UNITS[field], "source_model_parameter",
        "Published model input, not a measured source breath.", source="https://example.test/model",
        quote=f"{field}={value}", context="Dining event; conditional fully mixed model.")
        for field, value in (("breathing_flow_m3_s", 1.65 / 3600), ("quanta_emission_rate_s", 79.3 / 3600)))
    ledger.merge(ResearchResult(facts=facts, scenarios=({"subject": SOURCE, "role": "calibration",
        "description": "Documented dining event", "basis": "Source event"},)))
    return ledger, facts


def calculate(ledger, operator="cq_from_emission", fields=("breathing_flow_m3_s", "quanta_emission_rate_s")):
    options = numeric_references(ledger)
    selected = [next(key for key, fact in options.items() if fact.subject == SOURCE and fact.field == field)
                for field in fields]
    return execute_tool({"action": "calibrate_infectiousness", "subject": SOURCE,
        "operator": operator, "source_inputs": selected,
        "basis": "Conditional concentration; pulmonary flow represents source exhalation."},
        ledger, DEFAULT_METRICS, review=REVIEW)


def project(ledger, grade=None):
    ledger.merge(assess_numeric_target_relationships(ledger, grade))
    ledger.merge(project_source_target_values(ledger))
    return scientific_goal_status(ledger.result(), DEFAULT_METRICS)


class DerivedTargetValueTests(unittest.TestCase):
    def test_real_calibration_commit_fills_targets_refreshes_root_and_competes_by_relevance(self):
        ledger, facts = fixture()
        initial = ledger.result()
        requests, observed = [], []

        def grade(request):
            requests.append(request)
            value = request["source_profile"]["facts"][0]
            return {subject: (.9 if value["value"] == 29 else .8)
                    for subject in request["target_scenarios"]}

        run = ResearchRun(ResearchProviders(lambda _: {"findings": "Conditional values available."},
            lambda *_: (), lambda _: None, source_target_assessor=grade), ResearchLimits(),
            "Source calculations become conditional target inputs", DEFAULT_METRICS, initial=initial)
        parent, child = ResearchSource(run, initial), ResearchSource(run, initial)
        root = run.episode("evaluate_airborne_safety_guideline", parent,
            goal=EpisodeGoal.root(objective="Target progress", result_contract={}), focus=())
        subjects = tuple(row["subject"] for row in initial.scenarios if row.get("role", "target") == "target")
        count, launched = 0, False
        child_view = None
        original_commit = child.after_verdict

        def committed(*args, **kwargs):
            result = original_commit(*args, **kwargs)
            controller = next(c for path, c in run.live_controllers.items() if len(path) == 1)
            observed.append((controller.state(), child.inputs(child_view)))
            return result

        child.after_verdict = committed

        def child_propose(view):
            nonlocal count, child_view
            child_view = view
            count += 1
            if count == 1:
                result = calculate(child.ledger)
                self.assertEqual(current_support(result.calculations[0], child.ledger.assumptions,
                    ledger=child.ledger), 0)
                self.assertAlmostEqual(result.facts[0].value, 48.06060606060606)
                return child.leaf(lambda: result, "calibrate_infectiousness")
            return None

        def parent_propose(view):
            nonlocal launched
            if launched:
                observed.append((None, parent.inputs(view)))
                return None
            launched = True
            return parent.edge(run.child(view, child, "investigate", "Find concentration",
                contract=parent.contract("Find concentration", parameter=FIELD, subjects=subjects,
                                         deliverable="assumption_hypotheses"), proposal={}))

        child.propose, parent.propose = child_propose, parent_propose
        root.run(Context(tree=run.tree))
        first = observed[0][1]["global_scientific_goal"]
        self.assertAlmostEqual(first["target_inputs"][subjects[0]][FIELD]["value"], 48.06060606060606)
        self.assertEqual(first["target_inputs"][subjects[0]][FIELD]["source_applicability"]["score"], .8)
        self.assertEqual(first["recorded_target_inputs"]["recorded"], 2)
        baseline = scientific_goal_status(initial, DEFAULT_METRICS)
        self.assertEqual(first["recorded_target_inputs"]["required"], baseline["recorded_target_inputs"]["required"])
        self.assertEqual(observed[0][0].history, (0,))
        self.assertEqual(observed[0][0].score, first["weighted_score"])
        self.assertEqual(observed[-1][1]["global_scientific_goal"]["target_inputs"], first["target_inputs"])
        better = replace(facts[0], field=FIELD, value=29, unit=UNITS[FIELD],
                         quote="Concentration29", context="Closer quiet activity")
        run.global_goal.ledger.merge(ResearchResult(facts=(better,)))
        second = project(run.global_goal.ledger, grade)
        self.assertEqual(second["target_inputs"][subjects[0]][FIELD]["value"], 29)
        self.assertEqual(first["recorded_target_inputs"], second["recorded_target_inputs"])
        self.assertEqual(first["weighted_score"], second["weighted_score"])
        derived_request = next(r for r in requests if r["source_profile"]["facts"][0]["kind"] == "derived")
        lineage = derived_request["source_profile"]["derivation"]
        self.assertEqual(len(lineage["sources"]), 2)
        self.assertEqual(lineage["calculations"][0]["operator"], "cq_from_emission")
        self.assertFalse(lineage["calculations"][0]["support_complete"])
        self.assertIn("Conditional concentration", lineage["calculations"][0]["basis"])

    def test_invalid_upstream_withdraws_derived_candidate_and_downstream_premise(self):
        ledger, facts = fixture()
        result = calculate(ledger)
        ledger.merge(result)
        before = project(ledger)
        candidate = next(a for a in ledger.assumptions.values() if a.get("entry_type") == "source_candidate" and a["field"] == FIELD)
        self.assertEqual(candidate["derivation"]["input_fact_keys"], [result.facts[0].key])
        self.assertEqual(candidate["source_derivations"][0]["calculation_version"], result.calculations[0]["version"])
        self.assertEqual({e["field"] for e in candidate["evidence"].values()}, {f.field for f in facts})
        target = execute_tool({"action": "calculate_transmission_metrics", "subject": candidate["subject"],
            "metrics": ["quanta_emission_rate_s"],
            "conditions": {"uniform_breathing_and_masks": "Uniform conditional breathing and masks"},
            "basis": "Conditional main-table target inputs"},
            ledger, DEFAULT_METRICS, review=REVIEW)
        ledger.merge(target)
        self.assertTrue(current_premises(target.calculations[0], ledger.assumptions, ledger=ledger))
        self.assertNotIn(target.facts[0], eligible_source_values(ledger))
        invalidated = {**candidate, "invalidated_evidence": [facts[1].semantic_record()]}
        ledger.merge(ResearchResult(assumptions=(invalidated,)))
        after = project(ledger)
        self.assertNotIn(result.facts[0], eligible_source_values(ledger))
        self.assertEqual(before["recorded_target_inputs"]["recorded"] - after["recorded_target_inputs"]["recorded"], 1)
        self.assertTrue(ledger.assumptions[candidate["key"]]["source_candidate_unavailable"])
        self.assertFalse(current_premises(target.calculations[0], ledger.assumptions, ledger=ledger))

    def test_transitive_source_derivation_and_changed_bound_input(self):
        ledger, facts = fixture()
        ledger.scenarios[SOURCE]["fixed_parameters"] = {facts[0].field: {"value": facts[0].value, "unit": facts[0].unit}}
        first = calculate(ledger)
        ledger.merge(first)
        factors = tuple(replace(facts[1], field=field, value=value, unit=UNITS[field], quote=field)
                        for field, value in (("activity_aerosol_volume_fraction", 2e-6),
                                             ("baseline_aerosol_volume_fraction", 1e-6)))
        ledger.merge(ResearchResult(facts=factors))
        second = calculate(ledger, "cq_activity_rescaling", (FIELD, *(f.field for f in factors)))
        ledger.merge(second)
        self.assertIn(first.facts[0], eligible_source_values(ledger))
        self.assertIn(second.facts[0], eligible_source_values(ledger))
        before = project(ledger)
        self.assertEqual(before["recorded_target_inputs"]["recorded"], 2)
        ledger.scenarios[SOURCE]["fixed_parameters"][facts[0].field]["value"] *= 2
        after = project(ledger)
        self.assertNotIn(first.facts[0], eligible_source_values(ledger))
        self.assertNotIn(second.facts[0], eligible_source_values(ledger))
        self.assertEqual(after["recorded_target_inputs"]["recorded"], 1)

    def test_model_review_and_relevance_do_not_gate_executed_numeric_candidate(self):
        for grade in (None, lambda request: {subject: 0 for subject in request["target_scenarios"]}):
            with self.subTest(grade=grade):
                ledger, _ = fixture()
                result = calculate(ledger)
                record = {**result.calculations[0], "model_review": {
                    "admit": False, "reason": "Transfer context remains uncertain."}}
                ledger.merge(replace(result, calculations=(record,)))
                status = project(ledger, grade)
                group = next(row for row in status["independent_target_values"] if row["parameter"] == FIELD)
                self.assertTrue(group["found"])
                self.assertEqual(group["points"], 1)
                self.assertAlmostEqual(group["value"], 48.06060606060606)
                self.assertEqual(current_support(record, ledger.assumptions, ledger=ledger), 0)

    def test_failed_calculation_returns_no_numeric_record_or_target_candidate(self):
        ledger, facts = fixture()
        ledger.facts.pop(facts[0].key)
        ledger.merge(ResearchResult(facts=(replace(facts[0], value=0),)))
        with self.assertRaises((ValueError, ArithmeticError)):
            calculate(ledger)
        self.assertFalse(ledger.calculations)
        self.assertFalse(any(fact.kind == "derived" for fact in ledger.facts.values()))
        status = project(ledger)
        group = next(row for row in status["independent_target_values"] if row["parameter"] == FIELD)
        self.assertFalse(group["found"])

    def test_inconsistent_output_never_becomes_target_candidate(self):
        ledger, _ = fixture()
        result = calculate(ledger)
        record = {**result.calculations[0], "output": {FIELD: {"value": 999, "unit": UNITS[FIELD]}}}
        ledger.merge(replace(result, calculations=(record,)))
        status = project(ledger)
        group = next(row for row in status["independent_target_values"] if row["parameter"] == FIELD)
        self.assertFalse(group["found"])

    def test_production_noul_preserves_conditional_derivation_and_source_evidence(self):
        from episodic_workflow.configured_providers import WorkflowProviderConfig, firecrawl_argo_research_providers
        ledger, _ = fixture()
        ledger.merge(calculate(ledger))
        payloads = []

        def post(_endpoint, payload, **_kwargs):
            payloads.append(payload)
            return {"answers": {name: {"type": "noul", "noul": .7} for name in payload["questions"]}}

        config = WorkflowProviderConfig(firecrawl_api_key="fixture", llm_base_url="http://localhost:8000",
            llm_model="fixture", typesafe_api_key="fixture")
        with patch("episodic_workflow.configured_providers._llm_client", return_value=object()), \
             patch("episodic_workflow.task_executors.outbreak_search.jev.post_json", side_effect=post):
            provider = firecrawl_argo_research_providers(config)
            status = project(ledger, provider.source_target_assessor)
        event = next(payload["state"]["source_event"] for payload in payloads
                     if payload["state"]["assigned_parameter"] == FIELD)
        self.assertEqual(event["kinds"], ["derived"])
        self.assertEqual(event["source_scenario"]["role"], "calibration")
        self.assertEqual(event["derivation"]["calculations"][0]["operator"], "cq_from_emission")
        self.assertEqual({row["source"] for row in event["derivation"]["sources"]}, {"https://example.test/model"})
        self.assertEqual(next(row for row in status["independent_target_values"]
                              if row["parameter"] == FIELD)["source_applicability"]["score"], .7)

    def test_derived_candidate_model_view_does_not_leak_child_fact_identities(self):
        ledger, _ = fixture()
        ledger.merge(calculate(ledger))
        project(ledger)
        candidate = next(a for a in ledger.assumptions.values() if a.get("entry_type") == "source_candidate" and a["field"] == FIELD)
        view = assumption_model_view(candidate, evidence_options=ledger.evidence_options(), numeric_options=numeric_references(ledger))
        protected = set(ledger.facts)

        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    self.assertNotIn(key, protected)
                    check(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check(item)
            elif isinstance(value, str):
                self.assertNotIn(value, protected)

        check(view)
        self.assertTrue(view["source_derivations"])

    def test_source_best_guess_ancestry_survives_projection_and_invalidates_dependents(self):
        from episodic_workflow.case_reports.research.assumptions import unassessed_assumption
        ledger, facts = fixture()
        refs = {fact.field: key for key, fact in numeric_references(ledger).items() if fact.subject == SOURCE}
        guess = unassessed_assumption({"subject": SOURCE, "field": facts[1].field,
            "value": facts[1].value, "unit": facts[1].unit,
            "statement": "Estimate emission using the source model", "basis": "Source-based emission estimate",
            "entry_type": "best_guess", "source_refs": [], "input_refs": [refs[facts[1].field]],
            "added_assumptions": ["Use the native emission estimate conditionally"]}, ledger)
        ledger.merge(ResearchResult(assumptions=(guess,)))
        selected = next(key for key, fact in numeric_references(ledger).items() if fact.parents == (guess["key"],))
        result = execute_tool({"action": "calibrate_infectiousness", "subject": SOURCE,
            "operator": "cq_from_emission", "source_inputs": [refs[facts[0].field], selected],
            "basis": "Use estimated emission and native breathing"}, ledger, DEFAULT_METRICS, review=REVIEW)
        ledger.merge(result)
        before = project(ledger)
        candidate = next(a for a in ledger.assumptions.values()
                         if a.get("entry_type") == "source_candidate" and a["field"] == FIELD)
        self.assertEqual({row["field"] for row in candidate["evidence"].values()}, {fact.field for fact in facts})
        premise = candidate["source_derivations"][0]["input_assumptions"][0]
        self.assertEqual(premise["statement"], guess["statement"])
        self.assertEqual(premise["added_assumptions"], guess["derivation"]["added_assumptions"])
        self.assertEqual(premise["review"]["verdict"], "unassessed")
        ledger.merge(ResearchResult(assumptions=({**candidate,
            "invalidated_evidence": [facts[1].semantic_record()]},)))
        after = project(ledger)
        self.assertNotIn(result.facts[0], eligible_source_values(ledger))
        self.assertTrue(ledger.assumptions[candidate["key"]]["source_candidate_unavailable"])
        self.assertEqual(before["recorded_target_inputs"]["recorded"] - after["recorded_target_inputs"]["recorded"], 1)


if __name__ == "__main__":
    unittest.main()
