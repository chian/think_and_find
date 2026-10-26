"""Target arithmetic and review consume the main table, never alternative choices."""

import unittest

from episodic_workflow.case_reports.research.calculation_steps import calculation_step_status
from episodic_workflow.case_reports.research.engine import ResearchLimits, ResearchProviders, ResearchRun
from episodic_workflow.case_reports.research.goal_state import scientific_goal_status
from episodic_workflow.case_reports.research.model_defaults import initialize_model_defaults
from episodic_workflow.case_reports.research.records import Fact, Ledger, ResearchResult, identity, numeric_references
from episodic_workflow.case_reports.research.relevance import assess_numeric_target_relationships
from episodic_workflow.case_reports.research.scenario_sharing import initialize_parameter_scope
from episodic_workflow.case_reports.research.scenarios import initial_scenarios, load_scenarios
from episodic_workflow.case_reports.research.target_values import project_source_target_values
from episodic_workflow.case_reports.research.tools import execute_tool, resolve_calculation_inputs
from episodic_workflow.case_reports.research.workflow import GuidelineSource, InvestigationSource, calculation_leaf
from episodic_workflow.task_executors.scenario_metrics.evaluate import DEFAULT_METRICS, UNITS


MASK = "mask_penetration"
EMISSION = "emission_rate_quanta_s"
CQ = "quanta_concentration_exhaled_m3"
REVIEW = {"admit": True, "completed": True, "reason": "Conditional arithmetic is coherent"}


def fixture(*, concentration=True):
    initial = initialize_model_defaults(initial_scenarios(load_scenarios()[1:4:2]))
    initial = initialize_parameter_scope(initial, lambda request: {"features": [],
        "dependencies": [{"parameter": field, "factors": [], "reason": "Equivalent target fixtures"}
                         for field in request["parameters"]]})
    ledger = Ledger(initial)
    values = [(MASK, 1), (MASK, .7), ("breathing_flow_m3_s", .00025)]
    if concentration:
        values.append((CQ, 1))
    facts = tuple(Fact("source context", field, value, UNITS[field], "source_model_parameter",
        "Reported source parameter", source="https://example.test/values",
        quote=f"{field}={value}", context=f"Source parameter context for {field}={value}")
        for field, value in values)
    ledger.merge(ResearchResult(facts=facts))
    def grade(request):
        value = request["source_profile"]["facts"][0]
        return {subject: .2 if value["field"] == MASK and value["value"] == .7 else .9
                for subject in request["target_scenarios"]}
    ledger.merge(assess_numeric_target_relationships(ledger, grade))
    ledger.merge(project_source_target_values(ledger))
    return ledger


def action(ledger):
    return {"action": "calculate_transmission_metrics", "subject": next(iter(ledger.scenarios)),
        "metrics": ["quanta_emission_rate_s", EMISSION], "parameterization": "current main inputs",
        "conditions": {"uniform_breathing_and_masks": "Common breathing and mask factors within this conditional case."},
        "basis": "Evaluate the currently selected target inputs."}


class MainTableCalculationTests(unittest.TestCase):
    def test_reused_settling_ignores_unrelated_shared_claim_review_refresh(self):
        ledger = fixture()
        origin = next(row["subject"] for row in ledger.assumptions.values() if row["field"] == CQ)
        subject = next(subject for subject in ledger.scenarios if subject != origin)

        def add_values(values):
            ledger.merge(ResearchResult(facts=tuple(Fact("source context", field, value, UNITS[field],
                "source_model_parameter", "Source parameter for dependency regression",
                source="https://example.test/removal", quote=f"{field}={value}",
                context="The source parameter's physical context") for field, value in values)))
            ledger.merge(assess_numeric_target_relationships(ledger,
                lambda request: {target: .9 for target in request["target_scenarios"]}))
            ledger.merge(project_source_target_values(ledger))

        def calculate(metrics, conditions=None):
            result = execute_tool({"action": "calculate_geometry_and_removal", "subject": subject,
                "metrics": metrics, "conditions": conditions or {}, "basis": "Reuse actual upstream arithmetic",
                "parameterization": "removal chain"}, ledger, DEFAULT_METRICS, review=REVIEW)
            ledger.merge(result)
            return result.calculations[0]

        calculate([])
        add_values((("effective_radius_m", 1e-6),))
        settling = calculate(["settling_speed_m_s"], {"stokes_particles": "Spherical low-Reynolds-number particles"})
        self.assertNotIn(CQ, settling["derived_support"]["settling_speed_m_s"]["input_fields"])
        add_values((("deactivation_rate_s", .000175), ("filter_efficiency", 0),
                    ("recirculation_flow_m3_s", 0)))
        concentration = next(row for row in ledger.assumptions.values() if row["field"] == CQ)
        self.assertNotEqual(concentration["subject"], subject, "Exercise an actual shared input")
        concentration["version"] = identity([concentration["version"], "Same claim, refreshed evidence review"])
        removal = calculate(["removal_rate_s"])
        for field in ("removal_capacity_m3_s", "removal_rate_s"):
            dependency = removal["derived_support"][field]
            self.assertNotIn(concentration["key"], {link["assumption_key"]
                             for link in dependency["scenario_parameter_links"]})
        selected = {row["field"]: row for row in calculation_step_status(ledger.result(), DEFAULT_METRICS)["steps"]
                    if row["subject"] == subject}
        for field in ("settling_speed_m_s", "removal_capacity_m3_s", "removal_rate_s"):
            self.assertEqual(selected[field]["status"], "calculated_up_to_date")
            self.assertEqual(selected[field]["points"], .125)
            self.assertTrue(selected[field]["matches_main_selection"])
        # Actual upstream changes still invalidate both the reused value and its consequences.
        ledger.scenarios[subject]["model_defaults"]["air_dynamic_viscosity_pa_s"]["decision"] = "revisit"
        stale = {row["field"]: row for row in calculation_step_status(ledger.result(), DEFAULT_METRICS)["steps"]
                 if row["subject"] == subject}
        for field in ("settling_speed_m_s", "removal_capacity_m3_s", "removal_rate_s"):
            self.assertEqual(stale[field]["status"], "calculated_needs_updating")
            self.assertEqual(stale[field]["points"], .025)

    def test_hidden_lower_relevance_reference_cannot_override_arithmetic_or_current_credit(self):
        ledger = fixture()
        proposal = action(ledger)
        alternative = next(row for row in ledger.assumptions.values()
                           if row["field"] == MASK and row["value"] == .7)
        proposal["source_inputs"] = [alternative["key"], "invented-hidden-override"]
        before = scientific_goal_status(ledger.result(), DEFAULT_METRICS)
        self.assertEqual(before["target_inputs"][proposal["subject"]][MASK]["value"], 1)
        result = execute_tool(proposal, ledger, DEFAULT_METRICS, review=REVIEW)
        calculation = result.calculations[-1]
        self.assertEqual(calculation["input_values"][MASK], 1)
        self.assertAlmostEqual(calculation["output"][EMISSION]["value"], .00025)
        self.assertNotIn(alternative["key"], calculation["assumption_versions"])
        selected = before["target_inputs"][proposal["subject"]][MASK]["selected_assumption_key"]
        self.assertIn(selected, calculation["assumption_versions"])
        ledger.merge(result)
        step = next(row for row in calculation_step_status(ledger.result(), DEFAULT_METRICS)["steps"]
                    if row["subject"] == proposal["subject"] and row["field"] == EMISSION)
        self.assertEqual(step["status"], "calculated_up_to_date")
        self.assertEqual(step["points"], .125)
        self.assertAlmostEqual(step["computed_value"], .00025)

    def test_root_and_investigation_model_review_see_the_same_values_execution_uses(self):
        for kind in (GuidelineSource, InvestigationSource):
            with self.subTest(episode=kind.__name__):
                ledger, reviews = fixture(), []
                def agent(request):
                    if request["operation"] == "assess_assumption_reasonableness":
                        return {"verdict": "unsupported", "request_constraint": None,
                            "criteria": {name: {"grade": 0, "reason": "Unverified transfer", "evidence": []}
                                         for name in ("support", "applicability", "physical_consistency", "uncertainty")},
                            "limitations": ["Conditional input"], "alternatives": [], "missing_evidence": [],
                            "explanation": "No empirical support established", "supporting_evidence": [],
                            "contradicting_evidence": [], "invalid_evidence": []}
                    self.assertEqual(request["operation"], "review_research_model")
                    reviews.append(request)
                    return REVIEW
                run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None), ResearchLimits(),
                    "Main target inputs", DEFAULT_METRICS, initial=ledger.result())
                source = kind(run) if kind is GuidelineSource else kind(run, ledger.result())
                source.leaf = lambda callback, label: callback()
                proposal = action(ledger)
                result = calculation_leaf(source, proposal, Ledger(ledger.result()), source.context())
                self.assertEqual(len(reviews), 1)
                reviewed = reviews[0]
                self.assertNotIn("facts", reviewed["proposal"])
                self.assertEqual(reviewed["resolved_numeric_inputs"][MASK]["value"], 1)
                self.assertEqual(result.calculations[-1]["input_values"][MASK], 1)
                self.assertAlmostEqual(result.calculations[-1]["output"][EMISSION]["value"], .00025)

    def test_missing_main_value_cannot_be_supplied_by_a_raw_target_record(self):
        ledger = fixture(concentration=False)
        proposal = action(ledger)
        hidden = Fact(proposal["subject"], CQ, 999, UNITS[CQ], "user_assumption", "Unselected raw value")
        ledger.merge(ResearchResult(facts=(hidden,)))
        result = execute_tool(proposal, ledger, DEFAULT_METRICS, review=REVIEW)
        self.assertNotIn(CQ, result.calculations[-1]["input_values"])
        self.assertNotEqual(result.calculations[-1]["output"][EMISSION].get("status"), "calculated")

    def test_removed_selection_interfaces_and_target_transfer_cannot_bypass_main_table(self):
        from episodic_workflow.case_reports.research import modeling, tools
        self.assertFalse(hasattr(tools, "input_options"))
        self.assertFalse(hasattr(modeling, "select_facts"))
        ledger = fixture()
        proposal = {**action(ledger), "action": "calibrate_infectiousness",
                    "operator": "transfer_parameter"}
        with self.assertRaises(ValueError):
            resolve_calculation_inputs(proposal, ledger, DEFAULT_METRICS)

    def test_source_calibration_keeps_native_values_and_rejects_target_records(self):
        ledger = fixture()
        source = "independent calibration event"
        native = tuple(Fact(source, field, value, UNITS[field], "source_model_parameter",
            "Source-event conditional input", source="https://example.test/event",
            quote=f"{field}={value}", context="This source event")
            for field, value in (("breathing_flow_m3_s", .001),
                                 ("quanta_emission_rate_s", .05)))
        ledger.merge(ResearchResult(facts=native, scenarios=({"subject": source,
            "role": "calibration", "description": "Distinct native event"},)))
        proposal = {"action": "calibrate_infectiousness", "subject": source,
            "operator": "cq_from_emission", "source_inputs": [ref for ref, fact in numeric_references(ledger).items() if fact in native],
            "basis": "Use the source event's emission and breathing"}
        result = execute_tool(proposal, ledger, DEFAULT_METRICS, review=REVIEW)
        self.assertEqual(result.calculations[-1]["input_values"]["breathing_flow_m3_s"], .001)
        self.assertEqual(result.facts[0].value, 50)
        from episodic_workflow.case_reports.research import tools
        target_fact = next(fact for fact in tools.main_calculation_inputs(ledger, next(iter(ledger.scenarios)))
                           if fact.field == "breathing_flow_m3_s")
        with self.assertRaises(ValueError):
            execute_tool({**proposal, "source_inputs": [next(ref for ref, fact in numeric_references(ledger).items() if fact.key == target_fact.key), proposal["source_inputs"][1]]},
                         ledger, DEFAULT_METRICS, review=REVIEW)

    def test_source_to_source_transfer_preserves_native_value_without_target_override(self):
        ledger = fixture()
        destination = "separate source calibration"
        ledger.merge(ResearchResult(scenarios=({"subject": destination, "role": "calibration",
            "description": "A distinct calibration model"},)))
        references = numeric_references(ledger)
        native_ref = next(ref for ref, fact in references.items()
                          if fact.subject == "source context" and fact.field == MASK and fact.value == .7)
        proposal = {"action": "calibrate_infectiousness", "subject": destination,
            "operator": "transfer_parameter", "source_inputs": [native_ref],
            "basis": "Conditional transfer between source model contexts"}
        result = execute_tool(proposal, ledger, DEFAULT_METRICS, review=REVIEW)
        self.assertEqual((result.facts[0].subject, result.facts[0].value), (destination, .7))
        target_ref = next(ref for ref, fact in references.items()
                          if fact.subject == next(iter(ledger.scenarios)) and fact.field == MASK)
        with self.assertRaises(ValueError):
            execute_tool({**proposal, "source_inputs": [target_ref]}, ledger, DEFAULT_METRICS, review=REVIEW)

    def test_source_review_rebinds_local_labels_to_the_same_selected_records(self):
        from types import SimpleNamespace
        from episodic_workflow.case_reports.research.assumptions import RUBRIC, unassessed_assumption
        ledger = fixture()
        subject = "reviewed source event"
        flow = Fact(subject, "breathing_flow_m3_s", .001, "m3/s", "source_model_parameter",
            "Native source breathing", source="https://example.test/flow", quote="Flow .001 m3/s",
            context="The actual source event")
        ledger.merge(ResearchResult(facts=(flow,), scenarios=({"subject": subject, "role": "calibration",
            "description": "Source event with conditional emission"},)))
        claim = unassessed_assumption({"subject": subject, "field": "quanta_emission_rate_s",
            "value": .06, "unit": "quanta/s", "statement": "Conditional source emission",
            "basis": "Explicit source estimate", "entry_type": "best_guess",
            "source_refs": [], "input_refs": [], "added_assumptions": ["Conditional emission"]}, ledger)
        ledger.merge(ResearchResult(assumptions=(claim,)))
        before = numeric_references(ledger)
        selected_ref = next(ref for ref, fact in before.items() if fact.parents == (claim["key"],))
        model_requests = []
        def agent(request):
            if request["operation"] == "review_research_model":
                model_requests.append(request)
                return REVIEW
            refs = list(request["evidence"])
            return {"verdict": "provisional", "request_constraint": None,
                "criteria": {name: {"grade": 1, "reason": "Conditional source support", "evidence": refs[:1]}
                             for name in RUBRIC},
                "limitations": ["Source estimate remains conditional"], "alternatives": [],
                "missing_evidence": [], "explanation": "Conditional source input",
                "supporting_evidence": refs[:1], "contradicting_evidence": [], "invalid_evidence": []}
        source = SimpleNamespace(run=SimpleNamespace(complete=agent, metrics=DEFAULT_METRICS),
            leaf=lambda callback, _: callback(),
            failure_result=lambda exc, stage: ResearchResult(issues=(str(exc),)))
        result = calculation_leaf(source, {"action": "calibrate_infectiousness", "subject": subject,
            "operator": "cq_from_emission", "source_inputs": [next(ref for ref, fact in before.items()
                if fact.key == flow.key), selected_ref], "basis": "Keep exact source operands"}, ledger, {})
        after_ref = next(ref for ref, fact in numeric_references(ledger).items()
                         if fact.key == before[selected_ref].key)
        self.assertNotEqual(selected_ref, after_ref, "The fixture must exercise a shifted parent-local label")
        self.assertEqual(model_requests[0]["resolved_numeric_inputs"]["quanta_emission_rate_s"]["value"], .06)
        self.assertEqual(result.calculations[-1]["input_values"]["quanta_emission_rate_s"], .06)
        self.assertEqual(result.facts[-1].value, 60)


if __name__ == "__main__":
    unittest.main()
