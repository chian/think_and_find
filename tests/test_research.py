"""Controlled agency, closed child boundaries, and conditional scientific outputs."""

import json
import math
import unittest
from unittest.mock import patch
import io

from episodic_workflow.case_reports.research import ResearchLimits, ResearchProviders, run_guideline_research
from episodic_workflow.case_reports.research.records import Credit
from episodic_workflow.credit_assignment.research import ResearchController
from episodic_workflow.task_executors.scenario_metrics.evaluate import evaluate_parameters
from episodic_workflow.task_executors.scenario_metrics.physics import (
    exact_time_bound, integrated_concentration, threshold_distance,
)
from episodic_workflow.case_reports.research.admission import extract_and_admit, EVIDENCE_FIELDS
from episodic_workflow.case_reports.research.modeling import evaluate_model
from episodic_workflow.case_reports.research.records import Fact
from episodic_workflow.case_reports.research.records import Ledger, ResearchResult
from episodic_workflow.case_reports.research.assumptions import RUBRIC, assess_assumption
from episodic_workflow.task_executors.scenario_metrics.evaluate import UNITS
from episodic_workflow.task_executors.scenario_metrics.catalog import METRICS as CATALOG
from episodic_workflow.task_executors.scenario_metrics.evaluate import CONDITIONS as MODEL_CONDITIONS


PARAMETERS = {
    "breathing_flow_m3_s": 0.00015, "removal_capacity_m3_s": 0.1,
    "mask_penetration": 1, "relative_transmissibility": 1,
    "susceptible_count": 4, "occupant_count": 5, "exposure_time_s": 3600,
    "risk_tolerance": 0.1, "mouth_area_m2": 0.0002, "entrainment_coefficient": 0.1,
    "plume_encounter_probability": 0.1, "room_volume_m3": 100,
    "initial_concentration_quanta_m3": 0,
}
CONDITIONS = {name: "Explicit test scenario assumption" for name in (
    "well_mixed", "single_infectious_source", "uniform_breathing_and_masks", "continuous_unmasked_jet")}
METRICS = ("exposure_time_bound_s", "indoor_reproductive_number", "critical_distance_m", "risk_threshold_distance_m")
REQUEST = "Investigate conditional time, risk and distance estimates for a synthetic indoor scenario, with evidence for assumptions."
VALUES = {**PARAMETERS, "quanta_concentration_exhaled_m3": 30}
SOURCE_TEXT = "\n".join(f"{field} = {value} {UNITS[field]}." for field, value in VALUES.items())
SOURCE_TEXT += "\nThe synthetic scenario supports the stated well-mixed, single-source, uniform-breathing and continuous-unmasked-jet conditions."


def supported_review(request, *, grade=2, verdict="supported"):
    return {"verdict": verdict, "request_constraint": None,
            "criteria": {name: {"grade": grade, "reason": "Synthetic evidence establishes this criterion within this fixture",
                                "evidence": list(request["evidence"])} for name in RUBRIC},
            "limitations": ["Synthetic fixture; not empirical validation"],
            "alternatives": ["Other activities need separate evidence"], "missing_evidence": [],
             "supporting_evidence": list(request["evidence"]) if verdict in {"supported", "provisional"} else [],
             "contradicting_evidence": list(request["evidence"]) if verdict == "contradicted" else [],
             "invalid_evidence": [],
            "explanation": "The synthetic source supports this conditional assumption; it is not independent validation."}


def synthetic_information_review(request):
    """Fixture judgments only; never a production substitute for semantic review."""
    rows = []
    for field in request["focus"]:
        refs = [key for key in request["new_evidence"] if
                request["evidence"][key].get("field") == field or field in request["evidence"][key].get("fields", ())]
        rows.append({"field": field, "status": "resolved" if refs else "background",
                     "answer": "Synthetic fixture establishes this value or condition" if refs else "No new fixture evidence",
                     "evidence": refs, "limitations": ["Software fixture, not empirical validation"]})
    return {"judgments": rows}


def synthetic_synthesis(request):
    result, retained = request["admitted_result"], request["retained_parent_context"]
    def failures(options):
        return [{"references": [key], "meaning": item.get("why", "Fixture failure"),
                 "not_established": "No scientific disproof follows from a failed attempt"}
                for key, item in options.items()]
    return {"child_summary": {"findings": result["findings"], "uncertainties": result["unresolved"],
                              "failures": failures(request["child_diagnostics"])},
            "parent_summary": {"findings": list(dict.fromkeys([*retained["findings"], *result["findings"]])),
                               "uncertainties": list(dict.fromkeys([*retained["uncertainties"], *result["unresolved"]])),
                               "failures": failures(request["parent_diagnostics"])}}


class AdaptiveAgent:
    def __init__(self, subjects=("target",)):
        self.requests = []
        self.subjects = subjects
        self.query_count = 0

    def __call__(self, request):
        self.requests.append(request)
        op = request["operation"]
        if op == "propose_guideline_action":
            facts = request["admitted_facts"]
            common = {"objective": "Resolve the activity-specific exhaled concentration and its implications",
                      "focus": request["focus"]}
            subject = self.subjects[0]
            scenarios = [{"subject": name, "description": "Synthetic indoor study scenario",
                          "basis": "Scope aligned with the original synthetic request"} for name in self.subjects
                         if name not in request["context"]["defined_scenarios"]]
            assessments = request["assumption_options"].values()
            needed = [f for f in request["focus"] if f in VALUES or f in CONDITIONS]
            missing = [f for f in needed if not any(a["field"] == f and a["subject"] == subject and a["reasonableness"] > 0 for a in assessments)]
            if missing:
                return {**common, "action": "investigate", "focus": needed, "scenarios": scenarios, "subjects": [subject]}
            calculated = {m for c in request["calculations"].values() for m in c["output"]}
            action = "calculate_transmission_metrics" if "indoor_reproductive_number" not in calculated else "calculate_conditional_limits"
            return {**common, "action": action, "subject": subject,
                    "conditions": CONDITIONS, "basis": "Explicitly assessed synthetic assumptions"}
        if op == "propose_investigation_strategy":
            return {"approach": "Find evidence for the synthetic physical and infectiousness hypotheses",
                    "objective": "Resolve the requested assumptions", "candidates": [
                        {"subject": self.subjects[0], "field": field, "value": VALUES.get(field),
                         "unit": UNITS.get(field, "condition"),
                         "statement": CONDITIONS.get(field, f"Conditional {field}"),
                         "basis": "Synthetic hypothesis to be checked against retrieved evidence"}
                        for field in request["focus"] if field in VALUES or field in CONDITIONS]}
        if op == "propose_strategy_query":
            self.query_count += 1
            return {"query": f"synthetic evidence approach {self.query_count}", "objective": "Find applicable evidence"}
        if op == "propose_page_child":
            return {"option": "report", "question": "Read for activity-specific evidence"}
        if op == "review_scenario_scope":
            return {"admit": True, "reason": "Synthetic study scope matches the prompt"}
        if op == "assess_assumption_reasonableness":
            return supported_review(request)
        if op == "choose_research_source":
            return {"option": next(iter(request["candidates"])), "question": "Find the exhaled infectious concentration, not viral RNA"}
        if op == "propose_source_reading":
            return {"window": "window:1", "question": "Find exact activity-specific quanta concentration and limitations"}
        if op == "extract_research_evidence":
            if request["source"] == "request":
                return {"facts": [], "evidence": [], "findings": []}
            if request["source"].endswith("bad"):
                return {"facts": [], "findings": ["This source only reports viral RNA; use an infectivity study instead."]}
            if request["source_text"] == "Cq = 30 quanta/m3 for this activity." or "Cq = 30\nquanta/m3" in request["source_text"]:
                return {"facts": [{"subject": "reference", "field": "quanta_concentration_exhaled_m3",
                               "value": 30, "unit": "quanta/m3", "quote": "Cq = 30 quanta/m3 for this activity.",
                               "kind": "source_model_parameter", "context": "Same synthetic disease and activity as target",
                               "basis": "Activity-specific fitted concentration, not measured viral copies"}],
                    "findings": ["The activity matches, but this is a fitted infectiousness parameter."]}
            return {"facts": [{"subject": "reference", "field": field, "value": value, "unit": UNITS[field],
                               "quote": f"{field} = {value} {UNITS[field]}.", "kind": "source_model_parameter",
                               "context": "Synthetic fixture scenario", "basis": "Fitted synthetic parameter"}
                              for field, value in VALUES.items()],
                    "evidence": [{"subject": "reference", "claim": "Synthetic model conditions apply", "fields": list(CONDITIONS),
                                  "quote": SOURCE_TEXT.splitlines()[-1], "context": "Synthetic fixture scenario",
                                  "basis": "Explicit conditions, not demonstrated empirical uniformity"}],
                    "findings": ["These are fitted synthetic parameters, not independent validation."]}
        if op == "review_research_evidence":
            return {"decisions": [{"option": item["option"], "admit": True, "reason": "Exact reported value, compatible role and activity"}
                                  for item in request["candidates"]]}
        if op == "review_research_model":
            return {"admit": True, "reason": "Consistent synthetic model inputs", "limitations": ["Conditional model, not independent validation"]}
        if op == "assess_information_progress":
            return synthetic_information_review(request)
        if op == "synthesize_research_findings":
            return synthetic_synthesis(request)
        raise AssertionError(op)


def run_synthetic(agent=None, limits=None):
    agent = agent or AdaptiveAgent()
    providers = ResearchProviders(agent, lambda *_: (
        {"url": "https://example.test/bad", "title": "RNA", "text": "Only RNA copies are reported."},
        {"url": "https://example.test/good", "title": "Synthetic evidence", "text": SOURCE_TEXT},
    ), lambda _: None)
    return agent, run_guideline_research(research_request=REQUEST, providers=providers, metrics=METRICS,
        limits=limits or ResearchLimits())


class ResearchControlTests(unittest.TestCase):
    def test_numeric_yield_not_agent_stop_or_child_count(self):
        controller = ResearchController(("cq",), window=3, capacity=10)
        first = controller.observe("first", Credit((("cq", "one"),)), is_root=True)
        self.assertGreater(first.gain, 0)
        failure = controller.observe("failed", Credit(evaluated=False), is_root=True)
        self.assertFalse(failure.stop)
        self.assertEqual(len(controller.state().history), 2)
        for i in range(3):
            step = controller.observe(str(i), Credit((("cq", "one"),)), is_root=True)
        self.assertTrue(step.stop)
        self.assertEqual(step.reason, "saturated")
        self.assertEqual(controller.state().history, (0, .1, .1, .1, .1))

    def test_parent_recomputes_deduplicated_credit(self):
        c = ResearchController(("cq",), capacity=10)
        c.observe("child1", Credit((("cq", "one"), ("cq", "two"))), is_root=True)
        step = c.observe("child2", Credit((("cq", "two"),)), is_root=True)
        self.assertEqual(step.gain, 0)
        self.assertEqual(step.score, .2)

    def test_realized_source_findings_steer_next_selection(self):
        agent, (record, report) = run_synthetic()
        self.assertTrue(report.passed, report.as_record())
        selections = [r for r in agent.requests if r["operation"] == "choose_research_source"]
        self.assertGreaterEqual(len(selections), 2)
        self.assertIn("viral RNA", json.dumps(selections[1]["child_reports"]))
        later = [r for r in agent.requests if r["operation"] == "propose_guideline_action" and r.get("child_reports")]
        self.assertTrue(any("fitted" in json.dumps(r["child_reports"]) for r in later))
        self.assertEqual(record.scope_level, "evaluate_safety_guideline")
        child = record.unit_records[0].child
        self.assertEqual(child.path[:-1], record.path)
        self.assertEqual(child.unit_records[0].child.unit_records[0].child.scope_level, "search")
        self.assertEqual(record.ended_by, "yield_stop")
        self.assertIn(report.data["control"]["reason"], {"saturated", "complete"})
        json.dumps(report.as_record(), allow_nan=False)
        self.assertIn("critical_distance_m", report.markdown())
        for fact in report.data["evidence_and_assumptions"]:
            if fact["kind"].startswith("source_"):
                self.assertIn(fact["quote"], report.data["source_documents"][fact["source"]]["text"])

    def test_budget_is_not_convergence_or_success(self):
        _, (record, report) = run_synthetic(limits=ResearchLimits())
        self.assertFalse(report.passed)
        self.assertEqual(record.ended_by, "source_failed")
        self.assertIn("budget", report.data["control"]["reason"])

    def test_research_provider_composition_without_jev(self):
        from episodic_workflow.configured_providers import run_configured_guideline_research
        agent = AdaptiveAgent()
        env = {"FIRECRAWL_API_KEY": "test", "EPISODIC_LLM_BASE_URL": "https://example.test/v1", "EPISODIC_LLM_MODEL": "test"}
        results = ({"url": "https://example.test/good", "text": SOURCE_TEXT},)
        with patch("episodic_workflow.configured_providers.workflow_environment", return_value=env), \
             patch("episodic_workflow.task_executors.evidence_extraction.argo.ArgoClient.call", side_effect=lambda prompt, **_: json.dumps(agent(json.loads(prompt)))) as calls, \
             patch("episodic_workflow.task_executors.outbreak_search.firecrawl.firecrawl_search", return_value=results):
            _, report = run_configured_guideline_research(
                research_request=REQUEST, metrics=METRICS,
                limits=ResearchLimits())
        self.assertTrue(report.passed, report.as_record())
        self.assertTrue(all("JSON-escape backslashes" in call.kwargs["system_prompt"] for call in calls.call_args_list))
        for call in calls.call_args_list:
            system = call.kwargs["system_prompt"]
            self.assertIn("main table", system)
            self.assertIn("Within investigate", system)
            self.assertIn("values need not have been printed verbatim", system)
            self.assertIn("mandatory independent reasonableness assessment", system)
            self.assertIn("calculated without evidence support", system)
            self.assertIn("not permission to compute", system)
            self.assertNotIn("not admitted facts or calculation inputs until", system)
            self.assertNotIn("When proposing a scientific alternative, use only admitted facts", system)

    def test_one_scenario_cannot_complete_another_scenario(self):
        _, (record, report) = run_synthetic(AdaptiveAgent(("left", "right")))
        self.assertFalse(report.passed)
        self.assertTrue(report.data["scenarios"]["left"]["complete_parameterization_available"])
        self.assertFalse(report.data["scenarios"]["right"]["complete_parameterization_available"])
        self.assertEqual(record.controller_state.reason, "saturated")

    def test_failed_children_do_not_turn_into_evaluated_zero_at_parent(self):
        base = AdaptiveAgent()
        def agent(request):
            if request["operation"] == "extract_research_evidence":
                return {"facts": [{"field": "unsupported", "value": 12}]}
            return base(request)
        _, (record, report) = run_synthetic(agent, ResearchLimits())
        self.assertFalse(report.passed)
        self.assertTrue(all(v == 0 for v in record.controller_state.history))
        self.assertGreater(record.controller_state.failures, 0)


class ResearchAdmissionTests(unittest.TestCase):
    def extract(self, agent, text="Cq = 30 quanta/m3 for this activity."):
        return extract_and_admit(agent, text=text, source="https://example.test/good", focus=("quanta_concentration_exhaled_m3",),
                                 objective="Test evidence", context={}, question="Cq?")

    def test_review_cannot_rescue_a_fabricated_number(self):
        base = AdaptiveAgent()
        def agent(request):
            response = base(request)
            if request["operation"] == "extract_research_evidence":
                response["facts"][0]["value"] = 999
            return response
        result = self.extract(agent)
        self.assertFalse(result.evaluated)
        self.assertFalse(result.facts)
        self.assertNotIn("review_research_evidence", [r["operation"] for r in base.requests])

    def test_extraction_contract_declares_types_and_admissible_evidence_fields(self):
        base = AdaptiveAgent()
        self.extract(base)
        request = base.requests[0]
        schema = request["response_contract"]
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["facts"]["items"]["properties"]["value"]["type"], "number")
        fields = schema["properties"]["evidence"]["items"]["properties"]["fields"]["items"]["enum"]
        self.assertEqual(set(fields), EVIDENCE_FIELDS)
        self.assertIn("well_mixed", request["model_conditions"])
        self.assertIn("indoor_reproductive_number", fields)

    def test_numeric_strings_and_ranges_are_not_silently_coerced(self):
        for value in ("30", "20-40", None):
            with self.subTest(value=value):
                base = AdaptiveAgent()
                def agent(request):
                    response = base(request)
                    if request["operation"] == "extract_research_evidence":
                        response["facts"][0]["value"] = value
                    return response
                result = self.extract(agent)
                self.assertFalse(result.facts)
                self.assertFalse(result.evaluated)
                self.assertIn("finite JSON number", result.diagnostics[0]["why"])
                self.assertEqual(result.diagnostics[0]["proposal"]["value"], value)

    def test_ranges_and_output_equations_can_be_evidence_but_not_computed_answers(self):
        quote = "Synthetic source: room height is 2-6 m; R_in = Ns * sr * beta * time."
        def agent(request):
            if request["operation"] == "extract_research_evidence":
                return {"facts": [], "findings": [], "evidence": [{
                    "subject": "reference", "claim": "Reported height range and model relation",
                    "fields": ["room_height_m", "indoor_reproductive_number"], "quote": quote,
                    "context": "Synthetic model, not an observation", "basis": "Bounds and equation, not a scalar answer"}]}
            return {"decisions": [{"option": 0, "admit": True, "reason": "Supported synthetic relation"}]}
        result = self.extract(agent, quote)
        self.assertEqual(len(result.evidence), 1)
        self.assertFalse(result.facts)
        self.assertFalse(result.estimates)
        self.assertFalse(result.calculations)
        credit = result.credit(("indoor_reproductive_number",), scientific=True)
        self.assertFalse(credit.identities)
        self.assertFalse(any(row[2] for row in credit.qualities))

    def test_unknown_evidence_field_diagnostic_names_the_actual_invalid_field(self):
        result = self.extract(lambda _: {"evidence": [{"subject": "reference", "claim": "Some claim",
            "fields": ["wrong_field_name"], "quote": "Cq = 30 quanta/m3 for this activity.",
            "context": "Synthetic source", "basis": "Test"}]})
        self.assertIn("wrong_field_name", result.diagnostics[0]["why"])
        self.assertFalse(result.evidence)

    def test_reported_rate_notation_and_centimetre_velocity_preserve_dimensions(self):
        for field, unit, expected in (("air_exchange_rate_s", "/h", 2/3600),
                                      ("air_exchange_rate_s", "1 / h", 2/3600),
                                      ("air_exchange_rate_s", "1 / s", 2),
                                      ("air_exchange_rate_s", "h-1", 2/3600),
                                      ("air_exchange_rate_s", "h\u22121", 2/3600),
                                      ("air_exchange_rate_s", "h \u22121", 2/3600),
                                      ("air_exchange_rate_s", " h ^ -1 ", 2/3600),
                                      ("air_exchange_rate_s", "day-1", 2/86400),
                                      ("exposure_time_s", "days", 2*86400),
                                      ("neighbor_distance_m", "ft", 2*.3048),
                                      ("room_volume_m3", "ft\u00b3", 2*.3048**3),
                                      ("floor_area_m2", "ft^2", 2*.3048**2),
                                      ("floor_area_m2", "m 2", 2),
                                      ("floor_area_m2", "m^2", 2),
                                      ("room_volume_m3", "m 3", 2),
                                      ("floor_area_m2", "cm \u00b2", 2e-4),
                                      ("room_volume_m3", "m 2", None),
                                      ("floor_area_m2", "m 2/s", None),
                                      ("exposure_time_s", "m s", None),
                                      ("ventilation_flow_m3_s", "cfm", 2*.3048**3/60),
                                      ("relative_transmissibility", "dimensionless", 2),
                                      ("risk_tolerance", "%", .02),
                                      ("mask_penetration", "dimensionless", None),
                                      ("settling_speed_m_s", "cm/s", .02),
                                      ("quanta_concentration_exhaled_m3", "q/m3", 2),
                                      ("quanta_concentration_exhaled_m3", "quanta/m 3", 2),
                                      ("quanta_concentration_exhaled_m3", "infection quanta / m3 exhaled air", 2),
                                      ("breathing_flow_m3_s", "m 3 /h", 2/3600),
                                      ("breathing_flow_m3_s", "m 3 / s", 2),
                                      ("floor_area_m2", "m 2 /s", None),
                                      ("quanta_emission_rate_s", "q/h", 2/3600),
                                      ("occupant_count", "people", 2),
                                      ("air_dynamic_viscosity_pa_s", "\u03bc Pa\u00b7 s", 2e-6),
                                      ("air_dynamic_viscosity_pa_s", "\u00b5Pa*s", 2e-6),
                                      ("air_dynamic_viscosity_pa_s", "uPa*s", 2e-6),
                                      ("removal_rate_s", "\u03bc Pa\u00b7 s", None),
                                      ("removal_rate_s", "cm/s", None)):
            with self.subTest(field=field, unit=unit):
                quote = f"Synthetic reported {field}: 2 {unit}."
                def agent(request):
                    if request["operation"] == "extract_research_evidence":
                        return {"facts": [{"field": field, "value": 2, "unit": unit, "quote": quote,
                            "subject": "reference", "context": "Synthetic test", "basis": "Reported quantity",
                            "kind": "source_model_parameter"}]}
                    conversion = request["candidates"][0]["unit_conversion"]
                    self.assertEqual(conversion["printed_value"], 2)
                    self.assertEqual(conversion["printed_unit"], unit)
                    self.assertAlmostEqual(2 * conversion["factor"], expected)
                    self.assertAlmostEqual(request["candidates"][0]["normalized"]["value"], expected)
                    return {"decisions": [{"option": 0, "admit": True, "reason": "Compatible synthetic quantity"}]}
                result = self.extract(agent, quote)
                if expected is None:
                    self.assertFalse(result.facts)
                    self.assertIn("must be <= 1" if field == "mask_penetration" else "Unsupported unit conversion", result.issues[0])
                else:
                    self.assertAlmostEqual(result.facts[0].value, expected)
                    self.assertEqual(result.facts[0].quote, quote)

    def test_aerosol_volume_ratio_notation_has_cubic_dimension_conversion(self):
        for unit in ("(um/cm)^3", "(\u03bcm/cm)3", "(\u00b5m/cm)\u00b3", "um3/cm3", "\u03bcm^3/cm^3"):
            with self.subTest(unit=unit):
                quote = f"Synthetic reported volume fraction: 2 {unit}."
                def agent(request):
                    if request["operation"] == "extract_research_evidence":
                        return {"facts": [{"field": "aerosol_volume_fraction", "value": 2,
                            "unit": unit, "quote": quote, "subject": "reference", "context": "Synthetic test",
                            "basis": "Reported volume ratio", "kind": "source_model_parameter"}]}
                    candidate = request["candidates"][0]
                    self.assertTrue(math.isclose(candidate["unit_conversion"]["factor"], 1e-12, rel_tol=1e-12))
                    self.assertTrue(math.isclose(candidate["normalized"]["value"], 2e-12, rel_tol=1e-12))
                    return {"decisions": [{"option": 0, "admit": True, "reason": "Compatible synthetic ratio"}]}
                result = self.extract(agent, quote)
                self.assertTrue(math.isclose(result.facts[0].value, 2e-12, rel_tol=1e-12))
                self.assertEqual(result.facts[0].quote, quote)

    def test_whitespace_recovery_retains_exact_original_span(self):
        result = self.extract(AdaptiveAgent(), "Cq = 30\nquanta/m3 for this activity.")
        self.assertEqual(result.facts[0].quote, "Cq = 30\nquanta/m3 for this activity.")

    def test_applicability_rejection_does_not_admit_evidence(self):
        base = AdaptiveAgent()
        def agent(request):
            if request["operation"] == "review_research_evidence":
                return {"decisions": [{"option": 0, "admit": False, "reason": "Different activity"}]}
            return base(request)
        result = self.extract(agent)
        self.assertTrue(result.evaluated)
        self.assertFalse(result.facts)
        self.assertIn("Different activity", result.issues[0])

    def test_evaluation_does_not_require_evidence_support(self):
        facts = (Fact("target", "mask_penetration", .3, "fraction", "user_assumption", "test"),)
        result = evaluate_model(AdaptiveAgent(), {"subject": "target",
            "basis": "Synthetic hypothesis"}, facts, {}, METRICS)
        self.assertTrue(result.estimates)
        self.assertEqual(result.estimates[0]["reasonableness"], 0)
        self.assertTrue(result.estimates[0]["support_issues"])

    def test_llm_cannot_stop_loop(self):
        providers = ResearchProviders(lambda _: {"action": "stop"}, lambda *_: (), lambda _: None)
        record, report = run_guideline_research(research_request="Investigate distance limits",
            providers=providers, metrics=("critical_distance_m",), limits=ResearchLimits())
        self.assertFalse(report.passed)
        self.assertEqual(record.units_consumed, 3)
        self.assertEqual(record.controller_state.history, (0,))
        self.assertEqual(record.ended_by, "source_failed")


class ScenarioMetricTests(unittest.TestCase):
    def evaluate(self, parameters=None, metrics=METRICS, conditions=None):
        return evaluate_parameters(parameters or {**PARAMETERS, "quanta_concentration_exhaled_m3": 30},
            CONDITIONS if conditions is None else conditions, metrics, subject="test", label="defined test")

    def test_complete_catalog_executes_under_explicit_synthetic_conditions(self):
        values = {**VALUES, "floor_area_m2": 50, "room_height_m": 2,
            "air_exchange_rate_s": 1/3600, "recirculation_flow_m3_s": .05,
            "filter_efficiency": .9, "deactivation_rate_s": .3/3600,
            "effective_radius_m": 1e-6, "particle_density_kg_m3": 1000,
            "air_density_kg_m3": 1.2, "air_dynamic_viscosity_pa_s": 1.8e-5,
            "gravity_m_s2": 9.8, "clearance_fraction": .9,
            "initial_radius_m": 1e-5, "solute_volume_fraction": .01,
            "relative_humidity": .5, "evaporation_coefficient_m2_s": 4.2e-10,
            "microscopic_infectivity_per_virion": .1, "pathogen_concentration_virions_m3": 1e10,
            "aerosol_volume_fraction": 3e-8, "activity_aerosol_volume_fraction": 3e-8,
            "baseline_aerosol_volume_fraction": 3e-8, "neighbor_distance_m": 1.5,
            "breath_volume_m3": .0005}
        del values["removal_capacity_m3_s"]
        conditions = {name: "Explicit synthetic model condition; not empirical evidence" for name in MODEL_CONDITIONS}
        result = self.evaluate(values, tuple(CATALOG), conditions)
        self.assertFalse(result["issues"])
        self.assertEqual(set(result["metrics"]), set(CATALOG))
        for name, row in result["metrics"].items():
            with self.subTest(metric=name):
                self.assertEqual(row["status"], "calculated", row.get("missing"))
                self.assertEqual(row["unit"], CATALOG[name].unit)
                self.assertIn(row["limit"], {"finite", "unbounded"})
                if row["limit"] == "finite":
                    self.assertGreaterEqual(row["value"], 0)
                    if row["unit"] == "probability":
                        self.assertLessEqual(row["value"], 1)
        json.dumps(result, allow_nan=False)

    def test_distance_time_and_rin_are_distinct(self):
        m = self.evaluate()["metrics"]
        self.assertAlmostEqual(m["indoor_reproductive_number"]["value"], .0972)
        self.assertAlmostEqual(m["exposure_time_bound_s"]["value"], .1/(4*6.75e-6))
        self.assertAlmostEqual(m["critical_distance_m"]["value"], math.sqrt(.0002)/(.1*.0015))
        self.assertAlmostEqual(m["risk_threshold_distance_m"]["value"], .0972*.1*m["critical_distance_m"]["value"]/(4*(.1-.0972)))

    def test_clearance_requires_source_off_not_continuing_emissions(self):
        parameters = {"removal_rate_s": .002, "clearance_fraction": .9}
        metrics = ("air_clearance_time_s", "concentration_relaxation_time_s")
        conditions = {"well_mixed": "Synthetic spatially uniform room"}
        rows = self.evaluate(parameters, metrics, conditions)["metrics"]
        self.assertEqual(rows["concentration_relaxation_time_s"]["status"], "calculated")
        self.assertEqual(rows["air_clearance_time_s"]["missing"], ["condition:emissions_stopped"])
        conditions["emissions_stopped"] = "Synthetic source removed before the clearance interval"
        rows = self.evaluate(parameters, metrics, conditions)["metrics"]
        self.assertEqual(rows["air_clearance_time_s"]["status"], "calculated")
        self.assertAlmostEqual(rows["air_clearance_time_s"]["value"], -math.log(.1)/.002)

    def test_critical_aerosol_radius_requires_stokes_particles(self):
        parameters = {"air_exchange_rate_s": .002, "room_height_m": 3,
            "air_dynamic_viscosity_pa_s": 1.8e-5, "particle_density_kg_m3": 1000,
            "air_density_kg_m3": 1.2, "gravity_m_s2": 9.8}
        metric = "critical_aerosol_radius_m"
        row = self.evaluate(parameters, (metric,), {})["metrics"][metric]
        self.assertEqual(row["missing"], ["condition:stokes_particles"])
        row = self.evaluate(parameters, (metric,), {"stokes_particles": "Synthetic low Reynolds number spheres"})["metrics"][metric]
        self.assertEqual(row["status"], "calculated")

    def test_masked_jet_is_not_applicable(self):
        m = self.evaluate({**PARAMETERS, "quanta_concentration_exhaled_m3": 30, "mask_penetration": .3})["metrics"]
        self.assertEqual(m["critical_distance_m"]["status"], "inapplicable")
        self.assertEqual(m["exposure_time_bound_s"]["status"], "calculated")

    def test_known_exclusion_does_not_require_irrelevant_missing_parameters(self):
        for parameters, metric in (
            ({"mask_penetration": .3}, "critical_distance_m"),
            ({"initial_concentration_quanta_m3": 2}, "transient_interpolation_time_bound_s"),
        ):
            with self.subTest(metric=metric):
                row = evaluate_parameters(parameters, {}, (metric,), subject="test", label="Synthetic scope check")["metrics"][metric]
                self.assertEqual(row["status"], "inapplicable")
                self.assertIsNone(row["value"])
                self.assertEqual(row["missing"], [])
                self.assertEqual(row["applicability"]["inputs"], parameters)
                self.assertTrue(row["applicability"]["reason"])
        unknown = evaluate_parameters({}, {}, ("critical_distance_m",), subject="test", label="Unknown scope")
        self.assertEqual(unknown["metrics"]["critical_distance_m"]["status"], "missing_inputs")

    def test_missing_mask_state_is_not_evidence_that_jet_model_is_inapplicable(self):
        row = self.evaluate({"mouth_area_m2": .0002, "entrainment_coefficient": .1,
            "breathing_flow_m3_s": .0001, "removal_capacity_m3_s": .2},
            ("critical_distance_m",))["metrics"]["critical_distance_m"]
        self.assertEqual(row["status"], "missing_inputs")
        self.assertEqual(row["missing"], ["mask_penetration"])

    def test_inapplicable_dependency_is_not_misreported_as_missing_data(self):
        params = {**PARAMETERS, "quanta_concentration_exhaled_m3": 30, "neighbor_distance_m": .01}
        row = self.evaluate(params, ("jet_corrected_time_bound_s",))["metrics"]["jet_corrected_time_bound_s"]
        self.assertEqual(row["status"], "inapplicable")
        self.assertEqual(row["applicability"]["inputs"], {"neighbor_distance_m": .01})

    def test_no_silent_clean_start_or_model_assumptions(self):
        params = {k: v for k, v in PARAMETERS.items() if k != "initial_concentration_quanta_m3"}
        params["quanta_concentration_exhaled_m3"] = 30
        row = self.evaluate(params, ("transient_exposure_time_bound_s",))["metrics"]["transient_exposure_time_bound_s"]
        self.assertEqual(row["status"], "missing_inputs")
        self.assertIn("initial_concentration_quanta_m3", row["missing"])
        self.assertEqual(self.evaluate(conditions={})["metrics"]["indoor_reproductive_number"]["status"], "missing_inputs")

    def test_exact_transient_inversion_and_zero_removal(self):
        for rate in (0, 1e-12, .001, 1):
            t = exact_time_bound(.1, 4, .00015, 1, 1, .0045, 100, rate, 0)
            risk = 4*.00015*integrated_concentration(t, .0045, 100, rate, 0)
            self.assertAlmostEqual(risk, .1, places=10)

    def test_unbounded_distance_is_strict_json(self):
        result = self.evaluate({**PARAMETERS, "quanta_concentration_exhaled_m3": 30, "exposure_time_s": 10000})
        row = result["metrics"]["risk_threshold_distance_m"]
        self.assertIsNone(row["value"])
        self.assertEqual(row["limit"], "unbounded")
        json.dumps(result, allow_nan=False)
        self.assertTrue(math.isinf(threshold_distance(.1, .1, 4, .1, 5)))

    def test_inconsistent_total_removal_rejected(self):
        result = self.evaluate({**PARAMETERS, "quanta_concentration_exhaled_m3": 30, "removal_rate_s": .5})
        self.assertTrue(result["issues"])
        self.assertEqual(result["metrics"]["indoor_reproductive_number"]["status"], "inconsistent_inputs")
