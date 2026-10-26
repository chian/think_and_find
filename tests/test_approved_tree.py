"""Architecture and behavior fixtures for the user-approved tree, not live evidence."""

import json
import math
import hashlib
import unittest
from dataclasses import replace

from episodic_workflow.case_reports.research.acquisition import PageSource, ReportSource, LexicalProbeSource, SearchSource, SourceTableSource
from episodic_workflow.case_reports.research.documents import Document, rank_spans
from episodic_workflow.case_reports.research.engine import ResearchRun, ResearchLimits, ResearchProviders, TREE
from episodic_workflow.case_reports.research.records import Ledger, ResearchResult, Fact, condensed_diagnostics, evidence_inventory
from episodic_workflow.case_reports.research.report import ResearchReport, scientific_progress
from episodic_workflow.case_reports.research.tools import TOOLS, execute_tool
from episodic_workflow.case_reports.research.workflow import InvestigationSource, admit_scopes, run_guideline_research
from episodic_workflow.case_reports.research.assumptions import assess_assumption, assumption_candidate, candidate_response_schema
from episodic_workflow.task_executors.scenario_metrics.evaluate import UNITS, DEFAULT_METRICS
from episodic_workflow.task_executors.evidence_extraction.argo import parse_json_object
from episodic_workflow.method_loop import Context, EpisodeGoal, EpisodeTree
from episodic_workflow.tests.test_research import run_synthetic, AdaptiveAgent, supported_review, synthetic_information_review, SOURCE_TEXT, REQUEST, METRICS, VALUES, CONDITIONS
from episodic_workflow.tests.report_assertions import assert_admitted_research_provenance


FIELD = "quanta_concentration_exhaled_m3"


def walk(record):
    yield record
    for unit in record.unit_records:
        if unit.child:
            yield from walk(unit.child)


def local_run(agent, *, window=600):
    return ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None),
                       ResearchLimits(text_window_chars=window), "Synthetic reading task", (FIELD,))


def execute_binding(run, source, grain):
    goal = EpisodeGoal.root(objective="Resolve source evidence", result_contract={"field": FIELD})
    episode = run.episode(grain, source, goal=goal, focus=(FIELD,))
    pending, children = [grain], {}
    while pending:
        name = pending.pop()
        if TREE[name]:
            children[run.grains[name]] = tuple(run.grains[c] for c in TREE[name])
            pending.extend(TREE[name])
    tree = EpisodeTree(run.grains[grain], children)
    record = episode.run(Context(tree=tree))
    return record, source.build_result(record)


class ApprovedTreeTests(unittest.TestCase):
    def test_scope_rejection_keeps_the_reviewers_reason_for_the_parent(self):
        requests = []
        def agent(request):
            requests.append(request)
            return {"admit": False, "reason": "The proposal names a model topic, not a case receiving estimates."}
        run = local_run(agent)
        with self.assertRaisesRegex(ValueError, "model topic, not a case receiving estimates"):
            admit_scopes(run, {"scenarios": [{"subject": "topic", "description": "A model extension",
                "basis": "Synthetic scope proposal", "role": "target"}]}, ResearchResult(), {})
        self.assertEqual(requests[0]["operation"], "review_scenario_scope")
        self.assertIn("case or conditional experiment", requests[0]["instruction"])
        self.assertIn("explicitly conditional study case", requests[0]["instruction"])

    def test_parent_sees_current_conclusion_coverage_before_choosing_an_action(self):
        agent, (_, report) = run_synthetic()
        requests = [r for r in agent.requests if r["operation"] == "propose_guideline_action"]
        self.assertEqual(requests[0]["scientific_progress"]["targets"], {})
        self.assertTrue(any(r["scientific_progress"]["targets"].get("target", {}).get("assessed_numeric_fields")
                            and r["scientific_progress"]["calculated_quantities"] == 0 for r in requests))
        partial = [r["scientific_progress"]["targets"]["target"]["coverage"] for r in requests
                   if 0 < r["scientific_progress"]["calculated_quantities"] < len(METRICS)]
        self.assertTrue(partial)
        self.assertEqual(partial[0]["indoor_reproductive_number"]["status"], "calculated")
        self.assertEqual(partial[0]["critical_distance_m"]["status"], "not_evaluated")
        self.assertEqual(requests[-1]["scientific_progress"]["targets"]["target"]["coverage"],
                         report.data["scenarios"]["target"]["coverage"])

    def test_candidate_subject_contract_does_not_invent_missing_scenario_definitions(self):
        ledger = Ledger(ResearchResult(scenarios=({"subject": "reference", "description": "Synthetic scope"},)))
        candidate = {"subject": "reference:activity", "field": FIELD, "value": 30,
                     "unit": UNITS[FIELD], "statement": "Synthetic candidate", "basis": "Test"}
        with self.assertRaises(ValueError) as raised:
            assumption_candidate(candidate, ledger)
        self.assertIn("reference:activity", str(raised.exception))
        self.assertIn("Existing subjects: ['reference']", str(raised.exception))
        self.assertIn("scenarios list", str(raised.exception))
        self.assertEqual(list(ledger.scenarios), ["reference"])
        for branch in candidate_response_schema()["oneOf"]:
            self.assertIn("context.defined_scenarios", branch["properties"]["subject"]["description"])
            self.assertIn("scenarios list", branch["properties"]["subject"]["description"])

    def test_full_default_report_uses_reviewed_assumptions_across_all_calculation_groups(self):
        values = {**VALUES, "neighbor_distance_m": 1.5, "removal_rate_s": .001,
                  "ventilation_flow_m3_s": .08, "recirculation_flow_m3_s": .02,
                  "room_height_m": 2, "air_exchange_rate_s": .0008,
                  "particle_density_kg_m3": 1000, "air_density_kg_m3": 1.2,
                  "air_dynamic_viscosity_pa_s": 1.8e-5, "gravity_m_s2": 9.8,
                  "clearance_fraction": .9}
        conditions = {**CONDITIONS, "stokes_particles": "Explicit synthetic particle approximation",
                      "emissions_stopped": "Explicit synthetic source-off clearance interval"}
        source_text = "\n".join(f"{field} = {value} {UNITS[field]}." for field, value in values.items())
        condition_quote = "Synthetic conditions: " + "; ".join(conditions) + "."
        source_text += "\n" + condition_quote
        base = AdaptiveAgent()
        def agent(request):
            op = request["operation"]
            if op == "propose_guideline_action":
                base.requests.append(request)
                if not request["assumption_options"]:
                    return {"action": "investigate", "objective": "Assess all synthetic inputs",
                            "focus": [*values, *conditions], "subjects": ["target"], "child_model_calls": 90,
                            "scenarios": [{"subject": "target", "description": "Synthetic full-catalog room",
                                           "basis": "Offline contract test, not a real case", "role": "target"}]}
                completed = {c["tool"] for c in request["calculations"].values()}
                choices = ("calculate_geometry_and_removal", "calculate_transmission_metrics", "calculate_conditional_limits")
                tool = next((t for t in choices if t not in completed), choices[-1])
                facts = {f["field"]: key for key, f in request["admitted_facts"].items() if f["subject"] == "target"}
                return {"action": tool, "objective": "Calculate requested synthetic outputs", "subject": "target",
                        "facts": list(facts.values()), "conditions": conditions, "basis": "One coherent synthetic parameterization"}
            if op == "propose_investigation_strategy":
                base.requests.append(request)
                return {"approach": "Find synthetic support", "objective": "Resolve synthetic assumptions", "child_model_calls": 16,
                        "candidates": [{"subject": "target", "field": field, "value": values.get(field),
                            "unit": UNITS.get(field, "condition"), "statement": conditions.get(field, f"Conditional {field}"),
                            "basis": "Synthetic hypothesis, not an externally admitted input"} for field in (*values, *conditions)]}
            if op == "extract_research_evidence" and request["source"] != "request":
                base.requests.append(request)
                self.assertEqual(request["source_text"], source_text)
                return {"facts": [{"subject": "reference", "field": field, "value": value, "unit": UNITS[field],
                    "quote": f"{field} = {value} {UNITS[field]}.", "kind": "source_model_parameter",
                    "context": "Synthetic fixture", "basis": "Reported synthetic value"} for field, value in values.items()],
                    "evidence": [{"subject": "reference", "claim": "Synthetic model conditions", "fields": list(conditions),
                        "quote": condition_quote, "context": "Synthetic fixture", "basis": "Explicit synthetic conditions"}], "findings": []}
            return base(request)
        record, report = run_guideline_research(research_request=REQUEST,
            providers=ResearchProviders(agent, lambda *_: [{"url": "https://example.test/full", "text": source_text}], lambda _: None),
            limits=ResearchLimits())
        self.assertTrue(report.passed, report.data["unresolved"][:5])
        self.assertEqual(set(report.data["scenarios"]["target"]["coverage"]), set(DEFAULT_METRICS))
        self.assertTrue(all(r["status"] == "calculated" for r in report.data["scenarios"]["target"]["coverage"].values()))
        self.assertEqual(len(report.data["scenarios"]["target"]["alternatives"]), 1)
        tools = {u.unit_label for u in record.unit_records if not u.child}
        self.assertTrue({"calculate_geometry_and_removal", "calculate_transmission_metrics", "calculate_conditional_limits"} <= tools)
        self.assertFalse(base.requests[0]["admitted_facts"])
        assert_admitted_research_provenance(self, report.as_record())
        json.dumps(report.as_record(), allow_nan=False)

    def test_model_response_contracts_are_typed_schemas_not_placeholder_templates(self):
        agent, (_, report) = run_synthetic()
        self.assertTrue(report.passed)
        observed = set()
        for request in agent.requests:
            op, schema = request["operation"], request["response_contract"]
            observed.add(op)
            self.assertEqual(schema["type"], "object", op)
            self.assertFalse(schema["additionalProperties"], op)
            self.assertLessEqual(set(schema["required"]), set(schema["properties"]), op)
            if op == "review_research_evidence":
                decisions = schema["properties"]["decisions"]
                self.assertEqual(decisions["minItems"], len(request["candidates"]))
                self.assertEqual(decisions["maxItems"], len(request["candidates"]))
                self.assertEqual(decisions["items"]["properties"]["option"]["enum"], list(range(len(request["candidates"]))))
                self.assertEqual(decisions["items"]["properties"]["admit"]["type"], "boolean")
            elif op == "propose_page_child":
                self.assertEqual(set(schema["properties"]["option"]["enum"]), set(request["available_children"]))
                self.assertEqual(schema["allOf"][0]["then"]["required"], ["query"])
            elif op in {"review_scenario_scope", "review_research_model"}:
                self.assertEqual(schema["properties"]["admit"]["type"], "boolean")
                self.assertEqual(schema["properties"]["reason"]["minLength"], 1)
        self.assertTrue({"propose_strategy_query", "propose_page_child", "review_research_evidence",
                         "review_scenario_scope", "review_research_model", "synthesize_research_findings"} <= observed)

    def test_inferred_candidate_retains_evidence_linked_review_when_calculated(self):
        base = AdaptiveAgent()
        def agent(request):
            response = base(request)
            if request["operation"] == "propose_guideline_action" and response["action"] == "investigate":
                response["child_model_calls"] = 60
            elif request["operation"] == "propose_investigation_strategy":
                response["child_model_calls"] = 20
                self.assertIn("candidate may be an inferred conditional value", request["instruction"])
                for candidate in response["candidates"]:
                    if candidate["field"] == FIELD:
                        candidate["value"] = 25
                        candidate["basis"] = "Synthetic bounded inference from the reference value, not source extraction"
            elif request["operation"] == "assess_assumption_reasonableness" and request["candidate"]["field"] == FIELD:
                self.assertEqual(request["candidate"]["value"], 25)
                self.assertTrue(any(e.get("field") == FIELD and e.get("value") == 30 for e in request["evidence"].values()))
                response = supported_review(request, grade=1, verdict="provisional")
            return response
        _, (_, report) = run_synthetic(agent)
        self.assertTrue(report.passed, report.data["unresolved"])
        self.assertFalse(base.requests[0]["admitted_facts"])
        self.assertIn("an implementation plan or input glossary is not its result", base.requests[0]["instruction"])
        numeric = [f for f in report.data["evidence_and_assumptions"] if f["field"] == FIELD]
        self.assertTrue(any(f["kind"] == "source_model_parameter" and f["value"] == 30 for f in numeric))
        self.assertTrue(any(f["kind"] == "model_assumption" and f["value"] == 25 for f in numeric))
        self.assertFalse(any(f["kind"].startswith("source_") and f["value"] == 25 for f in numeric))
        for estimate in report.data["scenarios"]["target"]["alternatives"]:
            self.assertEqual(estimate["parameters"][FIELD], 25)
            self.assertEqual(estimate["current_reasonableness"], .5)
        assert_admitted_research_provenance(self, report.as_record())

    def test_child_allowances_leave_parent_resources_for_assessment_and_calculation(self):
        base = AdaptiveAgent()
        def agent(request):
            response = base(request)
            if request["operation"] == "propose_guideline_action" and response["action"] == "investigate":
                response["child_model_calls"] = 60
            elif request["operation"] == "propose_investigation_strategy":
                response["child_model_calls"] = 20
            return response
        _, (record, report) = run_synthetic(agent)
        self.assertTrue(report.passed, report.data["unresolved"])
        self.assertTrue(report.data["assumption_assessments"])
        self.assertTrue(report.data["calculations"])
        self.assertLess(report.data["control"]["model_calls"], 200)
        investigation = record.unit_records[0].child
        self.assertEqual(investigation.goal.as_record()["result_contract"]["operational_limits"]["requested_model_calls"], 60)
        for parent in walk(record):
            for unit in parent.unit_records:
                if not unit.child:
                    continue
                self.assertNotIn("resources", unit.episode_update.report.as_record()["completion"])
        for request in base.requests:
            resources = request["context"]["operational_resources"]
            self.assertGreaterEqual(resources["model_calls_remaining"], 0)
            self.assertGreater(resources["model_call_deadline"], resources["run_model_calls_used"])

    def test_descendant_cannot_extend_an_enclosing_allowance(self):
        base = AdaptiveAgent()
        def agent(request):
            response = base(request)
            if request["operation"] == "propose_guideline_action" and response["action"] == "investigate":
                response["child_model_calls"] = 8
            elif request["operation"] == "propose_investigation_strategy":
                response["child_model_calls"] = 1000
            return response
        _, (record, report) = run_synthetic(agent, ResearchLimits())
        investigation = record.unit_records[0].child
        strategy = investigation.unit_records[0].child
        granted = strategy.goal.as_record()["result_contract"]["operational_limits"]
        self.assertEqual(granted["requested_model_calls"], 1000)
        self.assertLess(granted["model_call_allowance"], 8)
        completion_past_deadline = []
        for attempt in report.data["model_attempts"]:
            deadline = attempt["request"]["context"]["operational_resources"]["model_call_deadline"]
            self.assertLessEqual(attempt["call"], 20)
            if attempt["phase"] == "work":
                self.assertLessEqual(attempt["call"], deadline)
            elif attempt["call"] > deadline:
                completion_past_deadline.append(attempt)
        self.assertTrue(completion_past_deadline)

    def test_invalid_child_allowance_does_not_consume_a_selected_source(self):
        for value in (None, True, 0, -1, "3", 1.5):
            with self.subTest(value=value):
                run = ResearchRun(ResearchProviders(lambda _: {"option": "source:1", "question": "Read",
                    "child_model_calls": value}, lambda *_: [{"url": "https://example.test/a"}],
                    lambda _: self.fail("Invalid allowance must not retrieve")), ResearchLimits(), REQUEST, (FIELD,))
                source = SearchSource(run, (), query="Synthetic query")
                _, result = execute_binding(run, source, "search")
                self.assertFalse(source.visited)
                self.assertTrue(any("positive integer" in issue for issue in result.issues))

    def test_parent_changes_index_does_not_repeat_full_evidence_bodies(self):
        agent, (record, report) = run_synthetic()
        found = False
        for parent in walk(record):
            for unit in parent.unit_records:
                if not unit.child:
                    continue
                changes = unit.episode_update.report.as_record()["changes"]
                self.assertEqual(set(changes), {"evidence_index", "assumptions", "interpretation"})
                for item in changes["evidence_index"]:
                    found = True
                    self.assertIn("fields", item)
                    self.assertIn("source", item)
                    self.assertNotIn("quote", item)
                    self.assertNotIn("claim", item)
                    self.assertNotIn("basis", item)
        self.assertTrue(found)
        for request in agent.requests:
            for name in ("admitted_facts", "evidence_options"):
                self.assertTrue(all("quote" not in row for row in request.get(name, {}).values()))
            if request["operation"] in {"propose_strategy_query", "choose_research_source", "propose_page_child"}:
                self.assertIn("evidence_inventory", request)
                self.assertNotIn("evidence_options", request)
            elif request["operation"] in {"propose_guideline_action", "propose_investigation_strategy"}:
                self.assertIn("evidence_options", request)
        reviews = [r for r in agent.requests if r["operation"] == "assess_assumption_reasonableness"]
        self.assertTrue(all("quote" in row for r in reviews for row in r["evidence"].values()))
        self.assertTrue(report.data["qualitative_evidence"][0]["quote"])
        assert_admitted_research_provenance(self, report.as_record())

    def test_change_index_preserves_source_and_evidence_roles(self):
        rows = [{"subject": "reference", "source": "https://example.test/a", "field": FIELD,
                 "kind": kind, "value": value} for kind, value in
                (("source_observation", 30), ("source_model_parameter", 30), ("source_observation", 40))]
        inventory = evidence_inventory(rows, [{"subject": "reference", "source": "https://example.test/a",
            "fields": [FIELD, "well_mixed"], "claim": "A synthetic context limitation"}])
        self.assertEqual(len(inventory), 3)
        self.assertEqual(inventory[0]["records"], 2)
        self.assertEqual(inventory[1]["kind"], "source_model_parameter")
        self.assertIn("well_mixed", inventory[2]["fields"])

    def test_inherited_failures_are_not_repeated_in_every_child_synthesis(self):
        base = AdaptiveAgent()
        inherited = {"stage": "retrieval", "category": "provider_failure", "why": "fixture inherited failure"}
        run = ResearchRun(ResearchProviders(base, lambda *_: [{"url": "https://example.test/good", "text": SOURCE_TEXT}],
                                           lambda _: None), ResearchLimits(), REQUEST, (FIELD,))
        source = SearchSource(run, ResearchResult(diagnostics=(inherited,), issues=(inherited["why"],)), query="Synthetic query")
        _, result = execute_binding(run, source, "search")
        self.assertIn(inherited["why"], result.issues)
        syntheses = [r for r in base.requests if r["operation"] == "synthesize_research_findings"]
        self.assertTrue(syntheses)
        for request in syntheses:
            self.assertNotIn(inherited["why"], json.dumps(request["admitted_result"]))

    def test_synthesis_failure_reaches_ancestor_diagnostics_without_losing_evidence(self):
        base = AdaptiveAgent()
        failed = False
        def agent(request):
            nonlocal failed
            if request["operation"] == "synthesize_research_findings" and not failed:
                failed = True
                return parse_json_object('{"findings":')
            return base(request)
        _, (_, report) = run_synthetic(agent)
        self.assertTrue(report.passed)
        self.assertTrue(any(d["stage"] == "synthesis" and d["category"] == "malformed_json"
                            for d in report.data["attempt_diagnostics"]))
        self.assertTrue(report.data["evidence_and_assumptions"])
        self.assertTrue(report.data["calculations"])

    def test_root_action_schema_names_focus_fields_and_current_fact_choices(self):
        agent, _ = run_synthetic()
        proposals = [r for r in agent.requests if r["operation"] == "propose_guideline_action"]
        for request in proposals:
            schema = request["response_contract"]
            self.assertEqual(schema["type"], "object")
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(schema["required"], ["action", "objective"])
            properties = schema["properties"]
            self.assertEqual(set(properties["action"]["enum"]), {"investigate", *TOOLS})
            focus = properties["focus"]
            self.assertEqual(focus["minItems"], 1)
            self.assertTrue({FIELD, "well_mixed", "critical_distance_m"} <= set(focus["items"]["enum"]))
            choices = properties["facts"]["items"]
            expected = {*request["admitted_facts"], *(k for k, a in request["assumption_options"].items() if a["value"] is not None)}
            self.assertEqual(set(choices["enum"]) if choices else set(), expected)
            self.assertEqual(properties["child_model_calls"]["minimum"], 1)
            self.assertNotIn("model_call_costs", request["context"])
            self.assertIn("not retrieved evidence", request["context"]["implemented_model"]["interpretation"])

    def test_invalid_root_focus_reports_offending_fields(self):
        def agent(request):
            if request["operation"] == "propose_guideline_action":
                return {"action": "investigate", "objective": "Synthetic malformed focus test",
                        "focus": [FIELD, "undeclared_parameter"]}
            if request["operation"] == "extract_research_evidence":
                return {"facts": [], "evidence": [], "findings": []}
            raise AssertionError(request["operation"])
        _, report = run_guideline_research(research_request=REQUEST,
            providers=ResearchProviders(agent, lambda *_: self.fail("Invalid focus must not open acquisition"), lambda _: None),
            metrics=METRICS, limits=ResearchLimits())
        self.assertFalse(report.passed)
        why = report.data["attempt_diagnostics"][0]["why"]
        self.assertIn("undeclared_parameter", why)
        self.assertIn("response_contract.properties.focus.items.enum", why)
        self.assertNotIn(repr(FIELD), why)

    def test_parent_contract_and_selected_subjects_reach_the_actual_model_requests(self):
        agent, (record, _) = run_synthetic(AdaptiveAgent(("selected", "other")))
        operations = {"propose_investigation_strategy", "propose_strategy_query", "choose_research_source",
                      "propose_page_child", "extract_research_evidence", "review_research_evidence",
                      "assess_assumption_reasonableness"}
        observed = set()
        for request in agent.requests:
            if request["operation"] not in operations or request.get("source") == "request":
                continue
            observed.add(request["operation"])
            context = request["context"]
            self.assertEqual(context["requested_subjects"], ["selected"])
            contract = context["parent_information_contract"]
            self.assertEqual(contract["measurements"]["subjects"], ["selected"])
            self.assertTrue(contract["decision"])
            self.assertTrue(contract["measurements"]["needed_fields"])
        self.assertEqual(observed, operations)
        for child in tuple(walk(record))[1:]:
            self.assertEqual(child.request.message.subjects, ("selected",))

    def test_invalid_numeric_dependency_is_rejected_before_acquisition_without_partial_candidates(self):
        for value in (None, "0.1", True):
            with self.subTest(value=value):
                requests = []
                def agent(request):
                    requests.append(request)
                    return {"approach": "Find source evidence", "objective": "Resolve assumptions", "candidates": [
                        {"subject": "target", "field": FIELD, "value": 30, "unit": "quanta/m3",
                         "statement": "Valid synthetic hypothesis", "basis": "Test"},
                        {"subject": "target", "field": "plume_encounter_probability", "value": value,
                         "unit": "fraction", "statement": "Unresolved numerical dependency", "basis": "Test"}]}
                run = ResearchRun(ResearchProviders(agent, lambda *_: self.fail("No acquisition before validation"),
                    lambda _: None), ResearchLimits(), REQUEST, (FIELD,))
                source = InvestigationSource(run, ResearchResult(scenarios=({"subject": "target"},)))
                _, result = execute_binding(run, source, "investigate")
                self.assertEqual([r["operation"] for r in requests], ["propose_investigation_strategy"])
                self.assertFalse(source.candidates)
                self.assertFalse(result.assumptions)
                self.assertFalse(result.evaluated)
                schema = requests[0]["response_contract"]["properties"]["candidates"]["items"]
                numeric, condition = schema["oneOf"]
                self.assertEqual(numeric["properties"]["value"]["type"], "number")
                self.assertEqual(condition["properties"]["value"]["type"], "null")
                self.assertIn("plume_encounter_probability", numeric["properties"]["field"]["enum"])
                self.assertNotIn("plume_encounter_probability", condition["properties"]["field"]["enum"])
                if value is None:
                    self.assertIn("unresolved information need", result.diagnostics[0]["why"])

    def test_strategy_evidence_focus_can_differ_without_replacing_investigation_goal(self):
        for focus in ((FIELD,), ("settling_speed_m_s",)):
            with self.subTest(focus=focus):
                requests = []
                def agent(request):
                    requests.append(request)
                    if request["operation"] == "propose_investigation_strategy":
                        return {"approach": "Seek targeted supporting evidence", "objective": "Resolve one evidence need",
                                "focus": list(focus), "candidates": []}
                    if request["operation"] == "propose_strategy_query":
                        return {"query": "Synthetic targeted query", "objective": "Find relevant evidence"}
                    raise AssertionError(request["operation"])
                run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None),
                                  ResearchLimits(), REQUEST, (FIELD,))
                source = InvestigationSource(run, ResearchResult(scenarios=({"subject": "target"},)))
                record, _ = execute_binding(run, source, "investigate")
                strategy_request = next(r for r in requests if r["operation"] == "propose_strategy_query")
                self.assertEqual(strategy_request["focus"], list(focus))
                self.assertEqual(strategy_request["context"]["parent_information_contract"]["measurements"]["needed_fields"], list(focus))
                self.assertEqual(source.focus, (FIELD,))
                self.assertEqual(record.unit_records[0].child.request.message.focus, focus)


    def test_failure_synthesis_condenses_proposals_without_hiding_failure_kinds(self):
        failures = [
            {"diagnostic_id": i, "stage": "proposal", "category": "invalid_action_contract",
             "why": "Missing option", "proposal": {"source_id": "source:1", "basis": "Long rejected rationale " * 200}}
            for i in range(1, 4)]
        failures.append({"diagnostic_id": 4, "stage": "retrieval", "category": "provider_failure", "why": "HTTP 403"})
        summary = condensed_diagnostics(failures)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["retained_records"], 3)
        self.assertIn("source_id", summary[0]["received_fields"])
        self.assertEqual(summary[1]["why"], "HTTP 403")
        self.assertNotIn("Long rejected rationale", json.dumps(summary))

    def test_malformed_json_is_auditable_not_evidence_or_zero_yield(self):
        raw = '{"action": "investigate", "findings": ["unverified claim"],}'
        calls = []
        def agent(request):
            calls.append(request)
            return parse_json_object(raw)
        record, report = run_guideline_research(research_request=REQUEST,
            providers=ResearchProviders(agent, lambda *_: (), lambda _: None), metrics=METRICS,
            limits=ResearchLimits())
        self.assertFalse(report.passed)
        self.assertEqual(record.controller_state.history, (0.0,))
        self.assertEqual(len(report.data["model_attempts"]), 3)
        for attempt in report.data["model_attempts"]:
            self.assertEqual(attempt["raw_response"], raw)
            self.assertEqual(attempt["request"], calls[attempt["call"] - 1])
            self.assertEqual(attempt["status"], "response_failed")
        diagnostics = report.data["attempt_diagnostics"]
        self.assertEqual([d["model_call"] for d in diagnostics], [1, 2, 3])
        self.assertEqual(len({d["diagnostic_id"] for d in diagnostics}), 3)
        self.assertTrue(all(d["category"] == "malformed_json" for d in diagnostics))
        self.assertTrue(all(d["proposal"] is None for d in diagnostics))
        self.assertEqual(sum(row["count"] for row in report.data["model_response_failures"]), 3)
        self.assertIn("malformed_json", json.dumps(calls[1]))
        self.assertNotIn("unverified claim", json.dumps(calls[1]))
        self.assertFalse(report.data["evidence_and_assumptions"])
        json.dumps(report.as_record(), allow_nan=False)

    def test_declared_tree_and_real_nested_contracts(self):
        agent, (record, report) = run_synthetic()
        self.assertTrue(report.passed)
        self.assertEqual(report.data["episode_tree"], {k: list(v) for k, v in TREE.items() if v})
        for parent in walk(record):
            self.assertIn(parent.scope_level, TREE)
            for unit in parent.unit_records:
                if unit.child:
                    child = unit.child
                    self.assertIn(child.scope_level, TREE[parent.scope_level])
                    self.assertEqual(child.goal.parent_goal_id, parent.goal.goal_id)
                    self.assertEqual(child.path[:-1], parent.path)
                    self.assertTrue(child.request.report_contract.decision)
                    self.assertEqual(set(unit.episode_update.report.values), set(child.request.report_contract.information))
                elif parent is record:
                    self.assertIn(unit.unit_label, TOOLS)
        operations = [r["operation"] for r in agent.requests]
        self.assertIn("assess_assumption_reasonableness", operations)
        for request in agent.requests:
            if "allowed_actions" in request:
                self.assertNotIn("assess_assumption", request["allowed_actions"])
                self.assertNotIn("render_report", request["allowed_actions"])
        self.assertFalse({"model_evaluation", "assumption_assessment", "source_reading"} & {r.scope_level for r in walk(record)})

    def test_tools_are_leaves_with_exact_assumption_versions(self):
        _, (record, report) = run_synthetic()
        tools = [u for u in record.unit_records if not u.child]
        self.assertTrue(tools)
        self.assertTrue(all(u.unit_label in TOOLS for u in tools))
        assessments = {a["key"]: a for a in report.data["assumption_assessments"]}
        for calculation in report.data["calculations"]:
            self.assertTrue(calculation["assumption_versions"])
            for key, version in calculation["assumption_versions"].items():
                self.assertEqual(assessments[key]["version"], version)
        json.dumps(record.as_record(), allow_nan=False)

    def test_paraphrasing_the_same_source_passage_does_not_manufacture_local_credit(self):
        evidence = {"subject": "case", "fields": [FIELD], "quote": "Reported evidence", "claim": "First wording"}
        first = ResearchResult(evidence=(evidence,)).credit((FIELD,))
        second = ResearchResult(evidence=({**evidence, "claim": "Different wording"},)).credit((FIELD,))
        self.assertEqual(first.identities, second.identities)

    def test_mandatory_reassessment_receives_contrary_evidence(self):
        calls = []
        def agent(request):
            calls.append(request)
            return supported_review(request, verdict="contradicted" if len(request["evidence"]) == 2 else "supported")
        run = local_run(agent)
        fact = Fact("reference", FIELD, 30, "quanta/m3", "source_model_parameter", "Fixture",
                    "https://example.test/one", "Cq = 30", "Matched activity")
        inherited = ResearchResult(facts=(fact,), scenarios=({"subject": "target", "description": "Fixture"},))
        source = InvestigationSource(run, inherited)
        source.focus = (FIELD,)
        candidate = {"subject": "target", "field": FIELD, "value": 30, "unit": "quanta/m3",
                     "statement": "Applicable target Cq", "basis": "Transfer hypothesis"}
        source.candidates[source.candidate_key(candidate)] = candidate
        first = source.assess(ResearchResult())
        self.assertEqual(first.assumptions[0]["reasonableness"], 1)
        source.staged.merge(first)
        contrary = {"subject": "reference", "fields": [FIELD], "claim": "Activity mismatch",
                    "source": "https://example.test/two", "quote": "Different activity", "context": "Mismatch", "basis": "Contrary evidence"}
        second = source.assess(ResearchResult(evidence=(contrary,)))
        self.assertEqual(second.assumptions[0]["reasonableness"], 0)
        self.assertNotEqual(first.assumptions[0]["version"], second.assumptions[0]["version"])
        self.assertEqual(len(calls[-1]["evidence"]), 2)

    def test_inherited_assessment_work_respects_parent_fields_and_subjects(self):
        ledger = CalculationToolTests().ledger({FIELD: 30, "breathing_flow_m3_s": .0001})
        ledger.merge(ResearchResult(scenarios=({"subject": "other", "description": "Separate synthetic scenario"},)))
        ledger.merge(assess_assumption(supported_review, {"subject": "other", "field": FIELD,
            "value": 40, "unit": UNITS[FIELD], "statement": "Separate synthetic hypothesis",
            "basis": "Fixture", "evidence": list(ledger.evidence_options())}, ledger, {}))
        before = dict(ledger.assumptions)
        calls = []
        def agent(request):
            calls.append(request)
            return supported_review(request, grade=1, verdict="provisional")
        source = InvestigationSource(local_run(agent), ledger.result())
        source.focus = (FIELD,)
        source.requested_subjects = ("case",)
        evidence = {"subject": "reference", "fields": ["breathing_flow_m3_s"],
                    "claim": "Different respiratory activity", "source": "https://example.test/contrary",
                    "quote": "Different activity", "context": "Synthetic limitation", "basis": "Cross-field mechanism"}
        result = source.assess(ResearchResult(evidence=(evidence,)))
        self.assertEqual([(r["candidate"]["subject"], r["candidate"]["field"]) for r in calls], [("case", FIELD)])
        self.assertTrue(any(e.get("claim") == evidence["claim"] for e in calls[0]["evidence"].values()))
        after = {a["key"]: a for a in result.assumptions}
        for key, previous in before.items():
            if previous["subject"] == "case" and previous["field"] == FIELD:
                self.assertEqual(after[key]["reasonableness"], .5)
            else:
                self.assertEqual(after[key], previous)

    def test_cross_field_evidence_reaches_mandatory_reasonableness_review(self):
        calls = []
        def agent(request):
            calls.append(request)
            return supported_review(request, verdict="provisional", grade=1)
        run = local_run(agent)
        fact = Fact("reference", "breathing_flow_m3_s", .0001, "m3/s", "source_observation", "Relevant mechanism",
                    "https://example.test/flow", "Flow 0.0001 m3/s", "Same activity")
        source = InvestigationSource(run, ResearchResult(facts=(fact,), scenarios=({"subject": "target"},)))
        source.focus = (FIELD,)
        candidate = {"subject": "target", "field": FIELD, "value": 30, "unit": "quanta/m3",
                     "statement": "Cq hypothesis", "basis": "Review physical reasoning"}
        source.candidates[source.candidate_key(candidate)] = candidate
        source.assess(ResearchResult())
        self.assertEqual(next(iter(calls[0]["evidence"].values()))["field"], "breathing_flow_m3_s")

    def test_failed_acquisition_does_not_hide_new_assessment_of_existing_evidence(self):
        run = local_run(supported_review)
        fact = Fact("reference", FIELD, 30, "quanta/m3", "source_model_parameter", "Synthetic source",
                    "https://example.test/source", "Cq = 30", "Synthetic activity")
        source = InvestigationSource(run, ResearchResult(facts=(fact,), scenarios=({"subject": "target"},)))
        source.focus = (FIELD,)
        candidate = {"subject": "target", "field": FIELD, "value": 30, "unit": "quanta/m3",
                     "statement": "Synthetic hypothesis", "basis": "Test"}
        source.candidates[source.candidate_key(candidate)] = candidate
        result = source.assess(ResearchResult(evaluated=False, issues=("Synthetic acquisition failure",)))
        self.assertTrue(result.evaluated)
        self.assertIn("Synthetic acquisition failure", result.issues)
        self.assertEqual(result.credit((FIELD,), scientific=True).qualities[0][2], 1)

    def test_reasonableness_contract_exposes_grade_verdict_consistency_without_coercion(self):
        requests = []
        def agent(request):
            requests.append(request)
            return supported_review(request, verdict="supported", grade=1)
        run = local_run(agent)
        fact = Fact("reference", FIELD, 30, "quanta/m3", "source_model_parameter", "Synthetic source",
                    "https://example.test/source", "Cq = 30", "Synthetic activity")
        source = InvestigationSource(run, ResearchResult(facts=(fact,), scenarios=({"subject": "target"},)))
        source.focus = (FIELD,)
        candidate = {"subject": "target", "field": FIELD, "value": 30, "unit": "quanta/m3",
                     "statement": "Synthetic target Cq", "basis": "Hypothesis"}
        source.candidates[source.candidate_key(candidate)] = candidate
        result = source.assess(ResearchResult())
        contract = requests[0]["response_contract"]
        self.assertEqual(contract["type"], "object")
        supported = contract["allOf"][0]
        self.assertEqual(supported["if"]["properties"]["verdict"]["const"], "supported")
        for criterion in supported["then"]["properties"]["criteria"]["properties"].values():
            self.assertEqual(criterion["properties"]["grade"]["const"], 2)
        self.assertIn("do not inflate grades", requests[0]["instruction"])
        self.assertEqual(result.assumptions[0]["review"]["verdict"], "unassessed")
        self.assertEqual(result.assumptions[0]["reasonableness"], 0)
        self.assertFalse(result.evaluated)
        self.assertIn("'support': 1", result.diagnostics[0]["why"])

    def test_one_failed_assessment_does_not_hide_a_successful_independent_judgment(self):
        other = "breathing_flow_m3_s"
        def agent(request):
            if request["candidate"]["field"] == other:
                return parse_json_object('{"verdict":')
            return supported_review(request)
        run = local_run(agent)
        fact = Fact("reference", FIELD, 30, "quanta/m3", "source_model_parameter", "Synthetic source",
                    "https://example.test/source", "Cq = 30", "Synthetic activity")
        source = InvestigationSource(run, ResearchResult(facts=(fact,), scenarios=({"subject": "target"},)))
        source.focus = (FIELD, other)
        for field, value in ((FIELD, 30), (other, .0001)):
            candidate = {"subject": "target", "field": field, "value": value, "unit": UNITS[field],
                         "statement": "Synthetic hypothesis", "basis": "Test"}
            source.candidates[source.candidate_key(candidate)] = candidate
        result = source.assess(ResearchResult())
        self.assertTrue(result.evaluated)
        self.assertEqual([a["field"] for a in result.assumptions if a["review"].get("completed", True)], [FIELD])
        self.assertEqual([a["field"] for a in result.assumptions if not a["review"].get("completed", True)], [other])
        qualities = {field: value for field, _, value in result.credit(source.focus, scientific=True).qualities}
        self.assertEqual(qualities, {FIELD: 1, other: 0})
        self.assertTrue(any(d["category"] == "malformed_json" for d in result.diagnostics))
        failure = next(d for d in result.diagnostics if d["category"] == "malformed_json")
        self.assertEqual(failure["subject"], "target")
        self.assertEqual(failure["field"], other)
        summary = condensed_diagnostics([failure])[0]
        self.assertEqual(summary["candidate"]["value"], .0001)
        self.assertEqual(summary["candidate"]["unit"], "m3/s")

    def test_parent_can_reinvestigate_and_recalculate_after_an_initial_prediction(self):
        base, revised = AdaptiveAgent(), False
        revised_text = f"{FIELD} = 25 quanta/m3 for the revised synthetic scenario."
        def agent(request):
            nonlocal revised
            op = request["operation"]
            if op == "propose_guideline_action":
                calculations = list(request["calculations"].values())
                if not revised and any(c["tool"] == "calculate_conditional_limits" for c in calculations):
                    revised = True
                    return {"action": "investigate", "objective": "Resolve new evidence about infectiousness",
                            "focus": [FIELD], "subjects": ["target"]}
                response = base(request)
                if revised and response["action"] != "investigate":
                    facts = {}
                    for key, fact in request["admitted_facts"].items():
                        if fact["subject"] == "target":
                            facts[fact["field"]] = key
                    response["facts"] = list(facts.values())
                    recent = [c for c in calculations if c["tool"] == "calculate_transmission_metrics"
                              and c["output"].get("indoor_reproductive_number", {}).get("value", 1) < .09]
                    response["action"] = "calculate_conditional_limits" if recent else "calculate_transmission_metrics"
                return response
            if revised and op == "propose_investigation_strategy":
                return {"approach": "Examine the new source and contrary evidence", "objective": "Assess revised Cq",
                        "candidates": [{"subject": "target", "field": FIELD, "value": 25, "unit": "quanta/m3",
                                        "statement": "Revised synthetic Cq", "basis": "New evidence, subject to assessment"}]}
            if revised and op == "extract_research_evidence" and request["source"].endswith("revised"):
                return {"facts": [{"subject": "reference", "field": FIELD, "value": 25, "unit": "quanta/m3",
                    "quote": revised_text, "kind": "source_model_parameter", "context": "Revised synthetic scenario",
                    "basis": "New synthetic evidence"}], "findings": []}
            if revised and op == "assess_assumption_reasonableness" and request["candidate"]["field"] == FIELD:
                return supported_review(request, verdict="contradicted" if request["candidate"]["value"] == 30 else "supported")
            return base(request)
        def search(*_):
            return ({"url": "https://example.test/revised" if revised else "https://example.test/original",
                     "text": revised_text if revised else SOURCE_TEXT},)
        record, report = run_guideline_research(research_request=REQUEST,
            providers=ResearchProviders(agent, search, lambda _: None), metrics=METRICS,
            limits=ResearchLimits())
        self.assertTrue(report.passed, report.data["unresolved"])
        sequence = [u.child.scope_level if u.child else u.unit_label for u in record.unit_records]
        self.assertEqual(sequence[:4], ["investigate", "calculate_transmission_metrics", "calculate_conditional_limits", "investigate"])
        alternatives = report.data["scenarios"]["target"]["alternatives"]
        self.assertTrue(any(e["current_reasonableness"] == 0 for e in alternatives))
        self.assertTrue(any(e["current_reasonableness"] > 0 and e["parameters"][FIELD] == 25 for e in alternatives))
        self.assertTrue(any(u.controller_step.gain < 0 for u in record.unit_records))


class ReadingBindingTests(unittest.TestCase):
    def test_source_choice_contract_and_feedback_name_the_required_response_field(self):
        requests = []
        def agent(request):
            requests.append(request)
            return {"source_id": "source:1", "question": "Find evidence"}
        run = ResearchRun(ResearchProviders(agent, lambda *_: [{"url": "https://example.test/source"}], lambda _: None),
                          ResearchLimits(), REQUEST, (FIELD,))
        source = SearchSource(run, (), query="synthetic evidence")
        _, result = execute_binding(run, source, "search")
        contract = requests[0]["response_contract"]
        self.assertEqual(contract["required"], ["option", "question"])
        self.assertEqual(contract["properties"]["option"]["enum"], ["source:1"])
        self.assertIn("Missing required top-level field 'option'", result.diagnostics[0]["why"])
        self.assertIn("source_id", result.diagnostics[0]["why"])
        self.assertFalse(source.visited)
        self.assertFalse(result.evaluated)

    def test_failed_review_preserves_extraction_for_study_without_admitting_it(self):
        quote = "Cq = 30 quanta/m3."
        candidate = {"subject": "reference", "field": FIELD, "value": 30, "unit": "quanta/m3",
                     "kind": "source_model_parameter", "basis": "Synthetic source",
                     "quote": quote, "context": "Synthetic activity"}
        def agent(request):
            if request["operation"] == "extract_research_evidence":
                return {"facts": [candidate], "findings": []}
            return parse_json_object('{"decisions": [')
        run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None),
                          ResearchLimits(), REQUEST, (FIELD,))
        document = Document("https://example.test/source", "Fixture", quote, 600)
        _, result = execute_binding(run, ReportSource(run, (), document=document), "report")
        self.assertEqual(run.model_attempts[0]["response"]["facts"], [candidate])
        self.assertEqual(run.model_attempts[1]["diagnostic"]["category"], "malformed_json")
        self.assertEqual(result.diagnostics[0]["operation"], "review_research_evidence")
        self.assertFalse(result.facts)
        self.assertFalse(result.evaluated)
        self.assertFalse(document.processed)

    def test_report_retains_context_across_bounded_windows_without_crediting_memory(self):
        requests = []
        def agent(request):
            requests.append(request)
            if request["operation"] == "extract_research_evidence":
                if not request["reading_memory"]["anchors"]:
                    key = next(k for k, s in request["source_spans"].items() if "Subject alpha" in s["text"])
                    memory = {"summary": "Alpha is the room subject", "unresolved_links": ["Find its Cq"],
                              "anchors": [{"span": key, "quote": "Subject alpha"}]}
                else:
                    memory = request["reading_memory"]
                return {"facts": [], "evidence": [], "findings": [], "reading_memory": memory}
            raise AssertionError(request["operation"])
        run = local_run(agent)
        text = "Subject alpha\n" + ("Source context without numerical evidence.\n" * 35)
        document = Document("https://example.test/report", "Long report", text, 600)
        source = ReportSource(run, (), document=document)
        record, result = execute_binding(run, source, "report")
        self.assertGreater(record.units_consumed, 1)
        self.assertTrue(requests[1]["reading_memory"]["anchors"])
        schema = requests[0]["response_contract"]
        self.assertEqual(schema["properties"]["reading_memory"]["properties"]["anchors"]["type"], "array")
        self.assertTrue(all(len(r["source_text"]) <= 603 for r in requests))
        self.assertFalse(result.facts)
        self.assertEqual(set(record.controller_state.history), {0})
        self.assertTrue(document.processed)

    def test_page_can_choose_table_then_lexical_probe(self):
        requests = []
        def agent(request):
            requests.append(request)
            op = request["operation"]
            if op == "propose_page_child":
                tables = [k for k in request["available_children"] if k.startswith("table:")]
                return {"option": tables[0] if tables else "lexical_probe", "question": "Inspect evidence", "query": "Cq"}
            if op == "propose_table_query":
                return {"rows": [next(iter(request["available_rows"]))], "question": "Interpret this row"}
            if op == "extract_research_evidence":
                return {"facts": [], "findings": []}
            if op == "synthesize_research_findings":
                return {"findings": [], "uncertainties": []}
            raise AssertionError(op)
        run = local_run(agent)
        source = PageSource(run, (), candidate={"url": "https://example.test/table", "text":
            "A detected table.\n| Activity | Cq |\n| --- | --- |\n| Speaking | 30 |\n| Quiet | 10 |\nMore prose for a lexical probe."})
        record, _ = execute_binding(run, source, "page")
        self.assertEqual([u.child.scope_level for u in record.unit_records], ["source_table", "lexical_probe"])
        self.assertEqual(record.unit_records[0].child.units_consumed, 2)
        self.assertTrue(all(u.child is None for u in record.unit_records[0].child.unit_records))
        later = next(r for r in requests if r["operation"] == "propose_page_child" and r["child_reports"])
        self.assertEqual(set(later["available_children"]), {"report", "lexical_probe"})

    def test_table_contract_distinguishes_row_keys_from_text_queries(self):
        proposals = ({"rows": [1]}, {"rows": []}, {"rows": None},
                     {"rows": ["table:1:row:99"]}, {"rows": [], "query": "Speaking"},
                     {"rows": ["table:1:row:1", "table:1:row:1"]})
        for proposal in proposals:
            with self.subTest(proposal=proposal):
                requests = []
                def agent(request):
                    requests.append(request)
                    return {**proposal, "question": "Inspect a synthetic row"}
                run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None),
                                  ResearchLimits(), REQUEST, (FIELD,))
                document = Document("https://example.test/table", "Fixture",
                    "| Activity | Cq |\n| --- | --- |\n| Speaking | 30 |\n", 600)
                source = SourceTableSource(run, (), document=document, table=document.tables[0])
                _, result = execute_binding(run, source, "source_table")
                schema = requests[0]["response_contract"]
                self.assertEqual(schema["oneOf"][0]["required"], ["rows", "question"])
                self.assertEqual(schema["oneOf"][1]["required"], ["query", "question"])
                self.assertIn("no rows were queried", result.diagnostics[0]["why"].lower())
                self.assertFalse(source.queried)
                self.assertFalse(result.evaluated)

    def test_table_text_query_can_find_rows_beyond_preview(self):
        requests = []
        def agent(request):
            requests.append(request)
            if request["operation"] == "propose_table_query":
                return {"query": "target activity", "column": "Activity", "question": "Find synthetic target row"}
            return {"facts": [], "evidence": [], "findings": []}
        run = ResearchRun(ResearchProviders(agent, lambda *_: (), lambda _: None),
                          ResearchLimits(), REQUEST, (FIELD,))
        text = "| Activity | Cq |\n| --- | --- |\n" + "| Other | 10 |\n" * 9 + "| Target activity | 30 |\n"
        document = Document("https://example.test/table", "Fixture", text, 600)
        source = SourceTableSource(run, (), document=document, table=document.tables[0])
        _, result = execute_binding(run, source, "source_table")
        self.assertNotIn("table:1:row:10", requests[0]["available_rows"])
        self.assertIn("| Target activity | 30 |", requests[1]["source_text"])
        self.assertEqual(source.queried, {"table:1:row:10"})
        self.assertTrue(result.evaluated)

    def test_failed_table_review_does_not_consume_selected_rows(self):
        requests, reviews = [], []
        quote = "| Speaking | 30 |"
        def agent(request):
            requests.append(request)
            if request["operation"] == "propose_table_query":
                self.assertIn("table:1:row:1", request["available_rows"])
                return {"rows": ["table:1:row:1"], "question": "Inspect synthetic Cq"}
            if request["operation"] == "extract_research_evidence":
                return {"facts": [{"subject": "case", "field": FIELD, "value": 30, "unit": "quanta/m3",
                    "quote": quote, "kind": "source_model_parameter", "context": "Synthetic source", "basis": "Test"}]}
            if request["operation"] == "review_research_evidence":
                reviews.append(request)
                if len(reviews) == 1:
                    return parse_json_object('{"decisions": [')
                return {"decisions": [{"option": 0, "admit": True, "reason": "Synthetic matching activity"}]}
            if request["operation"] == "assess_information_progress":
                return synthetic_information_review(request)
            raise AssertionError(request["operation"])
        run = local_run(agent)
        document = Document("https://example.test/table", "Fixture",
            "| Activity | Cq (quanta/m3) |\n| --- | --- |\n" + quote + "\n", 600)
        source = SourceTableSource(run, (), document=document, table=document.tables[0])
        record, result = execute_binding(run, source, "source_table")
        proposals = [r for r in requests if r["operation"] == "propose_table_query"]
        self.assertEqual(len(proposals), 2)
        self.assertIn("malformed_json", json.dumps(proposals[1]["attempt_diagnostics"]))
        self.assertEqual(record.controller_state.failures, 1)
        self.assertEqual(source.queried, {"table:1:row:1"})
        self.assertEqual(len(result.facts), 1)

    def test_lexical_probe_excludes_already_processed_chunks(self):
        run = local_run(lambda _: {"facts": [], "findings": []}, window=300)
        document = Document("https://example.test/probe", "Probe", "Cq context.\n" * 60, 300)
        document.processed.add(document.spans[0].key)
        source = LexicalProbeSource(run, (), document=document, query="Cq")
        self.assertNotIn(document.spans[0], source.ranked)
        record, _ = execute_binding(run, source, "lexical_probe")
        self.assertNotEqual(record.unit_records[0].unit_label, f"chunk:0:{document.spans[0].end}")

    def test_lexical_ranking_keeps_processed_neighbor_context(self):
        run = local_run(lambda _: {"facts": [], "findings": []}, window=300)
        document = Document("https://example.test/probe", "Probe", "unrelated context\n" * 90, 300)
        document.spans = tuple(replace(span, text="rare heading" if index == 8 else span.text)
                               for index, span in enumerate(document.spans))
        document.processed.update(span.key for span in document.spans[5:9])
        source = LexicalProbeSource(run, (), document=document, query="rare heading")
        self.assertEqual(source.ranked, tuple(span for span in rank_spans(document.spans, "rare heading")
                                             if span.key not in document.processed))
        self.assertEqual(source.ranked[0], document.spans[9])
        self.assertNotEqual(source.ranked, rank_spans(document.remaining(), "rare heading"))

    def test_retrieval_failure_is_not_a_zero_evidence_finding(self):
        run = local_run(lambda r: self.fail("No model needed when retrieval fails"))
        source = PageSource(run, (), candidate={"url": "https://example.test/unavailable"})
        record, result = execute_binding(run, source, "page")
        self.assertEqual(record.ended_by, "source_failed")
        self.assertFalse(result.evaluated)
        self.assertEqual(record.controller_state.history, (0,))
        self.assertIn("not negative evidence", result.diagnostics[0]["why"])
        self.assertEqual(result.diagnostics[0]["source"], "https://example.test/unavailable")
        self.assertEqual(result.diagnostics[0]["requested_fields"], [FIELD])

    def test_bad_memory_does_not_discard_valid_admitted_evidence(self):
        quote = "Cq = 30 quanta/m3."
        def agent(request):
            if request["operation"] == "extract_research_evidence":
                return {"facts": [{"subject": "case", "field": FIELD, "value": 30, "unit": "quanta/m3",
                    "quote": quote, "kind": "source_model_parameter", "context": "Synthetic source", "basis": "Test"}],
                    "reading_memory": {"summary": "Bad anchor", "unresolved_links": [],
                                       "anchors": [{"span": "nonexistent", "quote": "Invented passage"}]}}
            if request["operation"] == "review_research_evidence":
                return {"decisions": [{"option": 0, "admit": True, "reason": "Exact synthetic model value"}]}
            if request["operation"] == "assess_information_progress":
                return synthetic_information_review(request)
            raise AssertionError(request["operation"])
        run = local_run(agent)
        document = Document("https://example.test/memory", "Source", quote, 600)
        record, result = execute_binding(run, ReportSource(run, (), document=document), "report")
        self.assertTrue(result.evaluated)
        self.assertEqual(result.facts[0].value, 30)
        self.assertEqual(result.facts[0].source_version, document.sha256)
        self.assertTrue(any(d["category"] == "invalid_memory" for d in result.diagnostics))
        self.assertGreater(record.controller_state.history[-1], 0)

    def test_source_revisions_retain_both_original_texts(self):
        run = local_run(lambda _: {})
        first = PageSource(run, (), candidate={"url": "https://example.test/source", "text": "First version"})
        second = PageSource(run, (), candidate={"url": "https://example.test/source", "text": "Second version"})
        first.prepare()
        second.prepare()
        versions = run.documents["https://example.test/source"]["versions"]
        self.assertEqual(versions[first.document.sha256]["text"], "First version")
        self.assertEqual(versions[second.document.sha256]["text"], "Second version")


class CalculationToolTests(unittest.TestCase):
    def ledger(self, values, conditions=()):
        from episodic_workflow.task_executors.scenario_metrics.dependencies import CALCULATED_PARAMETER_FIELDS
        supplied = {field: {"value": value, "unit": UNITS[field]}
                    for field, value in values.items() if field in CALCULATED_PARAMETER_FIELDS}
        ledger = Ledger(ResearchResult(scenarios=({"subject": "case", "description": "Synthetic test room",
            "fixed_parameters": supplied, "assumption_needs": list(values)},)))
        from episodic_workflow.case_reports.research.scenarios import Scenario, initial_scenarios
        givens = initial_scenarios((Scenario("case", "Synthetic supplied inputs",
            tuple((field, row["value"]) for field, row in supplied.items()), variables=()),))
        ledger.merge(ResearchResult(facts=givens.facts, assumptions=givens.assumptions))
        for field, value in values.items():
            ledger.merge(ResearchResult(facts=(Fact("case", field, value, UNITS[field], "source_observation",
                "Synthetic fixture, not real evidence", "https://example.test/case",
                f"{field} = {value}", "Same synthetic event"),)))
        ledger.merge(ResearchResult(evidence=({"subject": "case", "fields": list(conditions), "claim": "Synthetic conditions",
            "source": "https://example.test/case", "quote": "Synthetic conditions", "context": "Test", "basis": "Test"},)))
        for field in (*values, *conditions):
            if field in supplied:
                continue
            ledger.merge(assess_assumption(supported_review, {"subject": "case", "field": field,
                "value": values.get(field), "unit": UNITS.get(field, "condition"), "statement": f"Assume {field}",
                "basis": "Synthetic fixture", "evidence": list(ledger.evidence_options())}, ledger, {}))
        return ledger

    def execute(self, ledger, name, fields, **kwargs):
        inputs = {}
        if ledger.scenarios["case"].get("role", "target") != "target":
            from episodic_workflow.case_reports.research.records import numeric_provenance
            native = {fact.field: key for key, fact in numeric_provenance(ledger).items()
                      if fact.subject == "case" and fact.field in fields}
            inputs["source_inputs"] = list(native.values())
        return execute_tool({"action": name, "subject": "case", **inputs,
            "basis": "Explicit synthetic model", **kwargs}, ledger, tuple(kwargs.get("metrics", ())),
            review={"admit": True, "reason": "Matched synthetic event; conditional check", "limitations": []})

    def test_geometry_calculation_is_available_without_an_episode_wrapper(self):
        values = {"floor_area_m2": 50, "room_height_m": 3, "removal_rate_s": .002}
        ledger = self.ledger(values)
        result = self.execute(ledger, "calculate_geometry_and_removal", values, metrics=[])
        derived = {f.field: f.value for f in result.facts}
        self.assertEqual(derived["room_volume_m3"], 150)
        self.assertAlmostEqual(derived["removal_capacity_m3_s"], .3)
        self.assertTrue(result.calculations[0]["assumption_versions"])

    def test_tools_merge_compatible_input_subsets_of_one_named_parameterization(self):
        conditions = {name: f"Assume {name}" for name in
                      ("well_mixed", "single_infectious_source", "uniform_breathing_and_masks")}
        values = {FIELD: 30, "susceptible_count": 4, "exposure_time_s": 3600,
                  "removal_capacity_m3_s": .2, "breathing_flow_m3_s": .00015,
                  "mask_penetration": 1, "relative_transmissibility": 1, "risk_tolerance": .1}
        ledger = self.ledger(values, conditions)
        for tool, metric, omitted in (
            ("calculate_transmission_metrics", "indoor_reproductive_number", "risk_tolerance"),
            ("calculate_conditional_limits", "exposure_time_bound_s", "exposure_time_s"),
        ):
            ledger.merge(self.execute(ledger, tool, [f for f in values if f != omitted],
                conditions=conditions, metrics=[metric], parameterization="shared assumptions"))
        self.assertEqual(len(ledger.estimates), 1)
        estimate = next(iter(ledger.estimates.values()))
        self.assertEqual(estimate["parameters"], values)
        self.assertEqual(set(estimate["metrics"]), {"indoor_reproductive_number", "exposure_time_bound_s"})
        self.assertTrue(all(m["status"] == "calculated" for m in estimate["metrics"].values()))
        self.assertEqual(len(estimate["calculation_versions"]), 2)

    def test_named_parameterization_does_not_merge_conflicting_values_or_versions(self):
        left = {"subject": "case", "label": "same", "parameterization": "same",
                "parameters": {FIELD: 30}, "conditions": {}, "metrics": {},
                "assumption_versions": {"support": "v1"}}
        for changes in ({"parameters": {FIELD: 40}}, {"assumption_versions": {"support": "v2"}},
                        {"parameterization": "alternative"}):
            with self.subTest(changes=changes):
                ledger = Ledger(ResearchResult(estimates=(left, {**left, **changes})))
                self.assertEqual(len(ledger.estimates), 2)

    def test_calibration_prediction_and_comparison_keep_training_dependence(self):
        conditions = {name: f"Assume {name}" for name in
                      ("well_mixed", "single_infectious_source", "uniform_breathing_and_masks")}
        values = {"attack_probability": .1, "susceptible_count": 10, "index_case_count": 1,
                  "exposure_time_s": 3600, "removal_capacity_m3_s": .2, "breathing_flow_m3_s": .00015,
                  "mask_penetration": 1, "relative_transmissibility": 1}
        ledger = self.ledger(values, conditions)
        ledger.scenarios["case"]["role"] = "calibration"
        calibration = self.execute(ledger, "calibrate_infectiousness", values,
                                  operator="cq_from_steady_outbreak", conditions=conditions)
        ledger.merge(calibration)
        expected_cq = -math.log1p(-.1) * .2 / (3600 * .00015**2)
        self.assertAlmostEqual(calibration.facts[0].value, expected_cq)
        prediction = self.execute(ledger, "calculate_transmission_metrics",
            (*values, FIELD), conditions=conditions, metrics=["individual_infection_probability"])
        ledger.merge(prediction)
        prediction_key = next(k for k, c in ledger.calculation_options().items() if c["tool"] == "calculate_transmission_metrics")
        observation_key = next(k for k, f in ledger.options().items() if f.field == "attack_probability" and f.kind == "source_observation")
        comparison = self.execute(ledger, "compare_predictions_with_observations", (),
            prediction=prediction_key, observation=observation_key, metric="individual_infection_probability")
        result = comparison.calculations[0]["output"]
        self.assertAlmostEqual(result["residual"], 0)
        self.assertEqual(result["category"], "calibration_consistency")
        self.assertFalse(result["independent_validation"])
        ledger.merge(comparison)
        credit = ledger.result().credit(("evidence_consistency",), scientific=True)
        self.assertEqual(credit.qualities[0][2], 0)
        # A changed claim cannot silently authenticate an old prediction.
        key = next(iter(prediction.calculations[0]["assumption_versions"]))
        changed = {**ledger.assumptions[key], "version": "changed-support-version", "statement": "Revised scope"}
        ledger.merge(ResearchResult(assumptions=(changed,)))
        historical = self.execute(ledger, "compare_predictions_with_observations", (),
            prediction=prediction_key, observation=observation_key, metric="individual_infection_probability")
        self.assertAlmostEqual(historical.calculations[0]["output"]["residual"], 0)
        self.assertEqual(historical.calculations[0]["assumption_versions"][key], prediction.calculations[0]["assumption_versions"][key])

    def test_unassessed_inputs_can_be_computed_without_earning_support(self):
        ledger = self.ledger({"floor_area_m2": 50, "room_height_m": 3})
        ledger.assumptions.clear()
        result = self.execute(ledger, "calculate_geometry_and_removal", ("floor_area_m2", "room_height_m"), metrics=[])
        self.assertEqual(result.facts[0].value, 150)
        self.assertEqual(len(result.calculations[0]["support_issues"]), 2)
        self.assertEqual(result.estimates[0]["reasonableness"], 0)


class ReportProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, (cls.record, original) = run_synthetic()
        cls.result = ResearchResult(
            estimates=tuple(e for s in original.data["scenarios"].values() for e in s["alternatives"]),
            assumptions=tuple(original.data["assumption_assessments"]),
            scenarios=tuple(s["definition"] for s in original.data["scenarios"].values()),
            calculations=tuple(original.data["calculations"]))

    def report(self, result, metrics=METRICS):
        run = ResearchRun(ResearchProviders(lambda _: {}, lambda *_: (), lambda _: None),
                          ResearchLimits(), REQUEST, metrics)
        return ResearchReport.from_result(result, self.record, run)

    def test_parent_progress_uses_current_support_without_running_a_kernel(self):
        original = json.dumps(self.result.synthesis(), sort_keys=True)
        progress = scientific_progress(self.result, METRICS)
        self.assertEqual(progress["calculated_quantities"], len(METRICS))
        self.assertEqual(progress["targets"]["target"]["coverage"],
                         self.report(self.result).data["scenarios"]["target"]["coverage"])
        changed = replace(self.result, assumptions=tuple({**a, "version": "changed", "statement": "Revised scope"}
                                                        for a in self.result.assumptions))
        stale = scientific_progress(changed, METRICS)
        self.assertEqual(stale["calculated_quantities"], 0)
        self.assertTrue(all(r["status"] == "unsupported_calculation" for r in stale["targets"]["target"]["coverage"].values()))
        self.assertEqual(json.dumps(self.result.synthesis(), sort_keys=True), original)

    def test_equation_evidence_and_conditions_are_not_numerical_parameterization(self):
        condition = next(a for a in self.result.assumptions if a["value"] is None)
        result = replace(self.result, estimates=(), calculations=(), assumptions=(condition,))
        progress = scientific_progress(result, METRICS)
        target = progress["targets"]["target"]
        self.assertEqual(target["assessed_numeric_fields"], [])
        self.assertEqual(target["assessed_condition_fields"], [condition["field"]])
        self.assertEqual(target["evaluated_parameterizations"], 0)
        self.assertFalse(target["complete_parameterization_available"])
        self.assertTrue(all(r["status"] == "not_evaluated" for r in target["coverage"].values()))

    def test_supported_inapplicability_resolves_scope_without_becoming_a_number(self):
        helper = CalculationToolTests()
        ledger = helper.ledger({"mask_penetration": .3})
        metric = "critical_distance_m"
        result = helper.execute(ledger, "calculate_conditional_limits", ("mask_penetration",), metrics=[metric])
        ledger.merge(result)
        report = self.report(ledger.result(), (metric,))
        self.assertTrue(report.passed)
        self.assertEqual(report.data["status"], "requested_model_inapplicable")
        coverage = report.data["scenarios"]["case"]["coverage"][metric]
        self.assertEqual(coverage["status"], "inapplicable")
        self.assertEqual(coverage["missing"], [])
        self.assertFalse(report.data["scenarios"]["case"]["comparisons"])
        self.assertIsNone(result.estimates[0]["metrics"][metric]["value"])
        self.assertIn("not applicable", report.markdown())
        self.assertIn("requires unmasked", report.markdown())
        self.assertEqual(ledger.result().credit((metric,), scientific=True).qualities[0][2], 1)
        stale = replace(ledger.result(), assumptions=tuple({**a, "version": "changed", "reasonableness": 0}
                                                         for a in ledger.assumptions.values()))
        withdrawn = self.report(stale, (metric,))
        self.assertFalse(withdrawn.passed)
        self.assertEqual(withdrawn.data["scenarios"]["case"]["coverage"][metric]["status"], "unsupported_applicability")
        self.assertEqual(stale.credit((metric,), scientific=True).qualities[0][2], 0)

    def test_bare_inapplicable_label_does_not_resolve_an_output(self):
        estimate = self.result.estimates[0]
        invalid = {**estimate, "metrics": {m: {"status": "inapplicable"} for m in METRICS}}
        result = replace(self.result, estimates=(invalid,))
        self.assertFalse(self.report(result).passed)
        self.assertTrue(all(q == 0 for _, _, q in result.credit(METRICS, scientific=True).qualities))

    def test_stale_calculations_are_history_not_current_coverage(self):
        assumptions = tuple({**a, "version": "new-support-version", "statement": "Revised scope"}
                            for a in self.result.assumptions)
        report = self.report(replace(self.result, assumptions=assumptions))
        self.assertFalse(report.passed)
        scenario = report.data["scenarios"]["target"]
        self.assertTrue(scenario["alternatives"])
        self.assertTrue(all(row["status"] == "unsupported_calculation" for row in scenario["coverage"].values()))
        self.assertFalse(scenario["comparisons"])
        self.assertIn("unsupported_calculation", report.markdown())

    def test_provenance_verifier_checks_admitted_versions_not_rejected_audit_proposals(self):
        source, original, latest = "https://example.test/revised", "Original evidence", "Revised evidence"
        v1, v2 = (hashlib.sha256(t.encode()).hexdigest() for t in (original, latest))
        fact = {"source": source, "source_version": v1, "quote": original, "kind": "source_evidence"}
        data = {"research_request": REQUEST, "source_documents": {source: {
            "text": latest, "sha256_utf8": v2, "versions": {v1: {"text": original}, v2: {"text": latest}}}},
            "evidence_and_assumptions": [], "qualitative_evidence": [fact],
            "assumption_assessments": [], "scenarios": {},
            "model_attempts": [{"response": {"quote": "Invented audit-only quote", "source": source}}]}
        assert_admitted_research_provenance(self, data)
        with self.assertRaises(AssertionError):
            assert_admitted_research_provenance(self, {**data, "qualitative_evidence": [{**fact, "quote": "Invented quotation"}]})
        with self.assertRaises(AssertionError):
            assert_admitted_research_provenance(self, {**data, "qualitative_evidence": [{**fact, "source_version": v2}]})

    def test_stale_alternative_cannot_expand_supported_sensitivity_range(self):
        previous = self.result.estimates[0]
        stale = {**previous, "label": "Historical alternative", "assumption_versions": {
            k: "old-version" for k in previous["assumption_versions"]}, "assumption_claims": {}}
        stale.pop("metric_support", None)
        report = self.report(replace(self.result, estimates=(*self.result.estimates, stale)))
        self.assertTrue(report.passed)
        scenario = report.data["scenarios"]["target"]
        self.assertFalse(scenario["comparisons"])
        self.assertEqual(len(scenario["alternatives"]), 2)

    def test_markdown_includes_comparison_status_and_unresolved_questions(self):
        estimate = self.result.estimates[0]
        comparison = {"tool": "compare_predictions_with_observations", "subject": "target",
            "assumption_versions": estimate["assumption_versions"], "output": {
                "metric": "individual_infection_probability", "predicted": .1, "observed": .2,
                "residual": -.1, "category": "out_of_calibration_comparison"}}
        result = replace(self.result, calculations=(*self.result.calculations, comparison),
                         issues=("Unresolved synthetic mechanism",))
        report = self.report(result)
        self.assertEqual(report.data["observation_comparisons"][0]["support_status"], "current")
        text = report.markdown()
        self.assertIn("Prediction and Observation Checks", text)
        self.assertIn("0.1 | 0.2 | -0.1", text)
        self.assertIn("out_of_calibration_comparison", text)
        self.assertIn("Unresolved synthetic mechanism", text)
        changed = tuple({**a, "version": "changed"} for a in result.assumptions)
        self.assertEqual(self.report(replace(result, assumptions=changed)).data["observation_comparisons"][0]["support_status"],
                         "stale_or_unsupported")

    def test_markdown_does_not_attribute_uncited_evidence_to_an_assumption(self):
        original = self.result.assumptions[0]
        extra = {"source": "https://example.test/uncited", "basis": "Considered but not cited"}
        assessment = {**original, "evidence": {**original["evidence"], "uncited": extra}}
        report = self.report(replace(self.result, assumptions=(assessment,)))
        self.assertIn("uncited", report.data["assumption_assessments"][0]["evidence"])
        self.assertNotIn("https://example.test/uncited", report.markdown())
        first_citation = next(iter(original["review"]["criteria"].values()))["evidence"][0]
        self.assertIn(original["evidence"][first_citation]["source"], report.markdown())
