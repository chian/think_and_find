"""Offline contract fixtures, not evidence of live scientific research quality."""

import io
import json
import unittest
from unittest.mock import patch

from episodic_workflow.case_reports.research import ResearchLimits, ResearchProviders, run_guideline_research
from episodic_workflow.case_reports.research.admission import extract_and_admit
from episodic_workflow.case_reports.research.assumptions import assess_assumption, frame_scenario
from episodic_workflow.case_reports.research.records import Credit, Fact, Ledger, ResearchResult
from episodic_workflow.credit_assignment.research import ResearchController
from episodic_workflow.cli import main
from episodic_workflow.tests.test_research import AdaptiveAgent, REQUEST, run_synthetic, supported_review


FIELD = "quanta_concentration_exhaled_m3"


def fixture_ledger():
    fact = Fact("reference", FIELD, 30, "quanta/m3", "source_model_parameter", "Synthetic reference estimate",
                "https://example.test/source", "Cq = 30 quanta/m3", "Synthetic activity")
    return Ledger(ResearchResult(facts=(fact,), scenarios=({"subject": "target", "description": "Synthetic scenario"},)))


def candidate(**changes):
    return {"subject": "target", "field": FIELD, "value": 30, "unit": "quanta/m3",
            "statement": "Conditional infectiousness for the target activity", "basis": "Explicit transfer hypothesis",
            "evidence": ["evidence:1"], **changes}


class AssumptionSupportTests(unittest.TestCase):
    def test_filled_fields_are_not_scientific_credit(self):
        ledger = fixture_ledger()
        credit = ledger.result().credit((FIELD,), scientific=True)
        controller = ResearchController((FIELD,))
        self.assertEqual(controller.observe("extraction", credit, is_root=True).score, 0)
        self.assertFalse(ledger.result().credit((FIELD,)).identities)

    def test_hypothesis_without_evidence_is_retained_but_not_credited(self):
        result = assess_assumption(lambda r: supported_review(r, grade=0, verdict="unsupported"),
                                   candidate(evidence=[]), fixture_ledger(), {})
        self.assertEqual(result.assumptions[0]["reasonableness"], 0)
        self.assertEqual(result.assumptions[0]["resolution_credit"], 0)
        self.assertFalse(result.facts)
        self.assertTrue(result.diagnostics[0]["why"])

    def test_reasonable_inference_need_not_be_printed_verbatim(self):
        result = assess_assumption(lambda r: supported_review(r, grade=1, verdict="provisional"),
                                   candidate(value=25), fixture_ledger(), {})
        self.assertEqual(result.facts[0].value, 25)
        self.assertEqual(result.assumptions[0]["reasonableness"], .5)
        self.assertEqual(result.facts[0].kind, "model_assumption")
        self.assertEqual(result.assumptions[0]["evidence"]["evidence:1"]["value"], 30)

    def test_assessment_summary_cites_basis_without_discarding_considered_evidence(self):
        ledger = fixture_ledger()
        ledger.merge(ResearchResult(facts=(Fact("other", FIELD, 50, "quanta/m3", "source_model_parameter",
            "Different synthetic activity", "https://example.test/other", "Cq = 50 quanta/m3", "Other activity"),)))
        requests = []
        def reviewer(request):
            requests.append(request)
            response = supported_review(request)
            for criterion in response["criteria"].values():
                criterion["evidence"] = ["evidence:1"]
            return response
        action = candidate(evidence=list(ledger.evidence_options()))
        ledger.merge(assess_assumption(reviewer, action, ledger, {}))
        summary = ledger.assumption_options()["assumption:1"]
        self.assertEqual(set(summary["evidence"]), {"evidence:1"})
        self.assertEqual(summary["considered_evidence_count"], 2)
        self.assertEqual(len(next(iter(ledger.assumptions.values()))["evidence"]), 2)
        self.assertEqual(len(ledger.evidence_options()), 2)
        assess_assumption(reviewer, action, ledger, {})
        self.assertEqual(len(requests[-1]["evidence"]), 2)
        self.assertEqual(requests[-1]["previous_assessments"], [summary])

    def test_positive_review_cannot_cite_missing_or_invented_evidence(self):
        for refs in ([], ["evidence:999"]):
            def reviewer(request):
                response = supported_review(request)
                response["criteria"]["support"]["evidence"] = refs
                return response
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                assess_assumption(reviewer, candidate(), fixture_ledger(), {})

    def test_bottleneck_prevents_one_strong_dimension_hiding_a_gap(self):
        def reviewer(request):
            response = supported_review(request, verdict="provisional")
            response["criteria"]["applicability"]["grade"] = 0
            return response
        result = assess_assumption(reviewer, candidate(), fixture_ledger(), {})
        self.assertEqual(result.assumptions[0]["reasonableness"], 0)
        self.assertFalse(result.facts)

    def test_unsupported_grade_cannot_claim_supported_verdict(self):
        with self.assertRaisesRegex(ValueError, "all rubric"):
            assess_assumption(lambda r: supported_review(r, grade=1), candidate(), fixture_ledger(), {})

    def test_refutation_is_learning_but_not_support_for_output(self):
        ledger = fixture_ledger()
        result = assess_assumption(lambda r: supported_review(r, verdict="contradicted"), candidate(), ledger, {})
        ledger.merge(result)
        self.assertFalse(result.facts)
        credit = ledger.result().credit((FIELD, "indoor_reproductive_number"), scientific=True)
        self.assertEqual([row[2] for row in credit.qualities], [1, 0])

    def test_repetition_or_rewording_does_not_manufacture_credit(self):
        ledger = fixture_ledger()
        controller = ResearchController((FIELD,))
        for n in range(2):
            result = assess_assumption(supported_review, candidate(statement=f"Paraphrase {n}"), ledger, {})
            ledger.merge(result)
            step = controller.observe(str(n), ledger.result().credit((FIELD,), scientific=True), is_root=True)
        self.assertEqual(step.gain, 0)
        self.assertEqual(len(ledger.assumptions), 1)

    def test_reassessment_withdraws_old_calculation_support(self):
        ledger = fixture_ledger()
        ledger.merge(assess_assumption(supported_review, candidate(), ledger, {}))
        key = next(iter(ledger.assumptions))
        estimate = {"subject": "target", "parameters": {FIELD: 30}, "conditions": {},
                    "metrics": {"indoor_reproductive_number": {"status": "calculated", "value": 1}},
                    "assumption_keys": [key]}
        ledger.merge(ResearchResult(estimates=(estimate,)))
        controller = ResearchController(("indoor_reproductive_number",))
        controller.observe("before", ledger.result().credit(controller.columns, scientific=True), is_root=True)
        ledger.merge(assess_assumption(lambda r: supported_review(r, verdict="contradicted"), candidate(), ledger, {}))
        step = controller.observe("after", ledger.result().credit(controller.columns, scientific=True), is_root=True)
        self.assertEqual(step.score, 0)
        self.assertEqual(step.gain, -1)
        self.assertFalse(step.stop)
        self.assertEqual(controller.state().history, (0, 1, 0))

    def test_output_count_does_not_increase_the_group_weight(self):
        for metrics in (("m1",), ("m1", "m2", "m3", "m4")):
            controller = ResearchController((FIELD, *metrics), groups=((FIELD,), metrics))
            score = controller.observe("assessed", Credit(qualities=((FIELD, "assessment", 1),)), is_root=True).score
            self.assertEqual(score, .5)

    def test_malformed_extraction_keeps_why_and_proposal_in_strict_json(self):
        result = extract_and_admit(lambda _: {"facts": [{"field": FIELD, "quote": "Cq = 30", "value": float("nan")}]},
                                   text="Cq = 30", source="https://example.test/source", focus=(FIELD,),
                                   objective="Extract evidence", context={}, question="Cq?")
        self.assertFalse(result.evaluated)
        self.assertIn("finite", result.diagnostics[0]["why"])
        self.assertEqual(result.diagnostics[0]["proposal"]["value"], "nan")
        json.dumps(result.as_record(), allow_nan=False)

    def test_scope_review_cannot_preload_parameters(self):
        with self.assertRaisesRegex(ValueError, "cannot inject"):
            frame_scenario(lambda _: self.fail("Must fail before review"),
                           {"subject": "target", "description": "A room", "basis": "Test", "parameters": {FIELD: 30}},
                           fixture_ledger(), {})

    def test_reframing_cannot_silently_reuse_old_assessments(self):
        with self.assertRaisesRegex(ValueError, "new subject"):
            frame_scenario(lambda _: self.fail("Must fail before review"),
                           {"subject": "target", "description": "A different activity", "basis": "Test"}, fixture_ledger(), {})

    def test_an_identical_scope_reuses_the_admitted_review_without_new_evidence(self):
        definition = {"subject": "target", "description": "Original scope", "basis": "Original request",
                      "role": "target", "scope_review": "Already reviewed"}
        ledger = Ledger(ResearchResult(scenarios=(definition,)))
        result = frame_scenario(lambda _: self.fail("An unchanged definition needs no new model review"),
                                definition, ledger, {})
        self.assertEqual(result.scenarios, (definition,))
        self.assertFalse(result.facts)
        self.assertFalse(result.assumptions)
        self.assertFalse(result.credit((FIELD,), scientific=True).qualities[0][2])

    def test_assumption_cannot_override_extracted_request_constraint(self):
        ledger = fixture_ledger()
        ledger.merge(ResearchResult(facts=(Fact("target", FIELD, 20, "quanta/m3", "request_constraint",
                                                "Explicit conditional request", "request", "Assume 20 quanta/m3", "Test condition"),)))
        with self.assertRaisesRegex(ValueError, "requester constraint"):
            assess_assumption(supported_review, candidate(value=30), ledger, {})

    def test_interpreting_a_supplied_number_is_not_scientific_progress(self):
        ledger = Ledger(ResearchResult(facts=(Fact("target", FIELD, 20, "quanta/m3", "request_constraint",
            "Explicit conditional request", "request", "Assume 20 quanta/m3", "Test condition"),),
            scenarios=({"subject": "target", "description": "Synthetic scenario"},)))
        result = assess_assumption(supported_review, candidate(value=20), ledger, {})
        self.assertEqual(result.assumptions[0]["role"], "request_constraint")
        self.assertEqual(result.assumptions[0]["resolution_credit"], 0)
        self.assertEqual(result.assumptions[0]["reasonableness"], 1)

    def test_explicit_request_model_condition_is_not_an_empirical_discovery(self):
        evidence = {"subject": "target", "fields": ["well_mixed"], "kind": "request_constraint",
                    "source": "request", "quote": "Assume a well-mixed room", "context": "Conditional scenario",
                    "claim": "The requester defines well-mixed conditions", "basis": "Explicit constraint"}
        ledger = Ledger(ResearchResult(evidence=(evidence,), scenarios=({"subject": "target"},)))
        result = assess_assumption(supported_review, {"subject": "target", "field": "well_mixed", "value": None,
            "statement": "Assume a well-mixed room", "basis": "Explicit request", "evidence": ["evidence:1"]}, ledger, {})
        self.assertEqual(result.assumptions[0]["resolution_credit"], 0)
        self.assertEqual(result.assumptions[0]["role"], "request_constraint")

    def test_request_scenario_class_can_bind_to_target_without_discovery_credit(self):
        evidence = {"subject": "requested scenario class", "fields": ["well_mixed"], "kind": "request_constraint",
                    "source": "request", "quote": "Assume well-mixed rooms", "context": "All target scenarios",
                    "claim": "The requester defines well-mixed conditions", "basis": "Explicit constraint"}
        source = {**evidence, "subject": "reference model", "kind": "source_evidence",
                  "source": "https://example.test/model", "quote": "Spatially uniform concentration",
                  "claim": "Synthetic model premise", "basis": "Synthetic source explanation"}
        ledger = Ledger(ResearchResult(evidence=(evidence, source), scenarios=({"subject": "target"},)))
        def reviewer(request):
            response = supported_review(request)
            for row in response["criteria"].values():
                row["evidence"] = ["evidence:2"]
            response["request_constraint"] = {"evidence": ["evidence:1"],
                "reason": "This target is a member of the request's explicitly defined well-mixed scenario class"}
            schema = request["response_contract"]
            self.assertIn("request_constraint", schema["required"])
            self.assertEqual(schema["properties"]["request_constraint"]["properties"]["evidence"]["items"]["enum"], ["evidence:1"])
            return response
        action = {"subject": "target", "field": "well_mixed", "value": None,
                  "statement": "Spatially uniform target background", "basis": "Requested model scope",
                  "evidence": list(ledger.evidence_options())}
        result = assess_assumption(reviewer, action, ledger, {})
        ledger.merge(result)
        assessment = result.assumptions[0]
        self.assertEqual(assessment["role"], "request_constraint")
        self.assertEqual(assessment["resolution_credit"], 0)
        self.assertEqual(assessment["reasonableness"], 1)
        self.assertIn("evidence:1", ledger.assumption_options()["assumption:1"]["evidence"])
        self.assertEqual(ledger.result().credit(("well_mixed",), scientific=True).qualities[0][2], 0)

    def test_request_classification_cannot_relabel_source_evidence_or_another_value(self):
        ledger = fixture_ledger()
        ledger.merge(ResearchResult(facts=(Fact("requested class", FIELD, 20, "quanta/m3", "request_constraint",
            "Synthetic requester condition", "request", "Assume 20 quanta/m3", "Synthetic condition"),)))
        for refs in (["evidence:1"], ["evidence:2"], ["evidence:999"], []):
            def reviewer(request):
                response = supported_review(request)
                response["request_constraint"] = {"evidence": refs, "reason": "Invalid fixture classification"}
                return response
            with self.subTest(refs=refs), self.assertRaisesRegex(ValueError, "admitted requester evidence"):
                assess_assumption(reviewer, candidate(evidence=list(ledger.evidence_options())), ledger, {})

    def test_request_scope_classification_cannot_be_omitted(self):
        def reviewer(request):
            response = supported_review(request)
            del response["request_constraint"]
            return response
        with self.assertRaisesRegex(ValueError, "must classify request_constraint"):
            assess_assumption(reviewer, candidate(), fixture_ledger(), {})

    def test_malformed_criterion_is_a_contract_failure(self):
        def reviewer(request):
            response = supported_review(request)
            response["criteria"]["support"] = None
            return response
        with self.assertRaisesRegex(ValueError, "criterion must be an object"):
            assess_assumption(reviewer, candidate(), fixture_ledger(), {})


class PromptBoundaryTests(unittest.TestCase):
    def test_initial_request_has_no_admitted_answers(self):
        agent, (_, report) = run_synthetic()
        first = agent.requests[0]
        self.assertEqual(first["context"]["research_request"], REQUEST)
        for key in ("admitted_facts", "evidence_options", "assumption_options"):
            self.assertEqual(first[key], {})
        self.assertEqual(first["context"]["defined_scenarios"], {})
        self.assertTrue(report.data["assumption_assessments"])
        self.assertTrue(report.data["qualitative_evidence"])
        self.assertTrue(any("diagnostics" in json.dumps(r.get("child_reports", {})) for r in agent.requests))

    def test_api_rejects_structured_or_seeded_answers(self):
        providers = ResearchProviders(lambda _: self.fail("Provider must not be called"), lambda *_: (), lambda _: None)
        for kw in ({"scenarios": {}}, {"seed_sources": ()}, {"disease_name": "test"}):
            with self.subTest(kw=kw), self.assertRaises(TypeError):
                run_guideline_research(research_request=REQUEST, providers=providers, **kw)
        with self.assertRaises(ValueError):
            run_guideline_research(research_request={"parameters": {}}, providers=providers)

    def test_cli_rejects_bypasses_before_provider_calls(self):
        for args in (("--target-scenario", "{}"), ("--scenarios-file", "x.json"),
                     ("--source-url", "https://example.test"), ("--legacy-workflow",)):
            with self.subTest(args=args), patch("episodic_workflow.cli.run_configured_safety_guideline_request") as run:
                error = io.StringIO()
                code = main([REQUEST, *args], stdout=io.StringIO(), stderr=error)
                self.assertEqual(code, 2)
                self.assertIn("disabled", error.getvalue())
                run.assert_not_called()

    def test_request_reading_cannot_label_its_contents_as_source_observations(self):
        def agent(_):
            return {"facts": [{"subject": "target", "field": FIELD, "value": 30, "unit": "quanta/m3",
                               "quote": "Cq = 30 quanta/m3", "kind": "source_observation", "context": "Request",
                               "basis": "Pretending a request is a measurement"}]}
        result = extract_and_admit(agent, text="Cq = 30 quanta/m3", source="request", focus=(FIELD,),
                                   objective="Interpret request", context={}, question="What is specified?")
        self.assertFalse(result.facts)
        self.assertFalse(result.evaluated)

    def test_invalid_action_reason_reaches_parent_without_becoming_evidence(self):
        agent = AdaptiveAgent()
        def failing(request):
            if request["operation"] == "propose_investigation_strategy":
                return {"action": "unknown", "why": "Malformed fixture action"}
            return agent(request)
        _, (_, report) = run_synthetic(failing, ResearchLimits())
        failures = report.data["attempt_diagnostics"]
        self.assertTrue(any(d["category"] == "invalid_action_contract" for d in failures))
        self.assertTrue(any((d.get("proposal") or {}).get("action") == "unknown" for d in failures))
        self.assertFalse(report.passed)
