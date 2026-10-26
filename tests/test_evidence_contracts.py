"""Evidence admission, calibration, and credit invariants on synthetic data."""

from __future__ import annotations

import copy
import math
import unittest

from episodic_workflow.case_reports.evidence_children import admit_exposure_time_from_extraction
from episodic_workflow.case_reports.evidence_extraction import (
    EvidenceSource,
    extract_exposure_time,
    extract_population_counts,
    extract_transmissibility_assumption,
    run_llm_field_extraction,
)
from episodic_workflow.case_reports.safety_guideline import (
    CUMULATIVE_EXPOSURE_MODELS,
    run_evaluate_safety_guideline_episode,
)
from episodic_workflow.credit_assignment.calculations import run_named_calculation
from episodic_workflow.credit_assignment.cq_calibration import CQ_CALIBRATION_CHILDREN, CQ_NORMALIZATION_CRITERIA
from episodic_workflow.tests.fixtures import FACTS, SCENARIO, SOURCE_TEXT, SOURCE_URL, applicability_response
from episodic_workflow.tests.report_assertions import assert_source_backed_calibration


def extracted_response(request, facts=FACTS):
    if request.get("operation") == "review_evidence_applicability":
        return applicability_response(request)
    return {"values": [
        {
            "field": name,
            "value": facts[name][0],
            "unit": facts[name][1],
            "normalized_value": facts[name][0],
            "normalized_unit": facts[name][1],
            "source_ref": request["source"]["source_ref"],
            "evidence_span": facts[name][2],
            "confidence": 1.0,
            "notes": [],
        }
        for name in request["required_fields"] if name in facts
    ]}


def run_synthetic(*, scenario=SCENARIO, disease="synthetic disease", model="steady_state_cumulative_exposure_bound", mutate=None, sources=None, requests=None, require_transmissibility_evidence=False):
    facts_by_ref = sources or {SOURCE_URL: FACTS}
    queries = []

    def search(query, limit):
        queries.append(query)
        return [
            {"source_ref": ref, "url": ref, "title": "Synthetic evidence", "text": " ".join(fact[2] for fact in facts.values())}
            for ref, facts in facts_by_ref.items()
        ][:limit]

    def extract(request):
        if requests is not None:
            requests.append(request)
        result = extracted_response(request, facts_by_ref[request["source"]["source_ref"]])
        if mutate:
            mutate(result)
        return result

    episode, report = run_evaluate_safety_guideline_episode(
        disease_name=disease, target_scenario=scenario, setting="test setting",
        search=search, llm=extract, transmission_model=model,
        require_transmissibility_evidence=require_transmissibility_evidence,
    )
    return episode, report, queries


def assert_credit_totals(test, report):
    test.assertAlmostEqual(report["score"], sum(child["score"] for child in report["child_reports"].values()))
    test.assertAlmostEqual(report["max_score"], sum(child["max_score"] for child in report["child_reports"].values()))
    if report["passed"]:
        test.assertEqual(set(report["credited_children"]), set(report["child_reports"]))
        test.assertTrue(all(child["passed"] for child in report["child_reports"].values()))


class ExtractionContractTests(unittest.TestCase):
    def setUp(self):
        self.source = EvidenceSource(source_ref=SOURCE_URL, text=SOURCE_TEXT)

    def test_request_declares_exact_normalized_units(self):
        requests = []

        def extractor(request):
            requests.append(request)
            return extracted_response(request)

        self.assertTrue(extract_population_counts(extractor, self.source).passed)
        self.assertEqual(requests[0]["normalized_units"], {name: "count" for name in ("attendee_count", "index_case_count", "secondary_case_count")})
        self.assertTrue(extract_exposure_time(extractor, self.source).passed)
        self.assertEqual(requests[1]["normalized_units"], {"exposure_time_s": "s"})

    def test_requests_disallow_exposed_total_confusion_and_circular_scaling(self):
        requests = []

        def extractor(request):
            requests.append(request)
            return extracted_response(request)

        extract_population_counts(extractor, self.source)
        extract_transmissibility_assumption(extractor, self.source)
        self.assertIn("attendee_count must include", requests[0]["instruction"])
        self.assertIn("susceptible_count", requests[0]["instruction"])
        self.assertIn("not a comparison of fitted", requests[1]["instruction"])
        self.assertIn("independent scalar input", requests[1]["instruction"])

    def test_bad_evidence_is_rejected(self):
        cases = (
            ({"source_ref": "https://unretrieved.example/"}, "source_ref_mismatch"),
            ({"evidence_span": "Invented evidence."}, "unsupported_evidence_span"),
            ({"value": float("inf")}, "invalid_extracted_value"),
            ({"normalized_value": float("nan")}, "invalid_extracted_value"),
            ({"confidence": 0.1}, "low_extraction_confidence"),
            ({"confidence": True}, "missing_extraction_field"),
        )
        for change, code in cases:
            with self.subTest(change=change):
                def extractor(request):
                    result = extracted_response(request)
                    result["values"][0].update(change)
                    return result
                report = extract_exposure_time(extractor, self.source)
                self.assertFalse(report.passed)
                self.assertIn(code, {error.code for error in report.errors})

    def test_duplicate_fields_fail(self):
        def extractor(request):
            result = extracted_response(request)
            result["values"] *= 2
            return result
        report = extract_exposure_time(extractor, self.source)
        self.assertFalse(report.passed)
        self.assertIn("duplicate_extracted_field", {error.code for error in report.errors})

    def test_nonexact_span_gets_one_same_source_repair(self):
        requests = []

        def extractor(request):
            requests.append(request)
            response = extracted_response(request)
            if "span_correction" not in request:
                response["values"][0]["evidence_span"] = "A paraphrased exposure duration."
            return response

        report = extract_exposure_time(extractor, self.source)
        self.assertTrue(report.passed)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["source"], requests[1]["source"])
        self.assertEqual(len(report.span_repair_history), 1)
        self.assertEqual(report.span_repair_history[0]["errors"][0]["code"], "unsupported_evidence_span")
        child = admit_exposure_time_from_extraction(report)
        self.assertTrue(child.passed)
        self.assertEqual(len(child.outputs["span_repair_history"]), 1)

    def test_span_repair_is_bounded_and_revalidates_every_field(self):
        for correction, code in (
            ({"evidence_span": "Still fabricated"}, "unsupported_evidence_span"),
            ({"source_ref": "unretrieved"}, "source_ref_mismatch"),
            ({"normalized_value": float("inf")}, "invalid_extracted_value"),
            ({"confidence": 0.1}, "low_extraction_confidence"),
        ):
            with self.subTest(correction=correction):
                requests = []

                def extractor(request):
                    requests.append(request)
                    response = extracted_response(request)
                    response["values"][0].update(correction if "span_correction" in request else {"evidence_span": "Fabricated"})
                    return response

                report = extract_exposure_time(extractor, self.source)
                self.assertFalse(report.passed)
                self.assertEqual(len(requests), 2)
                self.assertIn(code, {error.code for error in report.errors})

    def test_nonspan_errors_are_not_retried(self):
        requests = []

        def extractor(request):
            requests.append(request)
            response = extracted_response(request)
            response["values"][0]["source_ref"] = "unretrieved"
            return response

        report = extract_exposure_time(extractor, self.source)
        self.assertFalse(report.passed)
        self.assertEqual(len(requests), 1)
        self.assertFalse(report.span_repair_history)

    def test_span_repair_provider_failure_retains_diagnostics(self):
        def extractor(request):
            if "span_correction" in request:
                raise RuntimeError("Synthetic correction failure")
            response = extracted_response(request)
            response["values"][0]["evidence_span"] = "Fabricated"
            return response

        report = extract_exposure_time(extractor, self.source)
        self.assertFalse(report.passed)
        self.assertEqual(report.errors[0].code, "llm_error")
        self.assertEqual(len(report.span_repair_history), 1)

    def test_assumptions_require_exact_similarity_evidence(self):
        for basis in ({}, {"similarity_basis": "Same activity", "similarity_evidence_span": "Not in the source"}):
            with self.subTest(basis=basis):
                report = run_llm_field_extraction(
                    lambda request: {**extracted_response(request), **basis}, self.source,
                    output_name="duration", required_fields=("exposure_time_s",),
                    instruction="Extract duration", require_admission_basis=True,
                )
                self.assertFalse(report.passed)
                self.assertTrue(report.errors)

    def test_duration_conversion_is_independently_recomputed(self):
        source = EvidenceSource(source_ref=SOURCE_URL, text="The exposure lasted 2 hours.")
        for seconds, passed in ((7200, True), (2, False)):
            with self.subTest(seconds=seconds):
                def extractor(request):
                    result = extracted_response(request)
                    result["values"][0].update(value=2, unit="hours", normalized_value=seconds, evidence_span=source.text)
                    return result
                report = admit_exposure_time_from_extraction(extract_exposure_time(extractor, source))
                self.assertEqual(report.passed, passed)
                if passed:
                    self.assertEqual(report.outputs["exposure_time_s"], 7200)
                else:
                    self.assertIn("inconsistent_normalized_value", {error.code for error in report.errors})


class WorkflowContractTests(unittest.TestCase):
    def test_variant_rescaling_is_opt_in(self):
        facts = {**FACTS, "relative_transmissibility": (3, "scalar", "An independent input factor of 3 relative to the reference population was adopted.")}
        results = []
        for required in (False, True):
            with self.subTest(required=required):
                requests = []
                _, report, _ = run_synthetic(sources={SOURCE_URL: facts}, requests=requests, require_transmissibility_evidence=required)
                self.assertTrue(report.passed)
                requested = any("relative_transmissibility" in request["required_fields"] for request in requests)
                self.assertEqual(requested, required)
                results.append(report.outputs["estimate_pathogen_infectiousness"]["quanta_concentration_exhaled_m3"])
                assert_source_backed_calibration(self, report)
        self.assertAlmostEqual(results[0], 3 * results[1])

    def test_required_scaling_does_not_fall_back_to_identity(self):
        _, report, _ = run_synthetic(require_transmissibility_evidence=True)
        self.assertFalse(report.passed)
        self.assertTrue(any("relative_transmissibility" in field for field in report.missing_fields))

    def test_susceptible_count_does_not_subtract_the_index_case_again(self):
        facts = {name: fact for name, fact in FACTS.items() if name != "attendee_count"}
        facts["susceptible_count"] = (10, "count", "There were 10 susceptible people exposed to the index case.")
        _, report, _ = run_synthetic(sources={SOURCE_URL: facts})
        self.assertTrue(report.passed)
        nested = report.outputs["estimate_pathogen_infectiousness"]["infectiousness_estimation"]
        attempt = nested["outputs"]["calibrate_cq_from_outbreak"]["calibration_attempts"][0]
        attack = attempt["report"]["child_reports"]["build_attack_probability"]["outputs"]
        self.assertEqual(attack["susceptible_count"], 10)
        self.assertEqual(attack["attack_probability"], 0.2)
        self.assertNotIn("attendee_count", attack)
        assert_source_backed_calibration(self, report)

    def test_sibling_extractions_receive_the_credited_outbreak_context(self):
        requests = []
        _, report, _ = run_synthetic(requests=requests)
        self.assertTrue(report.passed)
        siblings = [request for request in requests if "exposure_time_s" in request["required_fields"]]
        self.assertTrue(siblings)
        for request in siblings:
            context = request["target_context"]
            self.assertEqual(tuple(context["attack_source_refs"]), (SOURCE_URL,))
            self.assertEqual(context["attack_probability_inputs"]["index_case_count"], 1)
            self.assertIn("attendee_count", context["attack_probability_inputs"]["extracted_values"])

    def test_credit_weights_match_criteria(self):
        for child in CQ_CALIBRATION_CHILDREN:
            if child.name in CQ_NORMALIZATION_CRITERIA:
                self.assertEqual(child.weight, sum(criterion.weight for criterion in CQ_NORMALIZATION_CRITERIA[child.name]))

    def test_calibration_equation_and_nested_sibling_credit(self):
        _, report, queries = run_synthetic()
        self.assertTrue(report.passed, report.as_record())
        self.assertTrue(all("synthetic disease" in query and "test setting" in query for query in queries))
        infectiousness = report.outputs["estimate_pathogen_infectiousness"]
        expected_cq = -math.log1p(-2 / 10) * 0.5 / (3600 * 0.0001**2)
        self.assertAlmostEqual(infectiousness["quanta_concentration_exhaled_m3"], expected_cq)
        expected_bound = 0.1 * 0.5 / (0.0001**2 * expected_cq)
        result = report.main_outputs["compute_safety_bound"]
        self.assertAlmostEqual(result["cumulative_exposure_bound_person_s"], expected_bound)
        nested = infectiousness["infectiousness_estimation"]
        assert_credit_totals(self, report.as_record())
        assert_credit_totals(self, nested)
        for attempt in nested["outputs"]["calibrate_cq_from_outbreak"]["calibration_attempts"]:
            assert_credit_totals(self, attempt["report"])
        self.assertIn(SOURCE_URL, report.source_refs)
        self.assertTrue(report.identities)
        assert_source_backed_calibration(self, report)

    def test_runtime_disease_is_not_a_fixed_case(self):
        for disease in ("synthetic pathogen alpha", "synthetic pathogen beta", "unseen airborne pathogen"):
            with self.subTest(disease=disease):
                _, report, queries = run_synthetic(disease=disease)
                self.assertTrue(report.passed)
                self.assertTrue(all(disease in query for query in queries))

    def test_unsupported_span_produces_repairable_parent_failure(self):
        def mutate(response):
            for value in response["values"]:
                value["evidence_span"] = "Invented evidence."
        _, report, _ = run_synthetic(mutate=mutate)
        self.assertFalse(report.passed)
        self.assertFalse(report.main_outputs)
        self.assertIn("unsupported_evidence_span", {error.code for error in report.errors})
        self.assertTrue(report.failed_fields)
        self.assertTrue(report.missing_fields)
        parent_child = report.child_reports["estimate_pathogen_infectiousness"]
        nested = parent_child["outputs"]["infectiousness_estimation"]
        self.assertGreater(nested["score"], 0)
        self.assertEqual(parent_child["score"], nested["score"])
        assert_credit_totals(self, report.as_record())

    def test_values_from_unrelated_events_are_not_combined(self):
        counts = {name: fact for name, fact in FACTS.items() if name.endswith("_count")}
        other = {name: fact for name, fact in FACTS.items() if name not in counts}
        _, report, _ = run_synthetic(sources={SOURCE_URL: counts, "https://evidence.example/other-event": other})
        self.assertFalse(report.passed)
        self.assertFalse(report.main_outputs)
        self.assertTrue(report.missing_fields or report.failed_fields)
        nested = report.outputs["estimate_pathogen_infectiousness"]["infectiousness_estimation"]
        calibration = nested["child_reports"]["calibrate_cq_from_outbreak"]
        attempts = calibration["outputs"]["calibration_attempts"]
        self.assertEqual(calibration["score"], max(attempt["report"]["score"] for attempt in attempts))

    def test_invalid_target_is_repairable(self):
        _, report, _ = run_synthetic(scenario={**SCENARIO, "breathing_flow_m3_s": -1})
        self.assertFalse(report.passed)
        self.assertTrue(any(error.field and "breathing_flow_m3_s" in error.field for error in report.errors))
        self.assertNotIn("compute_safety_bound", report.main_outputs)

    def test_zero_hazard_remains_unbounded_in_every_bound_model(self):
        for model in CUMULATIVE_EXPOSURE_MODELS:
            with self.subTest(model=model):
                _, report, _ = run_synthetic(scenario={**SCENARIO, "mask_penetration": 0}, model=model)
                self.assertTrue(report.passed, report.as_record())
                self.assertTrue(any(isinstance(value, float) and math.isinf(value) for value in report.main_outputs["compute_safety_bound"].values()))

    def test_unbounded_calculation_has_stable_credit_identity(self):
        inputs = {"cumulative_exposure_bound_person_s": math.inf, "susceptible_count": 10}
        first = run_named_calculation("steady_state_time_bound", inputs)
        second = run_named_calculation("steady_state_time_bound", copy.deepcopy(inputs))
        self.assertTrue(first.passed)
        self.assertEqual(first.value, math.inf)
        self.assertEqual(first.identities, second.identities)

    def test_small_nonzero_probabilities_do_not_become_zero_hazard(self):
        calibrated = run_named_calculation("quanta_concentration_from_attack_probability", {
            "attack_probability": 1e-20,
            "removal_capacity_m3_s": 1,
            "exposure_time_s": 1,
            "breathing_flow_m3_s": 1,
        })
        self.assertTrue(calibrated.passed)
        self.assertTrue(math.isclose(calibrated.value, 1e-20, rel_tol=1e-12))
        forward = run_named_calculation("first_transmission_probability", {"indoor_reproductive_number": 1e-20})
        self.assertTrue(forward.passed)
        self.assertTrue(math.isclose(forward.value, 1e-20, rel_tol=1e-12))
