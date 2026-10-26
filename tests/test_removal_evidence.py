"""Synthetic regression cases for removal reconstruction and applicability."""

from __future__ import annotations

import unittest

from episodic_workflow.case_reports.applicability import review_applicability
from episodic_workflow.case_reports.evidence_children import build_removal_capacity_from_extraction
from episodic_workflow.case_reports.evidence_extraction import (
    EvidenceSource,
    extract_equivalent_removal_capacity,
    extract_removal_capacity,
)
from episodic_workflow.tests.fixtures import FACTS, SOURCE_URL, applicability_response
from episodic_workflow.tests.report_assertions import assert_source_backed_calibration
from episodic_workflow.tests.test_evidence_contracts import extracted_response, run_synthetic


TOTAL_FACTS = {
    "room_volume_m3": (600, "m3", "The outbreak room volume was 600 m3."),
    "total_removal_rate_h": (3, "1/h", "The event model adopted a combined pathogen loss rate of 3 per hour, including ventilation, deposition and deactivation."),
}


def source_for(facts):
    return EvidenceSource(source_ref=SOURCE_URL, text=" ".join(fact[2] for fact in facts.values()))


class RemovalEvidenceTests(unittest.TestCase):
    def test_total_loss_rate_is_converted_and_multiplied_once(self):
        requests = []

        def extract(request):
            requests.append(request)
            return extracted_response(request, TOTAL_FACTS)

        extraction = extract_removal_capacity(extract, source_for(TOTAL_FACTS))
        report = build_removal_capacity_from_extraction(extraction)
        self.assertTrue(report.passed, report.as_record())
        self.assertEqual(requests[0]["required_fields"], ["removal_capacity_m3_s"])
        self.assertEqual(requests[1]["required_fields"], list(TOTAL_FACTS))
        self.assertEqual(requests[1]["normalized_units"]["total_removal_rate_h"], "1/h")
        self.assertEqual(report.outputs["removal_capacity_m3_s"], 0.5)
        self.assertEqual(report.outputs["normalized_room_fields"], {"room_volume_m3": 600, "total_removal_rate_s": 3 / 3600})
        self.assertEqual(report.outputs["evidence_basis"], "reported_total_removal_rate_times_volume")
        self.assertTrue(report.outputs["evidence_applicability"]["applicable"])
        self.assertEqual(set(report.outputs["extracted_values"]), set(TOTAL_FACTS))

    def test_total_loss_normalization_is_checked_independently(self):
        for field in TOTAL_FACTS:
            with self.subTest(field=field):
                def extract(request):
                    response = extracted_response(request, TOTAL_FACTS)
                    for value in response.get("values", ()):
                        if value["field"] == field:
                            value["normalized_value"] *= 2
                    return response

                report = build_removal_capacity_from_extraction(extract_removal_capacity(extract, source_for(TOTAL_FACTS)))
                self.assertFalse(report.passed)
                self.assertIn("inconsistent_normalized_value", {error.code for error in report.errors})

    def test_total_rate_path_reaches_the_final_bound(self):
        facts = {name: fact for name, fact in FACTS.items() if name != "removal_capacity_m3_s"}
        facts.update(TOTAL_FACTS)
        _, report, _ = run_synthetic(sources={SOURCE_URL: facts})
        self.assertTrue(report.passed, [error.as_record() for error in report.errors])
        assert_source_backed_calibration(self, report)

    def test_additional_flow_units_are_deterministic(self):
        for unit, raw, normalized in (("cfm", 100, 0.04719474432), ("ft3/min", 100, 0.04719474432), ("m3/min", 30, 0.5)):
            for wrong_value in (False, True):
                with self.subTest(unit=unit, wrong_value=wrong_value):
                    facts = {"removal_capacity_m3_s": (raw, unit, f"The measured whole-room equivalent clean-air capacity was {raw} {unit}.")}

                    def extract(request):
                        response = extracted_response(request, facts)
                        for value in response.get("values", ()):
                            value.update(normalized_unit="m3/s", normalized_value=normalized * (2 if wrong_value else 1))
                        return response

                    report = build_removal_capacity_from_extraction(extract_equivalent_removal_capacity(extract, source_for(facts)))
                    self.assertEqual(report.passed, not wrong_value, report.as_record())
                    if report.passed:
                        self.assertAlmostEqual(report.outputs["removal_capacity_m3_s"], normalized)

    def test_failed_applicability_is_retained_and_cannot_fall_back(self):
        requests = []

        def extract(request):
            requests.append(request)
            if request.get("operation") == "review_evidence_applicability":
                return applicability_response(request, evidence_role="recommendation", rationale="This is a recommendation, not the outbreak room's capacity.")
            return extracted_response(request)

        extraction = extract_removal_capacity(extract, source_for(FACTS))
        report = build_removal_capacity_from_extraction(extraction)
        self.assertFalse(report.passed)
        self.assertEqual(len(requests), 2)
        self.assertEqual(extraction.as_record()["applicability"]["evidence_role"], "recommendation")
        self.assertIn("inapplicable_evidence", {error.code for error in report.errors})


class ApplicabilityTests(unittest.TestCase):
    def review(self, *, changes=None, context=None, comparable=False, reviewer=None):
        return review_applicability(
            reviewer or (lambda request: applicability_response(request, **(changes or {}))),
            source=source_for(FACTS).as_record(), values={}, target_context=context or {},
            allow_comparable=comparable, removal_evidence=True,
        )

    def test_typed_compatibility_and_roles_are_required(self):
        cases = (
            ({"applicable": "true"}, "invalid_applicability_review"),
            ({"scale_compatible": False}, "inapplicable_evidence"),
            ({"mechanism_compatible": False}, "inapplicable_evidence"),
            ({"evidence_role": "recommendation"}, "inapplicable_evidence"),
            ({"evidence_role": "hypothetical"}, "inapplicable_evidence"),
            ({"evidence_role": "device_specification"}, "inapplicable_evidence"),
            ({"evidence_role": "comparable_observation"}, "inapplicable_evidence"),
            ({"rationale": ""}, "invalid_applicability_review"),
        )
        for changes, code in cases:
            with self.subTest(changes=changes):
                _, errors = self.review(changes=changes)
                self.assertIn(code, {error.code for error in errors})

    def test_source_and_target_refs_and_spans_must_be_exact(self):
        for changes, code in (
            ({"source_ref": "unretrieved"}, "source_ref_mismatch"),
            ({"source_evidence_span": "Fabricated span"}, "unsupported_evidence_span"),
            ({"target_ref": "unretrieved"}, "source_ref_mismatch"),
            ({"target_evidence_span": "Fabricated span"}, "unsupported_evidence_span"),
        ):
            with self.subTest(changes=changes):
                _, errors = self.review(changes=changes)
                self.assertIn(code, {error.code for error in errors})

    def test_comparable_source_requires_a_distinct_target(self):
        _, errors = self.review(comparable=True)
        self.assertEqual(errors[0].code, "missing_target_evidence")
        context = {"outbreak_sources": [{"source_ref": "target", "text": "The target room held the same activity."}]}
        result, errors = self.review(comparable=True, context=context, changes={"evidence_role": "comparable_observation"})
        self.assertFalse(errors)
        self.assertEqual(result["target_ref"], "target")
        _, errors = self.review(comparable=True, context=context, changes={"target_ref": SOURCE_URL})
        self.assertEqual(errors[0].code, "source_ref_mismatch")

    def test_reviewer_failure_is_structured(self):
        def fail(request):
            raise RuntimeError("Synthetic failure")

        _, errors = self.review(reviewer=fail)
        self.assertEqual(errors[0].code, "applicability_review_error")

    def test_nonfinite_target_is_a_structured_failure(self):
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                _, errors = self.review(context={"target_scenario": {"volume": value}})
                self.assertEqual(errors[0].code, "invalid_target_evidence")


if __name__ == "__main__":
    unittest.main()
