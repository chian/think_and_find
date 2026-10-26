"""Audit successful workflow reports against their retained source evidence."""

from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from collections.abc import Mapping
from statistics import median


def records(value):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from records(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from records(child)


def assert_admitted_research_provenance(test, data):
    documents = data["source_documents"]
    test.assertTrue(documents, "No retrieved source bodies retained in report")
    admitted = {name: data[name] for name in (
        "evidence_and_assumptions", "qualitative_evidence", "assumption_assessments", "scenarios")}
    for item in records(admitted):
        if not item.get("quote"):
            if item.get("kind") in {"source_observation", "source_model_parameter", "source_evidence"}:
                test.fail("Admitted source evidence has no quotation")
            continue
        source = item.get("source")
        if source == "request":
            test.assertIn(item["quote"], data["research_request"])
            continue
        test.assertIn(source, documents)
        version = item.get("source_version")
        test.assertTrue(version, "Admitted quotation has no source version")
        document = documents[source]
        versions = document.get("versions", {document["sha256_utf8"]: document})
        test.assertIn(version, versions)
        original = versions[version]["text"]
        test.assertEqual(version, hashlib.sha256(original.encode()).hexdigest())
        test.assertIn(item["quote"], original)


def assert_source_backed_calibration(test, report, *, expect_native_scaling=False):
    record = report.as_record()
    test.assertTrue(report.passed)
    sources = defaultdict(set)
    for item in records(record):
        if isinstance(item.get("source_ref"), str) and isinstance(item.get("text"), str):
            sources[item["source_ref"]].add(item["text"])
    test.assertTrue(sources, "No retrieved source bodies retained in report")
    infectiousness = record["outputs"]["estimate_pathogen_infectiousness"]
    estimation = infectiousness["infectiousness_estimation"]
    calibration = estimation["outputs"]["calibrate_cq_from_outbreak"]
    attempts = calibration["calibration_attempts"]
    successful = [attempt for attempt in attempts if attempt["passed"]]
    test.assertTrue(successful, "No searched outbreak was calibrated")
    test.assertEqual(calibration["calibration_summary"]["passed"], len(successful))
    concentrations = []
    for attempt in successful:
        children = attempt["report"]["child_reports"]
        test.assertTrue(all(child["passed"] for child in children.values()))
        evidence_count = 0
        for child in children.values():
            review = child["outputs"].get("evidence_applicability")
            if review:
                test.assertTrue(all(review[name] is True for name in ("applicable", "scale_compatible", "mechanism_compatible")))
                test.assertIn(review["evidence_role"], {"event_observation", "event_model_parameter", "comparable_observation"})
                for ref_key, span_key in (("source_ref", "source_evidence_span"), ("target_ref", "target_evidence_span")):
                    test.assertIn(review[ref_key], sources)
                    test.assertTrue(review[span_key])
                    test.assertTrue(any(review[span_key] in text for text in sources[review[ref_key]]))
            for value in child["outputs"].get("extracted_values", {}).values():
                ref = value["source_ref"]
                test.assertIn(ref, sources)
                test.assertIn(ref, report.source_refs)
                test.assertTrue(value["evidence_span"])
                test.assertTrue(any(value["evidence_span"] in text for text in sources[ref]))
                evidence_count += 1
        test.assertGreaterEqual(evidence_count, 6)
        attack = children["build_attack_probability"]["outputs"]
        removal_outputs = children["build_removal_capacity"]["outputs"]
        test.assertTrue(removal_outputs.get("evidence_applicability"), "Removal evidence has no applicability review")
        removal = removal_outputs["removal_capacity_m3_s"]
        if removal_outputs["evidence_basis"] == "reported_total_removal_rate_times_volume":
            quantities = removal_outputs["extracted_values"]
            test.assertAlmostEqual(removal, quantities["room_volume_m3"]["normalized_value"] * quantities["total_removal_rate_h"]["normalized_value"] / 3600)
        duration = children["admit_exposure_time"]["outputs"]["exposure_time_s"]
        behavior = children["admit_behavior_assumptions"]["outputs"]
        scaling = children["admit_population_or_variant_scaling"]["outputs"]["relative_transmissibility"]
        if expect_native_scaling:
            test.assertEqual(scaling, 1.0, "Unrequested variant/population rescaling")
        expected = -math.log1p(-attack["attack_probability"]) * removal / (
            attack["index_case_count"] * duration * behavior["breathing_flow_m3_s"]**2
            * behavior["mask_penetration"]**2 * scaling
        )
        actual = attempt["quanta_concentration_exhaled_m3"]
        test.assertTrue(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12))
        concentrations.append(actual)
    test.assertAlmostEqual(infectiousness["quanta_concentration_exhaled_m3"], median(concentrations))
    for parent in (record, estimation, *(attempt["report"] for attempt in successful)):
        test.assertAlmostEqual(parent["score"], sum(child["score"] for child in parent["child_reports"].values()))
        test.assertAlmostEqual(parent["max_score"], sum(child["max_score"] for child in parent["child_reports"].values()))
