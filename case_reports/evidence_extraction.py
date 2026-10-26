"""LLM-backed extraction envelopes for outbreak evidence fields."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from numbers import Real
from typing import Any, Callable, Mapping, Sequence

from episodic_workflow.credit_assignment.simple import CalculationError
from .applicability import review_applicability


LLMExtractCallable = Callable[[Mapping[str, Any]], Mapping[str, Any] | str]


POPULATION_COUNT_FIELDS = (
    "attendee_count",
    "index_case_count",
    "secondary_case_count",
)
POPULATION_ATTACK_RATE_FIELDS = (
    "attendee_count",
    "index_case_count",
    "attack_probability",
)
SUSCEPTIBLE_COUNT_FIELDS = (
    "susceptible_count",
    "index_case_count",
    "secondary_case_count",
)
SUSCEPTIBLE_ATTACK_RATE_FIELDS = (
    "susceptible_count",
    "index_case_count",
    "attack_probability",
)

EXPOSURE_TIME_FIELDS = ("exposure_time_s",)
REMOVAL_CAPACITY_FIELDS = ("removal_capacity_m3_s",)
TOTAL_REMOVAL_FIELDS = ("room_volume_m3", "total_removal_rate_h")
ROOM_REMOVAL_FIELDS = (
    "floor_area_m2",
    "room_height_m",
    "air_exchange_rate_h",
    "recirculation_flow_m3_s",
    "filter_efficiency",
    "deactivation_rate_h",
    "settling_speed_m_h",
)
VOLUME_REMOVAL_FIELDS = (
    "floor_area_m2",
    "room_volume_m3",
    "air_exchange_rate_h",
    "recirculation_flow_m3_s",
    "filter_efficiency",
    "deactivation_rate_h",
    "settling_speed_m_h",
)
ROOM_VENTILATION_FLOW_REMOVAL_FIELDS = (
    "floor_area_m2",
    "room_height_m",
    "ventilation_flow_m3_s",
    "recirculation_flow_m3_s",
    "filter_efficiency",
    "deactivation_rate_h",
    "settling_speed_m_h",
)
VOLUME_VENTILATION_FLOW_REMOVAL_FIELDS = (
    "floor_area_m2",
    "room_volume_m3",
    "ventilation_flow_m3_s",
    "recirculation_flow_m3_s",
    "filter_efficiency",
    "deactivation_rate_h",
    "settling_speed_m_h",
)
BEHAVIOR_FIELDS = (
    "breathing_flow_m3_s",
    "mask_penetration",
)
BEHAVIOR_FILTER_FIELDS = (
    "breathing_flow_m3_s",
    "mask_filter_efficiency",
)
BEHAVIOR_UNMASKED_FIELDS = (
    "breathing_flow_m3_s",
    "unmasked_indicator",
)
TRANSMISSIBILITY_FIELDS = ("relative_transmissibility",)
REMOVAL_EVIDENCE_FIELDS = frozenset(
    REMOVAL_CAPACITY_FIELDS + TOTAL_REMOVAL_FIELDS + ROOM_REMOVAL_FIELDS
    + VOLUME_REMOVAL_FIELDS + ROOM_VENTILATION_FLOW_REMOVAL_FIELDS
    + VOLUME_VENTILATION_FLOW_REMOVAL_FIELDS
)
MIN_EXTRACTED_VALUE_CONFIDENCE = 0.5

NORMALIZED_UNITS = {
    "attendee_count": "count",
    "index_case_count": "count",
    "secondary_case_count": "count",
    "susceptible_count": "count",
    "attack_probability": "fraction",
    "exposure_time_s": "s",
    "removal_capacity_m3_s": "m3/s",
    "floor_area_m2": "m2",
    "room_height_m": "m",
    "room_volume_m3": "m3",
    "air_exchange_rate_h": "1/h",
    "total_removal_rate_h": "1/h",
    "ventilation_flow_m3_s": "m3/s",
    "recirculation_flow_m3_s": "m3/s",
    "filter_efficiency": "fraction",
    "deactivation_rate_h": "1/h",
    "settling_speed_m_h": "m/h",
    "breathing_flow_m3_s": "m3/s",
    "mask_penetration": "fraction",
    "mask_filter_efficiency": "fraction",
    "unmasked_indicator": "scalar",
    "relative_transmissibility": "scalar",
}


@dataclass(frozen=True)
class EvidenceSource:
    """One retrieved passage or document that may support extracted fields."""

    source_ref: str
    text: str
    disease_name: str = ""
    title: str = ""
    url: str = ""
    doi: str = ""
    event_ref: str = ""
    year: str = ""
    query: str = ""
    purpose: str = ""
    queries: tuple[str, ...] = ()
    purposes: tuple[str, ...] = ()
    retrieval: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_ref:
            raise ValueError("source_ref must be populated")
        if not self.text:
            raise ValueError("text must be populated")

    def as_record(self) -> dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "disease_name": self.disease_name,
            "title": self.title,
            "url": self.url,
            "doi": self.doi,
            "event_ref": self.event_ref,
            "year": self.year,
            "query": self.query,
            "purpose": self.purpose,
            "queries": list(self.queries),
            "purposes": list(self.purposes),
            "retrieval": dict(self.retrieval),
            "text": self.text,
        }


@dataclass(frozen=True)
class ExtractedValue:
    """One value extracted by an LLM from a retrieved outbreak source."""

    field: str
    value: float
    unit: str
    normalized_value: float
    normalized_unit: str
    source_ref: str
    evidence_span: str
    confidence: float
    notes: tuple[str, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "unit": self.unit,
            "normalized_value": self.normalized_value,
            "normalized_unit": self.normalized_unit,
            "source_ref": self.source_ref,
            "evidence_span": self.evidence_span,
            "confidence": self.confidence,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ExtractionReport:
    """Parent-facing result from one LLM extraction child."""

    output_name: str
    passed: bool
    score: float
    max_score: float
    values: Mapping[str, ExtractedValue] = field(default_factory=dict)
    admission_basis: Mapping[str, str] = field(default_factory=dict)
    errors: tuple[CalculationError, ...] = ()
    stop: bool = True
    applicability: Mapping[str, Any] = field(default_factory=dict)
    span_repair_history: tuple[Mapping[str, Any], ...] = ()

    @property
    def normalized_score(self) -> float:
        if self.max_score == 0.0:
            return 0.0
        return self.score / self.max_score

    def as_record(self) -> dict[str, Any]:
        return {
            "output_name": self.output_name,
            "passed": self.passed,
            "score": self.score,
            "max_score": self.max_score,
            "normalized_score": self.normalized_score,
            "values": {
                name: value.as_record()
                for name, value in self.values.items()
            },
            "admission_basis": dict(self.admission_basis),
            "applicability": dict(self.applicability),
            "span_repair_history": list(self.span_repair_history),
            "errors": [error.as_record() for error in self.errors],
            "stop": self.stop,
        }


def extract_population_counts(
    llm: LLMExtractCallable,
    source: EvidenceSource,
) -> ExtractionReport:
    """Extract point-source outbreak population counts with an LLM."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="population_counts",
        required_fields=POPULATION_COUNT_FIELDS,
        instruction=(
            "Extract the outbreak attendee_count, index_case_count, and "
            "secondary_case_count from the source. Return only numeric counts "
            "explicitly supported by the passage. attendee_count must include "
            "the index case(s); a count of people exposed to the index case "
            "instead belongs to susceptible_count and must not be returned "
            "as attendee_count."
        ),
    )


def extract_population_attack_probability(
    llm: LLMExtractCallable,
    source: EvidenceSource,
) -> ExtractionReport:
    """Extract a reported per-susceptible attack probability."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="population_attack_probability",
        required_fields=POPULATION_ATTACK_RATE_FIELDS,
        instruction=(
            "Extract attendee_count, index_case_count, and an explicitly "
            "reported secondary attack probability for the outbreak. Return "
            "attack_probability normalized to a fraction of susceptible "
            "exposed people, excluding index cases."
        ),
    )


def extract_susceptible_counts(
    llm: LLMExtractCallable,
    source: EvidenceSource,
) -> ExtractionReport:
    """Extract directly reported susceptible and secondary counts."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="susceptible_counts",
        required_fields=SUSCEPTIBLE_COUNT_FIELDS,
        instruction=(
            "Extract the outbreak susceptible_count, index_case_count, and "
            "secondary_case_count from the source. Use susceptible_count only "
            "when the passage directly reports the exposed people who were "
            "susceptible after excluding index cases."
        ),
    )


def extract_susceptible_attack_probability(
    llm: LLMExtractCallable,
    source: EvidenceSource,
) -> ExtractionReport:
    """Extract susceptible count with a reported attack probability."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="susceptible_attack_probability",
        required_fields=SUSCEPTIBLE_ATTACK_RATE_FIELDS,
        instruction=(
            "Extract susceptible_count, index_case_count, and an explicitly "
            "reported secondary attack probability for the outbreak. Return "
            "susceptible_count only when it directly excludes index cases, and "
            "return attack_probability normalized to a fraction of susceptible "
            "exposed people."
        ),
    )


def extract_attack_probability_inputs(
    llm: LLMExtractCallable,
    source: EvidenceSource,
) -> ExtractionReport:
    """Extract counts or a reported attack rate for attack calibration."""

    reports: list[ExtractionReport] = []
    for extractor in (
        extract_population_counts,
        extract_population_attack_probability,
        extract_susceptible_counts,
        extract_susceptible_attack_probability,
    ):
        report = extractor(llm, source)
        if report.passed:
            return report
        reports.append(report)
        if not _alternate_extraction_allowed(
            report,
            missing_fields=(
                "attendee_count",
                "secondary_case_count",
                "attack_probability",
                "susceptible_count",
            ),
        ):
            return report

    last_report = reports[-1]
    return ExtractionReport(
        output_name=last_report.output_name,
        passed=False,
        score=max(report.score for report in reports),
        max_score=last_report.max_score,
        values=last_report.values,
        errors=tuple(
            error
            for report in reports
            for error in report.errors
        ),
    )


def extract_exposure_time(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract the outbreak exposure duration with an LLM."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="exposure_time",
        required_fields=EXPOSURE_TIME_FIELDS,
        instruction=(
            "Extract the exposure duration used for the outbreak "
            "calibration. Return exposure_time_s normalized to seconds and "
            "supported by the passage."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def extract_equivalent_removal_capacity(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract equivalent clean-air removal capacity with an LLM."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="removal_capacity",
        required_fields=REMOVAL_CAPACITY_FIELDS,
        instruction=(
            "Extract an equivalent virus-free-air or clean-air delivery "
            "capacity for the outbreak. Return removal_capacity_m3_s "
            "normalized to cubic meters per second and supported by the "
            "passage. Extract only a reported whole-room equivalent capacity, "
            "not a proposed upgrade, a device rating, a per-person flow, or "
            "a recommended flow per unit area. Do not compute capacity from "
            "other fields; those have a separate deterministic path."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def extract_removal_capacity(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract equivalent capacity or the fields needed to compute it."""

    equivalent = extract_equivalent_removal_capacity(
        llm,
        source,
        admit_assumption=admit_assumption,
        target_context=target_context,
    )
    if equivalent.passed:
        return equivalent
    if not _alternate_extraction_allowed(
        equivalent,
        missing_fields=("removal_capacity_m3_s",),
    ):
        return equivalent

    total = extract_total_removal_capacity(
        llm, source, admit_assumption=admit_assumption, target_context=target_context,
    )
    if total.passed:
        return total
    if not _alternate_extraction_allowed(total, missing_fields=TOTAL_REMOVAL_FIELDS):
        return total
    equivalent = replace(equivalent, errors=equivalent.errors + total.errors)

    room_fields = extract_room_removal_capacity(
        llm,
        source,
        admit_assumption=admit_assumption,
        target_context=target_context,
    )
    if room_fields.passed:
        return room_fields
    if not _alternate_extraction_allowed(
        room_fields,
        missing_fields=("room_height_m", "air_exchange_rate_h"),
    ):
        return ExtractionReport(
            output_name=room_fields.output_name,
            passed=False,
            score=room_fields.score,
            max_score=room_fields.max_score,
            values=room_fields.values,
            admission_basis=room_fields.admission_basis,
            applicability=room_fields.applicability,
            errors=equivalent.errors + room_fields.errors,
        )

    volume_fields = extract_volume_removal_capacity(
        llm,
        source,
        admit_assumption=admit_assumption,
        target_context=target_context,
    )
    if volume_fields.passed:
        return volume_fields

    if not _alternate_failures_allowed(
        (room_fields, volume_fields),
        missing_fields=(
            "room_height_m",
            "room_volume_m3",
            "air_exchange_rate_h",
        ),
    ):
        return ExtractionReport(
            output_name=volume_fields.output_name,
            passed=False,
            score=max(room_fields.score, volume_fields.score),
            max_score=volume_fields.max_score,
            values=volume_fields.values,
            admission_basis=volume_fields.admission_basis,
            applicability=volume_fields.applicability,
            errors=equivalent.errors + room_fields.errors + volume_fields.errors,
        )

    room_flow_fields = extract_room_ventilation_flow_removal_capacity(
        llm,
        source,
        admit_assumption=admit_assumption,
        target_context=target_context,
    )
    if room_flow_fields.passed:
        return room_flow_fields
    if not _alternate_extraction_allowed(
        room_flow_fields,
        missing_fields=("room_height_m",),
    ):
        return ExtractionReport(
            output_name=room_flow_fields.output_name,
            passed=False,
            score=max(
                room_fields.score,
                volume_fields.score,
                room_flow_fields.score,
            ),
            max_score=room_flow_fields.max_score,
            values=room_flow_fields.values,
            admission_basis=room_flow_fields.admission_basis,
            applicability=room_flow_fields.applicability,
            errors=(
                equivalent.errors
                + room_fields.errors
                + volume_fields.errors
                + room_flow_fields.errors
            ),
        )

    volume_flow_fields = extract_volume_ventilation_flow_removal_capacity(
        llm,
        source,
        admit_assumption=admit_assumption,
        target_context=target_context,
    )
    if volume_flow_fields.passed:
        return volume_flow_fields

    return ExtractionReport(
        output_name=volume_flow_fields.output_name,
        passed=False,
        score=max(
            room_fields.score,
            volume_fields.score,
            room_flow_fields.score,
            volume_flow_fields.score,
        ),
        max_score=volume_flow_fields.max_score,
        values=volume_flow_fields.values,
        admission_basis=volume_flow_fields.admission_basis,
        applicability=volume_flow_fields.applicability,
        errors=(
            equivalent.errors
            + room_fields.errors
            + volume_fields.errors
            + room_flow_fields.errors
            + volume_flow_fields.errors
        ),
    )


def extract_total_removal_capacity(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract room volume and a reported complete first-order loss rate."""

    return run_llm_field_extraction(
        llm, source,
        output_name="total_removal_capacity_fields",
        required_fields=TOTAL_REMOVAL_FIELDS,
        instruction=(
            "Extract the outbreak room_volume_m3 and total_removal_rate_h. "
            "The rate must explicitly include the modeled combined aerosol "
            "losses, not merely outdoor air changes or a proposed intervention. "
            "Use a reported point estimate or an explicitly adopted event-model "
            "parameter, never an arbitrarily chosen range endpoint or midpoint. "
            "Omit ambiguous values. Normalize volume to m3 and the rate to 1/h."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def extract_room_removal_capacity(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract floor-area and height removal-capacity primitives."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="room_removal_capacity_fields",
        required_fields=ROOM_REMOVAL_FIELDS,
        instruction=(
            "Extract floor_area_m2, room_height_m, air_exchange_rate_h, "
            "recirculation_flow_m3_s, filter_efficiency, "
            "deactivation_rate_h, and settling_speed_m_h for the room. "
            "Return floor area, height, and recirculated flow in SI units, "
            "rates or speeds per hour, and filter efficiency as a fraction. "
            "Use zero only when the passage explicitly supports no "
            "recirculated filtration. Every field must be explicitly "
            "supported by the passage."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def extract_volume_removal_capacity(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract direct room-volume removal-capacity primitives."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="volume_removal_capacity_fields",
        required_fields=VOLUME_REMOVAL_FIELDS,
        instruction=(
            "Extract floor_area_m2, room_volume_m3, air_exchange_rate_h, "
            "recirculation_flow_m3_s, filter_efficiency, "
            "deactivation_rate_h, and settling_speed_m_h for the room. "
            "Return floor area, room volume, and recirculated flow in SI "
            "units, rates or speeds per hour, and filter efficiency as a "
            "fraction. Use zero only when the passage explicitly supports no "
            "recirculated filtration. Every field must be explicitly "
            "supported by the passage."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def extract_room_ventilation_flow_removal_capacity(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract height-based room fields with outdoor-air flow."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="room_ventilation_flow_removal_capacity_fields",
        required_fields=ROOM_VENTILATION_FLOW_REMOVAL_FIELDS,
        instruction=(
            "Extract floor_area_m2, room_height_m, ventilation_flow_m3_s, "
            "recirculation_flow_m3_s, filter_efficiency, "
            "deactivation_rate_h, and settling_speed_m_h for the room. "
            "Return floor area, height, outdoor ventilation flow, and "
            "recirculated flow in SI units, deactivation and settling rates "
            "per hour, and filter efficiency as a fraction. Use zero only "
            "when the passage explicitly supports no recirculated "
            "filtration. Every field must be explicitly supported by the "
            "passage."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def extract_volume_ventilation_flow_removal_capacity(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract volume-based room fields with outdoor-air flow."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="volume_ventilation_flow_removal_capacity_fields",
        required_fields=VOLUME_VENTILATION_FLOW_REMOVAL_FIELDS,
        instruction=(
            "Extract floor_area_m2, room_volume_m3, ventilation_flow_m3_s, "
            "recirculation_flow_m3_s, filter_efficiency, "
            "deactivation_rate_h, and settling_speed_m_h for the room. "
            "Return floor area, room volume, outdoor ventilation flow, and "
            "recirculated flow in SI units, deactivation and settling rates "
            "per hour, and filter efficiency as a fraction. Use zero only "
            "when the passage explicitly supports no recirculated "
            "filtration. Every field must be explicitly supported by the "
            "passage."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def extract_behavior_assumptions(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract breathing and direct or inverted masking assumptions."""

    penetration = extract_behavior_penetration(
        llm,
        source,
        admit_assumption=admit_assumption,
        target_context=target_context,
    )
    if penetration.passed:
        return penetration
    if not _alternate_extraction_allowed(
        penetration,
        missing_fields=("mask_penetration",),
    ):
        return penetration

    filter_efficiency = extract_behavior_filter_efficiency(
        llm,
        source,
        admit_assumption=admit_assumption,
        target_context=target_context,
    )
    if filter_efficiency.passed:
        return filter_efficiency
    if not _alternate_extraction_allowed(
        filter_efficiency,
        missing_fields=("mask_filter_efficiency",),
    ):
        return ExtractionReport(
            output_name=filter_efficiency.output_name,
            passed=False,
            score=max(penetration.score, filter_efficiency.score),
            max_score=filter_efficiency.max_score,
            values=filter_efficiency.values,
            admission_basis=filter_efficiency.admission_basis,
            applicability=filter_efficiency.applicability,
            errors=penetration.errors + filter_efficiency.errors,
        )

    unmasked = extract_behavior_unmasked(
        llm,
        source,
        admit_assumption=admit_assumption,
        target_context=target_context,
    )
    if unmasked.passed:
        return unmasked

    return ExtractionReport(
        output_name=unmasked.output_name,
        passed=False,
        score=max(penetration.score, filter_efficiency.score, unmasked.score),
        max_score=unmasked.max_score,
        values=unmasked.values,
        admission_basis=unmasked.admission_basis,
        applicability=unmasked.applicability,
        errors=penetration.errors + filter_efficiency.errors + unmasked.errors,
    )


def extract_behavior_penetration(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract breathing flow and direct mask penetration with an LLM."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="behavior_assumptions",
        required_fields=BEHAVIOR_FIELDS,
        instruction=(
            "Extract breathing_flow_m3_s and mask_penetration for the "
            "outbreak calibration. Return normalized SI values supported by "
            "the passage."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def extract_behavior_filter_efficiency(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract breathing flow and mask filtration efficiency with an LLM."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="behavior_filter_assumptions",
        required_fields=BEHAVIOR_FILTER_FIELDS,
        instruction=(
            "Extract breathing_flow_m3_s and mask_filter_efficiency for the "
            "outbreak calibration. Return breathing_flow_m3_s in SI units and "
            "mask_filter_efficiency as the fraction of particles blocked by "
            "the mask. Every field must be explicitly supported by the "
            "passage."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def extract_behavior_unmasked(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract breathing flow and explicit unmasked-mask use."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="behavior_unmasked_assumptions",
        required_fields=BEHAVIOR_UNMASKED_FIELDS,
        instruction=(
            "Extract breathing_flow_m3_s and unmasked_indicator for the "
            "outbreak calibration. Return unmasked_indicator as scalar 1 "
            "only when the passage explicitly says masks or face coverings "
            "were not used, not worn, or absent. Every field must be "
            "explicitly supported by the passage."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def extract_transmissibility_assumption(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Extract population, variant, or disease transmissibility scaling."""

    return run_llm_field_extraction(
        llm,
        source,
        output_name="relative_transmissibility",
        required_fields=TRANSMISSIBILITY_FIELDS,
        instruction=(
            "Extract the scalar relative_transmissibility used to map this "
            "outbreak to the target Cq calibration. Return a positive "
            "normalized scalar supported by the passage. Require an "
            "independently specified INPUT scaling factor with an explicit "
            "reference population or variant, not a comparison of fitted "
            "quanta generation rates or attack-rate outputs from these "
            "outbreaks. Reusing a fitted quanta ratio here would be circular. "
            "Do not manufacture a scaling factor from a percentage increase "
            "or assume a reference variant not requested in target_context. "
            "Omit this field if the source does not supply an applicable "
            "independent scalar input."
        ),
        require_admission_basis=admit_assumption,
        target_context=target_context,
    )


def run_llm_field_extraction(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    output_name: str,
    required_fields: tuple[str, ...],
    instruction: str,
    require_admission_basis: bool = False,
    target_context: Mapping[str, Any] | None = None,
) -> ExtractionReport:
    """Call an injected LLM extractor and validate field-level outputs."""

    if llm is None:
        return _failed(
            output_name,
            required_fields,
            (
                CalculationError(
                    code="missing_llm",
                    field="llm",
                    message="an LLM extraction callable is required",
                ),
            ),
        )

    request = {
        "output_name": output_name,
        "required_fields": list(required_fields),
        "normalized_units": {
            name: NORMALIZED_UNITS[name]
            for name in required_fields if name in NORMALIZED_UNITS
        },
        "instruction": instruction,
        "confidence_threshold": MIN_EXTRACTED_VALUE_CONFIDENCE,
        "source": source.as_record(),
        "target_context": dict(target_context or {}),
        "assumption_admission": {
            "required": require_admission_basis,
            "required_fields": [
                "similarity_basis",
                "similarity_evidence_span",
            ],
        },
    }
    span_repair_history = ()
    for attempt in range(2):
        try:
            response = _coerce_llm_response(llm(request))
        except Exception as exc:
            return replace(_failed(
                output_name, required_fields,
                (CalculationError(code="llm_error", field="llm", message=str(exc)),),
            ), span_repair_history=span_repair_history)

        values, errors = _parse_extracted_values(
            response.get("values"), required_fields=required_fields,
            expected_source_ref=source.source_ref, source_text=source.text,
        )
        admission_basis, admission_errors = _parse_admission_basis(
            response, source_text=source.text, required=require_admission_basis,
        )
        errors += admission_errors
        if attempt or not any(error.code == "unsupported_evidence_span" for error in errors) or any(
            error.code not in {"unsupported_evidence_span", "missing_extracted_field"}
            for error in errors
        ):
            break
        # Re-extract once from the same source; never normalize or fuzzy-match a citation.
        diagnostic = {"rejected_response": dict(response), "errors": [error.as_record() for error in errors]}
        span_repair_history = (diagnostic,)
        request = {**request, "span_correction": {
            **diagnostic,
            "instruction": "Re-extract using exact contiguous source substrings, preserving characters and whitespace. Do not omit the numeric support to obtain a match. Omit unsupported values. All original validation still applies.",
        }}
    applicability: dict[str, Any] = {}
    removal_evidence = bool(REMOVAL_EVIDENCE_FIELDS.intersection(required_fields))
    if not errors and (require_admission_basis or removal_evidence or "relative_transmissibility" in required_fields):
        applicability, review_errors = review_applicability(
            llm,
            source=source.as_record(),
            values={name: value.as_record() for name, value in values.items()},
            target_context=dict(target_context or {}),
            allow_comparable=require_admission_basis,
            removal_evidence=removal_evidence,
        )
        errors += review_errors
    score = float(len(values))
    return ExtractionReport(
        output_name=output_name,
        passed=not errors and len(values) == len(required_fields),
        score=score,
        max_score=float(len(required_fields)),
        values=values,
        admission_basis=admission_basis,
        applicability=applicability,
        span_repair_history=span_repair_history,
        errors=errors,
    )


def evidence_source_from_candidate(
    candidate: Mapping[str, Any],
) -> EvidenceSource:
    """Build an extraction source from a searched candidate with text."""

    query = str(candidate.get("query") or "")
    purpose = str(candidate.get("purpose") or "")
    queries = _string_tuple(candidate.get("queries"))
    purposes = _string_tuple(candidate.get("purposes"))
    return EvidenceSource(
        source_ref=str(candidate.get("source_ref") or ""),
        text=str(
            candidate.get("text")
            or candidate.get("passage")
            or candidate.get("markdown")
            or candidate.get("content")
            or ""
        ),
        title=str(candidate.get("title") or ""),
        disease_name=str(candidate.get("disease_name") or ""),
        url=str(candidate.get("url") or ""),
        doi=str(candidate.get("doi") or ""),
        event_ref=str(candidate.get("event_ref") or ""),
        year=str(candidate.get("year") or ""),
        query=query,
        purpose=purpose,
        queries=queries or ((query,) if query else ()),
        purposes=purposes or ((purpose,) if purpose else ()),
        retrieval=_mapping(candidate.get("retrieval")),
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value if str(item))
    return (str(value),)


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    try:
        return dict(value)
    except Exception:
        return {}


def _alternate_extraction_allowed(
    report: ExtractionReport,
    *,
    missing_fields: tuple[str, ...],
) -> bool:
    """True when another extractor can repair only absent fields."""

    allowed = set(missing_fields)
    return bool(report.errors) and all(
        error.code == "missing_extracted_field" and error.field in allowed
        for error in report.errors
    )


def _alternate_failures_allowed(
    reports: tuple[ExtractionReport, ...],
    *,
    missing_fields: tuple[str, ...],
) -> bool:
    allowed = set(missing_fields)
    errors = tuple(error for report in reports for error in report.errors)
    return bool(errors) and all(
        error.code == "missing_extracted_field" and error.field in allowed
        for error in errors
    )


def _parse_extracted_values(
    raw_values: Any,
    *,
    required_fields: tuple[str, ...],
    expected_source_ref: str,
    source_text: str,
) -> tuple[dict[str, ExtractedValue], tuple[CalculationError, ...]]:
    errors: list[CalculationError] = []
    if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
        return (
            {},
            (
                CalculationError(
                    code="invalid_llm_output",
                    field="values",
                    message="values must be a list of extracted field records",
                ),
            ),
        )

    values: dict[str, ExtractedValue] = {}
    required = set(required_fields)
    for index, raw_value in enumerate(raw_values):
        value, value_errors = _parse_extracted_value(
            raw_value,
            index=index,
            required_fields=required,
            expected_source_ref=expected_source_ref,
            source_text=source_text,
        )
        errors.extend(value_errors)
        if value is None:
            continue
        if value.field in values:
            errors.append(
                CalculationError(
                    code="duplicate_extracted_field",
                    field=value.field,
                    message=f"{value.field} was extracted more than once",
                )
            )
            continue
        values[value.field] = value

    for field_name in required_fields:
        if field_name not in values:
            errors.append(
                CalculationError(
                    code="missing_extracted_field",
                    field=field_name,
                    message=f"{field_name} was not extracted",
                )
            )
    return values, tuple(errors)


def _parse_extracted_value(
    raw_value: Any,
    *,
    index: int,
    required_fields: set[str],
    expected_source_ref: str,
    source_text: str,
) -> tuple[ExtractedValue | None, tuple[CalculationError, ...]]:
    field_prefix = f"values[{index}]"
    if not isinstance(raw_value, Mapping):
        return (
            None,
            (
                CalculationError(
                    code="invalid_extracted_value",
                    field=field_prefix,
                    message=f"{field_prefix} must be an object",
                ),
            ),
        )

    errors: list[CalculationError] = []
    field_name = _required_string(
        raw_value.get("field"),
        f"{field_prefix}.field",
        errors,
    )
    if field_name and field_name not in required_fields:
        errors.append(
            CalculationError(
                code="unexpected_field",
                field=f"{field_prefix}.field",
                message=f"{field_name} is not a requested field",
            )
        )

    value = _real(raw_value.get("value"), f"{field_prefix}.value", errors)
    normalized_value = _real(
        raw_value.get("normalized_value"),
        f"{field_prefix}.normalized_value",
        errors,
    )
    unit = _required_string(
        raw_value.get("unit"),
        f"{field_prefix}.unit",
        errors,
    )
    normalized_unit = _required_string(
        raw_value.get("normalized_unit"),
        f"{field_prefix}.normalized_unit",
        errors,
    )
    source_ref = _required_string(
        raw_value.get("source_ref"),
        f"{field_prefix}.source_ref",
        errors,
    )
    evidence_span = _required_string(
        raw_value.get("evidence_span"),
        f"{field_prefix}.evidence_span",
        errors,
    )
    confidence = _probability(
        raw_value.get("confidence"),
        f"{field_prefix}.confidence",
        errors,
    )
    notes = _notes(raw_value.get("notes"), f"{field_prefix}.notes", errors)

    if source_ref and source_ref != expected_source_ref:
        errors.append(
            CalculationError(
                code="source_ref_mismatch",
                field=f"{field_prefix}.source_ref",
                message=f"source_ref must be {expected_source_ref}",
            )
        )
    if evidence_span and evidence_span not in source_text:
        errors.append(
            CalculationError(
                code="unsupported_evidence_span",
                field=f"{field_prefix}.evidence_span",
                message="evidence_span must be exact text from the source",
            )
        )
    if (
        field_name
        and confidence < MIN_EXTRACTED_VALUE_CONFIDENCE
        and not _has_error_for_field(errors, f"{field_prefix}.confidence")
    ):
        errors.append(
            CalculationError(
                code="low_extraction_confidence",
                field=field_name,
                message=(
                    f"{field_name} confidence must be at least "
                    f"{MIN_EXTRACTED_VALUE_CONFIDENCE}"
                ),
            )
        )

    if errors:
        return None, tuple(errors)

    return (
        ExtractedValue(
            field=field_name,
            value=value,
            unit=unit,
            normalized_value=normalized_value,
            normalized_unit=normalized_unit,
            source_ref=source_ref,
            evidence_span=evidence_span,
            confidence=confidence,
            notes=notes,
        ),
        (),
    )


def _coerce_llm_response(response: Mapping[str, Any] | str) -> Mapping[str, Any]:
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM JSON response was not valid JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("LLM JSON response must be an object")
        return parsed
    if not isinstance(response, Mapping):
        raise ValueError("LLM response must be a mapping or JSON object")
    return response


def _has_error_for_field(
    errors: Sequence[CalculationError],
    field: str,
) -> bool:
    return any(error.field == field for error in errors)


def _parse_admission_basis(
    response: Mapping[str, Any],
    *,
    source_text: str,
    required: bool,
) -> tuple[dict[str, str], tuple[CalculationError, ...]]:
    if not required:
        return {}, ()

    errors: list[CalculationError] = []
    similarity_basis = _admission_text(
        response.get("similarity_basis"),
        "similarity_basis",
        errors,
    )
    similarity_evidence_span = _admission_text(
        response.get("similarity_evidence_span"),
        "similarity_evidence_span",
        errors,
    )
    if not similarity_basis:
        errors.append(
            CalculationError(
                code="missing_assumption_basis",
                field="similarity_basis",
                message="assumption admission needs a similarity basis",
            )
        )
    if not similarity_evidence_span:
        errors.append(
            CalculationError(
                code="missing_assumption_basis",
                field="similarity_evidence_span",
                message="assumption admission needs a similarity evidence span",
            )
        )
    elif similarity_evidence_span not in source_text:
        errors.append(
            CalculationError(
                code="unsupported_evidence_span",
                field="similarity_evidence_span",
                message=(
                    "similarity_evidence_span must be exact text from the "
                    "source"
                ),
            )
        )

    if errors:
        return {}, tuple(errors)

    return {
        "similarity_basis": similarity_basis,
        "similarity_evidence_span": similarity_evidence_span,
    }, ()


def _admission_text(
    value: object,
    field: str,
    errors: list[CalculationError],
) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        errors.append(
            CalculationError(
                code="invalid_assumption_basis",
                field=field,
                message=f"{field} must be a string",
            )
        )
        return ""
    return value.strip()


def _required_string(
    value: object,
    field: str,
    errors: list[CalculationError],
) -> str:
    if value is None:
        errors.append(_missing(field))
        return ""
    if not isinstance(value, str):
        errors.append(
            CalculationError(
                code="invalid_extracted_value",
                field=field,
                message=f"{field} must be a string",
            )
        )
        return ""

    result = value.strip()
    if not result:
        errors.append(_missing(field))
    return result


def _real(
    value: object,
    field: str,
    errors: list[CalculationError],
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        errors.append(_missing(field))
        return 0.0
    result = float(value)
    if not math.isfinite(result):
        errors.append(
            CalculationError(
                code="invalid_extracted_value",
                field=field,
                message=f"{field} must be finite",
            )
        )
        return 0.0
    if result < 0.0:
        errors.append(
            CalculationError(
                code="invalid_extracted_value",
                field=field,
                message=f"{field} must be non-negative",
            )
        )
        return 0.0
    return result


def _probability(
    value: object,
    field: str,
    errors: list[CalculationError],
) -> float:
    result = _real(value, field, errors)
    if result > 1.0:
        errors.append(
            CalculationError(
                code="invalid_extracted_value",
                field=field,
                message=f"{field} must be in [0, 1]",
            )
        )
    return result


def _notes(
    value: object,
    field: str,
    errors: list[CalculationError],
) -> tuple[str, ...]:
    if value is None:
        errors.append(_missing(field))
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence):
        errors.append(
            CalculationError(
                code="invalid_extracted_value",
                field=field,
                message=f"{field} must be a list of strings",
            )
        )
        return ()

    notes: list[str] = []
    for index, note in enumerate(value):
        if not isinstance(note, str):
            errors.append(
                CalculationError(
                    code="invalid_extracted_value",
                    field=f"{field}[{index}]",
                    message=f"{field}[{index}] must be a string",
                )
            )
            continue
        notes.append(note)
    return tuple(notes)


def _missing(field: str) -> CalculationError:
    return CalculationError(
        code="missing_extraction_field",
        field=field,
        message=f"{field} is required",
    )


def _failed(
    output_name: str,
    required_fields: tuple[str, ...],
    errors: tuple[CalculationError, ...],
) -> ExtractionReport:
    return ExtractionReport(
        output_name=output_name,
        passed=False,
        score=0.0,
        max_score=float(len(required_fields)),
        errors=errors,
    )


__all__ = [
    "EvidenceSource",
    "BEHAVIOR_FILTER_FIELDS",
    "BEHAVIOR_FIELDS",
    "BEHAVIOR_UNMASKED_FIELDS",
    "EXPOSURE_TIME_FIELDS",
    "ExtractedValue",
    "ExtractionReport",
    "LLMExtractCallable",
    "POPULATION_COUNT_FIELDS",
    "POPULATION_ATTACK_RATE_FIELDS",
    "REMOVAL_CAPACITY_FIELDS",
    "ROOM_REMOVAL_FIELDS",
    "ROOM_VENTILATION_FLOW_REMOVAL_FIELDS",
    "SUSCEPTIBLE_ATTACK_RATE_FIELDS",
    "SUSCEPTIBLE_COUNT_FIELDS",
    "TRANSMISSIBILITY_FIELDS",
    "VOLUME_VENTILATION_FLOW_REMOVAL_FIELDS",
    "evidence_source_from_candidate",
    "extract_attack_probability_inputs",
    "extract_behavior_assumptions",
    "extract_behavior_filter_efficiency",
    "extract_behavior_penetration",
    "extract_behavior_unmasked",
    "extract_equivalent_removal_capacity",
    "extract_exposure_time",
    "extract_population_attack_probability",
    "extract_population_counts",
    "extract_removal_capacity",
    "extract_room_removal_capacity",
    "extract_room_ventilation_flow_removal_capacity",
    "extract_susceptible_attack_probability",
    "extract_susceptible_counts",
    "extract_transmissibility_assumption",
    "extract_volume_removal_capacity",
    "extract_volume_ventilation_flow_removal_capacity",
    "run_llm_field_extraction",
]
