"""Credit layout for calibrating Cq from one outbreak."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable, Mapping

from .simple import CalculationError


@dataclass(frozen=True)
class CqCalibrationChild:
    """One child expected by `calibrate_cq_from_outbreak`."""

    name: str
    output: str
    group: str
    does: str
    mode: str
    weight: float = 1.0
    requires_rarefaction: bool = False
    uses_search: bool = False
    main_output: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CqCalibrationChild.name must be populated")
        if not self.output:
            raise ValueError("CqCalibrationChild.output must be populated")
        if not self.group:
            raise ValueError("CqCalibrationChild.group must be populated")
        if not self.does:
            raise ValueError("CqCalibrationChild.does must be populated")
        if self.mode not in {
            "calculation",
            "evidence_normalization",
            "assumption_normalization",
        }:
            raise ValueError(f"unknown Cq child mode: {self.mode}")
        if self.weight <= 0.0:
            raise ValueError("CqCalibrationChild.weight must be positive")

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "output": self.output,
            "group": self.group,
            "does": self.does,
            "mode": self.mode,
            "weight": self.weight,
            "requires_rarefaction": self.requires_rarefaction,
            "uses_search": self.uses_search,
            "main_output": self.main_output,
        }


@dataclass(frozen=True)
class CqNormalizationCriterion:
    """One scored evidence or assumption-normalization criterion."""

    name: str
    field: str
    does: str
    weight: float = 1.0
    required: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CqNormalizationCriterion.name must be populated")
        if not self.field:
            raise ValueError("CqNormalizationCriterion.field must be populated")
        if not self.does:
            raise ValueError("CqNormalizationCriterion.does must be populated")
        if self.weight <= 0.0:
            raise ValueError("CqNormalizationCriterion.weight must be positive")

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "field": self.field,
            "does": self.does,
            "weight": self.weight,
            "required": self.required,
        }


@dataclass(frozen=True)
class CqCalibrationChildReport:
    """Progress and yield state returned by one Cq calibration child."""

    child: str
    output: str
    passed: bool
    score: float
    max_score: float
    should_yield: bool
    credited_criteria: tuple[str, ...] = ()
    missing_criteria: tuple[str, ...] = ()
    outputs: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[CalculationError, ...] = ()
    identities: tuple[str, ...] = ()
    rarefaction: Mapping[str, Any] | None = None
    stop: bool = True

    def as_record(self) -> dict[str, Any]:
        return {
            "child": self.child,
            "output": self.output,
            "passed": self.passed,
            "score": self.score,
            "max_score": self.max_score,
            "normalized_score": self.normalized_score,
            "should_yield": self.should_yield,
            "stop": self.stop,
            "credited_criteria": list(self.credited_criteria),
            "missing_criteria": list(self.missing_criteria),
            "outputs": dict(self.outputs),
            "errors": [error.as_record() for error in self.errors],
            "identities": list(self.identities),
            "rarefaction": (
                None if self.rarefaction is None else dict(self.rarefaction)
            ),
        }

    @property
    def normalized_score(self) -> float:
        if self.max_score == 0.0:
            return 0.0
        return self.score / self.max_score


@dataclass(frozen=True)
class CqRarefactionReport:
    """Numerical stop decision for a search-backed normalization child."""

    should_yield: bool
    score: float
    max_score: float | None
    recent_gain: float
    gain_threshold: float
    score_resolution: float
    efficiency_fraction: float
    productive_gain_baseline: float | None
    window_size: int
    observations: int
    outcome: str

    def __post_init__(self) -> None:
        if self.score < 0.0:
            raise ValueError("CqRarefactionReport.score must be non-negative")
        if not math.isfinite(self.score):
            raise ValueError("CqRarefactionReport.score must be finite")
        if self.max_score is not None:
            if not math.isfinite(self.max_score) or self.max_score <= 0.0:
                raise ValueError("CqRarefactionReport.max_score must be finite and positive")
            if self.score > self.max_score:
                raise ValueError("CqRarefactionReport.score cannot exceed max_score")
        if self.recent_gain < 0.0:
            raise ValueError("CqRarefactionReport.recent_gain must be non-negative")
        if not math.isfinite(self.recent_gain):
            raise ValueError("CqRarefactionReport.recent_gain must be finite")
        if self.gain_threshold < 0.0:
            raise ValueError("CqRarefactionReport.gain_threshold must be non-negative")
        if not math.isfinite(self.gain_threshold):
            raise ValueError("CqRarefactionReport.gain_threshold must be finite")
        if self.score_resolution <= 0.0:
            raise ValueError("CqRarefactionReport.score_resolution must be positive")
        if not math.isfinite(self.score_resolution):
            raise ValueError("CqRarefactionReport.score_resolution must be finite")
        if self.efficiency_fraction < 0.0:
            raise ValueError(
                "CqRarefactionReport.efficiency_fraction must be non-negative"
            )
        if (
            self.productive_gain_baseline is not None
            and self.productive_gain_baseline <= 0.0
        ):
            raise ValueError(
                "CqRarefactionReport.productive_gain_baseline must be positive"
            )
        if (
            self.productive_gain_baseline is not None
            and not math.isfinite(self.productive_gain_baseline)
        ):
            raise ValueError(
                "CqRarefactionReport.productive_gain_baseline must be finite"
            )
        if self.window_size <= 0:
            raise ValueError("CqRarefactionReport.window_size must be positive")
        if self.observations <= 0:
            raise ValueError("CqRarefactionReport.observations must be positive")
        if self.outcome not in {"continue", "complete", "saturated"}:
            raise ValueError(f"unknown Cq rarefaction outcome: {self.outcome}")

    def as_record(self) -> dict[str, object]:
        return {
            "should_yield": self.should_yield,
            "score": self.score,
            "max_score": self.max_score,
            "normalized_score": self.normalized_score,
            "recent_gain": self.recent_gain,
            "gain_threshold": self.gain_threshold,
            "score_resolution": self.score_resolution,
            "efficiency_fraction": self.efficiency_fraction,
            "productive_gain_baseline": self.productive_gain_baseline,
            "window_size": self.window_size,
            "observations": self.observations,
            "outcome": self.outcome,
        }

    @property
    def normalized_score(self) -> float | None:
        return self.score / self.max_score if self.max_score is not None else None


@dataclass(frozen=True)
class CqCalibrationError:
    """One child failure in a Cq calibration parent report."""

    child: str
    output: str
    code: str
    message: str
    field: str | None = None

    def as_record(self) -> dict[str, str]:
        record = {
            "child": self.child,
            "output": self.output,
            "code": self.code,
            "message": self.message,
        }
        if self.field:
            record["field"] = self.field
        return record


@dataclass(frozen=True)
class CqCalibrationCreditReport:
    """Summed mixed credit for `calibrate_cq_from_outbreak`."""

    passed: bool
    score: float
    max_score: float
    credited_children: tuple[str, ...] = ()
    missing_children: tuple[str, ...] = ()
    child_reports: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    outputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    main_outputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    admitted_assumptions: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    input_provenance: Mapping[str, Any] = field(default_factory=dict)
    missing_fields: tuple[str, ...] = ()
    failed_fields: tuple[str, ...] = ()
    errors: tuple[CqCalibrationError, ...] = ()
    identities: tuple[str, ...] = ()
    stop: bool = True

    @property
    def normalized_score(self) -> float:
        if self.max_score == 0.0:
            return 0.0
        return self.score / self.max_score

    def as_record(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "max_score": self.max_score,
            "normalized_score": self.normalized_score,
            "stop": self.stop,
            "credited_children": list(self.credited_children),
            "missing_children": list(self.missing_children),
            "child_reports": {
                child: dict(report)
                for child, report in self.child_reports.items()
            },
            "outputs": {
                child: dict(values)
                for child, values in self.outputs.items()
            },
            "main_outputs": {
                child: dict(values)
                for child, values in self.main_outputs.items()
            },
            "source_refs": list(self.source_refs),
            "admitted_assumptions": {
                name: dict(assumption)
                for name, assumption in self.admitted_assumptions.items()
            },
            "input_provenance": dict(self.input_provenance),
            "source_backed": bool(
                self.input_provenance.get("source_backed", False)
            ),
            "search_backed": bool(
                self.input_provenance.get("search_backed", False)
            ),
            "llm_extracted": bool(
                self.input_provenance.get("llm_extracted", False)
            ),
            "missing_fields": list(self.missing_fields),
            "failed_fields": list(self.failed_fields),
            "errors": [error.as_record() for error in self.errors],
            "identities": list(self.identities),
        }


CQ_CALIBRATION_CHILDREN: tuple[CqCalibrationChild, ...] = (
    CqCalibrationChild(
        name="find_candidate_outbreak_sources",
        output="candidate_outbreak_sources",
        group="outbreak_search",
        mode="evidence_normalization",
        does=(
            "Search for disease- and setting-scoped indoor outbreak reports "
            "with enough metadata and passages to seed extraction."
        ),
        weight=5.0,
        requires_rarefaction=True,
        uses_search=True,
    ),
    CqCalibrationChild(
        name="build_attack_probability",
        output="attack_probability",
        group="outbreak_attack_probability",
        mode="evidence_normalization",
        does=(
            "Find or admit a point-source outbreak, normalize secondary "
            "counts or an attack rate, and return the observed attack "
            "probability."
        ),
        weight=7.0,
        requires_rarefaction=True,
        uses_search=True,
    ),
    CqCalibrationChild(
        name="build_removal_capacity",
        output="removal_capacity_m3_s",
        group="room_reconstruction",
        mode="evidence_normalization",
        does=(
            "Find or admit room evidence, normalize the room and loss fields, "
            "and compute clean-air removal capacity."
        ),
        weight=7.0,
        requires_rarefaction=True,
        uses_search=True,
    ),
    CqCalibrationChild(
        name="admit_exposure_time",
        output="exposure_time_s",
        group="exposure",
        mode="evidence_normalization",
        does="Admit the outbreak exposure duration and normalize it to seconds.",
        weight=4.0,
        requires_rarefaction=True,
        uses_search=True,
    ),
    CqCalibrationChild(
        name="admit_behavior_assumptions",
        output="behavior_assumptions",
        group="behavior",
        mode="assumption_normalization",
        does=(
            "Admit breathing flow and mask penetration for the exposed "
            "population and activity."
        ),
        weight=5.0,
        requires_rarefaction=True,
        uses_search=True,
    ),
    CqCalibrationChild(
        name="admit_population_or_variant_scaling",
        output="relative_transmissibility",
        group="transmissibility",
        mode="assumption_normalization",
        does=(
            "Admit the scalar that maps the outbreak population or variant to "
            "the target disease calibration."
        ),
        weight=3.0,
        requires_rarefaction=True,
        uses_search=True,
    ),
    CqCalibrationChild(
        name="quanta_concentration_from_attack_probability",
        output="quanta_concentration_exhaled_m3",
        group="cq",
        mode="calculation",
        does=(
            "Compute Cq from attack probability, removal capacity, exposure "
            "time, index-case count, behavior assumptions, and "
            "transmissibility scaling."
        ),
        main_output=True,
    ),
)


CQ_NORMALIZATION_CRITERIA: Mapping[str, tuple[CqNormalizationCriterion, ...]] = {
    "find_candidate_outbreak_sources": (
        CqNormalizationCriterion(
            "scoped_queries_formulated",
            "search_specs",
            "Disease- and setting-scoped outbreak queries were formulated.",
        ),
        CqNormalizationCriterion(
            "search_results_returned",
            "search_records",
            "At least one query returned at least one candidate source.",
        ),
        CqNormalizationCriterion(
            "source_metadata_present",
            "candidate_outbreak_sources",
            "Candidate sources include identifiers and retrievable text.",
        ),
        CqNormalizationCriterion(
            "retrieved_text_present",
            "search_summary",
            "At least one candidate includes retrieved text for extraction.",
        ),
        CqNormalizationCriterion(
            "evidence_diversity_scored",
            "search_summary",
            "Candidates cover more than one query purpose or source.",
        ),
    ),
    "build_attack_probability": (
        CqNormalizationCriterion(
            "candidate_outbreak_sources_found",
            "candidate_outbreak_sources",
            "Search returned candidate outbreak sources.",
        ),
        CqNormalizationCriterion(
            "point_source_event_admitted",
            "outbreak_record",
            "One indoor point-source outbreak was admitted.",
        ),
        CqNormalizationCriterion(
            "attendee_count_admitted",
            "attendee_count",
            "The exposed attendee count was admitted.",
        ),
        CqNormalizationCriterion(
            "index_case_count_admitted",
            "index_case_count",
            "The index case count was admitted.",
        ),
        CqNormalizationCriterion(
            "secondary_cases_admitted",
            "secondary_case_count",
            (
                "Confirmed/probable secondary cases or a normalized "
                "secondary attack probability were admitted."
            ),
        ),
        CqNormalizationCriterion(
            "counts_cross_checked",
            "count_consistency",
            "Admitted outbreak counts are mutually consistent.",
        ),
        CqNormalizationCriterion(
            "attack_probability_computed",
            "attack_probability",
            "Attack probability was computed from the admitted counts.",
        ),
    ),
    "build_removal_capacity": (
        CqNormalizationCriterion(
            "room_geometry_admitted",
            "room_geometry",
            (
                "Room geometry or an equivalent reported virus-free-air "
                "capacity was admitted."
            ),
        ),
        CqNormalizationCriterion(
            "ventilation_basis_admitted",
            "air_purification_rate_s",
            "Ventilation, filtration, or deactivation evidence was admitted.",
        ),
        CqNormalizationCriterion(
            "particle_loss_basis_admitted",
            "settling_speed_m_s",
            "Particle settling evidence was extracted or admitted.",
        ),
        CqNormalizationCriterion(
            "evidence_basis_accepted",
            "evidence_basis",
            "Each room assumption has an accepted evidence basis.",
        ),
        CqNormalizationCriterion(
            "fields_normalized",
            "normalized_room_fields",
            "Room and removal fields were normalized to SI units.",
        ),
        CqNormalizationCriterion(
            "fields_cross_checked",
            "room_field_consistency",
            "Room geometry and removal fields are internally consistent.",
        ),
        CqNormalizationCriterion(
            "removal_capacity_computed",
            "removal_capacity_m3_s",
            "Clean-air removal capacity was computed.",
        ),
    ),
    "admit_exposure_time": (
        CqNormalizationCriterion(
            "raw_exposure_time_found",
            "raw_exposure_time",
            "A duration or event schedule was found.",
        ),
        CqNormalizationCriterion(
            "exposure_time_normalized",
            "exposure_time_s",
            "Exposure time was normalized to seconds.",
        ),
        CqNormalizationCriterion(
            "positive_duration",
            "exposure_time_s",
            "Exposure time is positive.",
        ),
        CqNormalizationCriterion(
            "event_time_scope_accepted",
            "time_scope",
            "The duration applies to the calibrated outbreak event.",
        ),
    ),
    "admit_behavior_assumptions": (
        CqNormalizationCriterion(
            "behavior_basis_chosen",
            "behavior_assumptions",
            "A sourced or assumed behavior basis was chosen.",
        ),
        CqNormalizationCriterion(
            "breathing_flow_admitted",
            "breathing_flow_m3_s",
            "A breathing flow value was admitted.",
        ),
        CqNormalizationCriterion(
            "breathing_flow_normalized",
            "breathing_flow_m3_s",
            "Breathing flow was normalized to cubic meters per second.",
        ),
        CqNormalizationCriterion(
            "mask_penetration_admitted",
            "mask_penetration",
            "A mask penetration value was admitted.",
        ),
        CqNormalizationCriterion(
            "mask_penetration_in_range",
            "mask_penetration",
            "Mask penetration is a probability.",
        ),
    ),
    "admit_population_or_variant_scaling": (
        CqNormalizationCriterion(
            "scaling_basis_chosen",
            "scaling_basis",
            "The population or variant scaling basis was chosen.",
        ),
        CqNormalizationCriterion(
            "relative_transmissibility_admitted",
            "relative_transmissibility",
            "A transmissibility scalar was admitted.",
        ),
        CqNormalizationCriterion(
            "relative_transmissibility_positive",
            "relative_transmissibility",
            "The transmissibility scalar is positive.",
        ),
    ),
}


def credit_cq_calibration_child(
    child: CqCalibrationChild,
    outputs: Mapping[str, Any],
    *,
    errors: tuple[CalculationError, ...] = (),
    identities: tuple[str, ...] = (),
) -> CqCalibrationChildReport:
    """Credit a one-shot Cq calibration child for returning its named output."""

    if errors:
        return _failed_child(child, errors)
    if child.output not in outputs:
        return _failed_child(
            child,
            (
                CalculationError(
                    code="missing_output",
                    field=child.output,
                    message=f"{child.name} did not return {child.output}",
                ),
            ),
        )
    if outputs[child.output] is None:
        return _failed_child(
            child,
            (
                CalculationError(
                    code="empty_output",
                    field=child.output,
                    message=f"{child.output} must not be None",
                ),
            ),
        )
    return CqCalibrationChildReport(
        child=child.name,
        output=child.output,
        passed=True,
        score=child.weight,
        max_score=child.weight,
        should_yield=True,
        outputs={child.output: outputs[child.output]},
        identities=identities,
    )


def score_cq_normalization_child(
    child: CqCalibrationChild,
    passed_criteria: Iterable[str],
    outputs: Mapping[str, Any],
    *,
    errors: tuple[CalculationError, ...] = (),
    identities: tuple[str, ...] = (),
    rarefaction: CqRarefactionReport | None = None,
) -> CqCalibrationChildReport:
    """Score a search-backed child and apply its rarefaction verdict."""

    try:
        criteria = CQ_NORMALIZATION_CRITERIA[child.name]
    except KeyError as exc:
        raise KeyError(f"{child.name} has no normalization criteria") from exc

    criteria_by_name = {criterion.name: criterion for criterion in criteria}
    credited_criteria = tuple(dict.fromkeys(passed_criteria))
    unknown_criteria = set(credited_criteria) - set(criteria_by_name)
    if unknown_criteria:
        raise KeyError(f"unknown criteria for {child.name}: {sorted(unknown_criteria)}")

    missing_criteria = tuple(
        criterion.name
        for criterion in criteria
        if criterion.required and criterion.name not in credited_criteria
    )
    score = sum(
        criteria_by_name[name].weight
        for name in credited_criteria
    )
    max_score = sum(criterion.weight for criterion in criteria)
    report_errors = (
        errors
        + tuple(
            CalculationError(
                code="missing_criterion",
                field=criteria_by_name[name].field,
                message=f"{name} is required for {child.name}",
            )
            for name in missing_criteria
        )
        + _output_errors(child, outputs)
        + _rarefaction_errors(child, rarefaction)
    )
    should_yield = rarefaction.should_yield if rarefaction is not None else False
    return CqCalibrationChildReport(
        child=child.name,
        output=child.output,
        passed=not report_errors and should_yield,
        score=score,
        max_score=max_score,
        should_yield=should_yield,
        credited_criteria=credited_criteria,
        missing_criteria=missing_criteria,
        outputs={name: value for name, value in outputs.items() if value is not None},
        errors=report_errors,
        identities=identities,
        rarefaction=(
            None if rarefaction is None else rarefaction.as_record()
        ),
    )


def cq_normalization_rarefaction(
    score_history: Iterable[float],
    *,
    max_score: float | None,
    score_resolution: float = 1.0,
    efficiency_fraction: float = 0.5,
    window_size: int = 2,
) -> CqRarefactionReport:
    """Stop on marginal saturation, or completion when a finite ceiling exists."""

    scores = tuple(float(score) for score in score_history)
    if not scores:
        raise ValueError("score_history must contain at least one score")
    if max_score is not None and (not math.isfinite(max_score) or max_score <= 0.0):
        raise ValueError("max_score must be finite and positive")
    if not math.isfinite(score_resolution) or score_resolution <= 0.0:
        raise ValueError("score_resolution must be finite and positive")
    if not math.isfinite(efficiency_fraction) or efficiency_fraction < 0.0:
        raise ValueError("efficiency_fraction must be finite and non-negative")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if any(
        not math.isfinite(score) or score < 0.0 or (max_score is not None and score > max_score)
        for score in scores
    ):
        raise ValueError("score_history values must be finite and in [0, max_score]")
    if any(next_score < score for score, next_score in zip(scores, scores[1:])):
        raise ValueError("score_history cannot decrease")

    score = scores[-1]
    baseline_index = max(0, len(scores) - window_size - 1)
    recent_gain = score - scores[baseline_index]
    score_gains = tuple(
        next_score - current_score
        for current_score, next_score in zip(scores, scores[1:])
    )
    productive_gains = tuple(
        gain for gain in score_gains[-window_size:] if gain > 0.0
    )
    productive_gain_baseline = (
        median(productive_gains) if productive_gains else None
    )
    threshold_candidates = [score_resolution]
    if productive_gain_baseline is not None:
        threshold_candidates.append(
            productive_gain_baseline * efficiency_fraction
        )
    gain_threshold = max(threshold_candidates)
    should_yield = max_score is not None and score >= max_score
    outcome = "complete" if should_yield else "continue"
    if not should_yield and len(scores) > window_size:
        should_yield = recent_gain < gain_threshold
        if should_yield:
            outcome = "saturated"

    return CqRarefactionReport(
        should_yield=should_yield,
        score=score,
        max_score=max_score,
        recent_gain=recent_gain,
        gain_threshold=gain_threshold,
        score_resolution=score_resolution,
        efficiency_fraction=efficiency_fraction,
        productive_gain_baseline=productive_gain_baseline,
        window_size=window_size,
        observations=len(scores),
        outcome=outcome,
    )


def sum_cq_calibration_credit(
    reports: (
        Mapping[str, CqCalibrationChildReport]
        | Iterable[CqCalibrationChildReport]
    ),
    *,
    children: tuple[CqCalibrationChild, ...] = CQ_CALIBRATION_CHILDREN,
) -> CqCalibrationCreditReport:
    """Sum deterministic and normalization child credit."""

    reports_by_child = _reports_by_child(reports)
    max_score = sum(child.weight for child in children)

    score = 0.0
    outputs: dict[str, Mapping[str, Any]] = {}
    main_outputs: dict[str, Mapping[str, Any]] = {}
    credited_children: list[str] = []
    missing_children: list[str] = []
    errors: list[CqCalibrationError] = []
    identities: list[str] = []
    child_reports: dict[str, Mapping[str, Any]] = {}
    source_refs: list[str] = []
    admitted_assumptions: dict[str, Mapping[str, Any]] = {}

    for child in children:
        report = reports_by_child.get(child.name)
        if report is None:
            missing_children.append(child.name)
            errors.append(
                CqCalibrationError(
                    child=child.name,
                    output=child.output,
                    code="missing_report",
                    message=f"{child.name} did not return a report",
                )
            )
            continue

        child_reports[child.name] = report.as_record()
        score += min(report.score, child.weight)
        if report.outputs:
            child_outputs = dict(report.outputs)
            outputs[child.name] = child_outputs
            source_refs.extend(_source_refs_from_outputs(child_outputs))
            admitted_assumptions.update(
                _admitted_assumptions_from_outputs(
                    child.name,
                    child_outputs,
                )
            )
        if not report.passed:
            errors.extend(_child_errors(child, report))
            continue

        if not report.should_yield:
            errors.append(
                CqCalibrationError(
                    child=child.name,
                    output=child.output,
                    code="rarefaction_not_yielded",
                    message=f"{child.name} did not receive a yield verdict",
                )
            )
            continue

        if child.output not in report.outputs:
            errors.append(
                CqCalibrationError(
                    child=child.name,
                    output=child.output,
                    code="missing_output",
                    message=f"{child.name} did not return {child.output}",
                    field=child.output,
                )
            )
            continue

        child_outputs = dict(report.outputs)
        if child.main_output:
            main_outputs[child.name] = child_outputs
        credited_children.append(child.name)
        identities.extend(report.identities)

    return CqCalibrationCreditReport(
        passed=len(credited_children) == len(children),
        score=score,
        max_score=max_score,
        credited_children=tuple(credited_children),
        missing_children=tuple(missing_children),
        child_reports=child_reports,
        outputs=outputs,
        main_outputs=main_outputs,
        errors=tuple(errors),
        identities=tuple(identities),
        source_refs=tuple(dict.fromkeys(source_refs)),
        admitted_assumptions=admitted_assumptions,
        missing_fields=_repair_fields(errors, missing=True),
        failed_fields=_repair_fields(errors, missing=False),
    )


def _admitted_assumptions_from_outputs(
    child_name: str,
    outputs: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    admitted = outputs.get("admitted_assumptions")
    if not isinstance(admitted, Mapping):
        return {}

    records: dict[str, Mapping[str, Any]] = {}
    for field, raw_assumption in admitted.items():
        if not isinstance(raw_assumption, Mapping):
            continue
        records[f"{child_name}.{field}"] = {
            "child": child_name,
            **dict(raw_assumption),
        }
    return records


def _source_refs_from_outputs(value: Any) -> tuple[str, ...]:
    refs: list[str] = []
    if isinstance(value, Mapping):
        for key, child_value in value.items():
            if key == "source_ref" and isinstance(child_value, str):
                refs.append(child_value)
            elif key == "source_refs":
                refs.extend(_source_ref_values(child_value))
            else:
                refs.extend(_source_refs_from_outputs(child_value))
    elif isinstance(value, tuple | list):
        for item in value:
            refs.extend(_source_refs_from_outputs(item))
    return tuple(ref for ref in refs if ref)


def _source_ref_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple | list):
        return tuple(str(item) for item in value if str(item))
    return ()


def _repair_fields(
    errors: Iterable[CqCalibrationError],
    *,
    missing: bool,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            f"{error.child}.{error.field or error.output}"
            for error in errors
            if _is_missing_error(error.code) is missing
        )
    )


def _is_missing_error(code: str) -> bool:
    return code in {
        "insufficient_assumption_sources",
        "missing_assumption_basis",
        "missing_assumption_sources",
        "missing_candidate_sources",
        "missing_child_report",
        "missing_criterion",
        "missing_extracted_field",
        "missing_extraction_field",
        "missing_input",
        "missing_llm",
        "missing_output",
        "missing_prerequisite_output",
        "missing_rarefaction",
        "missing_report",
        "missing_source_refs",
        "missing_target_scenario_input",
    }


def _failed_child(
    child: CqCalibrationChild,
    errors: tuple[CalculationError, ...],
) -> CqCalibrationChildReport:
    return CqCalibrationChildReport(
        child=child.name,
        output=child.output,
        passed=False,
        score=0.0,
        max_score=child.weight,
        should_yield=False,
        errors=errors,
    )


def _output_errors(
    child: CqCalibrationChild,
    outputs: Mapping[str, Any],
) -> tuple[CalculationError, ...]:
    if child.output not in outputs:
        return (
            CalculationError(
                code="missing_output",
                field=child.output,
                message=f"{child.name} did not return {child.output}",
            ),
        )
    if outputs[child.output] is None:
        return (
            CalculationError(
                code="empty_output",
                field=child.output,
                message=f"{child.output} must not be None",
            ),
        )
    return ()


def _rarefaction_errors(
    child: CqCalibrationChild,
    rarefaction: CqRarefactionReport | None,
) -> tuple[CalculationError, ...]:
    if child.requires_rarefaction and rarefaction is None:
        return (
            CalculationError(
                code="missing_rarefaction",
                field="rarefaction",
                message=f"{child.name} requires a rarefaction verdict",
            ),
        )
    return ()


def _reports_by_child(
    reports: (
        Mapping[str, CqCalibrationChildReport]
        | Iterable[CqCalibrationChildReport]
    ),
) -> dict[str, CqCalibrationChildReport]:
    if isinstance(reports, Mapping):
        return dict(reports)
    return {report.child: report for report in reports}


def _child_errors(
    child: CqCalibrationChild,
    report: CqCalibrationChildReport,
) -> tuple[CqCalibrationError, ...]:
    if not report.errors:
        return (
            CqCalibrationError(
                child=child.name,
                output=child.output,
                code="failed",
                message=f"{child.name} failed without a child error",
            ),
        )
    return tuple(
        CqCalibrationError(
            child=child.name,
            output=child.output,
            code=error.code,
            message=error.message,
            field=error.field,
        )
        for error in report.errors
    )
