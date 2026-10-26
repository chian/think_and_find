"""Build attack probability from LLM-extracted source text."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from episodic_workflow.credit_assignment.calculations import run_named_calculation
from episodic_workflow.credit_assignment.cq_calibration import (
    CQ_CALIBRATION_CHILDREN,
    CQ_NORMALIZATION_CRITERIA,
    CqCalibrationChild,
    CqCalibrationChildReport,
    cq_normalization_rarefaction,
    score_cq_normalization_child,
)
from episodic_workflow.credit_assignment.simple import CalculationError
from episodic_workflow.method_loop import (
    ClosedRecord,
    Context,
    Episode,
    EpisodeGoal,
    EpisodeRecord,
    EpisodeRequest,
    EpisodeTree,
    Grain,
    leaves,
)

from .evidence_extraction import (
    EvidenceSource,
    ExtractionReport,
    LLMExtractCallable,
    extract_attack_probability_inputs,
)


@dataclass(frozen=True)
class AttackProbabilityObservation:
    """One source-level extraction attempt for attack probability."""

    source_ref: str
    extraction: ExtractionReport
    report: CqCalibrationChildReport

    @property
    def passed(self) -> bool:
        return self.report.passed

    def as_record(self) -> dict[str, Any]:
        return {
            "source_ref": self.source_ref,
            "passed": self.passed,
            "extraction": self.extraction.as_record(),
            "report": self.report.as_record(),
        }


@dataclass(frozen=True)
class AttackProbabilityEpisodeStep:
    """Controller step for the attack-probability builder."""

    stop: bool
    requests_transition: bool = False

    def as_record(self) -> dict[str, bool]:
        return {
            "stop": self.stop,
            "requests_transition": self.requests_transition,
        }


@dataclass(frozen=True)
class AttackProbabilityEpisodeState:
    """Accumulated attack-probability extraction state."""

    source_count: int
    observations: tuple[AttackProbabilityObservation, ...] = ()
    stop: bool = False

    def to_report(self) -> CqCalibrationChildReport:
        for observation in self.observations:
            if observation.report.passed:
                return observation.report
        if not self.observations:
            return failed_build_attack_probability(
                (
                    CalculationError(
                        code="missing_candidate_sources",
                        field="candidate_outbreak_sources",
                        message="no retrieved sources were available",
                    ),
                )
            )
        return failed_build_attack_probability(
            tuple(
                error
                for observation in self.observations
                for error in observation.report.errors
            ),
            outputs={
                "failed_source_reports": _failed_source_reports(
                    self.observations
                ),
            },
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "source_count": self.source_count,
            "stop": self.stop,
            "observations": [
                observation.as_record()
                for observation in self.observations
            ],
            "report": self.to_report().as_record(),
        }


@dataclass(frozen=True)
class AttackProbabilityRequest(ClosedRecord):
    """Request for LLM-backed attack-probability construction."""

    sources: tuple[Mapping[str, Any], ...]

    def as_record(self) -> Mapping[str, Any]:
        return {"sources": [dict(source) for source in self.sources]}


@dataclass(frozen=True)
class AttackProbabilityRecord(ClosedRecord):
    """Closed result of attack-probability construction."""

    report: CqCalibrationChildReport

    def as_record(self) -> Mapping[str, Any]:
        return self.report.as_record()

    def audit_identifiers(self) -> tuple[str, ...]:
        return self.report.identities


class AttackProbabilityController:
    """Stop after the first credited source or after every source fails."""

    epoch = "attack_probability_from_evidence_v1"

    def __init__(
        self,
        *,
        source_count: int,
        observations: tuple[AttackProbabilityObservation, ...] = (),
        stop: bool = False,
    ) -> None:
        self._source_count = source_count
        self._observations = observations
        self._stop = stop

    def observe(
        self,
        unit_label: str,
        value: object,
        *,
        is_root: bool,
    ) -> AttackProbabilityEpisodeStep:
        if not isinstance(value, AttackProbabilityObservation):
            raise TypeError(
                "AttackProbabilityController expects "
                "AttackProbabilityObservation"
            )

        observations = self._observations + (value,)
        stop = value.passed or len(observations) >= self._source_count
        self._observations = observations
        self._stop = stop
        return AttackProbabilityEpisodeStep(stop=stop)

    def state(self) -> AttackProbabilityEpisodeState:
        return AttackProbabilityEpisodeState(
            source_count=self._source_count,
            observations=self._observations,
            stop=self._stop,
        )

    def transitioned(self, epoch: str) -> "AttackProbabilityController":
        return AttackProbabilityController(
            source_count=self._source_count,
            observations=self._observations,
            stop=self._stop,
        )


def _run_build_attack_probability_episode(
    llm: LLMExtractCallable | None,
    sources: Iterable[EvidenceSource],
) -> tuple[EpisodeRecord, CqCalibrationChildReport]:
    """Extract counts from retrieved text and compute attack probability."""

    source_tuple = tuple(sources)
    grain = Grain(
        name="build_attack_probability",
        unit="retrieved outbreak source",
        result="CqCalibrationChildReport",
        controller=lambda _path: AttackProbabilityController(
            source_count=len(source_tuple),
        ),
    )
    episode = Episode(
        grain=grain,
        key="build_attack_probability",
        source=leaves(
            source_tuple,
            extract=lambda source: source,
            accept=lambda _unit, source: _run_source(llm, source),
            result=lambda _unit, observation: observation,
            label=lambda source: source.source_ref,
        ),
        request=EpisodeRequest(
            goal=EpisodeGoal.root(
                objective={"operation": "build_attack_probability"},
                result_contract={
                    "result": "CqCalibrationChildReport",
                    "output": "attack_probability",
                },
            ),
            message=AttackProbabilityRequest(
                tuple(source.as_record() for source in source_tuple)
            ),
        ),
        build_result=_build_result,
    )
    tree = EpisodeTree(root=grain)
    episode_record = episode.run(Context(tree=tree))
    return episode_record, episode_record.controller_state.to_report()


def build_attack_probability_from_extraction(
    extraction: ExtractionReport,
) -> CqCalibrationChildReport:
    """Compute attack probability from a passed population extraction."""

    if not extraction.passed:
        return failed_build_attack_probability(extraction.errors)

    values, errors = _normalized_values(extraction)
    if errors:
        return failed_build_attack_probability(errors)

    if "susceptible_count" in values:
        susceptible_count = values["susceptible_count"]
        susceptible_identities = ()
    else:
        susceptible = run_named_calculation(
            "susceptible_count",
            {
                "attendee_count": values.get("attendee_count"),
                "index_case_count": values.get("index_case_count"),
            },
        )
        if not susceptible.passed:
            return failed_build_attack_probability(susceptible.errors)
        susceptible_count = susceptible.outputs["susceptible_count"]
        susceptible_identities = susceptible.identities

    if "secondary_case_count" in values:
        attack = run_named_calculation(
            "secondary_attack_probability",
            {
                "secondary_case_count": values.get("secondary_case_count"),
                "susceptible_count": susceptible_count,
            },
        )
        if not attack.passed:
            return failed_build_attack_probability(attack.errors)
        attack_probability = attack.outputs["attack_probability"]
        attack_identities = attack.identities
        secondary_case_count = values["secondary_case_count"]
    else:
        attack_probability = values["attack_probability"]
        attack_identities = ()
        secondary_case_count = None

    source_refs = tuple(
        dict.fromkeys(
            value.source_ref
            for value in extraction.values.values()
        )
    )
    child = _child()
    criteria = {
        criterion.name: criterion
        for criterion in CQ_NORMALIZATION_CRITERIA[child.name]
    }
    passed = tuple(criteria)
    rarefaction = cq_normalization_rarefaction(
        (float(len(passed)),),
        max_score=float(len(criteria)),
    )
    return score_cq_normalization_child(
        child,
        passed,
        {
            "candidate_outbreak_sources": source_refs,
            "outbreak_record": source_refs[0],
            **(
                {"attendee_count": values["attendee_count"]}
                if "attendee_count" in values
                else {}
            ),
            "index_case_count": values["index_case_count"],
            "secondary_case_count": secondary_case_count,
            "susceptible_count": susceptible_count,
            "count_consistency": True,
            "attack_probability": attack_probability,
            "source_refs": source_refs,
            "extracted_values": {
                name: value.as_record()
                for name, value in extraction.values.items()
            },
            **({"span_repair_history": list(extraction.span_repair_history)} if extraction.span_repair_history else {}),
        },
        identities=susceptible_identities + attack_identities,
        rarefaction=rarefaction,
    )


def _run_source(
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
) -> AttackProbabilityObservation:
    extraction = extract_attack_probability_inputs(llm, source)
    report = build_attack_probability_from_extraction(extraction)
    return AttackProbabilityObservation(
        source_ref=source.source_ref,
        extraction=extraction,
        report=report,
    )


def _normalized_values(
    extraction: ExtractionReport,
) -> tuple[dict[str, float], tuple[CalculationError, ...]]:
    values: dict[str, float] = {}
    errors: list[CalculationError] = []
    for field, value in extraction.values.items():
        if field == "attack_probability":
            normalized, field_errors = _normalized_fraction(
                field,
                value.value,
                value.unit,
                value.normalized_value,
                value.normalized_unit,
            )
        else:
            normalized, field_errors = _normalized_count(
                field,
                value.value,
                value.unit,
                value.normalized_value,
                value.normalized_unit,
            )

        values[field] = normalized
        errors.extend(field_errors)

    return values, tuple(errors)


def _normalized_count(
    field: str,
    raw_value: float,
    raw_unit: str,
    normalized_value: float,
    normalized_unit: str,
) -> tuple[float, tuple[CalculationError, ...]]:
    errors: list[CalculationError] = []
    if normalized_unit != "count":
        errors.append(
            CalculationError(
                code="unsupported_unit",
                field=field,
                message=f"{field} must be normalized to count",
            )
        )
    if _unit_key(raw_unit) not in {
        "count",
        "counts",
        "case",
        "cases",
        "person",
        "persons",
        "people",
        "attendee",
        "attendees",
    }:
        errors.append(
            CalculationError(
                code="unsupported_unit",
                field=field,
                message=f"{field} raw unit must be a count",
            )
        )
    if not _close(normalized_value, raw_value):
        errors.append(
            CalculationError(
                code="inconsistent_normalized_value",
                field=field,
                message=(
                    f"{field} normalized_value must match the raw count"
                ),
            )
        )
    if not _whole_number(normalized_value):
        errors.append(
            CalculationError(
                code="invalid_extracted_value",
                field=field,
                message=f"{field} must be a whole count",
            )
        )
    if normalized_value < 0.0:
        errors.append(
            CalculationError(
                code="invalid_extracted_value",
                field=field,
                message=f"{field} must be non-negative",
            )
        )
    if field in {
        "attendee_count",
        "index_case_count",
        "susceptible_count",
    } and normalized_value <= 0.0:
        errors.append(
            CalculationError(
                code="invalid_extracted_value",
                field=field,
                message=f"{field} must be positive",
            )
        )
    return normalized_value, tuple(errors)


def _normalized_fraction(
    field: str,
    raw_value: float,
    raw_unit: str,
    normalized_value: float,
    normalized_unit: str,
) -> tuple[float, tuple[CalculationError, ...]]:
    errors: list[CalculationError] = []
    if normalized_unit != "fraction":
        errors.append(
            CalculationError(
                code="unsupported_unit",
                field=field,
                message=f"{field} must be normalized to fraction",
            )
        )
    if not 0.0 <= normalized_value < 1.0:
        errors.append(
            CalculationError(
                code="invalid_extracted_value",
                field=field,
                message="attack_probability must be in [0, 1)",
            )
        )

    unit = _unit_key(raw_unit)
    if unit in {"fraction", "probability", "scalar", "1", "unitless"}:
        expected = raw_value
    elif unit in {"%", "percent", "percentage"}:
        expected = raw_value / 100.0
    else:
        errors.append(
            CalculationError(
                code="unsupported_unit",
                field=field,
                message=(
                    "attack_probability raw unit must be fraction or percent"
                ),
            )
        )
        expected = normalized_value

    if not _close(normalized_value, expected):
        errors.append(
            CalculationError(
                code="inconsistent_normalized_value",
                field=field,
                message=(
                    "attack_probability normalized_value must match the raw "
                    "fraction or percent value"
                ),
            )
        )

    return normalized_value, tuple(errors)


def _whole_number(value: float) -> bool:
    return _close(value, round(value))


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-9, abs(right) * 1e-9)


def _unit_key(unit: str) -> str:
    return (
        unit.strip()
        .lower()
        .replace(" ", "")
        .replace(".", "")
    )


def failed_build_attack_probability(
    errors: tuple[CalculationError, ...],
    *,
    outputs: Mapping[str, Any] | None = None,
) -> CqCalibrationChildReport:
    child = _child()
    return CqCalibrationChildReport(
        child=child.name,
        output=child.output,
        passed=False,
        score=0.0,
        max_score=child.weight,
        should_yield=True,
        outputs=dict(outputs or {}),
        errors=errors,
    )


def _failed_source_reports(
    observations: tuple[AttackProbabilityObservation, ...],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(observation.as_record() for observation in observations)


def _child() -> CqCalibrationChild:
    return next(
        child
        for child in CQ_CALIBRATION_CHILDREN
        if child.name == "build_attack_probability"
    )


def _build_result(record: EpisodeRecord) -> AttackProbabilityRecord:
    state = record.controller_state
    if not isinstance(state, AttackProbabilityEpisodeState):
        raise TypeError(
            "attack-probability Episode produced an unexpected state"
        )
    return AttackProbabilityRecord(state.to_report())


__all__ = [
    "AttackProbabilityController",
    "AttackProbabilityEpisodeState",
    "AttackProbabilityEpisodeStep",
    "AttackProbabilityObservation",
    "AttackProbabilityRecord",
    "AttackProbabilityRequest",
    "build_attack_probability_from_extraction",
    "failed_build_attack_probability",
]
