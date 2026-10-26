"""LLM-backed evidence children after candidate outbreak search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from episodic_workflow.credit_assignment.cq_calibration import (
    CQ_CALIBRATION_CHILDREN,
    CQ_NORMALIZATION_CRITERIA,
    CqCalibrationChild,
    CqCalibrationChildReport,
    cq_normalization_rarefaction,
    score_cq_normalization_child,
)
from episodic_workflow.credit_assignment.calculations import run_named_calculation
from episodic_workflow.credit_assignment.simple import CalculationError
from episodic_workflow.method_loop import (
    ChildEpisodeUnit,
    ClosedRecord,
    Context,
    Episode,
    EpisodeGoal,
    EpisodeRecord,
    EpisodeRequest,
    EpisodeTree,
    Grain,
    Leaf,
    leaves,
)
from .assumption_investigation import (
    AssumptionInvestigation, AssumptionServices, assumption_summary,
    branch_contract, latest_assessments,
)

from .evidence_extraction import (
    EvidenceSource,
    ExtractedValue,
    ExtractionReport,
    LLMExtractCallable,
    extract_behavior_assumptions,
    extract_exposure_time,
    extract_removal_capacity,
    extract_transmissibility_assumption,
)


Extractor = Callable[
    [LLMExtractCallable | None, EvidenceSource],
    ExtractionReport,
]
ReportBuilder = Callable[[ExtractionReport], CqCalibrationChildReport]


@dataclass(frozen=True)
class EvidenceChildSpec:
    """One LLM-backed Cq normalization child."""

    child: str
    extractor: Extractor
    build_report: ReportBuilder


@dataclass(frozen=True)
class EvidenceChildObservation(ClosedRecord):
    """One source-level extraction attempt for an evidence child."""

    child: str
    source_ref: str
    extraction: ExtractionReport | None
    report: CqCalibrationChildReport

    @property
    def passed(self) -> bool:
        return self.report.passed

    def as_record(self) -> dict[str, Any]:
        return {
            "child": self.child,
            "source_ref": self.source_ref,
            "passed": self.passed,
            "extraction": None if self.extraction is None else self.extraction.as_record(),
            "report": self.report.as_record(),
        }


@dataclass(frozen=True)
class EvidenceChildEpisodeStep:
    """Controller step for one evidence child."""

    stop: bool
    requests_transition: bool = False

    def as_record(self) -> dict[str, bool]:
        return {
            "stop": self.stop,
            "requests_transition": self.requests_transition,
        }


@dataclass(frozen=True)
class EvidenceChildEpisodeState:
    """Accumulated source attempts for one evidence child."""

    child: str
    source_count: int
    observations: tuple[EvidenceChildObservation, ...] = ()
    stop: bool = False

    def to_report(self) -> CqCalibrationChildReport:
        for observation in self.observations:
            if observation.report.passed:
                return observation.report
        if not self.observations:
            return failed_evidence_child(
                self.child,
                (
                    CalculationError(
                        code="missing_candidate_sources",
                        field="candidate_outbreak_sources",
                        message="no retrieved sources were available",
                    ),
                ),
            )

        return failed_evidence_child(
            self.child,
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
            "child": self.child,
            "source_count": self.source_count,
            "stop": self.stop,
            "observations": [
                observation.as_record()
                for observation in self.observations
            ],
            "report": self.to_report().as_record(),
        }


@dataclass(frozen=True)
class EvidenceChildRequest(ClosedRecord):
    """Request for one evidence child."""

    child: str
    sources: tuple[Mapping[str, Any], ...]

    def as_record(self) -> Mapping[str, Any]:
        return {
            "child": self.child,
            "sources": [dict(source) for source in self.sources],
        }


@dataclass(frozen=True)
class EvidenceChildRecord(ClosedRecord):
    """Closed result of one evidence child."""

    report: CqCalibrationChildReport

    def as_record(self) -> Mapping[str, Any]:
        return self.report.as_record()

    def audit_identifiers(self) -> tuple[str, ...]:
        return self.report.identities


class EvidenceChildController:
    """Stop after the first credited source or after every source fails."""

    epoch = "cq_evidence_child_v1"

    def __init__(
        self,
        *,
        child: str,
        source_count: int,
        observations: tuple[EvidenceChildObservation, ...] = (),
        stop: bool = False,
    ) -> None:
        self._child = child
        self._source_count = source_count
        self._observations = observations
        self._stop = stop

    def observe(
        self,
        unit_label: str,
        value: object,
        *,
        is_root: bool,
    ) -> EvidenceChildEpisodeStep:
        if not isinstance(value, EvidenceChildObservation):
            raise TypeError(
                "EvidenceChildController expects EvidenceChildObservation"
            )

        observations = self._observations + (value,)
        stop = value.passed or len(observations) >= self._source_count
        self._observations = observations
        self._stop = stop
        return EvidenceChildEpisodeStep(stop=stop)

    def state(self) -> EvidenceChildEpisodeState:
        return EvidenceChildEpisodeState(
            child=self._child,
            source_count=self._source_count,
            observations=self._observations,
            stop=self._stop,
        )

    def transitioned(self, epoch: str) -> "EvidenceChildController":
        return EvidenceChildController(
            child=self._child,
            source_count=self._source_count,
            observations=self._observations,
            stop=self._stop,
        )


def _run_build_removal_capacity_episode(
    llm: LLMExtractCallable | None,
    sources: Iterable[EvidenceSource],
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
    assumption_services: AssumptionServices | None = None,
) -> tuple[EpisodeRecord, CqCalibrationChildReport]:
    """Extract equivalent clean-air removal capacity."""

    return _run_evidence_child_episode(
        REMOVAL_CAPACITY_SPEC,
        llm,
        sources,
        admit_assumption=admit_assumption,
        target_context=target_context,
        assumption_services=assumption_services,
    )


def _run_admit_exposure_time_episode(
    llm: LLMExtractCallable | None,
    sources: Iterable[EvidenceSource],
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
    assumption_services: AssumptionServices | None = None,
) -> tuple[EpisodeRecord, CqCalibrationChildReport]:
    """Extract the outbreak exposure duration."""

    return _run_evidence_child_episode(
        EXPOSURE_TIME_SPEC,
        llm,
        sources,
        admit_assumption=admit_assumption,
        target_context=target_context,
        assumption_services=assumption_services,
    )


def _run_admit_behavior_assumptions_episode(
    llm: LLMExtractCallable | None,
    sources: Iterable[EvidenceSource],
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
    assumption_services: AssumptionServices | None = None,
) -> tuple[EpisodeRecord, CqCalibrationChildReport]:
    """Extract breathing and mask assumptions."""

    return _run_evidence_child_episode(
        BEHAVIOR_SPEC,
        llm,
        sources,
        admit_assumption=admit_assumption,
        target_context=target_context,
        assumption_services=assumption_services,
    )


def _run_admit_population_or_variant_scaling_episode(
    llm: LLMExtractCallable | None,
    sources: Iterable[EvidenceSource],
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
    assumption_services: AssumptionServices | None = None,
) -> tuple[EpisodeRecord, CqCalibrationChildReport]:
    """Extract relative transmissibility scaling."""

    return _run_evidence_child_episode(
        TRANSMISSIBILITY_SPEC,
        llm,
        sources,
        admit_assumption=admit_assumption,
        target_context=target_context,
        assumption_services=assumption_services,
    )


def _run_evidence_child_episode(
    spec: EvidenceChildSpec,
    llm: LLMExtractCallable | None,
    sources: Iterable[EvidenceSource],
    *,
    admit_assumption: bool = False,
    target_context: Mapping[str, Any] | None = None,
    assumption_services: AssumptionServices | None = None,
) -> tuple[EpisodeRecord, CqCalibrationChildReport]:
    """Run one LLM-backed evidence child over retrieved sources."""

    source_tuple = tuple(sources)
    investigation = None
    if admit_assumption:
        services = assumption_services or AssumptionServices()
        investigation = AssumptionInvestigation(
            fields=ASSUMPTION_FIELDS[spec.child], context=target_context or {},
            llm=llm, sources=source_tuple, search=services.search,
            retrieve=services.retrieve, search_limit=services.search_limit,
            limits=services.limits,
        )
    grain = Grain(
        name=spec.child,
        unit="retrieved outbreak source",
        result="CqCalibrationChildReport",
        controller=lambda _path: EvidenceChildController(
            child=spec.child,
            source_count=len(source_tuple) + int(investigation is not None),
        ),
    )
    episode = Episode(
        grain=grain,
        key=spec.child,
        source=EvidenceAdmissionSource(spec, llm, source_tuple, admit_assumption,
                                       target_context, investigation),
        request=EpisodeRequest(
            goal=EpisodeGoal.root(
                objective={"operation": spec.child},
                result_contract={
                    "result": "CqCalibrationChildReport",
                    "output": _child(spec.child).output,
                },
            ),
            message=EvidenceChildRequest(
                spec.child,
                tuple(source.as_record() for source in source_tuple),
            ),
        ),
        build_result=_build_result,
    )
    children = {} if investigation is None else {
        grain: (investigation.grain,), **investigation.children,
    }
    tree = EpisodeTree(root=grain, children=children)
    episode_record = episode.run(Context(tree=tree))
    return episode_record, episode_record.controller_state.to_report()


ASSUMPTION_FIELDS = {
    "build_removal_capacity": ("removal_capacity_m3_s",),
    "admit_exposure_time": ("exposure_time_s",),
    "admit_behavior_assumptions": ("breathing_flow_m3_s", "mask_penetration"),
    "admit_population_or_variant_scaling": ("relative_transmissibility",),
}


class EvidenceAdmissionSource:
    """Preserve source admission; attach justification only for absent evidence."""

    def __init__(self, spec, llm, sources, admit_assumption, context, investigation):
        self.spec, self.llm, self.sources = spec, llm, sources
        self.admit_assumption, self.context = admit_assumption, context
        self.investigation = investigation
        self.index = 0

    def next(self, view):
        if self.index < len(self.sources):
            source = self.sources[self.index]
            self.index += 1
            return Leaf(source, lambda value: _run_source(
                self.spec, self.llm, value, admit_assumption=self.admit_assumption,
                target_context=self.context,
            ), lambda _, observation: observation, source.source_ref)
        if self.investigation is None or self.index > len(self.sources):
            return None
        self.index += 1
        previous = view.controller_state.to_report()
        if previous.passed or not previous.errors or any(
            error.code not in {"missing_extracted_field", "missing_candidate_sources", "missing_assumption_basis"}
            for error in previous.errors
        ):
            return None
        contract = branch_contract(self.investigation.fields, self.context)
        child = self.investigation.open(view, contract)
        return ChildEpisodeUnit(child, self.receive, assumption_summary)

    def receive(self, result, completion, request):
        child = _child(self.spec.child)
        current = latest_assessments(result)
        admitted = {field: current[field] for field in self.investigation.fields
                    if field in current and current[field]["reasonableness"] > 0}
        errors = []
        for field in self.investigation.fields:
            if field not in admitted:
                errors.append(CalculationError("unjustified_assumption", f"No currently supported assumption for {field}", field))
            elif field != "mask_penetration" and admitted[field]["value"] <= 0:
                errors.append(CalculationError("invalid_assumption", f"{field} must be positive", field))
        outputs = {
            "assumption_investigation": result.as_record(),
            "assumption_episode": self.investigation.record.as_record(),
            "assumption_documents": dict(self.investigation.documents),
            "admitted_assumptions": {
                field: {**a, "kind": "inferred_assumption", "normalized_value": a["value"],
                        "normalized_unit": a["unit"]} for field, a in admitted.items()
            },
            "source_refs": tuple(dict.fromkeys(
                evidence["source"] for a in admitted.values()
                for evidence in a["evidence"].values() if evidence.get("source")
            )),
            "evidence_basis": "reviewed_conditional_inference_not_source_extraction",
        }
        if errors:
            report = failed_evidence_child(self.spec.child, tuple(errors), outputs=outputs)
        else:
            values = {field: a["value"] for field, a in admitted.items()}
            outputs.update(values)
            if self.spec.child == "admit_behavior_assumptions":
                outputs["behavior_assumptions"] = values
            quality = min(a["reasonableness"] for a in admitted.values())
            report = CqCalibrationChildReport(
                child=child.name, output=child.output, passed=True,
                score=child.weight * quality, max_score=child.weight, should_yield=True,
                credited_criteria=("reviewed_assumption_support",), outputs=outputs,
                rarefaction=self.investigation.record.controller_state.rarefaction,
            )
        return EvidenceChildObservation(self.spec.child, "assumption_investigation", None, report)


def build_removal_capacity_from_extraction(
    extraction: ExtractionReport,
) -> CqCalibrationChildReport:
    """Build a removal-capacity child from equivalent or room fields."""

    if "total_removal_rate_h" in extraction.values:
        return _build_total_removal_capacity(extraction)

    value, errors = _single_value(
        extraction,
        "removal_capacity_m3_s",
        "m3/s",
        positive=True,
    )
    if not errors and value is not None:
        removal_capacity, normalization_errors, identities = (
            _validate_flow_normalization(value, "removal_capacity_m3_s")
        )
        if normalization_errors:
            return failed_evidence_child(
                "build_removal_capacity",
                normalization_errors,
            )
        return _score_passed(
            "build_removal_capacity",
            {
                "room_geometry": {
                    "basis": "equivalent_virus_free_air_delivery",
                    "source_refs": _source_refs(extraction),
                },
                "air_purification_rate_s": {
                    "basis": "equivalent_virus_free_air_delivery",
                },
                "settling_speed_m_s": {
                    "basis": "included_in_equivalent_removal_capacity",
                },
                "evidence_basis": "equivalent_virus_free_air_delivery",
                "normalized_room_fields": {
                    "removal_capacity_m3_s": removal_capacity,
                },
                "room_field_consistency": True,
                "removal_capacity_m3_s": removal_capacity,
                "source_refs": _source_refs(extraction),
                "extracted_values": _extracted_values(extraction),
                **_admission_outputs(extraction),
            },
            identities=identities,
        )

    detailed = _build_detailed_removal_capacity(extraction)
    if detailed is not None:
        return detailed

    return failed_evidence_child("build_removal_capacity", errors)


def _build_total_removal_capacity(
    extraction: ExtractionReport,
) -> CqCalibrationChildReport:
    volume, volume_errors = _single_value(
        extraction, "room_volume_m3", "m3", positive=True,
    )
    rate, rate_errors = _single_value(
        extraction, "total_removal_rate_h", "1/h", positive=True,
    )
    if volume_errors or rate_errors:
        return failed_evidence_child("build_removal_capacity", volume_errors + rate_errors)
    volume_m3, errors = _validate_identity_normalization(
        volume, "room_volume_m3", {"m3", "cubicmeters", "cubicmetres"}, "m3",
    )
    rate_h, rate_errors, identities = _validate_per_hour_normalization(
        rate, "total_removal_rate_h",
    )
    if errors or rate_errors:
        return failed_evidence_child("build_removal_capacity", errors + rate_errors)
    converted = run_named_calculation("per_hour_to_per_second", {"rate_h": rate_h})
    if not converted.passed:
        return failed_evidence_child("build_removal_capacity", converted.errors)
    rate_s = converted.outputs["rate_s"]
    calculation = run_named_calculation("removal_capacity_from_total_loss_rate", {
        "room_volume_m3": volume_m3, "total_removal_rate_s": rate_s,
    })
    if not calculation.passed:
        return failed_evidence_child("build_removal_capacity", calculation.errors)
    return _score_passed(
        "build_removal_capacity",
        {
            "room_geometry": {"room_volume_m3": volume_m3, "source_refs": _source_refs(extraction)},
            "air_purification_rate_s": {"basis": "included_in_reported_total_loss"},
            "settling_speed_m_s": {"basis": "included_in_reported_total_loss"},
            "evidence_basis": "reported_total_removal_rate_times_volume",
            "normalized_room_fields": {"room_volume_m3": volume_m3, "total_removal_rate_s": rate_s},
            "room_field_consistency": True,
            **calculation.outputs,
            "source_refs": _source_refs(extraction),
            "extracted_values": _extracted_values(extraction),
            **_admission_outputs(extraction),
        },
        identities=identities + converted.identities + calculation.identities,
    )


def _build_detailed_removal_capacity(
    extraction: ExtractionReport,
) -> CqCalibrationChildReport | None:
    floor_area, floor_errors = _single_value(
        extraction,
        "floor_area_m2",
        "m2",
        positive=True,
    )
    height, height_errors = _optional_single_value(
        extraction,
        "room_height_m",
        "m",
        positive=True,
    )
    room_volume_value, room_volume_errors = _optional_single_value(
        extraction,
        "room_volume_m3",
        "m3",
        positive=True,
    )
    air_exchange, air_exchange_errors = _optional_single_value(
        extraction,
        "air_exchange_rate_h",
        "1/h",
    )
    ventilation_flow, ventilation_flow_errors = _optional_single_value(
        extraction,
        "ventilation_flow_m3_s",
        "m3/s",
    )
    deactivation, deactivation_errors = _single_value(
        extraction,
        "deactivation_rate_h",
        "1/h",
    )
    recirculation, recirculation_errors = _single_value(
        extraction,
        "recirculation_flow_m3_s",
        "m3/s",
    )
    filter_efficiency, filter_efficiency_errors = _single_value(
        extraction,
        "filter_efficiency",
        "fraction",
    )
    settling, settling_errors = _single_value(
        extraction,
        "settling_speed_m_h",
        "m/h",
    )
    errors = (
        floor_errors
        + height_errors
        + room_volume_errors
        + air_exchange_errors
        + ventilation_flow_errors
        + recirculation_errors
        + filter_efficiency_errors
        + deactivation_errors
        + settling_errors
    )
    if air_exchange is None and ventilation_flow is None:
        errors += (
            CalculationError(
                code="missing_extracted_field",
                field="air_exchange_rate_h",
                message=(
                    "air_exchange_rate_h or ventilation_flow_m3_s was not "
                    "extracted"
                ),
            ),
        )
    if (
        filter_efficiency is not None
        and not 0.0 <= filter_efficiency.normalized_value <= 1.0
    ):
        errors += (
            CalculationError(
                code="invalid_extracted_value",
                field="filter_efficiency",
                message="filter_efficiency must be in [0, 1]",
            ),
        )
    if errors:
        if extraction.passed:
            return failed_evidence_child("build_removal_capacity", errors)
        return None

    filter_fraction, filter_errors = _validate_fraction_normalization(
        filter_efficiency,
        "filter_efficiency",
    )
    if filter_errors:
        return failed_evidence_child("build_removal_capacity", filter_errors)

    recirculation_flow, flow_errors, flow_identities = (
        _validate_flow_normalization(
            recirculation,
            "recirculation_flow_m3_s",
        )
    )
    if flow_errors:
        return failed_evidence_child("build_removal_capacity", flow_errors)

    ventilation_flow_m3_s: float | None = None
    ventilation_flow_identities: tuple[str, ...] = ()
    if ventilation_flow is not None:
        (
            ventilation_flow_m3_s,
            ventilation_flow_errors,
            ventilation_flow_identities,
        ) = _validate_flow_normalization(
            ventilation_flow,
            "ventilation_flow_m3_s",
        )
        if ventilation_flow_errors:
            return failed_evidence_child(
                "build_removal_capacity",
                ventilation_flow_errors,
            )

    floor_area_m2, floor_area_unit_errors = _validate_identity_normalization(
        floor_area,
        "floor_area_m2",
        {
            "m2",
            "sqm",
            "sqmeter",
            "sqmeters",
            "squaremeter",
            "squaremeters",
            "sqmetre",
            "sqmetres",
            "squaremetre",
            "squaremetres",
        },
        "m2",
    )
    height_m, height_unit_errors = _validate_identity_normalization(
        height,
        "room_height_m",
        {"m", "meter", "meters", "metre", "metres"},
        "m",
    ) if height is not None else (None, ())
    room_volume_m3, room_volume_unit_errors = _validate_identity_normalization(
        room_volume_value,
        "room_volume_m3",
        {
            "m3",
            "cumeter",
            "cumeters",
            "cubicmeter",
            "cubicmeters",
            "cubicmetre",
            "cubicmetres",
        },
        "m3",
    ) if room_volume_value is not None else (None, ())
    air_exchange_h: float | None = None
    air_exchange_unit_errors: tuple[CalculationError, ...] = ()
    air_exchange_identities: tuple[str, ...] = ()
    if air_exchange is not None:
        air_exchange_h, air_exchange_unit_errors, air_exchange_identities = (
            _validate_per_hour_normalization(
                air_exchange,
                "air_exchange_rate_h",
            )
        )
    deactivation_h, deactivation_unit_errors, deactivation_identities = (
        _validate_per_hour_normalization(
            deactivation,
            "deactivation_rate_h",
        )
    )
    settling_m_h, settling_unit_errors, settling_identities = (
        _validate_speed_per_hour_normalization(
            settling,
            "settling_speed_m_h",
        )
    )
    unit_errors = (
        floor_area_unit_errors
        + height_unit_errors
        + room_volume_unit_errors
        + air_exchange_unit_errors
        + deactivation_unit_errors
        + settling_unit_errors
    )
    if unit_errors:
        return failed_evidence_child("build_removal_capacity", unit_errors)

    (
        room_volume,
        room_height_m,
        geometry_identities,
        geometry_errors,
    ) = _resolve_room_geometry(floor_area_m2, height_m, room_volume_m3)
    if geometry_errors:
        return failed_evidence_child("build_removal_capacity", geometry_errors)
    if air_exchange_h is not None:
        air_exchange_s = run_named_calculation(
            "per_hour_to_per_second",
            {"rate_h": air_exchange_h},
        )
    else:
        air_exchange_s = run_named_calculation(
            "air_exchange_rate",
            {
                "outdoor_air_flow_m3_s": ventilation_flow_m3_s,
                "room_volume_m3": room_volume.outputs["room_volume_m3"],
            },
        )
    deactivation_s = run_named_calculation(
        "per_hour_to_per_second",
        {"rate_h": deactivation_h},
    )
    settling_s = run_named_calculation(
        "meters_per_hour_to_meters_per_second",
        {"speed_m_h": settling_m_h},
    )
    if not air_exchange_s.passed:
        return failed_evidence_child(
            "build_removal_capacity",
            air_exchange_s.errors,
        )
    if not deactivation_s.passed:
        return failed_evidence_child(
            "build_removal_capacity",
            deactivation_s.errors,
        )
    if not settling_s.passed:
        return failed_evidence_child(
            "build_removal_capacity",
            settling_s.errors,
        )

    filtration = run_named_calculation(
        "filtration_loss_rate",
        {
            "recirculation_flow_m3_s": recirculation_flow,
            "filter_efficiency": filter_fraction,
            "room_volume_m3": room_volume.outputs["room_volume_m3"],
        },
    )
    if not filtration.passed:
        return failed_evidence_child(
            "build_removal_capacity",
            filtration.errors,
        )

    sedimentation = run_named_calculation(
        "sedimentation_rate",
        {
            "settling_speed_m_s": settling_s.outputs["speed_m_s"],
            "room_height_m": room_height_m,
        },
    )
    if not sedimentation.passed:
        return failed_evidence_child(
            "build_removal_capacity",
            sedimentation.errors,
        )

    purification = run_named_calculation(
        "air_purification_rate",
        {
            "air_exchange_rate_s": _air_exchange_rate_s(air_exchange_s),
            "filtration_rate_s": filtration.outputs["filtration_rate_s"],
            "deactivation_rate_s": deactivation_s.outputs["rate_s"],
        },
    )
    if not purification.passed:
        return failed_evidence_child(
            "build_removal_capacity",
            purification.errors,
        )

    total_removal = run_named_calculation(
        "total_removal_rate",
        {
            "air_exchange_rate_s": _air_exchange_rate_s(air_exchange_s),
            "filtration_rate_s": filtration.outputs["filtration_rate_s"],
            "sedimentation_rate_s": sedimentation.outputs[
                "sedimentation_rate_s"
            ],
            "deactivation_rate_s": deactivation_s.outputs["rate_s"],
        },
    )
    if not total_removal.passed:
        return failed_evidence_child(
            "build_removal_capacity",
            total_removal.errors,
        )

    capacity = run_named_calculation(
        "removal_capacity",
        {
            "room_volume_m3": room_volume.outputs["room_volume_m3"],
            "air_purification_rate_s": purification.outputs[
                "air_purification_rate_s"
            ],
            "settling_speed_m_s": settling_s.outputs["speed_m_s"],
            "floor_area_m2": floor_area_m2,
        },
    )
    if not capacity.passed:
        return failed_evidence_child("build_removal_capacity", capacity.errors)

    identities = (
        geometry_identities
        + air_exchange_identities
        + air_exchange_s.identities
        + ventilation_flow_identities
        + deactivation_identities
        + deactivation_s.identities
        + settling_identities
        + settling_s.identities
        + flow_identities
        + filtration.identities
        + sedimentation.identities
        + purification.identities
        + total_removal.identities
        + capacity.identities
    )
    normalized_room_fields: dict[str, float] = {
        "floor_area_m2": floor_area_m2,
        "room_height_m": room_height_m,
        "room_volume_m3": room_volume.outputs["room_volume_m3"],
        "air_exchange_rate_s": _air_exchange_rate_s(air_exchange_s),
        "recirculation_flow_m3_s": recirculation_flow,
        "filter_efficiency": filter_fraction,
        "deactivation_rate_s": deactivation_s.outputs["rate_s"],
        "deactivation_rate_h": deactivation_h,
        "settling_speed_m_s": settling_s.outputs["speed_m_s"],
        "settling_speed_m_h": settling_m_h,
        "sedimentation_rate_s": sedimentation.outputs[
            "sedimentation_rate_s"
        ],
        "removal_rate_s": total_removal.outputs["removal_rate_s"],
    }
    if air_exchange_h is not None:
        normalized_room_fields["air_exchange_rate_h"] = air_exchange_h
    if ventilation_flow_m3_s is not None:
        normalized_room_fields["ventilation_flow_m3_s"] = (
            ventilation_flow_m3_s
        )

    return _score_passed(
        "build_removal_capacity",
        {
            "room_geometry": {
                "basis": (
                    "room_area_height_and_loss_rates"
                    if height is not None
                    else "room_area_volume_and_loss_rates"
                ),
                "floor_area_m2": floor_area_m2,
                "room_height_m": room_height_m,
                "room_volume_m3": room_volume.outputs["room_volume_m3"],
                "source_refs": _source_refs(extraction),
            },
            "air_purification_rate_s": purification.outputs[
                "air_purification_rate_s"
            ],
            "filtration_rate_s": filtration.outputs["filtration_rate_s"],
            "settling_speed_m_s": settling_s.outputs["speed_m_s"],
            "evidence_basis": "room_geometry_and_loss_rates",
            "normalized_room_fields": normalized_room_fields,
            "room_field_consistency": True,
            "removal_capacity_m3_s": capacity.outputs[
                "removal_capacity_m3_s"
            ],
            "source_refs": _source_refs(extraction),
            "extracted_values": _extracted_values(extraction),
            **_admission_outputs(extraction),
        },
        identities=identities,
    )


def admit_exposure_time_from_extraction(
    extraction: ExtractionReport,
) -> CqCalibrationChildReport:
    """Build an exposure-time child from a passed extraction."""

    value, errors = _single_value(
        extraction,
        "exposure_time_s",
        "s",
        positive=True,
    )
    if errors:
        return failed_evidence_child("admit_exposure_time", errors)

    normalized_value, normalization_errors, identities = (
        _validate_duration_normalization(value)
    )
    if normalization_errors:
        return failed_evidence_child(
            "admit_exposure_time",
            normalization_errors,
        )

    return _score_passed(
        "admit_exposure_time",
        {
            "raw_exposure_time": value.value,
            "exposure_time_s": normalized_value,
            "time_scope": {
                "basis": "llm_extracted_outbreak_duration",
                "source_refs": _source_refs(extraction),
            },
            "source_refs": _source_refs(extraction),
            "extracted_values": _extracted_values(extraction),
            **_admission_outputs(extraction),
        },
        identities=identities,
    )


def admit_behavior_assumptions_from_extraction(
    extraction: ExtractionReport,
) -> CqCalibrationChildReport:
    """Build a behavior child from breathing and mask extractions."""

    if not extraction.passed:
        return failed_evidence_child(
            "admit_behavior_assumptions",
            extraction.errors,
        )

    breathing, breathing_errors = _single_value(
        extraction,
        "breathing_flow_m3_s",
        "m3/s",
        positive=True,
    )
    mask, mask_errors = _optional_single_value(
        extraction,
        "mask_penetration",
        "fraction",
    )
    mask_filter, mask_filter_errors = _optional_single_value(
        extraction,
        "mask_filter_efficiency",
        "fraction",
    )
    unmasked, unmasked_errors = _optional_single_value(
        extraction,
        "unmasked_indicator",
        "scalar",
    )
    errors = (
        breathing_errors
        + mask_errors
        + mask_filter_errors
        + unmasked_errors
    )
    if mask is None and mask_filter is None and unmasked is None:
        errors += (
            CalculationError(
                code="missing_extracted_field",
                field="mask_penetration",
                message=(
                    "mask_penetration, mask_filter_efficiency, or "
                    "unmasked_indicator was not extracted"
                ),
            ),
        )
    if mask is not None and not 0.0 < mask.normalized_value <= 1.0:
        errors += (
            CalculationError(
                code="invalid_extracted_value",
                field="mask_penetration",
                message="mask_penetration must be in (0, 1]",
            ),
        )
    if (
        mask_filter is not None
        and not 0.0 <= mask_filter.normalized_value < 1.0
    ):
        errors += (
            CalculationError(
                code="invalid_extracted_value",
                field="mask_filter_efficiency",
                message="mask_filter_efficiency must be in [0, 1)",
            ),
        )
    if errors:
        return failed_evidence_child("admit_behavior_assumptions", errors)

    mask_identities: tuple[str, ...] = ()
    mask_fraction = 0.0
    if mask is not None:
        mask_fraction, mask_errors = _validate_fraction_normalization(
            mask,
            "mask_penetration",
        )
        if mask_errors:
            return failed_evidence_child(
                "admit_behavior_assumptions",
                mask_errors,
            )
    if mask_filter is not None:
        filter_fraction, mask_filter_errors = _validate_fraction_normalization(
            mask_filter,
            "mask_filter_efficiency",
        )
        if mask_filter_errors:
            return failed_evidence_child(
                "admit_behavior_assumptions",
                mask_filter_errors,
            )

        inverted_fraction = 1.0 - filter_fraction
        if mask is not None and not _close(mask_fraction, inverted_fraction):
            return failed_evidence_child(
                "admit_behavior_assumptions",
                (
                    CalculationError(
                        code="inconsistent_extracted_value",
                        field="mask_penetration",
                        message=(
                            "mask_penetration must equal "
                            "1 - mask_filter_efficiency"
                        ),
                    ),
                ),
            )
        if mask is None:
            mask_fraction = inverted_fraction
            mask_identities = (
                "mask_penetration = 1 - mask_filter_efficiency",
            )
    if unmasked is not None:
        unmasked_indicator, unmasked_unit_errors = (
            _validate_scalar_normalization(
                unmasked,
                "unmasked_indicator",
            )
        )
        if unmasked_unit_errors:
            return failed_evidence_child(
                "admit_behavior_assumptions",
                unmasked_unit_errors,
            )
        if not _close(unmasked_indicator, 1.0):
            return failed_evidence_child(
                "admit_behavior_assumptions",
                (
                    CalculationError(
                        code="invalid_extracted_value",
                        field="unmasked_indicator",
                        message=(
                            "unmasked_indicator must be 1 when the source "
                            "explicitly supports absent mask use"
                        ),
                    ),
                ),
            )
        if not _close(mask_fraction, 0.0) and not _close(
            mask_fraction,
            1.0,
        ):
            return failed_evidence_child(
                "admit_behavior_assumptions",
                (
                    CalculationError(
                        code="inconsistent_extracted_value",
                        field="mask_penetration",
                        message=(
                            "unmasked_indicator requires mask_penetration "
                            "to equal 1"
                        ),
                    ),
                ),
            )
        mask_fraction = 1.0
        mask_identities += (
            "mask_penetration = 1 when unmasked_indicator = 1",
        )

    breathing_flow, flow_errors, identities = _validate_flow_normalization(
        breathing,
        "breathing_flow_m3_s",
    )
    if flow_errors:
        return failed_evidence_child("admit_behavior_assumptions", flow_errors)

    return _score_passed(
        "admit_behavior_assumptions",
        {
            "behavior_assumptions": {
                "basis": "llm_extracted_behavior",
                "breathing_flow_m3_s": breathing_flow,
                "mask_penetration": mask_fraction,
                "source_refs": _source_refs(extraction),
            },
            "breathing_flow_m3_s": breathing_flow,
            "mask_penetration": mask_fraction,
            "source_refs": _source_refs(extraction),
            "extracted_values": _extracted_values(extraction),
            **_admission_outputs(extraction),
        },
        identities=identities + mask_identities,
    )


def admit_transmissibility_from_extraction(
    extraction: ExtractionReport,
) -> CqCalibrationChildReport:
    """Build a transmissibility-scaling child from a passed extraction."""

    value, errors = _single_value(
        extraction,
        "relative_transmissibility",
        "scalar",
        positive=True,
    )
    if errors:
        return failed_evidence_child(
            "admit_population_or_variant_scaling",
            errors,
        )

    scalar, scalar_errors = _validate_scalar_normalization(
        value,
        "relative_transmissibility",
    )
    if scalar_errors:
        return failed_evidence_child(
            "admit_population_or_variant_scaling",
            scalar_errors,
        )

    return _score_passed(
        "admit_population_or_variant_scaling",
        {
            "scaling_basis": {
                "basis": "llm_extracted_relative_transmissibility",
                "source_refs": _source_refs(extraction),
            },
            "relative_transmissibility": scalar,
            "source_refs": _source_refs(extraction),
            "extracted_values": _extracted_values(extraction),
            **_admission_outputs(extraction),
        },
    )


def admit_neutral_transmissibility_scaling(
    outbreak_report: CqCalibrationChildReport | None = None,
) -> CqCalibrationChildReport:
    """Credit identity transmissibility when no rescaling was requested."""

    provenance_basis = (
        "No population, variant, or cross-disease rescaling was requested; "
        "the target Cq calibration uses the searched outbreak on its own "
        "transmissibility scale."
    )
    return _score_passed(
        "admit_population_or_variant_scaling",
        {
            "scaling_basis": {
                "basis": "neutral_identity_scaling",
                "provenance_basis": provenance_basis,
                **({"failed_outbreak_child": outbreak_report.as_record()} if outbreak_report is not None else {}),
            },
            "relative_transmissibility": 1.0,
            "admitted_assumptions": {
                "relative_transmissibility": {
                    "assumed_field": "relative_transmissibility",
                    "value": 1.0,
                    "unit": "scalar",
                    "normalized_value": 1.0,
                    "normalized_unit": "scalar",
                    "provenance_basis": provenance_basis,
                    "confidence": 1.0,
                    "notes": [
                        "Neutral identity scaling was used because the "
                        "request did not ask to adjust the outbreak to a "
                        "different population, variant, or pathogen."
                    ],
                },
            },
        },
    )


def failed_evidence_child(
    child_name: str,
    errors: tuple[CalculationError, ...],
    *,
    outputs: Mapping[str, Any] | None = None,
) -> CqCalibrationChildReport:
    """Return a yielded failed child report."""

    child = _child(child_name)
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


def _run_source(
    spec: EvidenceChildSpec,
    llm: LLMExtractCallable | None,
    source: EvidenceSource,
    *,
    admit_assumption: bool,
    target_context: Mapping[str, Any] | None,
) -> EvidenceChildObservation:
    extraction = spec.extractor(
        llm,
        source,
        admit_assumption=admit_assumption,
        target_context=target_context,
    )
    report = spec.build_report(extraction)
    return EvidenceChildObservation(
        child=spec.child,
        source_ref=source.source_ref,
        extraction=extraction,
        report=report,
    )


def _score_passed(
    child_name: str,
    outputs: Mapping[str, Any],
    *,
    identities: tuple[str, ...] = (),
) -> CqCalibrationChildReport:
    child = _child(child_name)
    criteria = {
        criterion.name: criterion
        for criterion in CQ_NORMALIZATION_CRITERIA[child_name]
    }
    passed = tuple(criteria)
    rarefaction = cq_normalization_rarefaction(
        (float(len(passed)),),
        max_score=float(len(criteria)),
    )
    return score_cq_normalization_child(
        child,
        passed,
        outputs,
        identities=identities,
        rarefaction=rarefaction,
    )


def _single_value(
    extraction: ExtractionReport,
    field: str,
    normalized_unit: str,
    *,
    positive: bool = False,
) -> tuple[ExtractedValue | None, tuple[CalculationError, ...]]:
    if not extraction.passed:
        return None, extraction.errors

    try:
        value = extraction.values[field]
    except KeyError:
        return (
            None,
            (
                CalculationError(
                    code="missing_extracted_field",
                    field=field,
                    message=f"{field} was not extracted",
                ),
            ),
        )

    if value.normalized_unit != normalized_unit:
        return (
            None,
            (
                CalculationError(
                    code="unsupported_unit",
                    field=field,
                    message=f"{field} must be normalized to {normalized_unit}",
                ),
            ),
        )
    if positive and value.normalized_value <= 0.0:
        return (
            None,
            (
                CalculationError(
                    code="invalid_extracted_value",
                    field=field,
                    message=f"{field} must be positive",
                ),
            ),
        )
    return value, ()


def _optional_single_value(
    extraction: ExtractionReport,
    field: str,
    normalized_unit: str,
    *,
    positive: bool = False,
) -> tuple[ExtractedValue | None, tuple[CalculationError, ...]]:
    if field not in extraction.values:
        return None, ()
    return _single_value(
        extraction,
        field,
        normalized_unit,
        positive=positive,
    )


def _resolve_room_geometry(
    floor_area_m2: float,
    height_m: float | None,
    room_volume_m3: float | None,
):
    if height_m is None and room_volume_m3 is None:
        return (
            None,
            0.0,
            (),
            (
                CalculationError(
                    code="missing_extracted_field",
                    field="room_height_m",
                    message="room_height_m or room_volume_m3 is required",
                ),
            ),
        )

    if height_m is None:
        height = run_named_calculation(
            "room_height",
            {
                "room_volume_m3": room_volume_m3,
                "floor_area_m2": floor_area_m2,
            },
        )
        if not height.passed:
            return None, 0.0, (), height.errors
        height_m = height.outputs["room_height_m"]
        volume = run_named_calculation(
            "room_volume",
            {
                "floor_area_m2": floor_area_m2,
                "room_height_m": height_m,
            },
        )
        if not volume.passed:
            return None, 0.0, (), volume.errors
        return (
            volume,
            height_m,
            height.identities + volume.identities,
            (),
        )

    volume = run_named_calculation(
        "room_volume",
        {
            "floor_area_m2": floor_area_m2,
            "room_height_m": height_m,
        },
    )
    if not volume.passed:
        return None, 0.0, (), volume.errors

    calculated_volume = volume.outputs["room_volume_m3"]
    if room_volume_m3 is not None and not _close(
        room_volume_m3,
        calculated_volume,
    ):
        return (
            None,
            0.0,
            (),
            (
                CalculationError(
                    code="inconsistent_normalized_value",
                    field="room_volume_m3",
                    message=(
                        "room_volume_m3 must match floor_area_m2 * "
                        "room_height_m"
                    ),
                ),
            ),
        )

    return volume, height_m, volume.identities, ()


def _air_exchange_rate_s(report) -> float:
    try:
        return report.outputs["air_exchange_rate_s"]
    except KeyError:
        return report.outputs["rate_s"]


def _validate_duration_normalization(
    value: ExtractedValue,
) -> tuple[float, tuple[CalculationError, ...], tuple[str, ...]]:
    unit = value.unit.strip().lower().rstrip(".")
    conversions = {
        "s": ("seconds", None, "seconds"),
        "sec": ("seconds", None, "seconds"),
        "secs": ("seconds", None, "seconds"),
        "second": ("seconds", None, "seconds"),
        "seconds": ("seconds", None, "seconds"),
        "min": ("minutes_to_seconds", "minutes", "seconds"),
        "mins": ("minutes_to_seconds", "minutes", "seconds"),
        "minute": ("minutes_to_seconds", "minutes", "seconds"),
        "minutes": ("minutes_to_seconds", "minutes", "seconds"),
        "h": ("hours_to_seconds", "hours", "seconds"),
        "hr": ("hours_to_seconds", "hours", "seconds"),
        "hrs": ("hours_to_seconds", "hours", "seconds"),
        "hour": ("hours_to_seconds", "hours", "seconds"),
        "hours": ("hours_to_seconds", "hours", "seconds"),
        "d": ("days_to_seconds", "days", "seconds"),
        "day": ("days_to_seconds", "days", "seconds"),
        "days": ("days_to_seconds", "days", "seconds"),
    }
    try:
        task_name, input_name, output_name = conversions[unit]
    except KeyError:
        return (
            0.0,
            (
                CalculationError(
                    code="unsupported_unit",
                    field="exposure_time_s",
                    message=(
                        "exposure_time_s raw unit must be seconds, minutes, "
                        "hours, or days"
                    ),
                ),
            ),
            (),
        )

    if input_name is None:
        expected = value.value
        identities: tuple[str, ...] = ()
    else:
        calculation = run_named_calculation(
            task_name,
            {input_name: value.value},
        )
        if not calculation.passed:
            return 0.0, calculation.errors, ()
        expected = calculation.outputs[output_name]
        identities = calculation.identities

    if _close(value.normalized_value, expected):
        return expected, (), identities

    return (
        0.0,
        (
            CalculationError(
                code="inconsistent_normalized_value",
                field="exposure_time_s",
                message=(
                    "exposure_time_s normalized_value must match the raw "
                    "duration converted to seconds"
                ),
            ),
        ),
        (),
    )


def _validate_flow_normalization(
    value: ExtractedValue,
    field: str,
) -> tuple[float, tuple[CalculationError, ...], tuple[str, ...]]:
    unit = _unit_key(value.unit)
    if unit in {"m3/s", "m3/sec", "m3/second", "m3/seconds"}:
        expected = value.value
        identities: tuple[str, ...] = ()
    elif unit in {"m3/h", "m3/hr", "m3/hour", "m3/hours"}:
        calculation = run_named_calculation(
            "cubic_meters_per_hour_to_cubic_meters_per_second",
            {"flow_m3_h": value.value},
        )
        if not calculation.passed:
            return 0.0, calculation.errors, ()
        expected = calculation.outputs["flow_m3_s"]
        identities = calculation.identities
    elif unit in {"l/s", "l/sec", "l/second", "l/seconds"}:
        calculation = run_named_calculation(
            "liters_per_second_to_cubic_meters_per_second",
            {"flow_l_s": value.value},
        )
        if not calculation.passed:
            return 0.0, calculation.errors, ()
        expected = calculation.outputs["flow_m3_s"]
        identities = calculation.identities
    elif unit in {"cfm", "ft3/min", "ft3/minute", "m3/min", "m3/minute"}:
        task_name, input_name = (
            ("cubic_meters_per_minute_to_cubic_meters_per_second", "flow_m3_min")
            if unit.startswith("m3")
            else ("cubic_feet_per_minute_to_cubic_meters_per_second", "flow_cfm")
        )
        calculation = run_named_calculation(task_name, {input_name: value.value})
        if not calculation.passed:
            return 0.0, calculation.errors, ()
        expected = calculation.outputs["flow_m3_s"]
        identities = calculation.identities
    else:
        return (
            0.0,
            (
                CalculationError(
                    code="unsupported_unit",
                    field=field,
                    message=(
                        f"{field} raw unit must be m3/s, m3/h, m3/min, L/s, or cfm"
                    ),
                ),
            ),
            (),
        )

    if _close(value.normalized_value, expected):
        return expected, (), identities

    return (
        0.0,
        (
            CalculationError(
                code="inconsistent_normalized_value",
                field=field,
                message=(
                    f"{field} normalized_value must match the raw flow "
                    "converted to m3/s"
                ),
            ),
        ),
        (),
    )


def _validate_per_hour_normalization(
    value: ExtractedValue,
    field: str,
) -> tuple[float, tuple[CalculationError, ...], tuple[str, ...]]:
    unit = _unit_key(value.unit)
    if unit in {
        "1/h",
        "/h",
        "h-1",
        "1/hr",
        "/hr",
        "hr-1",
        "1/hour",
        "/hour",
        "hour-1",
        "1/hours",
        "/hours",
        "hours-1",
        "perh",
        "perhr",
        "perhour",
        "perhours",
    }:
        expected = value.value
        identities: tuple[str, ...] = ()
    elif unit in {
        "1/s",
        "/s",
        "s-1",
        "1/sec",
        "/sec",
        "sec-1",
        "1/second",
        "/second",
        "second-1",
        "1/seconds",
        "/seconds",
        "seconds-1",
        "pers",
        "persec",
        "persecond",
        "perseconds",
    }:
        calculation = run_named_calculation(
            "per_second_to_per_hour",
            {"rate_s": value.value},
        )
        if not calculation.passed:
            return 0.0, calculation.errors, ()
        expected = calculation.outputs["rate_h"]
        identities = calculation.identities
    else:
        return (
            0.0,
            (
                CalculationError(
                    code="unsupported_unit",
                    field=field,
                    message=f"{field} raw unit must be 1/h or 1/s",
                ),
            ),
            (),
        )

    if _close(value.normalized_value, expected):
        return expected, (), identities

    return (
        0.0,
        (
            CalculationError(
                code="inconsistent_normalized_value",
                field=field,
                message=(
                    f"{field} normalized_value must match the raw rate "
                    "converted to 1/h"
                ),
            ),
        ),
        (),
    )


def _validate_speed_per_hour_normalization(
    value: ExtractedValue,
    field: str,
) -> tuple[float, tuple[CalculationError, ...], tuple[str, ...]]:
    unit = _unit_key(value.unit)
    if unit in {"m/h", "m/hr", "m/hour", "m/hours"}:
        expected = value.value
        identities: tuple[str, ...] = ()
    elif unit in {"m/s", "m/sec", "m/second", "m/seconds"}:
        calculation = run_named_calculation(
            "meters_per_second_to_meters_per_hour",
            {"speed_m_s": value.value},
        )
        if not calculation.passed:
            return 0.0, calculation.errors, ()
        expected = calculation.outputs["speed_m_h"]
        identities = calculation.identities
    else:
        return (
            0.0,
            (
                CalculationError(
                    code="unsupported_unit",
                    field=field,
                    message=f"{field} raw unit must be m/h or m/s",
                ),
            ),
            (),
        )

    if _close(value.normalized_value, expected):
        return expected, (), identities

    return (
        0.0,
        (
            CalculationError(
                code="inconsistent_normalized_value",
                field=field,
                message=(
                    f"{field} normalized_value must match the raw speed "
                    "converted to m/h"
                ),
            ),
        ),
        (),
    )


def _validate_fraction_normalization(
    value: ExtractedValue,
    field: str,
) -> tuple[float, tuple[CalculationError, ...]]:
    unit = _unit_key(value.unit)
    if unit in {"fraction", "probability", "scalar", "1", "unitless"}:
        expected = value.value
    elif unit in {"%", "percent", "percentage"}:
        expected = value.value / 100.0
    else:
        return (
            0.0,
            (
                CalculationError(
                    code="unsupported_unit",
                    field=field,
                    message=f"{field} raw unit must be fraction or percent",
                ),
            ),
        )

    if _close(value.normalized_value, expected):
        return expected, ()

    return (
        0.0,
        (
            CalculationError(
                code="inconsistent_normalized_value",
                field=field,
                message=(
                    f"{field} normalized_value must match the raw "
                    "fraction or percent value"
                ),
            ),
        ),
    )


def _validate_identity_normalization(
    value: ExtractedValue,
    field: str,
    units: set[str],
    unit_label: str,
) -> tuple[float, tuple[CalculationError, ...]]:
    if _unit_key(value.unit) not in units:
        return (
            0.0,
            (
                CalculationError(
                    code="unsupported_unit",
                    field=field,
                    message=f"{field} raw unit must be {unit_label}",
                ),
            ),
        )

    if _close(value.normalized_value, value.value):
        return value.value, ()

    return (
        0.0,
        (
            CalculationError(
                code="inconsistent_normalized_value",
                field=field,
                message=(
                    f"{field} normalized_value must match the raw "
                    f"{unit_label} value"
                ),
            ),
        ),
    )


def _validate_scalar_normalization(
    value: ExtractedValue,
    field: str,
) -> tuple[float, tuple[CalculationError, ...]]:
    unit = _unit_key(value.unit)
    if unit not in {
        "scalar",
        "1",
        "unitless",
        "fold",
        "folds",
        "x",
        "times",
    }:
        return (
            0.0,
            (
                CalculationError(
                    code="unsupported_unit",
                    field=field,
                    message=f"{field} raw unit must be a scalar",
                ),
            ),
        )

    if _close(value.normalized_value, value.value):
        return value.value, ()

    return (
        0.0,
        (
            CalculationError(
                code="inconsistent_normalized_value",
                field=field,
                message=(
                    f"{field} normalized_value must match the raw scalar"
                ),
            ),
        ),
    )


def _unit_key(unit: str) -> str:
    return (
        unit.strip()
        .lower()
        .replace(" ", "")
        .replace(".", "")
        .replace("^", "")
        .replace("³", "3")
        .replace("²", "2")
        .replace("⁻", "-")
        .replace("−", "-")
        .replace("¹", "1")
        .replace("liters", "l")
        .replace("liter", "l")
        .replace("litres", "l")
        .replace("litre", "l")
    )


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-9, abs(right) * 1e-9)


def _source_refs(extraction: ExtractionReport) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            value.source_ref
            for value in extraction.values.values()
        )
    )


def _extracted_values(
    extraction: ExtractionReport,
) -> dict[str, Mapping[str, Any]]:
    return {
        name: value.as_record()
        for name, value in extraction.values.items()
    }


def _admission_outputs(extraction: ExtractionReport) -> Mapping[str, Any]:
    review = {"evidence_applicability": dict(extraction.applicability)} if extraction.applicability else {}
    if extraction.span_repair_history:
        review["span_repair_history"] = list(extraction.span_repair_history)
    if not extraction.admission_basis:
        return review

    return {
        **review,
        "admitted_assumptions": {
            name: {
                "assumed_field": name,
                "value": value.value,
                "unit": value.unit,
                "normalized_value": value.normalized_value,
                "normalized_unit": value.normalized_unit,
                "source_ref": value.source_ref,
                "evidence_span": value.evidence_span,
                "confidence": value.confidence,
                "notes": list(value.notes),
                **dict(extraction.admission_basis),
            }
            for name, value in extraction.values.items()
        },
        "admission_basis": dict(extraction.admission_basis),
    }


def _failed_source_reports(
    observations: tuple[EvidenceChildObservation, ...],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(observation.as_record() for observation in observations)


def _child(name: str) -> CqCalibrationChild:
    return next(child for child in CQ_CALIBRATION_CHILDREN if child.name == name)


def _build_result(record: EpisodeRecord) -> EvidenceChildRecord:
    state = record.controller_state
    if not isinstance(state, EvidenceChildEpisodeState):
        raise TypeError("evidence child Episode produced an unexpected state")
    return EvidenceChildRecord(state.to_report())


REMOVAL_CAPACITY_SPEC = EvidenceChildSpec(
    child="build_removal_capacity",
    extractor=extract_removal_capacity,
    build_report=build_removal_capacity_from_extraction,
)
EXPOSURE_TIME_SPEC = EvidenceChildSpec(
    child="admit_exposure_time",
    extractor=extract_exposure_time,
    build_report=admit_exposure_time_from_extraction,
)
BEHAVIOR_SPEC = EvidenceChildSpec(
    child="admit_behavior_assumptions",
    extractor=extract_behavior_assumptions,
    build_report=admit_behavior_assumptions_from_extraction,
)
TRANSMISSIBILITY_SPEC = EvidenceChildSpec(
    child="admit_population_or_variant_scaling",
    extractor=extract_transmissibility_assumption,
    build_report=admit_transmissibility_from_extraction,
)


__all__ = [
    "BEHAVIOR_SPEC",
    "EXPOSURE_TIME_SPEC",
    "EvidenceChildController",
    "EvidenceChildEpisodeState",
    "EvidenceChildEpisodeStep",
    "EvidenceChildObservation",
    "EvidenceChildRecord",
    "EvidenceChildRequest",
    "EvidenceChildSpec",
    "REMOVAL_CAPACITY_SPEC",
    "TRANSMISSIBILITY_SPEC",
    "admit_behavior_assumptions_from_extraction",
    "admit_exposure_time_from_extraction",
    "admit_neutral_transmissibility_scaling",
    "admit_transmissibility_from_extraction",
    "build_removal_capacity_from_extraction",
    "failed_evidence_child",
]
