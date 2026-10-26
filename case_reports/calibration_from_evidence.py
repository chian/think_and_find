"""Searched and LLM-extracted Cq calibration episodes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from episodic_workflow.credit_assignment.cq_calibration import (
    CQ_CALIBRATION_CHILDREN,
    CqCalibrationChild,
    CqCalibrationChildReport,
    CqCalibrationCreditReport,
    credit_cq_calibration_child,
    sum_cq_calibration_credit,
)
from episodic_workflow.credit_assignment.calculations import run_named_calculation
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
from episodic_workflow.task_executors.outbreak_search import (
    RetrieveCallable,
    SearchCallable,
    SettingHints,
    setting_hints_text,
)

from .attack_probability_from_evidence import (
    _run_build_attack_probability_episode,
    failed_build_attack_probability,
)
from .assumption_search import (
    AssumptionSearchReport,
    _run_find_assumption_sources_episode,
    assumption_search_specs,
)
from .assumption_investigation import AssumptionServices
from .evidence_extraction import (
    EvidenceSource,
    LLMExtractCallable,
    evidence_source_from_candidate,
)
from .evidence_children import (
    _run_admit_behavior_assumptions_episode,
    _run_admit_exposure_time_episode,
    _run_admit_population_or_variant_scaling_episode,
    _run_build_removal_capacity_episode,
    admit_neutral_transmissibility_scaling,
    failed_evidence_child,
)
CALIBRATE_CQ_FROM_OUTBREAK_CHILDREN = (
    "build_attack_probability",
    "build_removal_capacity",
    "admit_exposure_time",
    "admit_behavior_assumptions",
    "admit_population_or_variant_scaling",
    "quanta_concentration_from_attack_probability",
)


@dataclass(frozen=True)
class CqEvidenceObservation:
    """One child report from disease-driven Cq calibration."""

    child: str
    report: CqCalibrationChildReport

    @property
    def passed(self) -> bool:
        return self.report.passed

    def as_record(self) -> dict[str, Any]:
        return {
            "child": self.child,
            "passed": self.passed,
            "report": self.report.as_record(),
        }


@dataclass(frozen=True)
class CqEvidenceEpisodeStep:
    """Controller step for disease-driven Cq calibration."""

    stop: bool
    requests_transition: bool = False

    def as_record(self) -> dict[str, bool]:
        return {
            "stop": self.stop,
            "requests_transition": self.requests_transition,
        }


@dataclass(frozen=True)
class CqEvidenceEpisodeState:
    """Accumulated state for a disease-driven Cq calibration attempt."""

    disease_name: str
    setting: str
    child_count: int
    children: tuple[str, ...] = CALIBRATE_CQ_FROM_OUTBREAK_CHILDREN
    observations: tuple[CqEvidenceObservation, ...] = ()
    stop: bool = False

    def to_report(self) -> CqCalibrationCreditReport:
        report = sum_cq_calibration_credit(
            (observation.report for observation in self.observations),
            children=_cq_children(self.children),
        )
        return replace(
            report,
            input_provenance=_runtime_search_input_provenance(
                report,
                disease_name=self.disease_name,
                setting=self.setting,
            ),
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "disease_name": self.disease_name,
            "setting": self.setting,
            "child_count": self.child_count,
            "children": list(self.children),
            "stop": self.stop,
            "observations": [
                observation.as_record()
                for observation in self.observations
            ],
            "report": self.to_report().as_record(),
        }


@dataclass(frozen=True)
class CqEvidenceRequest(ClosedRecord):
    """Request for searched Cq calibration from a disease name."""

    disease_name: str
    setting: str = ""
    search_limit: int = 10
    retrieval_enabled: bool = False
    require_transmissibility_evidence: bool = False

    def as_record(self) -> Mapping[str, Any]:
        return {
            "disease_name": self.disease_name,
            "setting": self.setting,
            "search_limit": self.search_limit,
            "retrieval_enabled": self.retrieval_enabled,
            "require_transmissibility_evidence": (
                self.require_transmissibility_evidence
            ),
        }


@dataclass(frozen=True)
class CqEvidenceRecord(ClosedRecord):
    """Closed result of disease-driven Cq calibration."""

    report: CqCalibrationCreditReport

    def as_record(self) -> Mapping[str, Any]:
        return self.report.as_record()

    def audit_identifiers(self) -> tuple[str, ...]:
        return self.report.identities


def _runtime_search_input_provenance(
    report: CqCalibrationCreditReport,
    *,
    disease_name: str,
    setting: str,
) -> dict[str, Any]:
    return {
        "input_mode": "runtime_search_llm_extraction",
        "source_backed": bool(report.source_refs),
        "search_backed": True,
        "llm_extracted": any(
            "extracted_values" in child_outputs
            for child_outputs in report.outputs.values()
        ),
        "disease_name": disease_name,
        "setting": setting,
        "source_refs": list(report.source_refs),
    }


class CqEvidenceController:
    """Stop when planned searched/extraction children have reported."""

    epoch = "cq_evidence_calibration_v1"

    def __init__(
        self,
        *,
        disease_name: str,
        setting: str,
        child_count: int,
        children: tuple[str, ...] = CALIBRATE_CQ_FROM_OUTBREAK_CHILDREN,
        observations: tuple[CqEvidenceObservation, ...] = (),
        stop: bool = False,
    ) -> None:
        self._disease_name = disease_name
        self._setting = setting
        self._child_count = child_count
        self._children = children
        self._observations = observations
        self._stop = stop

    def observe(
        self,
        unit_label: str,
        value: object,
        *,
        is_root: bool,
    ) -> CqEvidenceEpisodeStep:
        if not isinstance(value, CqEvidenceObservation):
            raise TypeError("CqEvidenceController expects CqEvidenceObservation")

        observations = self._observations + (value,)
        stop = len(observations) >= self._child_count
        self._observations = observations
        self._stop = stop
        return CqEvidenceEpisodeStep(stop=stop)

    def state(self) -> CqEvidenceEpisodeState:
        return CqEvidenceEpisodeState(
            disease_name=self._disease_name,
            setting=self._setting,
            child_count=self._child_count,
            children=self._children,
            observations=self._observations,
            stop=self._stop,
        )

    def transitioned(self, epoch: str) -> "CqEvidenceController":
        return CqEvidenceController(
            disease_name=self._disease_name,
            setting=self._setting,
            child_count=self._child_count,
            children=self._children,
            observations=self._observations,
            stop=self._stop,
        )


def _run_calibrate_cq_from_outbreak_with_search_report_episode(
    *,
    disease_name: str,
    outbreak_search_report: CqCalibrationChildReport,
    search: SearchCallable,
    llm: LLMExtractCallable | None,
    retrieve: RetrieveCallable | None = None,
    setting: SettingHints = "",
    search_limit: int = 10,
    require_transmissibility_evidence: bool = False,
) -> tuple[EpisodeRecord, CqCalibrationCreditReport]:
    """Calibrate Cq after `find_calibration_outbreaks` has searched."""

    return _run_cq_evidence_episode(
        operation="calibrate_cq_from_outbreak",
        disease_name=disease_name,
        search=search,
        llm=llm,
        retrieve=retrieve,
        setting=setting,
        search_limit=search_limit,
        require_transmissibility_evidence=require_transmissibility_evidence,
        children=CALIBRATE_CQ_FROM_OUTBREAK_CHILDREN,
        observations={
            "find_candidate_outbreak_sources": CqEvidenceObservation(
                child="find_candidate_outbreak_sources",
                report=outbreak_search_report,
            )
        },
    )


def _run_cq_evidence_episode(
    *,
    operation: str,
    disease_name: str,
    search: SearchCallable,
    llm: LLMExtractCallable | None,
    retrieve: RetrieveCallable | None,
    setting: SettingHints,
    search_limit: int,
    require_transmissibility_evidence: bool,
    children: tuple[str, ...],
    observations: dict[str, CqEvidenceObservation],
) -> tuple[EpisodeRecord, CqCalibrationCreditReport]:
    """Run the Cq evidence loop with a caller-selected child set."""

    setting_text = setting_hints_text(setting)
    request = CqEvidenceRequest(
        disease_name=disease_name,
        setting=setting_text,
        search_limit=search_limit,
        retrieval_enabled=retrieve is not None,
        require_transmissibility_evidence=require_transmissibility_evidence,
    )
    grain = Grain(
        name=operation,
        unit="Cq calibration child",
        result="CqCalibrationCreditReport",
        controller=lambda _path: CqEvidenceController(
            disease_name=disease_name,
            setting=setting_text,
            child_count=len(children),
            children=children,
        ),
    )
    episode = Episode(
        grain=grain,
        key=f"{disease_name}:{setting_text}",
        source=leaves(
            children,
            extract=lambda child: child,
            accept=lambda _unit, child: _run_child(
                child,
                disease_name=disease_name,
                setting=setting_text,
                search_limit=search_limit,
                search=search,
                retrieve=retrieve,
                llm=llm,
                observations=observations,
                require_transmissibility_evidence=(
                    require_transmissibility_evidence
                ),
            ),
            result=lambda _unit, observation: observation,
            label=lambda child: child,
        ),
        request=EpisodeRequest(
            goal=EpisodeGoal.root(
                objective={
                    "operation": "calibrate_cq_from_outbreak",
                    "disease_name": disease_name,
                    "setting": setting_text,
                    "require_transmissibility_evidence": (
                        require_transmissibility_evidence
                    ),
                },
                result_contract={
                    "result": "CqCalibrationCreditReport",
                    "main_output": "quanta_concentration_exhaled_m3",
                },
            ),
            message=request,
        ),
        build_result=_build_result,
    )
    tree = EpisodeTree(root=grain)
    episode_record = episode.run(Context(tree=tree))
    return episode_record, episode_record.controller_state.to_report()


def _run_child(
    child: str,
    *,
    disease_name: str,
    setting: str,
    search_limit: int,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    llm: LLMExtractCallable | None,
    observations: dict[str, CqEvidenceObservation],
    require_transmissibility_evidence: bool,
) -> CqEvidenceObservation:
    if child == "build_attack_probability":
        report = _run_attack_child(llm, observations)
    elif child == "build_removal_capacity":
        report = _run_evidence_child(
            child,
            _run_build_removal_capacity_episode,
            llm,
            observations,
            disease_name=disease_name,
            setting=setting,
            search=search,
            retrieve=retrieve,
            search_limit=search_limit,
            require_transmissibility_evidence=(
                require_transmissibility_evidence
            ),
        )
    elif child == "admit_exposure_time":
        report = _run_evidence_child(
            child,
            _run_admit_exposure_time_episode,
            llm,
            observations,
            disease_name=disease_name,
            setting=setting,
            search=search,
            retrieve=retrieve,
            search_limit=search_limit,
            require_transmissibility_evidence=(
                require_transmissibility_evidence
            ),
        )
    elif child == "admit_behavior_assumptions":
        report = _run_evidence_child(
            child,
            _run_admit_behavior_assumptions_episode,
            llm,
            observations,
            disease_name=disease_name,
            setting=setting,
            search=search,
            retrieve=retrieve,
            search_limit=search_limit,
            require_transmissibility_evidence=(
                require_transmissibility_evidence
            ),
        )
    elif child == "admit_population_or_variant_scaling":
        report = _run_evidence_child(
            child,
            _run_admit_population_or_variant_scaling_episode,
            llm,
            observations,
            disease_name=disease_name,
            setting=setting,
            search=search,
            retrieve=retrieve,
            search_limit=search_limit,
            require_transmissibility_evidence=(
                require_transmissibility_evidence
            ),
        )
    elif child == "quanta_concentration_from_attack_probability":
        report = _run_final_cq_child(observations)
    else:
        raise KeyError(f"unknown Cq evidence child: {child}")

    observation = CqEvidenceObservation(child=child, report=report)
    observations[child] = observation
    return observation


def _run_attack_child(
    llm: LLMExtractCallable | None,
    observations: Mapping[str, CqEvidenceObservation],
) -> CqCalibrationChildReport:
    try:
        search_report = observations["find_candidate_outbreak_sources"].report
    except KeyError:
        return failed_build_attack_probability(
            (
                CalculationError(
                    code="missing_child_report",
                    field="find_candidate_outbreak_sources",
                    message="search must pass before attack extraction",
                ),
            )
        )

    if not search_report.passed:
        return failed_build_attack_probability(search_report.errors)

    sources = _evidence_sources(
        search_report.outputs.get("candidate_outbreak_sources", ())
    )
    if not sources:
        return failed_build_attack_probability(
            (
                CalculationError(
                    code="missing_candidate_sources",
                    field="candidate_outbreak_sources",
                    message="search returned no extractable source text",
                ),
            )
        )

    _, report = _run_build_attack_probability_episode(llm, sources)
    return report


def _run_evidence_child(
    child_name: str,
    runner,
    llm: LLMExtractCallable | None,
    observations: Mapping[str, CqEvidenceObservation],
    *,
    disease_name: str,
    setting: str,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    search_limit: int,
    require_transmissibility_evidence: bool,
) -> CqCalibrationChildReport:
    sources = _sources_from_search(child_name, observations)
    if isinstance(sources, CqCalibrationChildReport):
        return sources

    if child_name == "admit_population_or_variant_scaling" and not require_transmissibility_evidence:
        return admit_neutral_transmissibility_scaling()

    attack = observations["build_attack_probability"].report
    _, report = runner(
        llm,
        sources,
        target_context={
            "disease_name": disease_name,
            "setting": setting,
            "attack_source_refs": attack.outputs.get("source_refs", ()),
            "attack_probability_inputs": dict(attack.outputs),
        },
    )
    if report.passed:
        return report
    if not _can_repair_with_assumption_search(report):
        return report

    return _run_assumption_fallback(
        child_name,
        runner,
        llm,
        observations,
        report,
        disease_name=disease_name,
        setting=setting,
        search=search,
        retrieve=retrieve,
        search_limit=search_limit,
    )


def _can_repair_with_assumption_search(
    report: CqCalibrationChildReport,
) -> bool:
    """Assumption search can repair absent facts, not invalid evidence."""

    return _missing_only(report)


def _missing_only(
    report: CqCalibrationChildReport,
    fields: set[str] | None = None,
) -> bool:
    return bool(report.errors) and all(
        error.code == "missing_extracted_field"
        and (fields is None or error.field in fields)
        for error in report.errors
    )


def _run_assumption_fallback(
    child_name: str,
    runner,
    llm: LLMExtractCallable | None,
    observations: Mapping[str, CqEvidenceObservation],
    outbreak_report: CqCalibrationChildReport,
    *,
    disease_name: str,
    setting: str,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    search_limit: int,
) -> CqCalibrationChildReport:
    if not assumption_search_specs(
        child_name,
        disease_name,
        setting=setting,
    ):
        return outbreak_report

    _, assumption_search = _run_find_assumption_sources_episode(
        child=child_name,
        disease_name=disease_name,
        setting=setting,
        search=search,
        retrieve=retrieve,
        limit=search_limit,
    )
    if not assumption_search.passed:
        return _failed_with_assumption_search(
            child_name,
            outbreak_report,
            assumption_search,
        )

    sources = _evidence_sources(
        assumption_search.outputs.get("assumption_sources", ())
    )
    if not sources:
        return _failed_with_assumption_search(
            child_name,
            outbreak_report,
            assumption_search,
            extra_errors=(
                CalculationError(
                    code="missing_assumption_sources",
                    field="assumption_sources",
                    message="assumption search returned no extractable text",
                ),
            ),
        )

    _, fallback_report = runner(
        llm,
        sources,
        admit_assumption=True,
        assumption_services=AssumptionServices(search, retrieve, search_limit),
        target_context={
            "disease_name": disease_name,
            "setting": setting,
            "child": child_name,
            "outbreak_sources": [
                source.as_record()
                for source in _sources_from_search(child_name, observations)
            ],
            "outbreak_search": _output(
                observations,
                "find_candidate_outbreak_sources",
                "search_summary",
            ),
            "attack_source_refs": _output(
                observations,
                "build_attack_probability",
                "source_refs",
            ) or (),
            "attack_probability_inputs": dict(
                observations["build_attack_probability"].report.outputs
            ),
            "failed_outbreak_child": outbreak_report.as_record(),
        },
    )
    if not fallback_report.passed:
        return _failed_with_assumption_search(
            child_name,
            fallback_report,
            assumption_search,
            extra_errors=outbreak_report.errors,
        )

    return replace(
        fallback_report,
        outputs={
            **dict(fallback_report.outputs),
            "assumption_search": assumption_search.as_record(),
        },
    )


def _failed_with_assumption_search(
    child_name: str,
    child_report: CqCalibrationChildReport,
    assumption_search: AssumptionSearchReport,
    *,
    extra_errors: tuple[CalculationError, ...] = (),
) -> CqCalibrationChildReport:
    return failed_evidence_child(
        child_name,
        child_report.errors + assumption_search.errors + extra_errors,
        outputs={
            **dict(child_report.outputs),
            "assumption_search": assumption_search.as_record(),
        },
    )


def _run_final_cq_child(
    observations: Mapping[str, CqEvidenceObservation],
) -> CqCalibrationChildReport:
    child = _child("quanta_concentration_from_attack_probability")
    values = {
        "attack_probability": _output(
            observations,
            "build_attack_probability",
            "attack_probability",
        ),
        "initial_infected_count": _output(
            observations,
            "build_attack_probability",
            "index_case_count",
        ),
        "removal_capacity_m3_s": _output(
            observations,
            "build_removal_capacity",
            "removal_capacity_m3_s",
        ),
        "exposure_time_s": _output(
            observations,
            "admit_exposure_time",
            "exposure_time_s",
        ),
        "breathing_flow_m3_s": _output(
            observations,
            "admit_behavior_assumptions",
            "breathing_flow_m3_s",
        ),
        "mask_penetration": _output(
            observations,
            "admit_behavior_assumptions",
            "mask_penetration",
        ),
        "relative_transmissibility": _output(
            observations,
            "admit_population_or_variant_scaling",
            "relative_transmissibility",
        ),
    }
    errors = tuple(
        CalculationError(
            code="missing_prerequisite_output",
            field=field,
            message=f"{field} is required before final Cq calculation",
        )
        for field, value in values.items()
        if value is None
    )
    if errors:
        return credit_cq_calibration_child(child, {}, errors=errors)

    calculation = run_named_calculation(
        "quanta_concentration_from_slow_incubation_attack_probability",
        values,
    )
    return credit_cq_calibration_child(
        child,
        calculation.outputs,
        errors=calculation.errors,
        identities=calculation.identities,
    )


def _sources_from_search(
    child_name: str,
    observations: Mapping[str, CqEvidenceObservation],
) -> tuple[EvidenceSource, ...] | CqCalibrationChildReport:
    try:
        search_report = observations["find_candidate_outbreak_sources"].report
    except KeyError:
        return failed_evidence_child(
            child_name,
            (
                CalculationError(
                    code="missing_child_report",
                    field="find_candidate_outbreak_sources",
                    message="search must pass before evidence extraction",
                ),
            ),
        )

    if not search_report.passed:
        return failed_evidence_child(
            child_name,
            search_report.errors,
        )

    try:
        attack_report = observations["build_attack_probability"].report
    except KeyError:
        return failed_evidence_child(
            child_name,
            (
                CalculationError(
                    code="missing_child_report",
                    field="build_attack_probability",
                    message=(
                        "attack probability must pass before extracting "
                        "sibling outbreak evidence"
                    ),
                ),
            ),
        )

    if not attack_report.passed:
        return failed_evidence_child(
            child_name,
            (
                CalculationError(
                    code="missing_child_report",
                    field="build_attack_probability",
                    message=(
                        "attack probability must pass before extracting "
                        "sibling outbreak evidence"
                    ),
                ),
            ),
        )

    attack_refs = _source_refs(attack_report.outputs.get("source_refs"))
    if not attack_refs:
        return failed_evidence_child(
            child_name,
            (
                CalculationError(
                    code="missing_source_refs",
                    field="build_attack_probability.source_refs",
                    message=(
                        "a credited attack-probability source ref is required "
                        "before extracting sibling outbreak evidence"
                    ),
                ),
            ),
        )

    sources = _evidence_sources(
        search_report.outputs.get("candidate_outbreak_sources", ())
    )
    sources = _same_event_sources(sources, attack_refs)
    if not sources:
        return failed_evidence_child(
            child_name,
            (
                CalculationError(
                    code="missing_candidate_sources",
                    field="candidate_outbreak_sources",
                    message="search returned no extractable source text",
                ),
            )
        )
    return sources


def _same_event_sources(
    sources: tuple[EvidenceSource, ...],
    attack_refs: tuple[str, ...],
) -> tuple[EvidenceSource, ...]:
    attack_event_refs = tuple(
        dict.fromkeys(
            source.event_ref
            for source in sources
            if source.source_ref in attack_refs and source.event_ref
        )
    )
    if not attack_event_refs:
        return tuple(
            source
            for source in sources
            if source.source_ref in attack_refs
        )

    accepted_event_refs = set(attack_event_refs)
    return tuple(
        source
        for source in sources
        if (
            source.source_ref in attack_refs
            or source.event_ref in accepted_event_refs
        )
    )


def _source_refs(refs: object) -> tuple[str, ...]:
    if isinstance(refs, str):
        return (refs,)
    if isinstance(refs, tuple | list):
        return tuple(str(ref) for ref in refs if str(ref))
    return ()


def _child(name: str) -> CqCalibrationChild:
    return next(
        child for child in CQ_CALIBRATION_CHILDREN if child.name == name
    )


def _cq_children(names: tuple[str, ...]) -> tuple[CqCalibrationChild, ...]:
    return tuple(_child(name) for name in names)


def _output(
    observations: Mapping[str, CqEvidenceObservation],
    child: str,
    output: str,
) -> Any:
    try:
        report = observations[child].report
    except KeyError:
        return None
    if not report.passed:
        return None
    return report.outputs.get(output)


def _evidence_sources(raw_candidates: object) -> tuple[EvidenceSource, ...]:
    if not isinstance(raw_candidates, tuple):
        return ()

    sources: list[EvidenceSource] = []
    for candidate in raw_candidates:
        if not isinstance(candidate, Mapping):
            continue
        try:
            sources.append(evidence_source_from_candidate(candidate))
        except ValueError:
            continue
    return tuple(sources)


def _build_result(record: EpisodeRecord) -> CqEvidenceRecord:
    state = record.controller_state
    if not isinstance(state, CqEvidenceEpisodeState):
        raise TypeError("Cq evidence Episode produced an unexpected state")
    return CqEvidenceRecord(state.to_report())


__all__ = [
    "CALIBRATE_CQ_FROM_OUTBREAK_CHILDREN",
    "CqEvidenceController",
    "CqEvidenceEpisodeState",
    "CqEvidenceEpisodeStep",
    "CqEvidenceObservation",
    "CqEvidenceRecord",
    "CqEvidenceRequest",
]
