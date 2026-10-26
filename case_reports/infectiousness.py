"""Episode for estimating pathogen infectiousness from outbreak evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real
from statistics import median
from typing import Any, Mapping

from episodic_workflow.credit_assignment.cq_calibration import (
    CqCalibrationChildReport,
    CqCalibrationCreditReport,
)
from episodic_workflow.credit_assignment.infectiousness import (
    INFECTIOUSNESS_CHILDREN,
    InfectiousnessChildReport,
    InfectiousnessCreditReport,
    fail_infectiousness_child,
    infectiousness_child,
    pass_infectiousness_child,
    sum_infectiousness_credit,
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
from episodic_workflow.task_executors.outbreak_search import (
    RetrieveCallable,
    SearchCallable,
    SettingHints,
    setting_hints_text,
)

from .calibration_from_evidence import (
    _run_calibrate_cq_from_outbreak_with_search_report_episode,
)
from .evidence_extraction import LLMExtractCallable
from .search_episodes import _run_find_calibration_outbreaks_episode


INFECTIOUSNESS_CHILD_NAMES = tuple(
    child.name for child in INFECTIOUSNESS_CHILDREN
)


@dataclass(frozen=True)
class InfectiousnessObservation:
    """One child report from pathogen-infectiousness estimation."""

    child: str
    report: InfectiousnessChildReport

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
class InfectiousnessEpisodeStep:
    """Controller step for infectiousness estimation."""

    stop: bool
    requests_transition: bool = False

    def as_record(self) -> dict[str, bool]:
        return {
            "stop": self.stop,
            "requests_transition": self.requests_transition,
        }


@dataclass(frozen=True)
class InfectiousnessEpisodeState:
    """Accumulated infectiousness-estimation child results."""

    disease_name: str
    setting: str
    child_count: int
    observations: tuple[InfectiousnessObservation, ...] = ()
    stop: bool = False

    def to_report(self) -> InfectiousnessCreditReport:
        return sum_infectiousness_credit(
            observation.report for observation in self.observations
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "disease_name": self.disease_name,
            "setting": self.setting,
            "child_count": self.child_count,
            "stop": self.stop,
            "observations": [
                observation.as_record()
                for observation in self.observations
            ],
            "report": self.to_report().as_record(),
        }


@dataclass(frozen=True)
class InfectiousnessRequest(ClosedRecord):
    """Request to estimate infectiousness from searched outbreak evidence."""

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
class InfectiousnessRecord(ClosedRecord):
    """Closed result of infectiousness estimation."""

    report: InfectiousnessCreditReport

    def as_record(self) -> Mapping[str, Any]:
        return self.report.as_record()

    def audit_identifiers(self) -> tuple[str, ...]:
        return self.report.identities


class InfectiousnessController:
    """Stop when planned infectiousness children have reported."""

    epoch = "estimate_pathogen_infectiousness_v1"

    def __init__(
        self,
        *,
        disease_name: str,
        setting: str,
        child_count: int,
        observations: tuple[InfectiousnessObservation, ...] = (),
        stop: bool = False,
    ) -> None:
        self._disease_name = disease_name
        self._setting = setting
        self._child_count = child_count
        self._observations = observations
        self._stop = stop

    def observe(
        self,
        unit_label: str,
        value: object,
        *,
        is_root: bool,
    ) -> InfectiousnessEpisodeStep:
        if not isinstance(value, InfectiousnessObservation):
            raise TypeError(
                "InfectiousnessController expects InfectiousnessObservation"
            )

        observations = self._observations + (value,)
        stop = len(observations) >= self._child_count
        self._observations = observations
        self._stop = stop
        return InfectiousnessEpisodeStep(stop=stop)

    def state(self) -> InfectiousnessEpisodeState:
        return InfectiousnessEpisodeState(
            disease_name=self._disease_name,
            setting=self._setting,
            child_count=self._child_count,
            observations=self._observations,
            stop=self._stop,
        )

    def transitioned(self, epoch: str) -> "InfectiousnessController":
        return InfectiousnessController(
            disease_name=self._disease_name,
            setting=self._setting,
            child_count=self._child_count,
            observations=self._observations,
            stop=self._stop,
        )


def _run_estimate_pathogen_infectiousness_episode(
    *,
    disease_name: str,
    search: SearchCallable,
    llm: LLMExtractCallable | None,
    retrieve: RetrieveCallable | None = None,
    setting: SettingHints = "",
    search_limit: int = 10,
    require_transmissibility_evidence: bool = False,
) -> tuple[EpisodeRecord, InfectiousnessCreditReport]:
    """Search for outbreak evidence and estimate a disease/activity Cq."""

    setting_text = setting_hints_text(setting)
    observations: dict[str, InfectiousnessObservation] = {}
    request = InfectiousnessRequest(
        disease_name=disease_name,
        setting=setting_text,
        search_limit=search_limit,
        retrieval_enabled=retrieve is not None,
        require_transmissibility_evidence=require_transmissibility_evidence,
    )
    grain = Grain(
        name="estimate_pathogen_infectiousness",
        unit="infectiousness child",
        result="InfectiousnessCreditReport",
        controller=lambda _path: InfectiousnessController(
            disease_name=disease_name,
            setting=setting_text,
            child_count=len(INFECTIOUSNESS_CHILD_NAMES),
        ),
    )
    episode = Episode(
        grain=grain,
        key=f"{disease_name}:{setting_text}",
        source=leaves(
            INFECTIOUSNESS_CHILD_NAMES,
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
                    "operation": "estimate_pathogen_infectiousness",
                    "disease_name": disease_name,
                    "setting": setting_text,
                    "require_transmissibility_evidence": (
                        require_transmissibility_evidence
                    ),
                },
                result_contract={
                    "result": "InfectiousnessCreditReport",
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
    child_name: str,
    *,
    disease_name: str,
    setting: str,
    search_limit: int,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    llm: LLMExtractCallable | None,
    observations: dict[str, InfectiousnessObservation],
    require_transmissibility_evidence: bool,
) -> InfectiousnessObservation:
    if child_name == "find_calibration_outbreaks":
        report = _run_find_calibration_outbreaks_child(
            disease_name=disease_name,
            setting=setting,
            search_limit=search_limit,
            search=search,
            retrieve=retrieve,
        )
    elif child_name == "calibrate_cq_from_outbreak":
        report = _run_calibrate_cq_child(
            disease_name=disease_name,
            setting=setting,
            search_limit=search_limit,
            search=search,
            retrieve=retrieve,
            llm=llm,
            observations=observations,
            require_transmissibility_evidence=(
                require_transmissibility_evidence
            ),
        )
    elif child_name == "rescale_cq_by_respiratory_activity":
        report = _run_activity_rescale_child(observations)
    elif child_name == "synthesize_pathogen_infectiousness":
        report = _run_synthesize_child(observations)
    else:
        raise KeyError(f"unknown infectiousness child: {child_name}")

    observation = InfectiousnessObservation(
        child=child_name,
        report=report,
    )
    observations[child_name] = observation
    return observation


def _run_find_calibration_outbreaks_child(
    *,
    disease_name: str,
    setting: str,
    search_limit: int,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
) -> InfectiousnessChildReport:
    _, report = _run_find_calibration_outbreaks_episode(
        search,
        disease_name,
        retrieve=retrieve,
        setting=setting,
        limit=search_limit,
    )
    return _adapt_search_report(report)


def _run_calibrate_cq_child(
    *,
    disease_name: str,
    setting: str,
    search_limit: int,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    llm: LLMExtractCallable | None,
    observations: Mapping[str, InfectiousnessObservation],
    require_transmissibility_evidence: bool,
) -> InfectiousnessChildReport:
    search_observation = _required_child(
        observations,
        "find_calibration_outbreaks",
        "calibrate_cq_from_outbreak",
    )
    if isinstance(search_observation, InfectiousnessChildReport):
        return search_observation

    child = infectiousness_child("calibrate_cq_from_outbreak")
    search_report = _to_cq_search_report(search_observation.report)
    candidate_groups = _candidate_source_groups(
        search_report.outputs.get("candidate_outbreak_sources", ())
    )
    if not candidate_groups:
        return fail_infectiousness_child(
            child,
            (
                CalculationError(
                    code="missing_candidate_sources",
                    field="candidate_outbreak_sources",
                    message="search returned no outbreak source groups",
                ),
            ),
        )

    attempts = []
    calibrated = []
    errors: list[CalculationError] = []
    identities: list[str] = []
    for group in candidate_groups:
        _, report = _run_calibrate_cq_from_outbreak_with_search_report_episode(
            disease_name=disease_name,
            outbreak_search_report=_to_group_cq_search_report(
                search_report,
                group,
            ),
            search=search,
            retrieve=retrieve,
            llm=llm,
            setting=setting,
            search_limit=search_limit,
            require_transmissibility_evidence=require_transmissibility_evidence,
        )
        value = _cq_output(report)
        event_ref = _candidate_group_ref(group)
        source_refs = _candidate_group_source_refs(group)
        attempts.append(
            {
                "event_ref": event_ref,
                "source_refs": source_refs,
                "passed": report.passed,
                "quanta_concentration_exhaled_m3": value,
                "report": report.as_record(),
            }
        )
        if report.passed and isinstance(value, Real):
            calibrated.append(
                {
                    "event_ref": event_ref,
                    "source_refs": source_refs,
                    "quanta_concentration_exhaled_m3": float(value),
                    "score": report.score,
                    "max_score": report.max_score,
                }
            )
            identities.extend(report.identities)
        else:
            errors.extend(_cq_errors(report, prefix=event_ref))

    outputs = {
        "quanta_concentration_exhaled_m3": (
            median(
                event["quanta_concentration_exhaled_m3"]
                for event in calibrated
            )
            if calibrated
            else None
        ),
        "calibrated_outbreaks": tuple(calibrated),
        "calibration_attempts": tuple(attempts),
        "calibration_summary": {
            "method": "median",
            "attempted": len(attempts),
            "passed": len(calibrated),
            "failed": len(attempts) - len(calibrated),
        },
        "cq_calibration": (
            attempts[0]["report"] if len(attempts) == 1 else None
        ),
    }
    if calibrated:
        return pass_infectiousness_child(
            child,
            outputs,
            identities=tuple(identities),
        )
    return replace(
        fail_infectiousness_child(child, tuple(errors), outputs=outputs),
        # Alternative outbreaks are attempts at the same child, not siblings.
        score=max(attempt["report"]["score"] for attempt in attempts),
    )


def _run_activity_rescale_child(
    observations: Mapping[str, InfectiousnessObservation],
) -> InfectiousnessChildReport:
    observation = _required_child(
        observations,
        "calibrate_cq_from_outbreak",
        "rescale_cq_by_respiratory_activity",
    )
    if isinstance(observation, InfectiousnessChildReport):
        return observation

    child = infectiousness_child("rescale_cq_by_respiratory_activity")
    value = observation.report.outputs.get("quanta_concentration_exhaled_m3")
    return pass_infectiousness_child(
        child,
        {
            "activity_adjusted_quanta_concentration_exhaled_m3": value,
            "activity_scaling": {
                "type": "identity",
                "value": 1.0,
                "notes": (
                    "No target respiratory-activity rescaling was requested."
                ),
            },
        },
        identities=observation.report.identities,
    )


def _run_synthesize_child(
    observations: Mapping[str, InfectiousnessObservation],
) -> InfectiousnessChildReport:
    observation = _required_child(
        observations,
        "rescale_cq_by_respiratory_activity",
        "synthesize_pathogen_infectiousness",
    )
    if isinstance(observation, InfectiousnessChildReport):
        return observation

    child = infectiousness_child("synthesize_pathogen_infectiousness")
    value = observation.report.outputs.get(
        "activity_adjusted_quanta_concentration_exhaled_m3"
    )
    calibration = observations.get("calibrate_cq_from_outbreak")
    calibrated_outbreaks = (
        ()
        if calibration is None
        else tuple(
            calibration.report.outputs.get("calibrated_outbreaks", ())
        )
    )
    return pass_infectiousness_child(
        child,
        {
            "quanta_concentration_exhaled_m3": value,
            "basis": (
                "median_calibrated_outbreaks"
                if len(calibrated_outbreaks) > 1
                else "single_calibrated_outbreak"
            ),
            "calibrated_outbreak_count": len(calibrated_outbreaks),
        },
        identities=observation.report.identities,
    )


def _required_child(
    observations: Mapping[str, InfectiousnessObservation],
    required_child: str,
    current_child: str,
) -> InfectiousnessObservation | InfectiousnessChildReport:
    child = infectiousness_child(current_child)
    try:
        observation = observations[required_child]
    except KeyError:
        return fail_infectiousness_child(
            child,
            (
                CalculationError(
                    code="missing_child_report",
                    field=required_child,
                    message=(
                        f"{required_child} must pass before {current_child}"
                    ),
                ),
            ),
        )

    if observation.report.passed:
        return observation

    return fail_infectiousness_child(
        child,
        (
            CalculationError(
                code="missing_child_report",
                field=required_child,
                message=f"{required_child} did not pass",
            ),
        ),
    )


def _adapt_search_report(
    report: CqCalibrationChildReport,
) -> InfectiousnessChildReport:
    child = infectiousness_child("find_calibration_outbreaks")
    if report.passed:
        return pass_infectiousness_child(
            child,
            report.outputs,
            identities=report.identities,
        )
    return fail_infectiousness_child(
        child,
        report.errors,
        outputs=report.outputs,
    )


def _to_cq_search_report(
    report: InfectiousnessChildReport,
) -> CqCalibrationChildReport:
    return CqCalibrationChildReport(
        child="find_candidate_outbreak_sources",
        output="candidate_outbreak_sources",
        passed=report.passed,
        score=report.score,
        max_score=report.max_score,
        should_yield=report.should_yield,
        outputs=report.outputs,
        errors=report.errors,
        identities=report.identities,
    )


def _cq_output(report: CqCalibrationCreditReport) -> object:
    child_outputs = report.main_outputs.get(
        "quanta_concentration_from_attack_probability",
        {},
    )
    return child_outputs.get("quanta_concentration_exhaled_m3")


def _to_group_cq_search_report(
    report: CqCalibrationChildReport,
    group: tuple[Mapping[str, Any], ...],
) -> CqCalibrationChildReport:
    return CqCalibrationChildReport(
        child=report.child,
        output=report.output,
        passed=report.passed,
        score=report.score,
        max_score=report.max_score,
        should_yield=report.should_yield,
        outputs={
            **dict(report.outputs),
            "candidate_outbreak_sources": group,
        },
        errors=report.errors,
        identities=report.identities,
    )


def _candidate_source_groups(
    raw_candidates: object,
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    if not isinstance(raw_candidates, tuple):
        return ()

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, Mapping):
            continue
        event_ref = str(
            candidate.get("event_ref")
            or candidate.get("outbreak_ref")
            or candidate.get("source_ref")
            or f"candidate:{index}"
        )
        groups.setdefault(event_ref, []).append(candidate)
    return tuple(tuple(group) for group in groups.values())


def _candidate_group_ref(group: tuple[Mapping[str, Any], ...]) -> str:
    for candidate in group:
        event_ref = str(
            candidate.get("event_ref")
            or candidate.get("outbreak_ref")
            or ""
        )
        if event_ref:
            return event_ref
    return "+".join(_candidate_group_source_refs(group))


def _candidate_group_source_refs(
    group: tuple[Mapping[str, Any], ...],
) -> tuple[str, ...]:
    return tuple(
        str(candidate.get("source_ref") or "")
        for candidate in group
        if str(candidate.get("source_ref") or "")
    )


def _cq_errors(
    report: CqCalibrationCreditReport,
    *,
    prefix: str = "",
) -> tuple[CalculationError, ...]:
    return tuple(
        CalculationError(
            code=error.code,
            field=".".join(
                part
                for part in (
                    prefix,
                    error.child,
                    error.field or error.output,
                )
                if part
            ),
            message=error.message,
        )
        for error in report.errors
    ) or (
        CalculationError(
            code="missing_output",
            field="quanta_concentration_exhaled_m3",
            message="Cq calibration did not return a credited Cq",
        ),
    )


def _build_result(record: EpisodeRecord) -> InfectiousnessRecord:
    state = record.controller_state
    if not isinstance(state, InfectiousnessEpisodeState):
        raise TypeError("infectiousness Episode produced an unexpected state")
    return InfectiousnessRecord(state.to_report())


__all__ = [
    "INFECTIOUSNESS_CHILD_NAMES",
    "InfectiousnessController",
    "InfectiousnessEpisodeState",
    "InfectiousnessEpisodeStep",
    "InfectiousnessObservation",
    "InfectiousnessRecord",
    "InfectiousnessRequest",
]
