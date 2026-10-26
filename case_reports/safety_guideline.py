"""Episode for a first generic indoor-airborne safety-guideline evaluation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from numbers import Real
from typing import Any, Mapping

from episodic_workflow.credit_assignment.calculations import run_named_calculation
from episodic_workflow.credit_assignment.guideline import (
    GUIDELINE_CHILDREN,
    GuidelineChildReport,
    GuidelineCreditReport,
    fail_guideline_child,
    guideline_child,
    pass_guideline_child,
    sum_guideline_credit,
)
from episodic_workflow.credit_assignment.infectiousness import (
    InfectiousnessCreditReport,
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

from .assumption_search import (
    AssumptionSearchReport,
    _run_find_assumption_sources_episode,
)
from .assumption_investigation import AssumptionServices
from .evidence_children import (
    _run_admit_behavior_assumptions_episode,
    _run_build_removal_capacity_episode,
)
from .evidence_extraction import (
    EvidenceSource,
    LLMExtractCallable,
    evidence_source_from_candidate,
)
from .infectiousness import _run_estimate_pathogen_infectiousness_episode


GUIDELINE_CHILD_NAMES = tuple(child.name for child in GUIDELINE_CHILDREN)
STEADY_STATE_CUMULATIVE_EXPOSURE = "steady_state_cumulative_exposure_bound"
STEADY_STATE_TIME_BOUND = "steady_state_time_bound"
STEADY_STATE_SUSCEPTIBLE_BOUND = "steady_state_susceptible_bound"
STEADY_STATE_INDOOR_REPRODUCTIVE_NUMBER = (
    "steady_state_indoor_reproductive_number"
)
STEADY_STATE_FIRST_TRANSMISSION_PROBABILITY = (
    "steady_state_first_transmission_probability"
)
STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER = (
    "steady_state_jet_corrected_indoor_reproductive_number"
)
TRANSIENT_CUMULATIVE_EXPOSURE = "transient_cumulative_exposure_bound"
TRANSIENT_TIME_BOUND = "transient_time_bound"
TRANSIENT_SUSCEPTIBLE_BOUND = "transient_susceptible_bound"
RELATIVE_RISK_INDEX = "relative_risk_index"
GUIDELINE_OUTPUTS = {
    STEADY_STATE_CUMULATIVE_EXPOSURE: (
        "cumulative_exposure_bound_person_s",
        "person_s",
    ),
    STEADY_STATE_TIME_BOUND: ("exposure_time_bound_s", "s"),
    STEADY_STATE_SUSCEPTIBLE_BOUND: ("susceptible_bound", "person"),
    STEADY_STATE_INDOOR_REPRODUCTIVE_NUMBER: (
        "indoor_reproductive_number",
        "scalar",
    ),
    STEADY_STATE_FIRST_TRANSMISSION_PROBABILITY: (
        "first_transmission_probability",
        "probability",
    ),
    STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER: (
        "jet_corrected_indoor_reproductive_number",
        "scalar",
    ),
    TRANSIENT_CUMULATIVE_EXPOSURE: (
        "cumulative_exposure_bound_person_s",
        "person_s",
    ),
    TRANSIENT_TIME_BOUND: ("exposure_time_bound_s", "s"),
    TRANSIENT_SUSCEPTIBLE_BOUND: ("susceptible_bound", "person"),
    RELATIVE_RISK_INDEX: ("relative_risk_index", "scalar"),
}
STEADY_STATE_GUIDELINE_OUTPUTS = GUIDELINE_OUTPUTS
CUMULATIVE_EXPOSURE_MODELS = frozenset(
    {
        STEADY_STATE_CUMULATIVE_EXPOSURE,
        STEADY_STATE_TIME_BOUND,
        STEADY_STATE_SUSCEPTIBLE_BOUND,
        TRANSIENT_CUMULATIVE_EXPOSURE,
        TRANSIENT_TIME_BOUND,
        TRANSIENT_SUSCEPTIBLE_BOUND,
    }
)
FIXED_EXPOSURE_TIME_MODELS = frozenset(
    {
        STEADY_STATE_SUSCEPTIBLE_BOUND,
        TRANSIENT_CUMULATIVE_EXPOSURE,
        TRANSIENT_SUSCEPTIBLE_BOUND,
    }
)
FIXED_SUSCEPTIBLE_MODELS = frozenset(
    {
        STEADY_STATE_TIME_BOUND,
        TRANSIENT_TIME_BOUND,
    }
)
INDOOR_REPRODUCTIVE_NUMBER_MODELS = frozenset(
    {
        STEADY_STATE_INDOOR_REPRODUCTIVE_NUMBER,
        STEADY_STATE_FIRST_TRANSMISSION_PROBABILITY,
        STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER,
    }
)
TRANSIENT_MODELS = frozenset(
    {
        TRANSIENT_CUMULATIVE_EXPOSURE,
        TRANSIENT_TIME_BOUND,
        TRANSIENT_SUSCEPTIBLE_BOUND,
    }
)
EQ5_HAZARD_MODELS = (
    CUMULATIVE_EXPOSURE_MODELS | INDOOR_REPRODUCTIVE_NUMBER_MODELS
)
JET_CORRECTION_TARGET_FIELDS = (
    "plume_encounter_probability",
    "mouth_area_m2",
    "entrainment_coefficient",
    "neighbor_distance_m",
    "well_mixed_dilution_factor",
)
TARGET_ASSUMPTION_HINT_FIELDS = (
    "setting",
    "setting_hint",
    "room",
    "room_type",
    "building",
    "building_type",
    "occupancy",
    "activity",
    "masking",
    "mask_type",
    "ventilation",
    "hvac",
    "filter",
    "building_code",
    "country",
    "region",
)


@dataclass(frozen=True)
class GuidelineObservation:
    """One child report from the safety-guideline evaluation."""

    child: str
    report: GuidelineChildReport

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
class GuidelineEpisodeStep:
    """Controller step for the safety-guideline parent."""

    stop: bool
    requests_transition: bool = False

    def as_record(self) -> dict[str, bool]:
        return {
            "stop": self.stop,
            "requests_transition": self.requests_transition,
        }


@dataclass(frozen=True)
class GuidelineEpisodeState:
    """Accumulated safety-guideline child results."""

    disease_name: str
    setting: str
    child_count: int
    observations: tuple[GuidelineObservation, ...] = ()
    stop: bool = False

    def to_report(self) -> GuidelineCreditReport:
        return sum_guideline_credit(
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
class GuidelineRequest(ClosedRecord):
    """Request to evaluate a steady-state safety guideline."""

    disease_name: str
    setting: str
    target_scenario: Mapping[str, Any]
    transmission_model: str
    search_limit: int = 10
    retrieval_enabled: bool = False
    require_transmissibility_evidence: bool = False
    research_request: str | None = None

    def as_record(self) -> Mapping[str, Any]:
        return {
            "disease_name": self.disease_name,
            "setting": self.setting,
            "target_scenario": dict(self.target_scenario),
            "transmission_model": self.transmission_model,
            "search_limit": self.search_limit,
            "retrieval_enabled": self.retrieval_enabled,
            "require_transmissibility_evidence": (
                self.require_transmissibility_evidence
            ),
            "research_request": self.research_request,
        }


@dataclass(frozen=True)
class GuidelineRecord(ClosedRecord):
    """Closed result of a safety-guideline evaluation."""

    report: GuidelineCreditReport

    def as_record(self) -> Mapping[str, Any]:
        return self.report.as_record()

    def audit_identifiers(self) -> tuple[str, ...]:
        return self.report.identities


class GuidelineController:
    """Stop when planned guideline children have reported."""

    epoch = "evaluate_safety_guideline_v1"

    def __init__(
        self,
        *,
        disease_name: str,
        setting: str,
        child_count: int,
        observations: tuple[GuidelineObservation, ...] = (),
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
    ) -> GuidelineEpisodeStep:
        if not isinstance(value, GuidelineObservation):
            raise TypeError(
                "GuidelineController expects GuidelineObservation"
            )

        observations = self._observations + (value,)
        stop = len(observations) >= self._child_count
        self._observations = observations
        self._stop = stop
        return GuidelineEpisodeStep(stop=stop)

    def state(self) -> GuidelineEpisodeState:
        return GuidelineEpisodeState(
            disease_name=self._disease_name,
            setting=self._setting,
            child_count=self._child_count,
            observations=self._observations,
            stop=self._stop,
        )

    def transitioned(self, epoch: str) -> "GuidelineController":
        return GuidelineController(
            disease_name=self._disease_name,
            setting=self._setting,
            child_count=self._child_count,
            observations=self._observations,
            stop=self._stop,
        )


def run_evaluate_safety_guideline_episode(
    *,
    disease_name: str = "",
    target_scenario: Mapping[str, Any] | None = None,
    search: SearchCallable,
    llm: LLMExtractCallable | None,
    retrieve: RetrieveCallable | None = None,
    setting: SettingHints = "",
    search_limit: int = 10,
    require_transmissibility_evidence: bool = False,
    transmission_model: str = STEADY_STATE_CUMULATIVE_EXPOSURE,
    research_request: str | None = None,
    requested_model: str | None = None,
) -> tuple[EpisodeRecord, GuidelineCreditReport]:
    """Estimate infectiousness and compute a safety-guideline result."""

    setting_text = setting_hints_text(setting)
    if research_request is not None and (disease_name or target_scenario or setting):
        raise ValueError("A prompt request cannot be combined with supplied scenario values or scope")
    scenario = dict(target_scenario or {})
    observations: dict[str, GuidelineObservation] = {}
    interpretation = None
    interpretation_error = None

    def run_child(child):
        nonlocal disease_name, setting_text, transmission_model, interpretation, interpretation_error
        if research_request is not None and interpretation is None and interpretation_error is None:
            from .request_admission import interpret_guideline_request
            try:
                interpretation = interpret_guideline_request(
                    llm, research_request, GUIDELINE_OUTPUTS, requested_model=requested_model,
                )
                disease_name = interpretation["disease_name"]
                setting_text = interpretation["setting"]
                transmission_model = interpretation["transmission_model"]
                scenario.update(interpretation["target_scenario"])
            except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
                interpretation_error = CalculationError("request_interpretation_failed", str(exc), "research_request")
        if interpretation_error is not None:
            observation = GuidelineObservation(child, fail_guideline_child(
                guideline_child(child), (interpretation_error,),
                outputs={"original_request": research_request},
            ))
            observations[child] = observation
            return observation
        observation = _run_child(
            child, disease_name=disease_name, setting=setting_text,
            search_limit=search_limit, search=search, retrieve=retrieve, llm=llm,
            target_scenario=scenario, transmission_model=transmission_model,
            observations=observations,
            require_transmissibility_evidence=require_transmissibility_evidence,
        )
        if interpretation is not None and child == "estimate_pathogen_infectiousness":
            observation = replace(observation, report=replace(observation.report, outputs={
                **observation.report.outputs, "request_interpretation": interpretation,
            }))
            observations[child] = observation
        return observation

    request = GuidelineRequest(
        disease_name=disease_name,
        setting=setting_text,
        target_scenario=scenario,
        transmission_model=transmission_model,
        search_limit=search_limit,
        retrieval_enabled=retrieve is not None,
        require_transmissibility_evidence=require_transmissibility_evidence,
        research_request=research_request,
    )
    grain = Grain(
        name="evaluate_safety_guideline",
        unit="safety guideline child",
        result="GuidelineCreditReport",
        controller=lambda _path: GuidelineController(
            disease_name=disease_name,
            setting=setting_text,
            child_count=len(GUIDELINE_CHILD_NAMES),
        ),
    )
    episode = Episode(
        grain=grain,
        key=f"{disease_name}:{setting_text}",
        source=leaves(
            GUIDELINE_CHILD_NAMES,
            extract=lambda child: child,
            accept=lambda _unit, child: run_child(child),
            result=lambda _unit, observation: observation,
            label=lambda child: child,
        ),
        request=EpisodeRequest(
            goal=EpisodeGoal.root(
                objective={
                    "operation": "evaluate_safety_guideline",
                    "research_request": research_request,
                    "disease_name": disease_name,
                    "setting": setting_text,
                    "transmission_model": transmission_model,
                    "require_transmissibility_evidence": (
                        require_transmissibility_evidence
                    ),
                },
                result_contract={
                    "result": "GuidelineCreditReport",
                    "main_output": "safety_guideline_result",
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
    target_scenario: Mapping[str, Any],
    transmission_model: str,
    observations: dict[str, GuidelineObservation],
    require_transmissibility_evidence: bool,
) -> GuidelineObservation:
    if child_name == "estimate_pathogen_infectiousness":
        report = _run_infectiousness_child(
            disease_name=disease_name,
            setting=setting,
            search_limit=search_limit,
            search=search,
            retrieve=retrieve,
            llm=llm,
            require_transmissibility_evidence=(
                require_transmissibility_evidence
            ),
        )
    elif child_name == "admit_target_room_removal":
        report = _admit_target_room_removal(
            target_scenario,
            disease_name=disease_name,
            setting=setting,
            search=search,
            retrieve=retrieve,
            llm=llm,
            search_limit=search_limit,
        )
    elif child_name == "admit_target_exposure_scenario":
        report = _admit_target_exposure_scenario(
            target_scenario,
            transmission_model,
            disease_name=disease_name,
            setting=setting,
            search=search,
            retrieve=retrieve,
            llm=llm,
            search_limit=search_limit,
        )
    elif child_name == "select_transmission_model":
        report = _select_transmission_model(transmission_model)
    elif child_name == "compute_safety_bound":
        report = _compute_safety_bound(observations)
    else:
        raise KeyError(f"unknown guideline child: {child_name}")

    observation = GuidelineObservation(child=child_name, report=report)
    observations[child_name] = observation
    return observation


def _run_infectiousness_child(
    *,
    disease_name: str,
    setting: str,
    search_limit: int,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    llm: LLMExtractCallable | None,
    require_transmissibility_evidence: bool,
) -> GuidelineChildReport:
    child = guideline_child("estimate_pathogen_infectiousness")
    _, report = _run_estimate_pathogen_infectiousness_episode(
        disease_name=disease_name,
        setting=setting,
        search_limit=search_limit,
        search=search,
        retrieve=retrieve,
        llm=llm,
        require_transmissibility_evidence=require_transmissibility_evidence,
    )
    outputs = {
        "quanta_concentration_exhaled_m3": _infectiousness_output(report),
        "infectiousness_estimation": report.as_record(),
    }
    if report.passed:
        return pass_guideline_child(
            child,
            outputs,
            identities=report.identities,
        )
    return replace(
        fail_guideline_child(child, _infectiousness_errors(report), outputs=outputs),
        score=report.score,
    )


def _admit_target_room_removal(
    target_scenario: Mapping[str, Any],
    *,
    disease_name: str,
    setting: str,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    llm: LLMExtractCallable | None,
    search_limit: int,
) -> GuidelineChildReport:
    child = guideline_child("admit_target_room_removal")
    if "removal_capacity_m3_s" not in target_scenario:
        report = _reconstruct_target_room_removal(target_scenario)
        if report.passed or not _missing_target_only(report):
            return report

        return _admit_target_room_removal_from_assumption_search(
            report,
            target_scenario,
            disease_name=disease_name,
            setting=setting,
            search=search,
            retrieve=retrieve,
            llm=llm,
            search_limit=search_limit,
        )

    missing = _missing_target_fields(
        target_scenario,
        ("removal_capacity_m3_s",),
    )
    if missing:
        return fail_guideline_child(child, missing)

    removal_capacity = _positive_target(
        target_scenario["removal_capacity_m3_s"],
        "removal_capacity_m3_s",
    )
    if isinstance(removal_capacity, CalculationError):
        return fail_guideline_child(child, (removal_capacity,))

    return pass_guideline_child(
        child,
        {
            "removal_capacity_m3_s": removal_capacity,
            "target_room_removal": {
                "basis": "user_supplied_target_scenario",
                "fields": ("removal_capacity_m3_s",),
            },
        },
    )


def _reconstruct_target_room_removal(
    target_scenario: Mapping[str, Any],
) -> GuidelineChildReport:
    child = guideline_child("admit_target_room_removal")
    errors = list(
        _missing_target_fields(
            target_scenario,
            (
                "floor_area_m2",
                "recirculation_flow_m3_s",
                "filter_efficiency",
                "deactivation_rate_h",
                "settling_speed_m_h",
            ),
        )
    )
    if (
        "room_volume_m3" not in target_scenario
        and "room_height_m" not in target_scenario
    ):
        errors.append(
            CalculationError(
                code="missing_target_scenario_input",
                field="room_volume_m3",
                message=(
                    "room_volume_m3 or room_height_m must be supplied or "
                    "admitted before guideline calculation"
                ),
            )
        )
    if (
        "air_exchange_rate_h" not in target_scenario
        and "outdoor_air_flow_m3_s" not in target_scenario
    ):
        errors.append(
            CalculationError(
                code="missing_target_scenario_input",
                field="air_exchange_rate_h",
                message=(
                    "air_exchange_rate_h or outdoor_air_flow_m3_s must be "
                    "supplied or admitted before guideline calculation"
                ),
            )
        )
    if errors:
        return fail_guideline_child(child, tuple(errors))

    values, errors = _admit_room_removal_numbers(target_scenario)
    if errors:
        return fail_guideline_child(child, tuple(errors))

    reports, outputs = _run_target_removal_calculations(values)
    calculation_errors = tuple(
        error
        for report in reports
        for error in report.errors
    )
    if calculation_errors:
        return fail_guideline_child(child, calculation_errors)

    removal_capacity = outputs.get("removal_capacity_m3_s")
    if removal_capacity is None:
        return fail_guideline_child(
            child,
            (
                CalculationError(
                    code="missing_output",
                    field="removal_capacity_m3_s",
                    message="target room reconstruction did not compute removal capacity",
                ),
            ),
        )
    if removal_capacity <= 0.0:
        return fail_guideline_child(
            child,
            (
                CalculationError(
                    code="invalid_target_scenario_input",
                    field="removal_capacity_m3_s",
                    message="computed removal_capacity_m3_s must be positive",
                ),
            ),
        )

    return pass_guideline_child(
        child,
        {
            "removal_capacity_m3_s": removal_capacity,
            "target_room_removal": {
                "basis": "user_supplied_detailed_target_scenario",
                "fields": tuple(sorted(values)),
            },
            "normalized_room_fields": outputs,
        },
        identities=tuple(
            identity
            for report in reports
            for identity in report.identities
        ),
    )


def _admit_target_room_removal_from_assumption_search(
    failed_direct_report: GuidelineChildReport,
    target_scenario: Mapping[str, Any],
    *,
    disease_name: str,
    setting: str,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    llm: LLMExtractCallable | None,
    search_limit: int,
) -> GuidelineChildReport:
    child = guideline_child("admit_target_room_removal")
    assumption_search = _find_target_assumption_sources(
        child.name,
        disease_name=disease_name,
        setting=_target_assumption_setting(setting, target_scenario),
        search=search,
        retrieve=retrieve,
        search_limit=search_limit,
    )
    failure_outputs = {
        "assumption_search": assumption_search.as_record(),
    }
    if not assumption_search.passed:
        return fail_guideline_child(
            child,
            failed_direct_report.errors + assumption_search.errors,
            outputs=failure_outputs,
        )

    sources = _guideline_assumption_sources(
        assumption_search.outputs.get("assumption_sources", ())
    )
    if not sources:
        return fail_guideline_child(
            child,
            failed_direct_report.errors
            + (
                CalculationError(
                    code="missing_assumption_sources",
                    field="assumption_sources",
                    message=(
                        "target room assumption search returned no "
                        "extractable text"
                    ),
                ),
            ),
            outputs=failure_outputs,
        )

    _, removal_report = _run_build_removal_capacity_episode(
        llm,
        sources,
        admit_assumption=True,
        assumption_services=AssumptionServices(search, retrieve, search_limit),
        target_context=_target_assumption_context(
            child.name,
            target_scenario,
            failed_direct_report.errors,
            disease_name=disease_name,
            setting=setting,
        ),
    )
    outputs = {
        **failure_outputs,
        "assumption_report": removal_report.as_record(),
    }
    if not removal_report.passed:
        return fail_guideline_child(
            child,
            failed_direct_report.errors + removal_report.errors,
            outputs={
                **outputs,
                **dict(removal_report.outputs),
            },
        )

    removal_capacity = removal_report.outputs.get("removal_capacity_m3_s")
    if removal_capacity is None:
        return fail_guideline_child(
            child,
            (
                CalculationError(
                    code="missing_output",
                    field="removal_capacity_m3_s",
                    message=(
                        "target room assumption admission did not compute "
                        "removal capacity"
                    ),
                ),
            ),
            outputs=outputs,
        )

    return pass_guideline_child(
        child,
        {
            **outputs,
            "removal_capacity_m3_s": removal_capacity,
            "target_room_removal": {
                "basis": "evidence_admitted_target_room_assumption",
                "source_refs": tuple(
                    removal_report.outputs.get("source_refs", ())
                ),
            },
            **_selected_outputs(
                removal_report.outputs,
                (
                    "admission_basis",
                    "admitted_assumptions",
                    "evidence_basis",
                    "extracted_values",
                    "normalized_room_fields",
                    "source_refs",
                ),
            ),
        },
        identities=removal_report.identities,
    )


def _admit_target_exposure_scenario(
    target_scenario: Mapping[str, Any],
    transmission_model: str,
    *,
    disease_name: str,
    setting: str,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    llm: LLMExtractCallable | None,
    search_limit: int,
) -> GuidelineChildReport:
    child = guideline_child("admit_target_exposure_scenario")
    fields = (
        "breathing_flow_m3_s",
        "mask_penetration",
    )
    requires_risk_tolerance = transmission_model in CUMULATIVE_EXPOSURE_MODELS
    if requires_risk_tolerance:
        fields = (*fields, "risk_tolerance")
    if transmission_model in FIXED_SUSCEPTIBLE_MODELS:
        fields = (*fields, "susceptible_count")
    elif transmission_model in FIXED_EXPOSURE_TIME_MODELS:
        fields = (*fields, "exposure_time_s")
    elif transmission_model in INDOOR_REPRODUCTIVE_NUMBER_MODELS:
        fields = (*fields, "susceptible_count", "exposure_time_s")
        if (
            transmission_model
            == STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER
        ):
            fields = (*fields, *JET_CORRECTION_TARGET_FIELDS)
    elif transmission_model == RELATIVE_RISK_INDEX:
        fields = (*fields, "occupant_count", "exposure_time_s")

    errors: list[CalculationError] = []
    behavior_outputs: dict[str, Any] = {}
    behavior_identities: tuple[str, ...] = ()
    behavior_fields = ("breathing_flow_m3_s", "mask_penetration")
    missing_behavior = _missing_target_fields(
        target_scenario,
        behavior_fields,
    )
    if missing_behavior:
        behavior_outputs, behavior_errors, behavior_identities = (
            _admit_target_behavior_from_assumption_search(
                missing_behavior,
                target_scenario,
                disease_name=disease_name,
                setting=setting,
                search=search,
                retrieve=retrieve,
                llm=llm,
                search_limit=search_limit,
            )
        )
        errors.extend(behavior_errors)
    else:
        breathing_flow = _positive_target(
            target_scenario["breathing_flow_m3_s"],
            "breathing_flow_m3_s",
        )
        if isinstance(breathing_flow, CalculationError):
            errors.append(breathing_flow)

        mask_penetration = _probability_target(
            target_scenario["mask_penetration"],
            "mask_penetration",
        )
        if isinstance(mask_penetration, CalculationError):
            errors.append(mask_penetration)

        if not isinstance(breathing_flow, CalculationError):
            behavior_outputs["breathing_flow_m3_s"] = breathing_flow
        if not isinstance(mask_penetration, CalculationError):
            behavior_outputs["mask_penetration"] = mask_penetration

    missing = _missing_target_fields(
        target_scenario,
        tuple(field for field in fields if field not in behavior_fields),
    )
    errors.extend(missing)

    risk_tolerance = None
    if requires_risk_tolerance and "risk_tolerance" in target_scenario:
        risk_tolerance = _non_negative_target(
            target_scenario["risk_tolerance"],
            "risk_tolerance",
        )
        if isinstance(risk_tolerance, CalculationError):
            errors.append(risk_tolerance)

    extra_outputs: dict[str, float] = {}
    if transmission_model in FIXED_SUSCEPTIBLE_MODELS:
        if "susceptible_count" in target_scenario:
            susceptible_count = _positive_target(
                target_scenario["susceptible_count"],
                "susceptible_count",
            )
            if isinstance(susceptible_count, CalculationError):
                errors.append(susceptible_count)
            else:
                extra_outputs["susceptible_count"] = susceptible_count
    elif transmission_model in FIXED_EXPOSURE_TIME_MODELS:
        if "exposure_time_s" in target_scenario:
            exposure_time = _positive_target(
                target_scenario["exposure_time_s"],
                "exposure_time_s",
            )
            if isinstance(exposure_time, CalculationError):
                errors.append(exposure_time)
            else:
                extra_outputs["exposure_time_s"] = exposure_time
    elif transmission_model in INDOOR_REPRODUCTIVE_NUMBER_MODELS:
        if "susceptible_count" in target_scenario:
            susceptible_count = _positive_target(
                target_scenario["susceptible_count"],
                "susceptible_count",
            )
            if isinstance(susceptible_count, CalculationError):
                errors.append(susceptible_count)
            else:
                extra_outputs["susceptible_count"] = susceptible_count
        if "exposure_time_s" in target_scenario:
            exposure_time = _positive_target(
                target_scenario["exposure_time_s"],
                "exposure_time_s",
            )
            if isinstance(exposure_time, CalculationError):
                errors.append(exposure_time)
            else:
                extra_outputs["exposure_time_s"] = exposure_time
        if (
            transmission_model
            == STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER
        ):
            if "plume_encounter_probability" in target_scenario:
                plume_encounter_probability = _probability_target(
                    target_scenario["plume_encounter_probability"],
                    "plume_encounter_probability",
                )
                if isinstance(plume_encounter_probability, CalculationError):
                    errors.append(plume_encounter_probability)
                else:
                    extra_outputs["plume_encounter_probability"] = (
                        plume_encounter_probability
                    )
            for field in (
                "mouth_area_m2",
                "entrainment_coefficient",
                "neighbor_distance_m",
                "well_mixed_dilution_factor",
            ):
                if field in target_scenario:
                    value = _positive_target(target_scenario[field], field)
                    if isinstance(value, CalculationError):
                        errors.append(value)
                    else:
                        extra_outputs[field] = value
    elif transmission_model == RELATIVE_RISK_INDEX:
        if "occupant_count" in target_scenario:
            occupant_count = _non_negative_target(
                target_scenario["occupant_count"],
                "occupant_count",
            )
            if isinstance(occupant_count, CalculationError):
                errors.append(occupant_count)
            else:
                extra_outputs["occupant_count"] = occupant_count
        if "exposure_time_s" in target_scenario:
            exposure_time = _non_negative_target(
                target_scenario["exposure_time_s"],
                "exposure_time_s",
            )
            if isinstance(exposure_time, CalculationError):
                errors.append(exposure_time)
            else:
                extra_outputs["exposure_time_s"] = exposure_time

    if errors:
        return fail_guideline_child(
            child,
            tuple(errors),
            outputs=behavior_outputs,
        )

    outputs = {
        "target_exposure_scenario": {
            "basis": "user_supplied_target_scenario",
            "fields": fields,
        },
        **behavior_outputs,
        **extra_outputs,
    }
    if risk_tolerance is not None:
        outputs["risk_tolerance"] = risk_tolerance

    return pass_guideline_child(
        child,
        outputs,
        identities=behavior_identities,
    )


def _admit_target_behavior_from_assumption_search(
    missing_behavior: tuple[CalculationError, ...],
    target_scenario: Mapping[str, Any],
    *,
    disease_name: str,
    setting: str,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    llm: LLMExtractCallable | None,
    search_limit: int,
) -> tuple[dict[str, Any], tuple[CalculationError, ...], tuple[str, ...]]:
    assumption_search = _find_target_assumption_sources(
        "admit_target_exposure_scenario",
        disease_name=disease_name,
        setting=_target_assumption_setting(setting, target_scenario),
        search=search,
        retrieve=retrieve,
        search_limit=search_limit,
    )
    outputs: dict[str, Any] = {
        "assumption_search": assumption_search.as_record(),
    }
    if not assumption_search.passed:
        return (
            outputs,
            missing_behavior + assumption_search.errors,
            (),
        )

    sources = _guideline_assumption_sources(
        assumption_search.outputs.get("assumption_sources", ())
    )
    if not sources:
        return (
            outputs,
            missing_behavior
            + (
                CalculationError(
                    code="missing_assumption_sources",
                    field="assumption_sources",
                    message=(
                        "target behavior assumption search returned no "
                        "extractable text"
                    ),
                ),
            ),
            (),
        )

    _, behavior_report = _run_admit_behavior_assumptions_episode(
        llm,
        sources,
        admit_assumption=True,
        assumption_services=AssumptionServices(search, retrieve, search_limit),
        target_context=_target_assumption_context(
            "admit_target_exposure_scenario",
            target_scenario,
            missing_behavior,
            disease_name=disease_name,
            setting=setting,
        ),
    )
    outputs["assumption_report"] = behavior_report.as_record()
    outputs.update(
        _selected_outputs(
            behavior_report.outputs,
            (
                "admission_basis",
                "admitted_assumptions",
                "behavior_assumptions",
                "breathing_flow_m3_s",
                "extracted_values",
                "mask_penetration",
                "source_refs",
            ),
        )
    )
    if not behavior_report.passed:
        return outputs, missing_behavior + behavior_report.errors, ()

    return outputs, (), behavior_report.identities


def _select_transmission_model(
    transmission_model: str,
) -> GuidelineChildReport:
    child = guideline_child("select_transmission_model")
    if transmission_model not in GUIDELINE_OUTPUTS:
        return fail_guideline_child(
            child,
            (
                CalculationError(
                    code="unsupported_transmission_model",
                    field="transmission_model",
                    message=(
                        f"{transmission_model} is not a supported guideline "
                        "output"
                    ),
                ),
            ),
        )

    return pass_guideline_child(
        child,
        {
            "transmission_model": transmission_model,
            "relative_transmissibility": 1.0,
            "model_basis": {
                "equation": _model_equation(transmission_model),
                "required_outputs": _model_required_outputs(
                    transmission_model
                ),
                "deterministic_calculations": _model_calculations(
                    transmission_model
                ),
                "relative_transmissibility_basis": (
                    "pathogen infectiousness has already been synthesized for "
                    "the target disease and activity"
                ),
            },
        },
    )


def _model_equation(transmission_model: str) -> str:
    if transmission_model == STEADY_STATE_INDOOR_REPRODUCTIVE_NUMBER:
        return "steady-state indoor reproductive number"
    if transmission_model == STEADY_STATE_FIRST_TRANSMISSION_PROBABILITY:
        return "steady-state first-transmission probability"
    if (
        transmission_model
        == STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER
    ):
        return "steady-state respiratory-jet-corrected reproductive number"
    if transmission_model in TRANSIENT_MODELS:
        return "transient well-mixed exposure correction"
    if transmission_model == RELATIVE_RISK_INDEX:
        return "relative risk index"
    return "steady-state cumulative exposure bound"


def _model_required_outputs(transmission_model: str) -> tuple[str, ...]:
    required = [
        "estimate_pathogen_infectiousness.quanta_concentration_exhaled_m3",
        "admit_target_exposure_scenario.breathing_flow_m3_s",
        "admit_target_exposure_scenario.mask_penetration",
    ]

    if transmission_model in EQ5_HAZARD_MODELS:
        required.extend(
            (
                "admit_target_room_removal.removal_capacity_m3_s",
                "select_transmission_model.relative_transmissibility",
            )
        )
    if transmission_model in CUMULATIVE_EXPOSURE_MODELS:
        required.append("admit_target_exposure_scenario.risk_tolerance")

    if transmission_model in FIXED_SUSCEPTIBLE_MODELS:
        required.append("admit_target_exposure_scenario.susceptible_count")
    elif transmission_model in FIXED_EXPOSURE_TIME_MODELS:
        required.append("admit_target_exposure_scenario.exposure_time_s")
    elif transmission_model == RELATIVE_RISK_INDEX:
        required.extend(
            (
                "admit_target_exposure_scenario.occupant_count",
                "admit_target_exposure_scenario.exposure_time_s",
                "admit_target_room_removal."
                "normalized_room_fields.air_exchange_rate_s",
                "admit_target_room_removal."
                "normalized_room_fields.room_volume_m3",
            )
        )
    elif transmission_model in INDOOR_REPRODUCTIVE_NUMBER_MODELS:
        required.extend(
            (
                "admit_target_exposure_scenario.susceptible_count",
                "admit_target_exposure_scenario.exposure_time_s",
            )
        )
        if (
            transmission_model
            == STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER
        ):
            required.extend(
                f"admit_target_exposure_scenario.{field}"
                for field in JET_CORRECTION_TARGET_FIELDS
            )

    if transmission_model in TRANSIENT_MODELS:
        required.append(
            "admit_target_room_removal.normalized_room_fields.removal_rate_s"
        )

    return tuple(required)


def _model_calculations(transmission_model: str) -> tuple[str, ...]:
    if transmission_model == RELATIVE_RISK_INDEX:
        return (RELATIVE_RISK_INDEX,)

    calculations = ["steady_state_inhalation_transmission_rate"]
    if transmission_model in INDOOR_REPRODUCTIVE_NUMBER_MODELS:
        calculations.append("indoor_reproductive_number")
        if transmission_model == STEADY_STATE_FIRST_TRANSMISSION_PROBABILITY:
            calculations.append("first_transmission_probability")
        elif (
            transmission_model
            == STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER
        ):
            calculations.extend(
                (
                    "jet_dilution_ratio",
                    "respiratory_jet_multiplier",
                    "jet_corrected_reproductive_number",
                )
            )
    elif transmission_model in CUMULATIVE_EXPOSURE_MODELS:
        calculations.append(STEADY_STATE_CUMULATIVE_EXPOSURE)
        if transmission_model != STEADY_STATE_CUMULATIVE_EXPOSURE:
            calculations.append(transmission_model)

    return tuple(calculations)


def _compute_safety_bound(
    observations: Mapping[str, GuidelineObservation],
) -> GuidelineChildReport:
    child = guideline_child("compute_safety_bound")
    exposure_values = {
        "breathing_flow_m3_s": _output(
            observations,
            "admit_target_exposure_scenario",
            "breathing_flow_m3_s",
        ),
        "quanta_concentration_exhaled_m3": _output(
            observations,
            "estimate_pathogen_infectiousness",
            "quanta_concentration_exhaled_m3",
        ),
        "mask_penetration": _output(
            observations,
            "admit_target_exposure_scenario",
            "mask_penetration",
        ),
    }
    transmission_model = _output(
        observations,
        "select_transmission_model",
        "transmission_model",
    )
    all_values = {
        **exposure_values,
        "transmission_model": transmission_model,
    }
    needs_eq5_hazard = transmission_model in EQ5_HAZARD_MODELS
    if needs_eq5_hazard:
        all_values.update(
            {
                "removal_capacity_m3_s": _output(
                    observations,
                    "admit_target_room_removal",
                    "removal_capacity_m3_s",
                ),
                "relative_transmissibility": _output(
                    observations,
                    "select_transmission_model",
                    "relative_transmissibility",
                ),
            }
        )
    if transmission_model in CUMULATIVE_EXPOSURE_MODELS:
        all_values["risk_tolerance"] = _output(
            observations,
            "admit_target_exposure_scenario",
            "risk_tolerance",
        )
    if transmission_model in FIXED_SUSCEPTIBLE_MODELS:
        all_values["susceptible_count"] = _output(
            observations,
            "admit_target_exposure_scenario",
            "susceptible_count",
        )
    elif transmission_model in FIXED_EXPOSURE_TIME_MODELS:
        all_values["exposure_time_s"] = _output(
            observations,
            "admit_target_exposure_scenario",
            "exposure_time_s",
        )
    elif transmission_model == RELATIVE_RISK_INDEX:
        all_values["occupant_count"] = _output(
            observations,
            "admit_target_exposure_scenario",
            "occupant_count",
        )
        all_values["exposure_time_s"] = _output(
            observations,
            "admit_target_exposure_scenario",
            "exposure_time_s",
        )
        all_values["air_exchange_rate_s"] = _room_output(
            observations,
            "air_exchange_rate_s",
        )
        all_values["room_volume_m3"] = _room_output(
            observations,
            "room_volume_m3",
        )
    elif transmission_model in INDOOR_REPRODUCTIVE_NUMBER_MODELS:
        all_values["susceptible_count"] = _output(
            observations,
            "admit_target_exposure_scenario",
            "susceptible_count",
        )
        all_values["exposure_time_s"] = _output(
            observations,
            "admit_target_exposure_scenario",
            "exposure_time_s",
        )
        if (
            transmission_model
            == STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER
        ):
            for field in JET_CORRECTION_TARGET_FIELDS:
                all_values[field] = _output(
                    observations,
                    "admit_target_exposure_scenario",
                    field,
                )
    if transmission_model in TRANSIENT_MODELS:
        all_values["concentration_relaxation_rate_s"] = _room_output(
            observations,
            "removal_rate_s",
        )

    errors = tuple(
        CalculationError(
            code="missing_prerequisite_output",
            field=field,
            message=f"{field} is required before safety-bound calculation",
        )
        for field, value in all_values.items()
        if value is None
    )
    if errors:
        return fail_guideline_child(child, errors)

    outputs = {
        "transmission_model": transmission_model,
    }
    identities: list[str] = []
    if transmission_model == RELATIVE_RISK_INDEX:
        selected_calculation = run_named_calculation(
            RELATIVE_RISK_INDEX,
            {
                "occupant_count": all_values["occupant_count"],
                "exposure_time_s": all_values["exposure_time_s"],
                **exposure_values,
                "air_exchange_rate_s": all_values["air_exchange_rate_s"],
                "room_volume_m3": all_values["room_volume_m3"],
            },
        )
        outputs.update(selected_calculation.outputs)
        identities.extend(selected_calculation.identities)
        if not selected_calculation.passed:
            return fail_guideline_child(
                child,
                selected_calculation.errors,
                outputs=outputs,
            )
    else:
        hazard_values = {
            **exposure_values,
            "removal_capacity_m3_s": all_values["removal_capacity_m3_s"],
            "relative_transmissibility": all_values[
                "relative_transmissibility"
            ],
        }
        transmission_rate = run_named_calculation(
            "steady_state_inhalation_transmission_rate",
            hazard_values,
        )
        identities.extend(transmission_rate.identities)
        if not transmission_rate.passed:
            return fail_guideline_child(
                child,
                transmission_rate.errors,
                outputs=outputs,
            )
        outputs.update(transmission_rate.outputs)

    if transmission_model in INDOOR_REPRODUCTIVE_NUMBER_MODELS:
        selected_calculation = run_named_calculation(
            "indoor_reproductive_number",
            {
                "susceptible_count": all_values["susceptible_count"],
                "transmission_rate_s": (
                    transmission_rate.outputs["transmission_rate_s"]
                ),
                "exposure_time_s": all_values["exposure_time_s"],
            },
        )
        outputs.update(selected_calculation.outputs)
        identities.extend(selected_calculation.identities)
        if not selected_calculation.passed:
            return fail_guideline_child(
                child,
                selected_calculation.errors,
                outputs=outputs,
            )
        if (
            transmission_model
            == STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER
        ):
            outputs["long_range_indoor_reproductive_number"] = (
                selected_calculation.outputs["indoor_reproductive_number"]
            )
            jet_dilution = run_named_calculation(
                "jet_dilution_ratio",
                {
                    "mouth_area_m2": all_values["mouth_area_m2"],
                    "entrainment_coefficient": all_values[
                        "entrainment_coefficient"
                    ],
                    "neighbor_distance_m": all_values["neighbor_distance_m"],
                },
            )
            outputs.update(jet_dilution.outputs)
            identities.extend(jet_dilution.identities)
            if not jet_dilution.passed:
                return fail_guideline_child(
                    child,
                    jet_dilution.errors,
                    outputs=outputs,
                )

            jet_multiplier = run_named_calculation(
                "respiratory_jet_multiplier",
                {
                    "plume_encounter_probability": all_values[
                        "plume_encounter_probability"
                    ],
                    "mouth_area_m2": all_values["mouth_area_m2"],
                    "entrainment_coefficient": all_values[
                        "entrainment_coefficient"
                    ],
                    "neighbor_distance_m": all_values["neighbor_distance_m"],
                    "well_mixed_dilution_factor": all_values[
                        "well_mixed_dilution_factor"
                    ],
                    "susceptible_count": all_values["susceptible_count"],
                },
            )
            outputs.update(jet_multiplier.outputs)
            identities.extend(jet_multiplier.identities)
            if not jet_multiplier.passed:
                return fail_guideline_child(
                    child,
                    jet_multiplier.errors,
                    outputs=outputs,
                )

            jet_corrected = run_named_calculation(
                "jet_corrected_reproductive_number",
                {
                    "indoor_reproductive_number": (
                        selected_calculation.outputs[
                            "indoor_reproductive_number"
                        ]
                    ),
                    "jet_multiplier": jet_multiplier.outputs[
                        "jet_multiplier"
                    ],
                },
            )
            outputs["jet_corrected_indoor_reproductive_number"] = (
                jet_corrected.outputs.get("indoor_reproductive_number")
            )
            identities.extend(jet_corrected.identities)
            if not jet_corrected.passed:
                return fail_guideline_child(
                    child,
                    jet_corrected.errors,
                    outputs=outputs,
                )
        elif (
            transmission_model
            == STEADY_STATE_FIRST_TRANSMISSION_PROBABILITY
        ):
            probability = run_named_calculation(
                "first_transmission_probability",
                {
                    "indoor_reproductive_number": (
                        selected_calculation.outputs[
                            "indoor_reproductive_number"
                        ]
                    ),
                },
            )
            outputs.update(probability.outputs)
            identities.extend(probability.identities)
            if not probability.passed:
                return fail_guideline_child(
                    child,
                    probability.errors,
                    outputs=outputs,
                )
    elif transmission_model in CUMULATIVE_EXPOSURE_MODELS:
        calculation = run_named_calculation(
            STEADY_STATE_CUMULATIVE_EXPOSURE,
            {
                "risk_tolerance": all_values["risk_tolerance"],
                **hazard_values,
            },
        )
        outputs.update(calculation.outputs)
        identities.extend(calculation.identities)
        if not calculation.passed:
            return fail_guideline_child(
                child,
                calculation.errors,
                outputs=outputs,
            )
        if transmission_model in TRANSIENT_MODELS:
            outputs[
                "steady_state_cumulative_exposure_bound_person_s"
            ] = calculation.outputs["cumulative_exposure_bound_person_s"]

    if transmission_model == TRANSIENT_CUMULATIVE_EXPOSURE:
        selected_calculation = run_named_calculation(
            TRANSIENT_CUMULATIVE_EXPOSURE,
            {
                "steady_state_bound_person_s": (
                    calculation.outputs["cumulative_exposure_bound_person_s"]
                ),
                "concentration_relaxation_rate_s": all_values[
                    "concentration_relaxation_rate_s"
                ],
                "exposure_time_s": all_values["exposure_time_s"],
            },
        )
        outputs.update(selected_calculation.outputs)
        identities.extend(selected_calculation.identities)
        if not selected_calculation.passed:
            return fail_guideline_child(
                child,
                selected_calculation.errors,
                outputs=outputs,
            )
    elif transmission_model in {
        STEADY_STATE_TIME_BOUND,
        TRANSIENT_TIME_BOUND,
    }:
        selected_inputs = {
            "susceptible_count": all_values["susceptible_count"],
        }
        if transmission_model == TRANSIENT_TIME_BOUND:
            selected_inputs.update(
                {
                    "steady_state_bound_person_s": (
                        calculation.outputs[
                            "cumulative_exposure_bound_person_s"
                        ]
                    ),
                    "concentration_relaxation_rate_s": all_values[
                        "concentration_relaxation_rate_s"
                    ],
                }
            )
        else:
            selected_inputs["cumulative_exposure_bound_person_s"] = (
                calculation.outputs["cumulative_exposure_bound_person_s"]
            )
        selected_calculation = run_named_calculation(
            transmission_model,
            selected_inputs,
        )
        outputs.update(selected_calculation.outputs)
        identities.extend(selected_calculation.identities)
        if not selected_calculation.passed:
            return fail_guideline_child(
                child,
                selected_calculation.errors,
                outputs=outputs,
            )
    elif transmission_model in {
        STEADY_STATE_SUSCEPTIBLE_BOUND,
        TRANSIENT_SUSCEPTIBLE_BOUND,
    }:
        selected_inputs = {
            "exposure_time_s": all_values["exposure_time_s"],
        }
        if transmission_model == TRANSIENT_SUSCEPTIBLE_BOUND:
            selected_inputs.update(
                {
                    "steady_state_bound_person_s": (
                        calculation.outputs[
                            "cumulative_exposure_bound_person_s"
                        ]
                    ),
                    "concentration_relaxation_rate_s": all_values[
                        "concentration_relaxation_rate_s"
                    ],
                }
            )
        else:
            selected_inputs["cumulative_exposure_bound_person_s"] = (
                calculation.outputs["cumulative_exposure_bound_person_s"]
            )
        selected_calculation = run_named_calculation(
            transmission_model,
            selected_inputs,
        )
        outputs.update(selected_calculation.outputs)
        identities.extend(selected_calculation.identities)
        if not selected_calculation.passed:
            return fail_guideline_child(
                child,
                selected_calculation.errors,
                outputs=outputs,
            )
    elif transmission_model not in GUIDELINE_OUTPUTS:
        return fail_guideline_child(
            child,
            (
                CalculationError(
                    code="unsupported_transmission_model",
                    field="transmission_model",
                    message=(
                        f"{transmission_model} is not a supported "
                        "guideline output"
                    ),
                ),
            ),
            outputs=outputs,
        )

    output_name, unit = GUIDELINE_OUTPUTS[transmission_model]
    outputs["safety_guideline_result"] = {
        "transmission_model": transmission_model,
        "output": output_name,
        "value": outputs[output_name],
        "unit": unit,
    }

    return pass_guideline_child(
        child,
        outputs,
        identities=tuple(identities),
    )


def _missing_target_fields(
    target_scenario: Mapping[str, Any],
    fields: tuple[str, ...],
) -> tuple[CalculationError, ...]:
    return tuple(
        CalculationError(
            code="missing_target_scenario_input",
            field=field,
            message=(
                f"{field} must be supplied or admitted before guideline "
                "calculation"
            ),
        )
        for field in fields
        if field not in target_scenario
    )


def _missing_target_only(report: GuidelineChildReport) -> bool:
    return bool(report.errors) and all(
        error.code == "missing_target_scenario_input"
        for error in report.errors
    )


def _find_target_assumption_sources(
    child: str,
    *,
    disease_name: str,
    setting: str,
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    search_limit: int,
) -> AssumptionSearchReport:
    _, report = _run_find_assumption_sources_episode(
        child=child,
        disease_name=disease_name,
        setting=setting,
        search=search,
        retrieve=retrieve,
        limit=search_limit,
    )
    return report


def _guideline_assumption_sources(
    raw_candidates: object,
) -> tuple[EvidenceSource, ...]:
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


def _target_assumption_context(
    child: str,
    target_scenario: Mapping[str, Any],
    errors: tuple[CalculationError, ...],
    *,
    disease_name: str,
    setting: str,
) -> Mapping[str, Any]:
    return {
        "disease_name": disease_name,
        "setting": setting,
        "child": child,
        "target_scenario": dict(target_scenario),
        "missing_fields": tuple(
            error.field
            for error in errors
            if error.field
            and error.code == "missing_target_scenario_input"
        ),
    }


def _target_assumption_setting(
    setting: str,
    target_scenario: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return scoped search hints from the outbreak setting and target room."""

    hints = _setting_parts(setting)
    for field in TARGET_ASSUMPTION_HINT_FIELDS:
        hints += _setting_parts(target_scenario.get(field))
    return tuple(dict.fromkeys(hints))


def _setting_parts(raw: object) -> tuple[str, ...]:
    if raw is None or isinstance(raw, (bytes, bytearray, Mapping)):
        return ()
    if isinstance(raw, str):
        value = raw.strip()
        return (value,) if value else ()
    if isinstance(raw, Sequence):
        return tuple(
            part
            for part in (
                str(value).strip()
                for value in raw
                if not isinstance(value, (bytes, bytearray, Mapping))
            )
            if part
        )
    return ()


def _selected_outputs(
    outputs: Mapping[str, Any],
    keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        key: outputs[key]
        for key in keys
        if key in outputs
    }


def _admit_room_removal_numbers(
    target_scenario: Mapping[str, Any],
) -> tuple[dict[str, float], tuple[CalculationError, ...]]:
    validators = {
        "floor_area_m2": _positive_target,
        "room_height_m": _positive_target,
        "room_volume_m3": _positive_target,
        "air_exchange_rate_h": _non_negative_target,
        "outdoor_air_flow_m3_s": _non_negative_target,
        "recirculation_flow_m3_s": _non_negative_target,
        "filter_efficiency": _probability_target,
        "deactivation_rate_h": _non_negative_target,
        "settling_speed_m_h": _non_negative_target,
    }
    errors: list[CalculationError] = []
    values: dict[str, float] = {}
    for field, validator in validators.items():
        if field not in target_scenario:
            continue
        result = validator(target_scenario[field], field)
        if isinstance(result, CalculationError):
            errors.append(result)
        else:
            values[field] = result
    return values, tuple(errors)


def _run_target_removal_calculations(
    values: Mapping[str, float],
):
    reports = []
    outputs: dict[str, float] = {"floor_area_m2": values["floor_area_m2"]}
    if "room_volume_m3" in values:
        room_volume_m3 = values["room_volume_m3"]
        outputs["room_volume_m3"] = room_volume_m3
    else:
        volume = run_named_calculation("room_volume", values)
        reports.append(volume)
        room_volume_m3 = volume.outputs.get("room_volume_m3")
    if room_volume_m3 is None:
        return tuple(reports), outputs
    outputs["room_volume_m3"] = room_volume_m3

    if "room_height_m" in values:
        room_height_m = values["room_height_m"]
        outputs["room_height_m"] = room_height_m
    else:
        height = run_named_calculation(
            "room_height",
            {
                "room_volume_m3": room_volume_m3,
                "floor_area_m2": values["floor_area_m2"],
            },
        )
        reports.append(height)
        room_height_m = height.outputs.get("room_height_m")
        if room_height_m is not None:
            outputs["room_height_m"] = room_height_m

    if "air_exchange_rate_h" in values:
        air_exchange = run_named_calculation(
            "per_hour_to_per_second",
            {"rate_h": values["air_exchange_rate_h"]},
        )
        reports.append(air_exchange)
        air_exchange_rate_s = air_exchange.outputs.get("rate_s")
        if air_exchange_rate_s is not None:
            outputs["air_exchange_rate_s"] = air_exchange_rate_s
    else:
        air_exchange = run_named_calculation(
            "air_exchange_rate",
            {
                "outdoor_air_flow_m3_s": values["outdoor_air_flow_m3_s"],
                "room_volume_m3": room_volume_m3,
            },
        )
        reports.append(air_exchange)
        air_exchange_rate_s = air_exchange.outputs.get("air_exchange_rate_s")
        if air_exchange_rate_s is not None:
            outputs["air_exchange_rate_s"] = air_exchange_rate_s

    filtration = run_named_calculation(
        "filtration_loss_rate",
        {
            "recirculation_flow_m3_s": values["recirculation_flow_m3_s"],
            "filter_efficiency": values["filter_efficiency"],
            "room_volume_m3": room_volume_m3,
        },
    )
    deactivation = run_named_calculation(
        "per_hour_to_per_second",
        {"rate_h": values["deactivation_rate_h"]},
    )
    settling = run_named_calculation(
        "meters_per_hour_to_meters_per_second",
        {"speed_m_h": values["settling_speed_m_h"]},
    )
    reports.extend((filtration, deactivation, settling))
    outputs.update(filtration.outputs)
    if "rate_s" in deactivation.outputs:
        outputs["deactivation_rate_s"] = deactivation.outputs["rate_s"]
    if "speed_m_s" in settling.outputs:
        outputs["settling_speed_m_s"] = settling.outputs["speed_m_s"]
    if (
        air_exchange_rate_s is None
        or "filtration_rate_s" not in filtration.outputs
        or "rate_s" not in deactivation.outputs
        or "speed_m_s" not in settling.outputs
    ):
        return tuple(reports), outputs

    if room_height_m is not None:
        sedimentation = run_named_calculation(
            "sedimentation_rate",
            {
                "settling_speed_m_s": settling.outputs["speed_m_s"],
                "room_height_m": room_height_m,
            },
        )
        reports.append(sedimentation)
        outputs.update(sedimentation.outputs)

    purification = run_named_calculation(
        "air_purification_rate",
        {
            "air_exchange_rate_s": air_exchange_rate_s,
            "filtration_rate_s": filtration.outputs["filtration_rate_s"],
            "deactivation_rate_s": deactivation.outputs["rate_s"],
        },
    )
    reports.append(purification)
    outputs.update(purification.outputs)
    if "air_purification_rate_s" not in purification.outputs:
        return tuple(reports), outputs

    if "sedimentation_rate_s" in outputs:
        total_removal = run_named_calculation(
            "total_removal_rate",
            {
                "air_exchange_rate_s": air_exchange_rate_s,
                "filtration_rate_s": filtration.outputs[
                    "filtration_rate_s"
                ],
                "sedimentation_rate_s": outputs["sedimentation_rate_s"],
                "deactivation_rate_s": deactivation.outputs["rate_s"],
            },
        )
        reports.append(total_removal)
        outputs.update(total_removal.outputs)

    removal = run_named_calculation(
        "removal_capacity",
        {
            "room_volume_m3": room_volume_m3,
            "air_purification_rate_s": purification.outputs[
                "air_purification_rate_s"
            ],
            "settling_speed_m_s": settling.outputs["speed_m_s"],
            "floor_area_m2": values["floor_area_m2"],
        },
    )
    reports.append(removal)
    outputs.update(removal.outputs)
    return tuple(reports), outputs


def _positive_target(
    value: object,
    field: str,
) -> float | CalculationError:
    numeric = _real_target(value, field)
    if isinstance(numeric, CalculationError):
        return numeric
    if numeric <= 0.0:
        return CalculationError(
            code="invalid_target_scenario_input",
            field=field,
            message=f"{field} must be positive",
        )
    return numeric


def _non_negative_target(
    value: object,
    field: str,
) -> float | CalculationError:
    numeric = _real_target(value, field)
    if isinstance(numeric, CalculationError):
        return numeric
    if numeric < 0.0:
        return CalculationError(
            code="invalid_target_scenario_input",
            field=field,
            message=f"{field} must be non-negative",
        )
    return numeric


def _probability_target(
    value: object,
    field: str,
) -> float | CalculationError:
    numeric = _non_negative_target(value, field)
    if isinstance(numeric, CalculationError):
        return numeric
    if numeric > 1.0:
        return CalculationError(
            code="invalid_target_scenario_input",
            field=field,
            message=f"{field} must be in [0, 1]",
        )
    return numeric


def _real_target(
    value: object,
    field: str,
) -> float | CalculationError:
    if isinstance(value, bool) or not isinstance(value, Real):
        return CalculationError(
            code="invalid_target_scenario_input",
            field=field,
            message=f"{field} must be a real number",
        )
    numeric = float(value)
    if math.isnan(numeric):
        return CalculationError(
            code="invalid_target_scenario_input",
            field=field,
            message=f"{field} must not be NaN",
        )
    return numeric


def _infectiousness_output(report: InfectiousnessCreditReport) -> Any:
    child_outputs = report.main_outputs.get(
        "synthesize_pathogen_infectiousness",
        {},
    )
    return child_outputs.get("quanta_concentration_exhaled_m3")


def _infectiousness_errors(
    report: InfectiousnessCreditReport,
) -> tuple[CalculationError, ...]:
    return tuple(
        CalculationError(
            code=error.code,
            field=".".join(
                part
                for part in (error.child, error.field or error.output)
                if part
            ),
            message=error.message,
        )
        for error in report.errors
    ) or (
        CalculationError(
            code="missing_output",
            field="quanta_concentration_exhaled_m3",
            message=(
                "infectiousness estimation did not return a credited Cq"
            ),
        ),
    )


def _output(
    observations: Mapping[str, GuidelineObservation],
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


def _room_output(
    observations: Mapping[str, GuidelineObservation],
    field: str,
) -> Any:
    normalized_room_fields = _output(
        observations,
        "admit_target_room_removal",
        "normalized_room_fields",
    )
    if isinstance(normalized_room_fields, Mapping):
        return normalized_room_fields.get(field)
    return None


def _build_result(record: EpisodeRecord) -> GuidelineRecord:
    state = record.controller_state
    if not isinstance(state, GuidelineEpisodeState):
        raise TypeError("guideline Episode produced an unexpected state")
    return GuidelineRecord(state.to_report())


__all__ = [
    "GUIDELINE_CHILD_NAMES",
    "GUIDELINE_OUTPUTS",
    "RELATIVE_RISK_INDEX",
    "STEADY_STATE_CUMULATIVE_EXPOSURE",
    "STEADY_STATE_FIRST_TRANSMISSION_PROBABILITY",
    "STEADY_STATE_GUIDELINE_OUTPUTS",
    "STEADY_STATE_INDOOR_REPRODUCTIVE_NUMBER",
    "STEADY_STATE_JET_CORRECTED_INDOOR_REPRODUCTIVE_NUMBER",
    "STEADY_STATE_SUSCEPTIBLE_BOUND",
    "STEADY_STATE_TIME_BOUND",
    "TRANSIENT_CUMULATIVE_EXPOSURE",
    "TRANSIENT_SUSCEPTIBLE_BOUND",
    "TRANSIENT_TIME_BOUND",
    "GuidelineController",
    "GuidelineEpisodeState",
    "GuidelineEpisodeStep",
    "GuidelineObservation",
    "GuidelineRecord",
    "GuidelineRequest",
    "run_evaluate_safety_guideline_episode",
]
