"""Reusable searched assumption-source discovery for Cq children."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from episodic_workflow.credit_assignment.cq_calibration import (
    cq_normalization_rarefaction,
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
    OutbreakSearchSpec,
    RetrieveCallable,
    SearchCallable,
    SettingHints,
    candidate_sources_from_search_records,
    result_with_retrieved_text,
    search_results_from_provider,
    setting_hints_text,
    source_ref,
)


@dataclass(frozen=True)
class AssumptionSearchReport:
    """Scored assumption-search result for one Cq child."""

    child: str
    passed: bool
    score: float
    max_score: float
    outputs: Mapping[str, Any]
    errors: tuple[CalculationError, ...] = ()
    rarefaction: Mapping[str, Any] | None = None

    def as_record(self) -> dict[str, Any]:
        return {
            "child": self.child,
            "passed": self.passed,
            "score": self.score,
            "max_score": self.max_score,
            "outputs": dict(self.outputs),
            "errors": [error.as_record() for error in self.errors],
            "rarefaction": (
                None if self.rarefaction is None else dict(self.rarefaction)
            ),
        }


@dataclass(frozen=True)
class AssumptionSearchObservation:
    """One scoped assumption query and its retrieved candidates."""

    search: Mapping[str, str]
    results: tuple[Mapping[str, Any], ...] = ()
    errors: tuple[CalculationError, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "search": dict(self.search),
            "results": [dict(result) for result in self.results],
            "errors": [error.as_record() for error in self.errors],
        }


@dataclass(frozen=True)
class AssumptionSearchEpisodeStep:
    """Controller step for assumption source discovery."""

    stop: bool
    requests_transition: bool = False

    def as_record(self) -> dict[str, bool]:
        return {
            "stop": self.stop,
            "requests_transition": self.requests_transition,
        }


@dataclass(frozen=True)
class AssumptionSearchEpisodeState:
    """Accumulated assumption-search state."""

    child: str
    disease_name: str
    setting: str
    query_count: int
    observations: tuple[AssumptionSearchObservation, ...] = ()
    stop: bool = False

    def to_report(self) -> AssumptionSearchReport:
        return score_assumption_sources(
            self.child,
            self._search_records(),
            extra_outputs={"search_specs": self._search_specs()},
            errors=tuple(
                error
                for observation in self.observations
                for error in observation.errors
            ),
            score_history=self._score_history(),
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "child": self.child,
            "disease_name": self.disease_name,
            "setting": self.setting,
            "query_count": self.query_count,
            "stop": self.stop,
            "observations": [
                observation.as_record()
                for observation in self.observations
            ],
            "report": self.to_report().as_record(),
        }

    def _search_specs(self) -> tuple[Mapping[str, str], ...]:
        return tuple(observation.search for observation in self.observations)

    def _search_records(
        self,
        observations: tuple[AssumptionSearchObservation, ...] | None = None,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "search": observation.search,
                "results": observation.results,
            }
            for observation in (
                self.observations if observations is None else observations
            )
        )

    def _score_history(self) -> tuple[float, ...]:
        scores: list[float] = []
        for end in range(1, len(self.observations) + 1):
            observations = self.observations[:end]
            report = score_assumption_sources(
                self.child,
                self._search_records(observations),
                extra_outputs={
                    "search_specs": tuple(
                        observation.search for observation in observations
                    ),
                },
            )
            scores.append(report.score)
        return tuple(scores[:-1])


@dataclass(frozen=True)
class AssumptionSearchRequest(ClosedRecord):
    """Admitted request for one assumption-search episode."""

    child: str
    disease_name: str
    setting: str
    limit: int

    def as_record(self) -> Mapping[str, Any]:
        return {
            "child": self.child,
            "disease_name": self.disease_name,
            "setting": self.setting,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class AssumptionSearchRecord(ClosedRecord):
    """Closed result of searched assumption discovery."""

    report: AssumptionSearchReport

    def as_record(self) -> Mapping[str, Any]:
        return self.report.as_record()


class AssumptionSearchController:
    """Stop after every scoped assumption query has reported."""

    epoch = "cq_assumption_search_v1"

    def __init__(
        self,
        *,
        child: str,
        disease_name: str,
        setting: str,
        query_count: int,
        observations: tuple[AssumptionSearchObservation, ...] = (),
        stop: bool = False,
    ) -> None:
        self._child = child
        self._disease_name = disease_name
        self._setting = setting
        self._query_count = query_count
        self._observations = observations
        self._stop = stop

    def observe(
        self,
        unit_label: str,
        value: object,
        *,
        is_root: bool,
    ) -> AssumptionSearchEpisodeStep:
        if not isinstance(value, AssumptionSearchObservation):
            raise TypeError(
                "AssumptionSearchController expects "
                "AssumptionSearchObservation"
            )

        observations = self._observations + (value,)
        report = AssumptionSearchEpisodeState(
            child=self._child,
            disease_name=self._disease_name,
            setting=self._setting,
            query_count=self._query_count,
            observations=observations,
        ).to_report()
        rarefaction = report.rarefaction or {}
        stop = bool(rarefaction.get("should_yield")) or (
            len(observations) >= self._query_count
        )
        self._observations = observations
        self._stop = stop
        return AssumptionSearchEpisodeStep(stop=stop)

    def state(self) -> AssumptionSearchEpisodeState:
        return AssumptionSearchEpisodeState(
            child=self._child,
            disease_name=self._disease_name,
            setting=self._setting,
            query_count=self._query_count,
            observations=self._observations,
            stop=self._stop,
        )

    def transitioned(self, epoch: str) -> "AssumptionSearchController":
        return AssumptionSearchController(
            child=self._child,
            disease_name=self._disease_name,
            setting=self._setting,
            query_count=self._query_count,
            observations=self._observations,
            stop=self._stop,
        )


def _run_find_assumption_sources_episode(
    *,
    child: str,
    disease_name: str,
    setting: SettingHints,
    search: SearchCallable,
    retrieve: RetrieveCallable | None = None,
    limit: int = 5,
) -> tuple[EpisodeRecord, AssumptionSearchReport]:
    """Run child-scoped searches for assumption evidence."""

    if limit <= 0:
        raise ValueError("limit must be positive")

    setting_text = setting_hints_text(setting)
    specs = assumption_search_specs(child, disease_name, setting=setting_text)
    request = AssumptionSearchRequest(
        child=child,
        disease_name=disease_name,
        setting=setting_text,
        limit=limit,
    )
    grain = Grain(
        name=f"{child}_assumption_search",
        unit="scoped assumption search",
        result="AssumptionSearchReport",
        controller=lambda _path: AssumptionSearchController(
            child=child,
            disease_name=disease_name,
            setting=setting_text,
            query_count=len(specs),
        ),
    )
    episode = Episode(
        grain=grain,
        key=f"{disease_name}:{setting_text}:{child}",
        source=leaves(
            specs,
            extract=lambda spec: spec,
            accept=lambda _unit, spec: _run_search(search, retrieve, spec, limit),
            result=lambda _unit, observation: observation,
            label=lambda spec: spec.purpose,
        ),
        request=EpisodeRequest(
            goal=EpisodeGoal.root(
                objective={
                    "operation": "find_assumption_sources",
                    "child": child,
                    "disease_name": disease_name,
                    "setting": setting_text,
                },
                result_contract={
                    "result": "AssumptionSearchReport",
                    "output": "assumption_sources",
                },
            ),
            message=request,
        ),
        build_result=_build_result,
    )
    tree = EpisodeTree(root=grain)
    episode_record = episode.run(Context(tree=tree))
    return episode_record, episode_record.controller_state.to_report()


def assumption_search_specs(
    child: str,
    disease_name: str,
    *,
    setting: SettingHints = "",
) -> tuple[OutbreakSearchSpec, ...]:
    """Return child-scoped assumption searches."""

    disease = str(disease_name or "").strip()
    setting_text = setting_hints_text(setting)
    scoped = " ".join(part for part in (disease, setting_text) if part)
    if not scoped:
        scoped = disease or "airborne disease"

    if child in {"build_removal_capacity", "admit_target_room_removal"}:
        return (
            OutbreakSearchSpec(
                query=(
                    f"{scoped} indoor ventilation ACH room volume clean air "
                    "delivery outdoor air flow recirculation filter "
                    "efficiency particle deposition"
                ),
                purpose="assume_removal_capacity",
                disease_name=disease,
            ),
            OutbreakSearchSpec(
                query=(
                    f"{scoped} equivalent virus-free-air clean-air delivery "
                    "ventilation filtration deposition deactivation"
                ),
                purpose="assume_removal_capacity",
                disease_name=disease,
            ),
            OutbreakSearchSpec(
                query=(
                    f"{scoped} comparable indoor room height room volume air "
                    "exchange outdoor ventilation flow recirculation HEPA "
                    "filter efficiency settling speed"
                ),
                purpose="assume_removal_capacity",
                disease_name=disease,
            ),
        )
    if child in {
        "admit_behavior_assumptions",
        "admit_target_exposure_scenario",
    }:
        return (
            OutbreakSearchSpec(
                query=(
                    f"{scoped} breathing flow mask penetration mask "
                    "filtration unmasked airborne transmission quanta "
                    "assumptions"
                ),
                purpose="assume_behavior",
                disease_name=disease,
            ),
            OutbreakSearchSpec(
                query=(
                    f"{scoped} respiratory minute ventilation breathing flow "
                    "mask penetration mask filter efficiency Wells Riley "
                    "quanta assumption"
                ),
                purpose="assume_behavior",
                disease_name=disease,
            ),
            OutbreakSearchSpec(
                query=(
                    f"{scoped} face covering penetration filtration "
                    "efficiency no masks inhalation rate quiet seated "
                    "breathing flow airborne quanta"
                ),
                purpose="assume_behavior",
                disease_name=disease,
            ),
        )
    if child == "admit_population_or_variant_scaling":
        return (
            OutbreakSearchSpec(
                query=(
                    f"{scoped} relative transmissibility variant population "
                    "airborne outbreak scalar"
                ),
                purpose="assume_transmissibility",
                disease_name=disease,
            ),
            OutbreakSearchSpec(
                query=(
                    f"{scoped} variant relative transmissibility population "
                    "susceptibility airborne quanta scaling"
                ),
                purpose="assume_transmissibility",
                disease_name=disease,
            ),
            OutbreakSearchSpec(
                query=(
                    f"{scoped} outbreak calibration variant growth advantage "
                    "relative infectiousness scalar"
                ),
                purpose="assume_transmissibility",
                disease_name=disease,
            ),
        )
    return ()


def score_assumption_sources(
    child: str,
    search_records: tuple[dict[str, Any], ...],
    *,
    extra_outputs: Mapping[str, Any] | None = None,
    errors: tuple[CalculationError, ...] = (),
    score_history: tuple[float, ...] = (),
) -> AssumptionSearchReport:
    """Score reusable assumption-search output with rarefaction."""

    outputs: dict[str, Any] = dict(extra_outputs or {})
    candidates = candidate_sources_from_search_records(search_records)
    outputs.update(
        {
            "search_records": search_records,
            "assumption_sources": candidates,
        }
    )

    score = float(
        bool(outputs.get("search_specs"))
        + bool(candidates)
        + any(
            candidate.get("source_ref")
            and (candidate.get("title")
                 or candidate.get("doi")
                 or candidate.get("url"))
            for candidate in candidates
        )
        + any(candidate.get("text") for candidate in candidates)
    )
    rarefaction = cq_normalization_rarefaction(
        score_history + (score,),
        max_score=4.0,
    )
    if errors:
        outputs["provider_errors"] = tuple(
            error.as_record() for error in errors
        )

    report_errors = _blocking_provider_errors(
        errors,
        score=score,
        max_score=4.0,
    )
    if score < 4.0:
        report_errors += (
            CalculationError(
                code="insufficient_assumption_sources",
                field="assumption_sources",
                message=(
                    f"{child} did not find retrievable assumption evidence"
                ),
            ),
        )

    return AssumptionSearchReport(
        child=child,
        passed=not report_errors and rarefaction.should_yield,
        score=score,
        max_score=4.0,
        outputs=outputs,
        errors=report_errors,
        rarefaction=rarefaction.as_record(),
    )


def _blocking_provider_errors(
    errors: tuple[CalculationError, ...],
    *,
    score: float,
    max_score: float,
) -> tuple[CalculationError, ...]:
    """Provider errors only block if they prevented enough scored evidence."""

    if score >= max_score:
        return ()
    return errors


def _run_search(
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    spec: OutbreakSearchSpec,
    limit: int,
) -> AssumptionSearchObservation:
    try:
        raw_results = search_results_from_provider(
            search(spec.query, limit)
        )
    except Exception as exc:
        return AssumptionSearchObservation(
            search=spec.as_record(),
            errors=(
                CalculationError(
                    code="search_error",
                    field=spec.purpose,
                    message=str(exc),
                ),
            ),
        )

    results: list[Mapping[str, Any]] = []
    errors: list[CalculationError] = []
    for raw_result in raw_results:
        try:
            results.append(result_with_retrieved_text(raw_result, retrieve))
        except Exception as exc:
            results.append(raw_result)
            errors.append(
                CalculationError(
                    code="retrieval_error",
                    field=_result_ref(raw_result),
                    message=str(exc),
                )
            )

    return AssumptionSearchObservation(
        search=spec.as_record(),
        results=tuple(results),
        errors=tuple(errors),
    )


def _result_ref(result: Mapping[str, Any]) -> str:
    return source_ref(result)


def _build_result(record: EpisodeRecord) -> AssumptionSearchRecord:
    state = record.controller_state
    if not isinstance(state, AssumptionSearchEpisodeState):
        raise TypeError("assumption search Episode produced an unexpected state")
    return AssumptionSearchRecord(state.to_report())


__all__ = [
    "AssumptionSearchController",
    "AssumptionSearchEpisodeState",
    "AssumptionSearchEpisodeStep",
    "AssumptionSearchObservation",
    "AssumptionSearchRecord",
    "AssumptionSearchReport",
    "AssumptionSearchRequest",
    "assumption_search_specs",
    "score_assumption_sources",
]
