"""Episode run for searched outbreak-source discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from episodic_workflow.credit_assignment.cq_calibration import (
    CqCalibrationChildReport,
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
    outbreak_search_specs,
    result_with_retrieved_text,
    search_results_from_provider,
    setting_hints_text,
    source_ref,
)

from .outbreak_discovery import score_candidate_outbreak_sources


@dataclass(frozen=True)
class OutbreakSearchObservation:
    """One scoped outbreak query and its retrieved source candidates."""

    search: Mapping[str, str]
    results: tuple[Mapping[str, Any], ...] = ()
    errors: tuple[CalculationError, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_record(self) -> dict[str, Any]:
        return {
            "search": dict(self.search),
            "results": [dict(result) for result in self.results],
            "errors": [error.as_record() for error in self.errors],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class OutbreakSearchEpisodeStep:
    """Controller step for source discovery."""

    stop: bool
    requests_transition: bool = False

    def as_record(self) -> dict[str, bool]:
        return {
            "stop": self.stop,
            "requests_transition": self.requests_transition,
        }


@dataclass(frozen=True)
class OutbreakSearchEpisodeState:
    """Accumulated search observations for one disease and setting."""

    disease_name: str
    setting: str
    query_count: int
    observations: tuple[OutbreakSearchObservation, ...] = ()
    stop: bool = False

    def to_report(self) -> CqCalibrationChildReport:
        return score_candidate_outbreak_sources(
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
        return tuple(
            observation.search
            for observation in self.observations
        )

    def _search_records(
        self,
        observations: tuple[OutbreakSearchObservation, ...] | None = None,
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
            report = score_candidate_outbreak_sources(
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
class OutbreakSearchRequest(ClosedRecord):
    """Admitted request for searched outbreak-source discovery."""

    disease_name: str
    setting: str = ""
    limit: int = 10
    retrieval_enabled: bool = False

    def as_record(self) -> Mapping[str, Any]:
        return {
            "disease_name": self.disease_name,
            "setting": self.setting,
            "limit": self.limit,
            "retrieval_enabled": self.retrieval_enabled,
        }


@dataclass(frozen=True)
class OutbreakSearchRecord(ClosedRecord):
    """Closed result of searched outbreak-source discovery."""

    report: CqCalibrationChildReport

    def as_record(self) -> Mapping[str, Any]:
        return self.report.as_record()

    def audit_identifiers(self) -> tuple[str, ...]:
        return self.report.identities


class OutbreakSearchController:
    """Stop after every scoped outbreak query has reported."""

    epoch = "outbreak_search_v1"

    def __init__(
        self,
        *,
        disease_name: str,
        setting: str,
        query_count: int,
        observations: tuple[OutbreakSearchObservation, ...] = (),
        stop: bool = False,
    ) -> None:
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
    ) -> OutbreakSearchEpisodeStep:
        if not isinstance(value, OutbreakSearchObservation):
            raise TypeError(
                "OutbreakSearchController expects OutbreakSearchObservation"
            )

        observations = self._observations + (value,)
        report = OutbreakSearchEpisodeState(
            disease_name=self._disease_name,
            setting=self._setting,
            query_count=self._query_count,
            observations=observations,
        ).to_report()
        stop = report.should_yield or len(observations) >= self._query_count
        self._observations = observations
        self._stop = stop
        return OutbreakSearchEpisodeStep(stop=stop)

    def state(self) -> OutbreakSearchEpisodeState:
        return OutbreakSearchEpisodeState(
            disease_name=self._disease_name,
            setting=self._setting,
            query_count=self._query_count,
            observations=self._observations,
            stop=self._stop,
        )

    def transitioned(self, epoch: str) -> "OutbreakSearchController":
        return OutbreakSearchController(
            disease_name=self._disease_name,
            setting=self._setting,
            query_count=self._query_count,
            observations=self._observations,
            stop=self._stop,
        )


def _run_find_calibration_outbreaks_episode(
    search: SearchCallable,
    disease_name: str,
    *,
    retrieve: RetrieveCallable | None = None,
    setting: SettingHints = "",
    limit: int = 10,
) -> tuple[EpisodeRecord, CqCalibrationChildReport]:
    """Search for calibration outbreaks at the infectiousness level."""

    return _run_outbreak_search_episode(
        search,
        disease_name,
        retrieve=retrieve,
        setting=setting,
        limit=limit,
        operation="find_calibration_outbreaks",
    )


def _run_outbreak_search_episode(
    search: SearchCallable,
    disease_name: str,
    *,
    retrieve: RetrieveCallable | None,
    setting: SettingHints,
    limit: int,
    operation: str,
) -> tuple[EpisodeRecord, CqCalibrationChildReport]:
    """Run scoped outbreak searches under the requested task name."""

    if limit <= 0:
        raise ValueError("limit must be positive")

    setting_text = setting_hints_text(setting)
    specs = outbreak_search_specs(disease_name, setting=setting_text)
    request = OutbreakSearchRequest(
        disease_name=disease_name,
        setting=setting_text,
        limit=limit,
        retrieval_enabled=retrieve is not None,
    )
    grain = Grain(
        name=operation,
        unit="scoped outbreak search",
        result="CqCalibrationChildReport",
        controller=lambda _path: OutbreakSearchController(
            disease_name=disease_name,
            setting=setting_text,
            query_count=len(specs),
        ),
    )
    episode = Episode(
        grain=grain,
        key=f"{disease_name}:{setting_text}",
        source=leaves(
            specs,
            extract=lambda spec: spec,
            accept=lambda _unit, spec: _run_search(
                search,
                retrieve,
                spec,
                limit,
            ),
            result=lambda _unit, observation: observation,
            label=lambda spec: spec.purpose,
        ),
        request=EpisodeRequest(
            goal=EpisodeGoal.root(
                objective={
                    "operation": operation,
                    "disease_name": disease_name,
                    "setting": setting_text,
                },
                result_contract={
                    "result": "CqCalibrationChildReport",
                    "output": "candidate_outbreak_sources",
                },
            ),
            message=request,
        ),
        build_result=_build_result,
    )
    tree = EpisodeTree(root=grain)
    episode_record = episode.run(Context(tree=tree))
    return episode_record, episode_record.controller_state.to_report()


def _run_search(
    search: SearchCallable,
    retrieve: RetrieveCallable | None,
    spec: OutbreakSearchSpec,
    limit: int,
) -> OutbreakSearchObservation:
    try:
        raw_results = search_results_from_provider(
            search(spec.query, limit)
        )
    except Exception as exc:
        return OutbreakSearchObservation(
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

    return OutbreakSearchObservation(
        search=spec.as_record(),
        results=tuple(results),
        errors=tuple(errors),
    )


def _result_ref(result: Mapping[str, Any]) -> str:
    return source_ref(result)


def _build_result(record: EpisodeRecord) -> OutbreakSearchRecord:
    state = record.controller_state
    if not isinstance(state, OutbreakSearchEpisodeState):
        raise TypeError("outbreak search Episode produced an unexpected state")
    return OutbreakSearchRecord(state.to_report())


__all__ = [
    "OutbreakSearchController",
    "OutbreakSearchEpisodeState",
    "OutbreakSearchEpisodeStep",
    "OutbreakSearchObservation",
    "OutbreakSearchRecord",
    "OutbreakSearchRequest",
]
