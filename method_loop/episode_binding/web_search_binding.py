"""Local search-page assessment and selection for evidence acquisition."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from ..episode import EpisodeGoal
from ...task_executors.evidence_extraction.argo import parse_json_object
from ...task_executors.outbreak_search.queries import (
    search_results_from_provider,
    source_locator,
    source_text,
)

SEARCH_PAGE_SELECTION_VERSION = "local_model_planned_optional_jev_v2"


@dataclass(frozen=True)
class PageCandidateAssessment:
    provider_rank: int
    relevance_probability: float
    window_probabilities: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_rank": self.provider_rank,
            "relevance_probability": self.relevance_probability,
            "window_probabilities": list(self.window_probabilities),
        }


@dataclass(frozen=True)
class SearchPageProposal:
    option_id: str
    provider_rank: int
    objective: str = ""
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(vars(self))


async def assess_page_candidate(
    jev: Any,
    *,
    goal: EpisodeGoal,
    query: str,
    result: Mapping[str, Any],
    provider_rank: int,
    assessment_instructions: str | None = None,
    assessment_criteria: Mapping[str, str] | None = None,
    standing_task: str | None = None,
    information_need: str | None = None,
) -> PageCandidateAssessment:
    """Measure every text window, retaining the strongest relevance evidence."""

    text = source_text(result) or str(result.get("description") or result.get("snippet") or "")
    state = {
        "search_query": query,
        "candidate": {
            "provider_rank": provider_rank,
            "title": str(result.get("title") or ""),
            "url": source_locator(result),
        },
    }
    if standing_task is None:
        state["search_goal"] = goal.as_record()
    else:
        if not isinstance(standing_task, str) or not standing_task.strip():
            raise ValueError("Standing assessment task must be text")
        if not isinstance(information_need, str) or not information_need.strip():
            raise ValueError("Page assessment information need must be text")
        state["information_need"] = information_need
    instructions = assessment_instructions or (
        "Does candidate_text_window contain evidence that could fill at "
        "least one result requested by search_goal, for an identifiable "
        "subject of that goal? Judge possible contribution to the "
        "declared result, not general topical similarity."
    )
    if standing_task is not None:
        instructions = "Standing research task: " + standing_task + "\n\n" + instructions
    # UTF-8 byte windows give a conservative token ceiling without a tokenizer
    # dependency. Reserve space for the typed question and goal metadata.
    metadata_bytes = len(json.dumps(state, ensure_ascii=False).encode("utf-8"))
    text_budget = 48_000 - metadata_bytes
    if text_budget < 4:
        raise ValueError("Search assessment metadata exceeds its request budget")
    windows = _text_windows(text, text_budget)
    probabilities = []
    for index, window in enumerate(windows, start=1):
        probability = await jev.noul(
            state={
                **state,
                "candidate_text_window": window,
                "window_index": index,
                "window_count": len(windows),
            },
            instructions=instructions,
            criteria=assessment_criteria or {
                "true": "The text contains a reported value, an evidence-based basis for a permitted estimate, or identity/context connecting a value to a requested subject.",
                "false": "The text cannot contribute any requested result even if it mentions the broad topic.",
            },
        )
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError("Candidate relevance probability must be between zero and one")
        probabilities.append(float(probability))
    return PageCandidateAssessment(provider_rank, max(probabilities), tuple(probabilities))


def _text_windows(text: str, byte_budget: int) -> list[str]:
    windows = []
    start = 0
    while start < len(text):
        end = min(len(text), start + byte_budget)
        while len(json.dumps(text[start:end], ensure_ascii=False).encode("utf-8")) > byte_budget:
            end = start + (end - start) // 2
            if end == start:
                raise ValueError("Search text window budget is too small")
        windows.append(text[start:end])
        start = end
    return windows or [""]


async def propose_search_page(
    llm: Any,
    *,
    goal: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    previous_pages: list[Mapping[str, Any]],
) -> SearchPageProposal:
    options = {item["option_id"]: item for item in candidates}
    if not options:
        raise ValueError("Search page proposal requires at least one candidate")
    prompt = (
        "Choose exactly one option_id from the unprocessed candidates. Use "
        "their Jev measurements, titles, descriptions, provider ranks, and "
        "previous pages to choose the next page worth processing. You select "
        "a page; you do not decide whether acquisition continues. Return one "
        "JSON object with option_id, objective, and rationale.\n\n"
        + json.dumps({"search_goal": dict(goal), "candidates": candidates, "previous_pages": previous_pages}, ensure_ascii=False)
    )
    response = parse_json_object(await llm.call_async(
        prompt,
        system_prompt="Choose one page from the supplied candidate catalog. Return one JSON object.",
    ))
    option_id = response.get("option_id")
    if not isinstance(option_id, str) or option_id not in options:
        raise ValueError("Search page proposer selected an undeclared option_id")
    for name in ("objective", "rationale"):
        if not isinstance(response.get(name, ""), str):
            raise ValueError(f"Search page proposal {name} must be text")
    return SearchPageProposal(
        option_id,
        options[option_id]["provider_rank"],
        response.get("objective", ""),
        response.get("rationale", ""),
    )


class SearchPageProposer:
    """Buffer one search and select each candidate at most once."""

    def __init__(
        self,
        *,
        query: str,
        limit: int,
        provider_search: Callable[[str, int], Iterable[Mapping[str, Any]]],
        llm: Any = None,
        jev: Any = None,
        select_page: Callable | None = None,
        assessment_instructions: str | None = None,
        assessment_criteria: Mapping[str, str] | None = None,
        standing_task: str | None = None,
        information_need: str | None = None,
    ) -> None:
        self.query = query
        self.limit = limit
        self.provider_search = provider_search
        self.llm = llm
        self.jev = jev
        self.select_page = select_page
        self.assessment_instructions=assessment_instructions
        self.assessment_criteria=assessment_criteria
        self.standing_task=standing_task
        self.information_need=information_need
        self._results: tuple[Mapping[str, Any], ...] | None = None
        self._assessments: dict[int, PageCandidateAssessment] = {}
        self._selected: set[int] = set()
        self._history: list[Mapping[str, Any]] = []

    async def next(self, goal: EpisodeGoal) -> Mapping[str, Any] | None:
        if self._results is None:
            results = search_results_from_provider(self.provider_search(self.query, self.limit))
            self._results = tuple({source_locator(r): r for r in results if source_locator(r)}.values())
            for rank, result in enumerate(self._results, start=1):
                if self.jev is not None:
                    self._assessments[rank] = await assess_page_candidate(
                        self.jev, goal=goal, query=self.query, result=result, provider_rank=rank,
                        assessment_instructions=self.assessment_instructions,
                        assessment_criteria=self.assessment_criteria,
                        standing_task=self.standing_task,
                        information_need=self.information_need,
                    )
        pending = [rank for rank in range(1, len(self._results) + 1) if rank not in self._selected]
        if not pending or len(self._selected) >= self.limit:
            return None
        candidates = []
        for rank in pending:
            result = self._results[rank - 1]
            candidates.append({
                "option_id": f"page:{rank}",
                "provider_rank": rank,
                "title": str(result.get("title") or ""),
                "url": source_locator(result),
                "description": str(result.get("description") or result.get("snippet") or ""),
                "jev_assessment": self._assessments[rank].to_dict() if rank in self._assessments else None,
            })
        if self.select_page is not None:
            response = self.select_page(goal.as_record(), candidates, list(self._history))
            options = {row["option_id"]: row for row in candidates}
            option = response.get("option_id")
            if not isinstance(option, str) or option not in options:
                raise ValueError("Search page proposer selected an undeclared option_id")
            if any(not isinstance(response.get(key), str) or not response[key].strip()
                   for key in ("objective", "rationale")):
                raise ValueError("Search page proposal requires objective and rationale")
            proposal = SearchPageProposal(option, options[option]["provider_rank"],
                                         response["objective"], response["rationale"])
        else:
            proposal = await propose_search_page(
                self.llm, goal=goal.as_record(), candidates=candidates, previous_pages=self._history,
            )
        rank = proposal.provider_rank
        self._selected.add(rank)
        self._history.append(proposal.to_dict())
        return {
            **dict(self._results[rank - 1]),
            "provider_rank": rank,
            "page_selection": {
                "version": SEARCH_PAGE_SELECTION_VERSION,
                "assessment": self._assessments[rank].to_dict() if rank in self._assessments else None,
                "binding": "SearchPageProposer",
                "proposal": proposal.to_dict(),
            },
        }
