"""Bind provider searches to local Jev assessment and page selection."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Iterable, Mapping

from ...method_loop import EpisodeGoal

ProviderSearch = Callable[[str, int], Iterable[Mapping[str, Any]]]


def bound_search_callable(provider_search: ProviderSearch, llm: Any, *, jev: Any):
    def search(query: str, limit: int):
        return asyncio.run(bound_search(
            provider_search=provider_search,
            llm=llm,
            jev=jev,
            query=query,
            limit=limit,
        ))

    return search


async def bound_search(
    *,
    provider_search: ProviderSearch,
    llm: Any,
    jev: Any,
    query: str,
    limit: int,
) -> tuple[Mapping[str, Any], ...]:
    from ...method_loop.episode_binding import SearchPageProposer

    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("search limit must be a positive integer")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("search query must be non-empty text")
    goal = EpisodeGoal.root(
        objective={
            "query": query,
            "purpose": "find airborne outbreak evidence or assumption evidence",
        },
        result_contract={"result": "candidate evidence pages"},
    )
    proposer = SearchPageProposer(
        query=query, limit=limit, provider_search=provider_search, llm=llm, jev=jev,
    )
    selected = []
    for _ in range(limit):
        item = await proposer.next(goal)
        if item is None:
            break
        selected.append(item)
    return tuple(selected)
