"""Repository-owned Firecrawl Search and Scrape adapters."""

from __future__ import annotations

from typing import Any, Mapping

from .._http import ProviderError, http_url, post_json
from .queries import source_locator, source_text
from .selection import bound_search_callable

FIRECRAWL_BASE_URL = "https://api.firecrawl.dev/v2"
FIRECRAWL_SEARCH_MAX_BATCH_SIZE = 100


def firecrawl_search_callable(
    api_key: str,
    *,
    page_selection_llm: Any,
    page_assessor: Any,
    scrape_results: bool = True,
    base_url: str = FIRECRAWL_BASE_URL,
    timeout: float = 300.0,
):
    """Return a binding-mediated `search(query, limit)` callable."""

    return bound_search_callable(
        lambda query, limit: firecrawl_search(
            query,
            api_key=api_key,
            limit=limit,
            scrape_results=scrape_results,
            base_url=base_url,
            timeout=timeout,
        ),
        page_selection_llm,
        jev=page_assessor,
    )


def firecrawl_retrieve_callable(
    api_key: str,
    *,
    base_url: str = FIRECRAWL_BASE_URL,
    timeout: float = 300.0,
):
    """Return a `retrieve(candidate)` callable backed by Firecrawl."""

    def retrieve(candidate: Mapping[str, Any]):
        url = source_locator(candidate)
        if not url:
            raise ValueError("Firecrawl retrieval needs a candidate URL")
        return firecrawl_scrape(
            url, api_key=api_key, base_url=base_url, timeout=timeout,
        )

    return retrieve


def firecrawl_search(
    query: str,
    *,
    api_key: str,
    limit: int | None,
    scrape_results: bool = True,
    base_url: str = FIRECRAWL_BASE_URL,
    timeout: float = 300.0,
) -> tuple[Mapping[str, Any], ...]:
    """Issue one Firecrawl v2 Search request."""

    query = _required_text(query, "query")
    limit = FIRECRAWL_SEARCH_MAX_BATCH_SIZE if limit is None else limit
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= FIRECRAWL_SEARCH_MAX_BATCH_SIZE
    ):
        raise ValueError("Firecrawl search limit must be an integer from 1 to 100")
    payload: dict[str, Any] = {"query": query, "limit": limit, "sources": ["web"]}
    if scrape_results:
        payload["scrapeOptions"] = {"formats": ["markdown"]}
    body = post_json(
        f"{http_url(base_url, 'Firecrawl base URL')}/search",
        payload,
        api_key=_required_text(api_key, "firecrawl_api_key"),
        timeout=timeout,
    )
    data = body.get("data")
    records = data.get("web") if isinstance(data, Mapping) else data
    if not isinstance(records, list) or any(
        not isinstance(item, Mapping) for item in records
    ):
        raise ProviderError(
            "Firecrawl search data must contain a list of web result objects"
        )
    return tuple(
        _with_firecrawl_retrieval(record, mode="search", url=source_locator(record))
        if source_text(record) else dict(record)
        for record in records
    )


def firecrawl_scrape(
    url: str,
    *,
    api_key: str,
    base_url: str = FIRECRAWL_BASE_URL,
    timeout: float = 300.0,
) -> Mapping[str, Any]:
    """Issue one Firecrawl v2 Scrape request."""

    body = post_json(
        f"{http_url(base_url, 'Firecrawl base URL')}/scrape",
        {"url": _required_text(url, "url"), "formats": ["markdown"]},
        api_key=_required_text(api_key, "firecrawl_api_key"),
        timeout=timeout,
    )
    result = body.get("data")
    if not isinstance(result, Mapping) or not source_text(result):
        raise ProviderError("Firecrawl did not return retrieved text")
    return _with_firecrawl_retrieval(result, mode="scrape", url=url)


def _with_firecrawl_retrieval(
    result: Mapping[str, Any],
    *,
    mode: str,
    url: str = "",
) -> dict[str, Any]:
    record = dict(result)
    retrieval = {
        "provider": "firecrawl",
        "mode": mode,
    }
    if url:
        retrieval["url"] = url

    existing = record.get("retrieval")
    if isinstance(existing, Mapping):
        retrieval.update(dict(existing))
    record["retrieval"] = retrieval
    return record


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


__all__ = [
    "firecrawl_retrieve_callable",
    "firecrawl_scrape",
    "firecrawl_search",
    "firecrawl_search_callable",
]
