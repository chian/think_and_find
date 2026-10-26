"""Scoped literature-search helpers for outbreak calibration."""

from .queries import (
    OutbreakSearchSpec,
    RetrieveCallable,
    SearchCallable,
    SettingHints,
    candidate_sources_from_search_records,
    outbreak_search_specs,
    result_with_retrieved_text,
    search_results_from_provider,
    setting_hints_text,
    source_doi,
    source_locator,
    source_ref,
    source_text,
)
from .firecrawl import (
    firecrawl_retrieve_callable,
    firecrawl_scrape,
    firecrawl_search,
    firecrawl_search_callable,
)
from .selection import bound_search, bound_search_callable

__all__ = [
    "OutbreakSearchSpec",
    "RetrieveCallable",
    "SearchCallable",
    "SettingHints",
    "candidate_sources_from_search_records",
    "firecrawl_retrieve_callable",
    "firecrawl_scrape",
    "firecrawl_search",
    "firecrawl_search_callable",
    "bound_search",
    "bound_search_callable",
    "outbreak_search_specs",
    "result_with_retrieved_text",
    "search_results_from_provider",
    "setting_hints_text",
    "source_doi",
    "source_locator",
    "source_ref",
    "source_text",
]
