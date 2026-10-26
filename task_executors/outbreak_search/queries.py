"""Build scoped search calls for calibration outbreaks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


SearchCallable = Callable[[str, int], Iterable[Mapping[str, Any]]]
SettingHints = str | Sequence[str] | None
RetrieveCallable = Callable[
    [Mapping[str, Any]],
    Mapping[str, Any] | str | None,
]


RETRIEVED_TEXT_KEYS = (
    "text",
    "passage",
    "markdown",
    "content",
    "html",
    "abstract",
)
PREVIEW_TEXT_KEYS = (
    "snippet",
    "description",
    "summary",
)
URL_LOCATOR_KEYS = (
    "url",
    "link",
    "href",
    "sourceURL",
    "sourceUrl",
    "source_url",
    "canonicalUrl",
    "canonical_url",
)
SOURCE_REF_KEYS = (
    "source_ref",
    "source_id",
    "id",
    "doi",
    "DOI",
    *URL_LOCATOR_KEYS,
)
DOI_KEYS = (
    "doi",
    "DOI",
)
EVENT_REF_KEYS = (
    "event_ref",
    "event_id",
    "event_key",
    "outbreak_ref",
    "outbreak_id",
    "outbreak_key",
)
EXPLICIT_SOURCE_REF_KEYS = ("source_ref",)
PROVIDER_ID_KEYS = (
    "source_id",
    "id",
)
PROVENANCE_KEYS = tuple(
    dict.fromkeys(
        (
            *SOURCE_REF_KEYS,
            "title",
            "name",
            *EVENT_REF_KEYS,
            "year",
            "publication_year",
            "published_year",
        )
    )
)


@dataclass(frozen=True)
class OutbreakSearchSpec:
    """One scoped search query for finding outbreak calibration records."""

    query: str
    purpose: str
    disease_name: str

    def as_record(self) -> dict[str, str]:
        return {
            "query": self.query,
            "purpose": self.purpose,
            "disease_name": self.disease_name,
        }


def outbreak_search_specs(
    disease_name: str,
    *,
    setting: SettingHints = "",
) -> tuple[OutbreakSearchSpec, ...]:
    """Return targeted outbreak-search specs for one disease."""

    disease = _required_text(disease_name, "disease_name")
    setting_text = setting_hints_text(setting)
    setting_clause = f" {setting_text}" if setting_text else ""
    return (
        OutbreakSearchSpec(
            query=(
                f"{disease}{setting_clause} indoor outbreak attack rate "
                "index case ventilation duration"
            ),
            purpose="find_outbreak_counts",
            disease_name=disease,
        ),
        OutbreakSearchSpec(
            query=(
                f"{disease}{setting_clause} airborne transmission outbreak "
                "room volume air exchange rate secondary attack rate"
            ),
            purpose="find_room_reconstruction",
            disease_name=disease,
        ),
        OutbreakSearchSpec(
            query=(
                f"{disease}{setting_clause} superspreading event enclosed "
                "space attendance symptom onset seating chart"
            ),
            purpose="find_point_source_event",
            disease_name=disease,
        ),
    )


def search_results_from_provider(raw_results: object) -> tuple[dict[str, Any], ...]:
    """Return copied provider results after validating their shape."""

    if (
        isinstance(raw_results, str)
        or isinstance(raw_results, Mapping)
        or not isinstance(raw_results, Iterable)
    ):
        raise TypeError("search results must be an iterable of objects")

    results: list[dict[str, Any]] = []
    for index, result in enumerate(raw_results):
        if not isinstance(result, Mapping):
            raise TypeError(f"search result {index} must be an object")
        results.append(dict(result))
    return tuple(results)


def setting_hints_text(setting: SettingHints = "") -> str:
    """Normalize optional setting hints to the text appended to searches."""

    if setting is None:
        return ""
    if isinstance(setting, str):
        return setting.strip()
    return " ".join(
        part
        for part in (str(raw).strip() for raw in setting)
        if part
    )


def candidate_sources_from_search_records(
    search_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return compact candidate sources from scoped search result bundles."""

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    index_by_key: dict[str, int] = {}
    for bundle in search_records:
        if not isinstance(bundle, Mapping):
            continue
        search = _mapping(bundle.get("search"))
        disease_name = str(search.get("disease_name") or "")
        purpose = str(search.get("purpose") or "")
        query = str(search.get("query") or "")
        for rank, result in _ranked_result_mappings(bundle.get("results")):
            if not _has_candidate_identity(result):
                continue
            source_ref = _source_ref(result)
            key = source_ref.lower()
            candidate = _candidate_from_result(
                result,
                source_ref=source_ref,
                disease_name=disease_name,
                query=query,
                purpose=purpose,
                rank=rank,
            )
            if key in seen:
                _merge_candidate(candidates[index_by_key[key]], candidate)
                continue
            seen.add(key)
            index_by_key[key] = len(candidates)
            candidates.append(candidate)
    return tuple(candidates)


def _candidate_from_result(
    result: Mapping[str, Any],
    *,
    source_ref: str,
    disease_name: str,
    query: str,
    purpose: str,
    rank: int,
) -> dict[str, Any]:
    title = _first_identity_text(result, ("title", "name"))
    text = _retrieved_text(result)
    snippet = _first_text(result, PREVIEW_TEXT_KEYS)
    return {
        "source_ref": source_ref,
        "disease_name": disease_name,
        "title": title,
        "url": source_locator(result),
        "doi": source_doi(result),
        "event_ref": _first_identity_text(result, EVENT_REF_KEYS),
        "year": _first_identity_text(
            result,
            ("year", "publication_year", "published_year"),
        ),
        "text": text,
        "snippet": snippet,
        "query": query,
        "queries": (query,),
        "purpose": purpose,
        "purposes": (purpose,),
        "rank": rank,
        "retrieval": _mapping(result.get("retrieval")),
    }


def _merge_candidate(
    existing: dict[str, Any],
    candidate: Mapping[str, Any],
) -> None:
    for field in (
        "disease_name",
        "title",
        "url",
        "doi",
        "event_ref",
        "year",
        "text",
        "snippet",
    ):
        if not existing.get(field) and candidate.get(field):
            existing[field] = candidate[field]
    if (
        candidate.get("text")
        and len(str(candidate["text"])) > len(str(existing.get("text") or ""))
    ):
        existing["text"] = candidate["text"]

    existing["queries"] = tuple(
        dict.fromkeys(
            tuple(existing.get("queries") or ())
            + tuple(candidate.get("queries") or ())
        )
    )
    existing["purposes"] = tuple(
        dict.fromkeys(
            tuple(existing.get("purposes") or ())
            + tuple(candidate.get("purposes") or ())
        )
    )
    existing["retrieval"] = {
        **_mapping(existing.get("retrieval")),
        **_mapping(candidate.get("retrieval")),
    }


def result_with_retrieved_text(
    result: Mapping[str, Any],
    retrieve: RetrieveCallable | None,
) -> dict[str, Any]:
    """Attach retrieved text to a search result when no text is inline."""

    out = dict(result)
    if retrieve is None or _has_retrieved_text(out):
        return out

    retrieved = retrieve(out)
    if retrieved is None:
        return out
    if isinstance(retrieved, str):
        retrieved = {"text": retrieved}
    if not isinstance(retrieved, Mapping):
        raise TypeError("retrieved source must be a mapping, string, or None")

    merged = _merge_retrieved_result(out, dict(retrieved))
    if _has_retrieved_text(merged):
        retrieval = {
            **_mapping(out.get("retrieval")),
            **_mapping(merged.get("retrieval")),
        }
        retrieval.setdefault("mode", "provider")
        merged["retrieval"] = retrieval
    return merged


def _merge_retrieved_result(
    result: Mapping[str, Any],
    retrieved: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge fetched text without erasing search-provider provenance."""

    merged = {**result, **retrieved}
    for key in PROVENANCE_KEYS:
        if _first_text(result, (key,)):
            merged[key] = result[key]

    result_metadata = result.get("metadata")
    retrieved_metadata = retrieved.get("metadata")
    if isinstance(result_metadata, Mapping):
        merged["metadata"] = dict(result_metadata)
        if isinstance(retrieved_metadata, Mapping):
            merged["metadata"].update(dict(retrieved_metadata))
        for key in PROVENANCE_KEYS:
            if _first_text(result_metadata, (key,)):
                merged["metadata"][key] = result_metadata[key]
    return merged


def _has_retrieved_text(result: Mapping[str, Any]) -> bool:
    return bool(_retrieved_text(result))


def _retrieved_text(result: Mapping[str, Any]) -> str:
    text = _first_text(result, RETRIEVED_TEXT_KEYS)
    if text:
        return text

    for nested_key in ("scrape", "data"):
        nested = result.get(nested_key)
        if isinstance(nested, Mapping):
            text = _first_text(nested, RETRIEVED_TEXT_KEYS)
            if text:
                return text
    return ""


def source_text(result: Mapping[str, Any]) -> str:
    """Return full inline or fetched text from a search/retrieval result."""

    return _retrieved_text(result)


def source_locator(result: Mapping[str, Any]) -> str:
    """Return a retrievable URL-like locator from a provider result."""

    return _first_identity_text(result, URL_LOCATOR_KEYS)


def source_doi(result: Mapping[str, Any]) -> str:
    """Return a DOI from a provider result."""

    return _first_identity_text(result, DOI_KEYS)


def source_ref(result: Mapping[str, Any]) -> str:
    """Return the stable source reference used to deduplicate a result."""

    return _source_ref(result)


def _required_text(value: object, name: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _source_ref(result: Mapping[str, Any]) -> str:
    value = _first_identity_text(result, EXPLICIT_SOURCE_REF_KEYS)
    if value:
        return value

    value = source_doi(result) or source_locator(result)
    if value:
        return value

    value = _first_identity_text(result, PROVIDER_ID_KEYS)
    if value:
        return value

    digest = hashlib.sha256(
        repr(sorted(result.items())).encode("utf-8")
    ).hexdigest()[:16]
    return f"search_result:{digest}"


def _has_candidate_identity(result: Mapping[str, Any]) -> bool:
    return bool(
        _first_identity_text(result, EXPLICIT_SOURCE_REF_KEYS)
        or source_doi(result)
        or source_locator(result)
        or _first_identity_text(result, PROVIDER_ID_KEYS)
    )


def _first_identity_text(result: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    text = _first_text(result, keys)
    if text:
        return text

    metadata = result.get("metadata")
    if isinstance(metadata, Mapping):
        return _first_text(metadata, keys)
    return ""


def _first_text(result: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = result.get(key)
        if not isinstance(value, str):
            continue
        text = value.strip()
        if text:
            return text
    return ""


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        try:
            return dict(value)
        except Exception:
            return {}
    return {}


def _ranked_result_mappings(
    value: object,
) -> tuple[tuple[int, dict[str, Any]], ...]:
    if (
        isinstance(value, str)
        or isinstance(value, Mapping)
        or not isinstance(value, Iterable)
    ):
        return ()

    return tuple(
        (rank, copied)
        for rank, result in enumerate(value, start=1)
        if isinstance(result, Mapping)
        for copied in (_mapping(result),)
        if copied
    )


