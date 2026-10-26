"""Scoring for runtime outbreak discovery."""

from __future__ import annotations

from typing import Any, Mapping

from episodic_workflow.credit_assignment.cq_calibration import (
    CQ_CALIBRATION_CHILDREN,
    CQ_NORMALIZATION_CRITERIA,
    CqCalibrationChild,
    CqCalibrationChildReport,
    cq_normalization_rarefaction,
    score_cq_normalization_child,
)
from episodic_workflow.credit_assignment.simple import CalculationError
from episodic_workflow.task_executors.outbreak_search import (
    candidate_sources_from_search_records,
)


def score_candidate_outbreak_sources(
    search_records: tuple[dict[str, Any], ...],
    *,
    extra_outputs: Mapping[str, Any] | None = None,
    errors: tuple[CalculationError, ...] = (),
    score_history: tuple[float, ...] = (),
) -> CqCalibrationChildReport:
    """Score normalized source candidates from an outbreak-search run."""

    child = _child("find_candidate_outbreak_sources")
    outputs: dict[str, Any] = dict(extra_outputs or {})
    passed = []
    if outputs.get("search_specs"):
        passed.append("scoped_queries_formulated")

    candidates = candidate_sources_from_search_records(search_records)
    summary = _search_summary(candidates)
    outputs.update(
        {
            "search_records": search_records,
            "candidate_outbreak_sources": candidates,
            "search_summary": summary,
        }
    )

    if candidates:
        passed.append("search_results_returned")
    if _has_source_metadata(candidates):
        passed.append("source_metadata_present")
    if summary["unique_sources"]:
        passed.append("retrieved_text_present")
    if summary["unique_sources"] >= 2 or summary["covered_purposes"] >= 2:
        passed.append("evidence_diversity_scored")

    criteria = {
        criterion.name: criterion
        for criterion in CQ_NORMALIZATION_CRITERIA[child.name]
    }
    score = sum(criteria[name].weight for name in passed)
    max_score = sum(criterion.weight for criterion in criteria.values())
    score_resolution = min(criterion.weight for criterion in criteria.values())
    rarefaction = cq_normalization_rarefaction(
        score_history + (score,),
        max_score=max_score,
        score_resolution=score_resolution,
    )
    if errors:
        outputs["provider_errors"] = tuple(
            error.as_record() for error in errors
        )
    return score_cq_normalization_child(
        child,
        passed,
        outputs,
        errors=_blocking_provider_errors(
            errors,
            score=score,
            max_score=max_score,
        ),
        rarefaction=rarefaction,
    )


def _search_summary(
    candidates: tuple[Mapping[str, Any], ...],
) -> dict[str, Any]:
    usable_candidates = tuple(
        candidate for candidate in candidates if candidate.get("text")
    )
    purposes = {
        str(purpose)
        for candidate in usable_candidates
        for purpose in (
            tuple(candidate.get("purposes") or ())
            or (candidate.get("purpose"),)
        )
        if purpose
    }
    unique_sources = len(
        {
            str(candidate.get("source_ref") or "")
            for candidate in usable_candidates
            if candidate.get("source_ref")
        }
    )
    return {
        "unique_sources": unique_sources,
        "covered_purposes": len(purposes),
        "disease_relevant_sources": unique_sources,
    }


def _has_source_metadata(candidates: tuple[Mapping[str, Any], ...]) -> bool:
    return any(
        candidate.get("source_ref")
        and (candidate.get("title") or candidate.get("doi") or candidate.get("url"))
        and candidate.get("text")
        for candidate in candidates
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


def _child(name: str) -> CqCalibrationChild:
    return next(child for child in CQ_CALIBRATION_CHILDREN if child.name == name)


__all__ = [
    "score_candidate_outbreak_sources",
]
