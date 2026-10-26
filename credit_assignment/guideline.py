"""Credit layout for a first indoor safety-guideline parent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .simple import CalculationError
from .infectiousness import INFECTIOUSNESS_CHILDREN


@dataclass(frozen=True)
class GuidelineChild:
    """One task expected by the safety-guideline parent."""

    name: str
    output: str
    does: str
    mode: str
    main_output: bool = False
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("GuidelineChild.name must be populated")
        if not self.output:
            raise ValueError("GuidelineChild.output must be populated")
        if not self.does:
            raise ValueError("GuidelineChild.does must be populated")
        if self.mode not in {
            "evidence",
            "admission",
            "selection",
            "calculation",
        }:
            raise ValueError(f"unknown guideline mode: {self.mode}")
        if self.weight <= 0.0:
            raise ValueError("GuidelineChild.weight must be positive")

    def as_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "output": self.output,
            "does": self.does,
            "mode": self.mode,
            "main_output": self.main_output,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class GuidelineChildReport:
    """Report returned by one safety-guideline child."""

    child: str
    output: str
    passed: bool
    score: float
    max_score: float
    should_yield: bool
    outputs: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[CalculationError, ...] = ()
    identities: tuple[str, ...] = ()

    @property
    def normalized_score(self) -> float:
        if self.max_score == 0.0:
            return 0.0
        return self.score / self.max_score

    def as_record(self) -> dict[str, Any]:
        return {
            "child": self.child,
            "output": self.output,
            "passed": self.passed,
            "score": self.score,
            "max_score": self.max_score,
            "normalized_score": self.normalized_score,
            "should_yield": self.should_yield,
            "outputs": dict(self.outputs),
            "errors": [error.as_record() for error in self.errors],
            "identities": list(self.identities),
        }


@dataclass(frozen=True)
class GuidelineError:
    """One child failure in a safety-guideline parent report."""

    child: str
    output: str
    code: str
    message: str
    field: str | None = None

    def as_record(self) -> dict[str, str]:
        record = {
            "child": self.child,
            "output": self.output,
            "code": self.code,
            "message": self.message,
        }
        if self.field:
            record["field"] = self.field
        return record


@dataclass(frozen=True)
class GuidelineCreditReport:
    """Summed credit for a safety-guideline evaluation."""

    passed: bool
    score: float
    max_score: float
    credited_children: tuple[str, ...] = ()
    missing_children: tuple[str, ...] = ()
    child_reports: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    outputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    main_outputs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    source_refs: tuple[str, ...] = ()
    admitted_assumptions: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )
    missing_fields: tuple[str, ...] = ()
    failed_fields: tuple[str, ...] = ()
    errors: tuple[GuidelineError, ...] = ()
    identities: tuple[str, ...] = ()

    @property
    def normalized_score(self) -> float:
        if self.max_score == 0.0:
            return 0.0
        return self.score / self.max_score

    def as_record(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "max_score": self.max_score,
            "normalized_score": self.normalized_score,
            "credited_children": list(self.credited_children),
            "missing_children": list(self.missing_children),
            "child_reports": {
                child: dict(report)
                for child, report in self.child_reports.items()
            },
            "outputs": {
                child: dict(values)
                for child, values in self.outputs.items()
            },
            "main_outputs": {
                child: dict(values)
                for child, values in self.main_outputs.items()
            },
            "source_refs": list(self.source_refs),
            "admitted_assumptions": {
                name: dict(assumption)
                for name, assumption in self.admitted_assumptions.items()
            },
            "missing_fields": list(self.missing_fields),
            "failed_fields": list(self.failed_fields),
            "errors": [error.as_record() for error in self.errors],
            "identities": list(self.identities),
        }


GUIDELINE_CHILDREN: tuple[GuidelineChild, ...] = (
    GuidelineChild(
        name="estimate_pathogen_infectiousness",
        output="quanta_concentration_exhaled_m3",
        mode="evidence",
        weight=sum(child.weight for child in INFECTIOUSNESS_CHILDREN),
        does=(
            "Search for outbreak evidence and synthesize the infectiousness "
            "parameter consumed by the guideline."
        ),
    ),
    GuidelineChild(
        name="admit_target_room_removal",
        output="removal_capacity_m3_s",
        mode="admission",
        does="Admit the target room removal capacity supplied by the caller.",
    ),
    GuidelineChild(
        name="admit_target_exposure_scenario",
        output="target_exposure_scenario",
        mode="admission",
        does=(
            "Admit the target risk tolerance, breathing flow, and mask "
            "penetration supplied by the caller."
        ),
    ),
    GuidelineChild(
        name="select_transmission_model",
        output="transmission_model",
        mode="selection",
        does="Select the indoor-airborne equation used for the output.",
    ),
    GuidelineChild(
        name="compute_safety_bound",
        output="safety_guideline_result",
        mode="calculation",
        main_output=True,
        does=(
            "Compute the credited indoor airborne safety-guideline result."
        ),
    ),
)


def guideline_child(name: str) -> GuidelineChild:
    """Return the guideline child with the requested task name."""

    return next(child for child in GUIDELINE_CHILDREN if child.name == name)


def pass_guideline_child(
    child: GuidelineChild,
    outputs: Mapping[str, Any],
    *,
    identities: tuple[str, ...] = (),
) -> GuidelineChildReport:
    """Credit a guideline child that returned its contracted output."""

    if child.output not in outputs or outputs[child.output] is None:
        return fail_guideline_child(
            child,
            (
                CalculationError(
                    code="missing_output",
                    field=child.output,
                    message=f"{child.name} did not return {child.output}",
                ),
            ),
            outputs=outputs,
        )

    return GuidelineChildReport(
        child=child.name,
        output=child.output,
        passed=True,
        score=child.weight,
        max_score=child.weight,
        should_yield=True,
        outputs=dict(outputs),
        identities=identities,
    )


def fail_guideline_child(
    child: GuidelineChild,
    errors: tuple[CalculationError, ...],
    *,
    outputs: Mapping[str, Any] | None = None,
) -> GuidelineChildReport:
    """Return a repairable failed guideline child report."""

    return GuidelineChildReport(
        child=child.name,
        output=child.output,
        passed=False,
        score=0.0,
        max_score=child.weight,
        should_yield=False,
        outputs={} if outputs is None else dict(outputs),
        errors=errors,
    )


def sum_guideline_credit(
    reports: Iterable[GuidelineChildReport],
    *,
    children: tuple[GuidelineChild, ...] = GUIDELINE_CHILDREN,
) -> GuidelineCreditReport:
    """Sum child credit for the guideline parent."""

    reports_by_child = {report.child: report for report in reports}
    max_score = sum(child.weight for child in children)

    score = 0.0
    credited_children: list[str] = []
    missing_children: list[str] = []
    child_reports: dict[str, Mapping[str, Any]] = {}
    outputs: dict[str, Mapping[str, Any]] = {}
    main_outputs: dict[str, Mapping[str, Any]] = {}
    errors: list[GuidelineError] = []
    identities: list[str] = []
    source_refs: list[str] = []
    admitted_assumptions: dict[str, Mapping[str, Any]] = {}

    for child in children:
        report = reports_by_child.get(child.name)
        if report is None:
            missing_children.append(child.name)
            errors.append(
                GuidelineError(
                    child=child.name,
                    output=child.output,
                    code="missing_report",
                    message=f"{child.name} did not return a report",
                )
            )
            continue

        score += min(report.score, child.weight)
        child_reports[child.name] = report.as_record()
        if report.outputs:
            outputs[child.name] = dict(report.outputs)
            source_refs.extend(_source_refs_from_outputs(report.outputs))
            admitted_assumptions.update(
                _admitted_assumptions_from_outputs(
                    child.name,
                    report.outputs,
                )
            )

        if not report.passed:
            errors.extend(
                GuidelineError(
                    child=child.name,
                    output=child.output,
                    field=error.field,
                    code=error.code,
                    message=error.message,
                )
                for error in report.errors
            )
            continue

        if child.output not in report.outputs:
            errors.append(
                GuidelineError(
                    child=child.name,
                    output=child.output,
                    code="missing_output",
                    field=child.output,
                    message=f"{child.name} did not return {child.output}",
                )
            )
            continue

        credited_children.append(child.name)
        identities.extend(report.identities)
        if child.main_output:
            main_outputs[child.name] = dict(report.outputs)

    return GuidelineCreditReport(
        passed=len(credited_children) == len(children),
        score=score,
        max_score=max_score,
        credited_children=tuple(credited_children),
        missing_children=tuple(missing_children),
        child_reports=child_reports,
        outputs=outputs,
        main_outputs=main_outputs,
        source_refs=tuple(dict.fromkeys(source_refs)),
        missing_fields=_repair_fields(errors, missing=True),
        failed_fields=_repair_fields(errors, missing=False),
        errors=tuple(errors),
        identities=tuple(identities),
        admitted_assumptions=admitted_assumptions,
    )


def _admitted_assumptions_from_outputs(
    child_name: str,
    outputs: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            admitted = value.get("admitted_assumptions")
            if isinstance(admitted, Mapping):
                for field, raw_assumption in admitted.items():
                    if not isinstance(raw_assumption, Mapping):
                        continue
                    records[f"{child_name}.{field}"] = {
                        **dict(raw_assumption),
                        "child": _qualified_assumption_child(
                            child_name,
                            str(field),
                            raw_assumption,
                        ),
                    }

            for key, child_value in value.items():
                if key != "admitted_assumptions":
                    collect(child_value)
        elif isinstance(value, tuple | list):
            for item in value:
                collect(item)

    collect(outputs)
    return records


def _qualified_assumption_child(
    parent_child: str,
    field: str,
    assumption: Mapping[str, Any],
) -> str:
    raw_child = assumption.get("child")
    if isinstance(raw_child, str) and raw_child:
        return f"{parent_child}.{raw_child}"

    return parent_child


def _source_refs_from_outputs(value: Any) -> tuple[str, ...]:
    refs: list[str] = []
    if isinstance(value, Mapping):
        for key, child_value in value.items():
            if key == "source_ref" and isinstance(child_value, str):
                refs.append(child_value)
            elif key == "source_refs":
                refs.extend(_source_ref_values(child_value))
            else:
                refs.extend(_source_refs_from_outputs(child_value))
    elif isinstance(value, tuple | list):
        for item in value:
            refs.extend(_source_refs_from_outputs(item))
    return tuple(ref for ref in refs if ref)


def _source_ref_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, tuple | list):
        return tuple(str(item) for item in value if str(item))
    return ()


def _repair_fields(
    errors: Iterable[GuidelineError],
    *,
    missing: bool,
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            f"{error.child}.{error.field or error.output}"
            for error in errors
            if _is_missing_error(error.code) is missing
        )
    )


def _is_missing_error(code: str) -> bool:
    return code in {
        "insufficient_assumption_sources",
        "missing_assumption_basis",
        "missing_assumption_sources",
        "missing_candidate_sources",
        "missing_child_report",
        "missing_criterion",
        "missing_extracted_field",
        "missing_extraction_field",
        "missing_input",
        "missing_llm",
        "missing_output",
        "missing_prerequisite_output",
        "missing_rarefaction",
        "missing_report",
        "missing_source_refs",
        "missing_target_scenario_input",
    }


__all__ = [
    "GUIDELINE_CHILDREN",
    "GuidelineChild",
    "GuidelineChildReport",
    "GuidelineCreditReport",
    "GuidelineError",
    "fail_guideline_child",
    "guideline_child",
    "pass_guideline_child",
    "sum_guideline_credit",
]
