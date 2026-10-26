"""Pass/fail credit for one deterministic calculation.

The first executor layer is intentionally mechanical:

    inputs -> calculation function -> one named output -> pass/fail credit

There is no model judge here.  A function either completes with an admissible
numeric output or returns a small error report that a parent can inspect.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class CalculationError:
    """One machine-readable reason a calculation failed."""

    code: str
    message: str
    field: str | None = None

    def as_record(self) -> dict[str, str]:
        record = {
            "code": self.code,
            "message": self.message,
        }
        if self.field:
            record["field"] = self.field
        return record


@dataclass(frozen=True)
class CalculationReport:
    """Minimal parent-facing result for a one-shot calculation."""

    task: str
    output: str
    passed: bool
    score: float
    unit: str | None = None
    outputs: Mapping[str, float] = field(default_factory=dict)
    errors: tuple[CalculationError, ...] = ()
    identities: tuple[str, ...] = ()
    stop: bool = True

    @property
    def value(self) -> float | None:
        return self.outputs.get(self.output)

    def as_record(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "output": self.output,
            "output_name": self.output,
            "value": self.value,
            "unit": self.unit,
            "passed": self.passed,
            "score": self.score,
            "stop": self.stop,
            "outputs": dict(self.outputs),
            "errors": [error.as_record() for error in self.errors],
            "identities": list(self.identities),
        }


@dataclass(frozen=True)
class CalculationTask:
    """Metadata needed to run and judge a one-output calculation function."""

    name: str
    output: str
    function: Callable[..., float]
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...] = ()
    unit: str | None = None
    min_output: float | None = 0.0
    max_output: float | None = None
    allow_infinite_output: bool = False
    identity_version: str = "v1"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CalculationTask.name must be populated")
        if not self.output:
            raise ValueError("CalculationTask.output must be populated")
        if self.unit is not None and not self.unit:
            raise ValueError("CalculationTask.unit must be populated when set")
        if self.min_output is not None and not math.isfinite(self.min_output):
            raise ValueError("CalculationTask.min_output must be finite")
        if self.max_output is not None and not math.isfinite(self.max_output):
            raise ValueError("CalculationTask.max_output must be finite")
        if (
            self.min_output is not None
            and self.max_output is not None
            and self.min_output > self.max_output
        ):
            raise ValueError("CalculationTask.min_output cannot exceed max_output")

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.required_inputs + self.optional_inputs


def calculation_task(
    name: str,
    output: str,
    function: Callable[..., float],
    *,
    unit: str | None = None,
    min_output: float | None = 0.0,
    max_output: float | None = None,
    allow_infinite_output: bool = False,
    identity_version: str = "v1",
) -> CalculationTask:
    """Create task metadata from a function's signature."""

    required_inputs: list[str] = []
    optional_inputs: list[str] = []
    signature = inspect.signature(function)
    for parameter in signature.parameters.values():
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError(
                f"{name} has unsupported parameter kind for {parameter.name}"
            )
        if parameter.default is inspect.Parameter.empty:
            required_inputs.append(parameter.name)
        else:
            optional_inputs.append(parameter.name)
    return CalculationTask(
        name=name,
        output=output,
        function=function,
        required_inputs=tuple(required_inputs),
        optional_inputs=tuple(optional_inputs),
        unit=unit or _OUTPUT_UNITS.get(output),
        min_output=min_output,
        max_output=max_output,
        allow_infinite_output=allow_infinite_output,
        identity_version=identity_version,
    )


def credit_identity(task: CalculationTask, inputs: Mapping[str, Any]) -> str:
    """Return a stable credit identity for the task and accepted inputs."""

    payload = {
        "task": task.name,
        "output": task.output,
        "version": task.identity_version,
        "inputs": {
            name: _json_safe(inputs[name])
            for name in task.inputs
            if name in inputs
        },
    }
    digest = hashlib.sha256(
        json.dumps(payload, allow_nan=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{task.name}:{task.output}:{task.identity_version}:{digest}"


def run_calculation(
    task: CalculationTask,
    inputs: Mapping[str, Any],
) -> CalculationReport:
    """Run one task once and return a pass/fail report."""

    errors = tuple(
        CalculationError(
            code="missing_input",
            field=name,
            message=f"{name} is required",
        )
        for name in task.required_inputs
        if name not in inputs
    )
    if errors:
        return _failed(task, errors)

    try:
        value = task.function(
            **{name: inputs[name] for name in task.inputs if name in inputs}
        )
    except ValueError as exc:
        return _failed(
            task,
            (
                CalculationError(
                    code="invalid_input",
                    message=str(exc),
                ),
            ),
        )
    except Exception as exc:
        return _failed(
            task,
            (
                CalculationError(
                    code="calculation_error",
                    message=str(exc),
                ),
            ),
        )

    output_error = _judge_output(task, value)
    if output_error is not None:
        return _failed(task, (output_error,))

    numeric_value = float(value)
    return CalculationReport(
        task=task.name,
        output=task.output,
        passed=True,
        score=1.0,
        unit=task.unit,
        outputs={task.output: numeric_value},
        identities=(credit_identity(task, inputs),),
    )


def _failed(
    task: CalculationTask,
    errors: tuple[CalculationError, ...],
) -> CalculationReport:
    return CalculationReport(
        task=task.name,
        output=task.output,
        passed=False,
        score=0.0,
        unit=task.unit,
        errors=errors,
    )


def _judge_output(
    task: CalculationTask,
    value: Any,
) -> CalculationError | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return CalculationError(
            code="invalid_output",
            field=task.output,
            message=f"{task.output} must be a real number",
        )
    numeric_value = float(value)
    if task.allow_infinite_output:
        if math.isnan(numeric_value):
            return CalculationError(
                code="invalid_output",
                field=task.output,
                message=f"{task.output} must not be NaN",
            )
    elif not math.isfinite(numeric_value):
        return CalculationError(
            code="invalid_output",
            field=task.output,
            message=f"{task.output} must be finite",
        )
    if task.min_output is not None and numeric_value < task.min_output:
        return CalculationError(
            code="invalid_output",
            field=task.output,
            message=f"{task.output} must be >= {task.min_output}",
        )
    if task.max_output is not None and numeric_value > task.max_output:
        return CalculationError(
            code="invalid_output",
            field=task.output,
            message=f"{task.output} must be <= {task.max_output}",
        )
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            return {"nonfinite_number": repr(number)}
        return number
    if isinstance(value, str) or value is None:
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    return repr(value)


_OUTPUT_UNITS: Mapping[str, str] = {
    "air_exchange_rate_s": "1/s",
    "air_purification_rate_s": "1/s",
    "attack_probability": "probability",
    "concentration_quanta_m3": "quanta/m3",
    "cumulative_exposure_bound_person_s": "person_s",
    "exposure_time_bound_s": "s",
    "filtration_rate_s": "1/s",
    "first_transmission_probability": "probability",
    "flow_l_s": "L/s",
    "flow_m3_h": "m3/h",
    "flow_m3_s": "m3/s",
    "indoor_reproductive_number": "scalar",
    "jet_dilution_ratio": "scalar",
    "jet_multiplier": "scalar",
    "quanta_concentration_exhaled_m3": "quanta/m3",
    "quanta_emission_rate_s": "quanta/s",
    "rate_d": "1/d",
    "rate_h": "1/h",
    "rate_s": "1/s",
    "relative_risk_index": "scalar",
    "removal_capacity_m3_s": "m3/s",
    "removal_rate_s": "1/s",
    "room_height_m": "m",
    "room_volume_m3": "m3",
    "secondary_case_count": "count",
    "sedimentation_rate_s": "1/s",
    "seconds": "s",
    "settling_speed_m_h": "m/h",
    "settling_speed_m_s": "m/s",
    "speed_m_h": "m/h",
    "speed_m_s": "m/s",
    "steady_state_concentration_quanta_m3": "quanta/m3",
    "susceptible_bound": "person",
    "susceptible_count": "count",
    "transmission_rate_s": "1/s",
    "well_mixed_dilution_factor": "scalar",
}
