"""Shared numeric validation for calculation kernels."""

from __future__ import annotations

import math
from numbers import Real


def finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def non_negative(value: object, name: str) -> float:
    result = finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def positive(value: object, name: str) -> float:
    result = finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def probability(value: object, name: str) -> float:
    result = finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result

