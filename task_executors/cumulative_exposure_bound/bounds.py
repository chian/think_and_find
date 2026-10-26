"""Invert steady risk equations into occupancy-time bounds."""

from __future__ import annotations

import math
from numbers import Real

from .._validation import non_negative, positive, probability


def steady_state_cumulative_exposure_bound(
    risk_tolerance: float,
    removal_capacity_m3_s: float,
    breathing_flow_m3_s: float,
    quanta_concentration_exhaled_m3: float,
    mask_penetration: float = 1.0,
    relative_transmissibility: float = 1.0,
) -> float:
    """Return the steady maximum value of (N - 1) * tau, in person-seconds."""

    denominator = (
        positive(breathing_flow_m3_s, "breathing_flow_m3_s") ** 2
        * probability(mask_penetration, "mask_penetration") ** 2
        * non_negative(
            quanta_concentration_exhaled_m3,
            "quanta_concentration_exhaled_m3",
        )
        * non_negative(relative_transmissibility, "relative_transmissibility")
    )
    if denominator == 0.0:
        return math.inf
    return (
        non_negative(risk_tolerance, "risk_tolerance")
        * positive(removal_capacity_m3_s, "removal_capacity_m3_s")
        / denominator
    )


def conservative_cumulative_exposure_bound(
    risk_tolerance: float,
    air_exchange_rate_s: float,
    room_volume_m3: float,
    breathing_flow_m3_s: float,
    quanta_concentration_exhaled_m3: float,
    mask_penetration: float = 1.0,
    relative_transmissibility: float = 1.0,
) -> float:
    """Return the ventilation-only maximum value of N * tau."""

    return steady_state_cumulative_exposure_bound(
        risk_tolerance,
        non_negative(air_exchange_rate_s, "air_exchange_rate_s")
        * positive(room_volume_m3, "room_volume_m3"),
        breathing_flow_m3_s,
        quanta_concentration_exhaled_m3,
        mask_penetration,
        relative_transmissibility,
    )


def transient_cumulative_exposure_bound(
    steady_state_bound_person_s: float,
    concentration_relaxation_rate_s: float,
    exposure_time_s: float,
) -> float:
    """Return the transient maximum value of Ns * tau."""

    bound = _non_negative_or_infinite(
        steady_state_bound_person_s,
        "steady_state_bound_person_s",
    )
    if bound == 0.0 or math.isinf(bound):
        return bound
    return bound * (
        1.0
        + 1.0
        / (
            positive(
                concentration_relaxation_rate_s,
                "concentration_relaxation_rate_s",
            )
            * positive(exposure_time_s, "exposure_time_s")
        )
    )


def steady_state_time_bound(
    cumulative_exposure_bound_person_s: float,
    susceptible_count: float,
) -> float:
    """Return the steady maximum exposure time for a susceptible count."""

    return _non_negative_or_infinite(
        cumulative_exposure_bound_person_s,
        "cumulative_exposure_bound_person_s",
    ) / positive(susceptible_count, "susceptible_count")


def transient_time_bound(
    steady_state_bound_person_s: float,
    concentration_relaxation_rate_s: float,
    susceptible_count: float,
) -> float:
    """Return the transient maximum exposure time for a susceptible count."""

    bound = _non_negative_or_infinite(
        steady_state_bound_person_s,
        "steady_state_bound_person_s",
    )
    if bound == 0.0 or math.isinf(bound):
        return bound

    relaxation_rate = positive(
        concentration_relaxation_rate_s,
        "concentration_relaxation_rate_s",
    )
    susceptibles = positive(susceptible_count, "susceptible_count")
    return (
        bound
        + math.sqrt(bound**2 + 4.0 * susceptibles * bound / relaxation_rate)
    ) / (2.0 * susceptibles)


def steady_state_susceptible_bound(
    cumulative_exposure_bound_person_s: float,
    exposure_time_s: float,
) -> float:
    """Return the steady maximum susceptible count for an exposure time."""

    return _non_negative_or_infinite(
        cumulative_exposure_bound_person_s,
        "cumulative_exposure_bound_person_s",
    ) / positive(exposure_time_s, "exposure_time_s")


def transient_susceptible_bound(
    steady_state_bound_person_s: float,
    concentration_relaxation_rate_s: float,
    exposure_time_s: float,
) -> float:
    """Return the transient maximum susceptible count for an exposure time."""

    transient_bound = transient_cumulative_exposure_bound(
        steady_state_bound_person_s,
        concentration_relaxation_rate_s,
        exposure_time_s,
    )
    return transient_bound / positive(exposure_time_s, "exposure_time_s")


def _non_negative_or_infinite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if math.isnan(result) or result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result

