"""Analytic concentration curves for one well-mixed radius bin."""

from __future__ import annotations

import math

from .._validation import non_negative, positive


def steady_state_concentration(
    emission_rate_quanta_s: float,
    room_volume_m3: float,
    removal_rate_s: float,
) -> float:
    """Return Cs = P / (lambda * V)."""

    return non_negative(emission_rate_quanta_s, "emission_rate_quanta_s") / (
        positive(removal_rate_s, "removal_rate_s")
        * positive(room_volume_m3, "room_volume_m3")
    )


def concentration_at_time(
    time_s: float,
    emission_rate_quanta_s: float,
    room_volume_m3: float,
    removal_rate_s: float,
    *,
    initial_concentration_quanta_m3: float = 0.0,
) -> float:
    """Return C(t) for a constant source in a linear well-mixed room."""

    steady = steady_state_concentration(
        emission_rate_quanta_s,
        room_volume_m3,
        removal_rate_s,
    )
    initial = non_negative(
        initial_concentration_quanta_m3, "initial_concentration_quanta_m3"
    )
    return steady + (initial - steady) * math.exp(
        -positive(removal_rate_s, "removal_rate_s")
        * non_negative(time_s, "time_s")
    )


def dilution_factor(
    breathing_flow_m3_s: float,
    room_volume_m3: float,
    removal_rate_s: float,
) -> float:
    """Return the well-mixed dilution factor, Qb / (lambda * V)."""

    return positive(breathing_flow_m3_s, "breathing_flow_m3_s") / (
        positive(removal_rate_s, "removal_rate_s")
        * positive(room_volume_m3, "room_volume_m3")
    )

