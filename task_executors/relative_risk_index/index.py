"""Relative risk index for comparing events."""

from __future__ import annotations

from .._validation import non_negative, positive, probability


def relative_risk_index(
    occupant_count: float,
    exposure_time_s: float,
    quanta_concentration_exhaled_m3: float,
    breathing_flow_m3_s: float,
    mask_penetration: float,
    air_exchange_rate_s: float,
    room_volume_m3: float,
) -> float:
    """Return IR = N * tau * Cq * Qb**2 * pm**2 / (lambda_a * V)."""

    return (
        non_negative(occupant_count, "occupant_count")
        * non_negative(exposure_time_s, "exposure_time_s")
        * non_negative(
            quanta_concentration_exhaled_m3,
            "quanta_concentration_exhaled_m3",
        )
        * positive(breathing_flow_m3_s, "breathing_flow_m3_s") ** 2
        * probability(mask_penetration, "mask_penetration") ** 2
        / (
            positive(air_exchange_rate_s, "air_exchange_rate_s")
            * positive(room_volume_m3, "room_volume_m3")
        )
    )
