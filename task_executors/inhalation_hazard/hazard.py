"""Convert airborne infectious quanta into per-susceptible hazard."""

from __future__ import annotations

from .._validation import non_negative, positive, probability


def inhalation_transmission_rate(
    airborne_quanta_concentration_m3: float,
    breathing_flow_m3_s: float,
    inhalation_mask_penetration: float = 1.0,
    relative_transmissibility: float = 1.0,
) -> float:
    """Return beta_a from quanta concentration already present in room air."""

    return (
        non_negative(
            airborne_quanta_concentration_m3,
            "airborne_quanta_concentration_m3",
        )
        * positive(breathing_flow_m3_s, "breathing_flow_m3_s")
        * probability(inhalation_mask_penetration, "inhalation_mask_penetration")
        * non_negative(relative_transmissibility, "relative_transmissibility")
    )


def steady_state_inhalation_transmission_rate(
    quanta_concentration_exhaled_m3: float,
    breathing_flow_m3_s: float,
    removal_capacity_m3_s: float,
    mask_penetration: float = 1.0,
    relative_transmissibility: float = 1.0,
) -> float:
    """Return the steady well-mixed hazard for equal exhale and inhale masks."""

    return (
        non_negative(
            quanta_concentration_exhaled_m3,
            "quanta_concentration_exhaled_m3",
        )
        * positive(breathing_flow_m3_s, "breathing_flow_m3_s") ** 2
        * probability(mask_penetration, "mask_penetration") ** 2
        * non_negative(relative_transmissibility, "relative_transmissibility")
        / positive(removal_capacity_m3_s, "removal_capacity_m3_s")
    )

