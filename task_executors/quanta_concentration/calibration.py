"""Back-calculate exhaled quanta concentration from outbreak summaries."""

from __future__ import annotations

import math

from .._validation import non_negative, positive, probability


def quanta_concentration_from_reproductive_number(
    indoor_reproductive_number: float,
    removal_capacity_m3_s: float,
    susceptible_count: float,
    exposure_time_s: float,
    breathing_flow_m3_s: float,
    mask_penetration: float = 1.0,
    relative_transmissibility: float = 1.0,
) -> float:
    """Solve the steady well-mixed risk equations backward for C_q."""

    denominator = (
        positive(susceptible_count, "susceptible_count")
        * positive(exposure_time_s, "exposure_time_s")
        * positive(breathing_flow_m3_s, "breathing_flow_m3_s") ** 2
        * probability(mask_penetration, "mask_penetration") ** 2
        * positive(relative_transmissibility, "relative_transmissibility")
    )
    return (
        non_negative(indoor_reproductive_number, "indoor_reproductive_number")
        * positive(removal_capacity_m3_s, "removal_capacity_m3_s")
        / denominator
    )


def quanta_concentration_from_aggregate_reproductive_number(
    indoor_reproductive_number: float,
    removal_capacity_m3_s: float,
    exposure_time_s: float,
    breathing_flow_m3_s: float,
    mask_penetration: float = 1.0,
    relative_transmissibility: float = 1.0,
) -> float:
    """Infer C_q from an already aggregated indoor reproductive number."""

    denominator = (
        positive(exposure_time_s, "exposure_time_s")
        * positive(breathing_flow_m3_s, "breathing_flow_m3_s") ** 2
        * probability(mask_penetration, "mask_penetration") ** 2
        * positive(relative_transmissibility, "relative_transmissibility")
    )
    return (
        non_negative(indoor_reproductive_number, "indoor_reproductive_number")
        * positive(removal_capacity_m3_s, "removal_capacity_m3_s")
        / denominator
    )


def quanta_concentration_from_attack_probability(
    attack_probability: float,
    removal_capacity_m3_s: float,
    exposure_time_s: float,
    breathing_flow_m3_s: float,
    mask_penetration: float = 1.0,
    relative_transmissibility: float = 1.0,
) -> float:
    """Infer C_q from a per-susceptible attack probability."""

    attack = probability(attack_probability, "attack_probability")
    if attack == 1.0:
        raise ValueError("attack_probability must be less than 1")
    denominator = (
        positive(exposure_time_s, "exposure_time_s")
        * positive(breathing_flow_m3_s, "breathing_flow_m3_s") ** 2
        * probability(mask_penetration, "mask_penetration") ** 2
        * positive(relative_transmissibility, "relative_transmissibility")
    )
    return -math.log1p(-attack) * positive(
        removal_capacity_m3_s, "removal_capacity_m3_s"
    ) / denominator


def quanta_concentration_from_slow_incubation_attack_probability(
    attack_probability: float,
    initial_infected_count: float,
    removal_capacity_m3_s: float,
    exposure_time_s: float,
    breathing_flow_m3_s: float,
    mask_penetration: float = 1.0,
    relative_transmissibility: float = 1.0,
) -> float:
    """Infer C_q from a slow-incubation attack probability."""

    return quanta_concentration_from_attack_probability(
        attack_probability,
        removal_capacity_m3_s,
        exposure_time_s
        * positive(initial_infected_count, "initial_infected_count"),
        breathing_flow_m3_s,
        mask_penetration,
        relative_transmissibility,
    )


def quanta_concentration_from_population_transmission_rate(
    population_transmission_rate_s: float,
    total_occupant_count: float,
    removal_capacity_m3_s: float,
    breathing_flow_m3_s: float,
    mask_penetration: float = 1.0,
    relative_transmissibility: float = 1.0,
) -> float:
    """Infer C_q from a fitted population-scale N * beta rate."""

    denominator = (
        positive(total_occupant_count, "total_occupant_count")
        * positive(breathing_flow_m3_s, "breathing_flow_m3_s") ** 2
        * probability(mask_penetration, "mask_penetration") ** 2
        * positive(relative_transmissibility, "relative_transmissibility")
    )
    return positive(
        population_transmission_rate_s,
        "population_transmission_rate_s",
    ) * positive(removal_capacity_m3_s, "removal_capacity_m3_s") / denominator


def quanta_emission_rate(
    quanta_concentration_exhaled_m3: float,
    breathing_flow_m3_s: float,
) -> float:
    """Return quanta emission rate, lambda_q = Cq * Qb."""

    return non_negative(
        quanta_concentration_exhaled_m3,
        "quanta_concentration_exhaled_m3",
    ) * positive(breathing_flow_m3_s, "breathing_flow_m3_s")


def quanta_concentration_from_emission_rate(
    quanta_emission_rate_s: float,
    breathing_flow_m3_s: float,
) -> float:
    """Return Cq from a quanta emission rate and breathing flow."""

    return non_negative(
        quanta_emission_rate_s,
        "quanta_emission_rate_s",
    ) / positive(breathing_flow_m3_s, "breathing_flow_m3_s")


def rescaled_quanta_concentration(
    baseline_quanta_concentration_exhaled_m3: float,
    activity_aerosol_volume_fraction: float,
    baseline_aerosol_volume_fraction: float,
) -> float:
    """Rescale a baseline Cq by activity-specific infectious aerosol volume."""

    return non_negative(
        baseline_quanta_concentration_exhaled_m3,
        "baseline_quanta_concentration_exhaled_m3",
    ) * positive(
        activity_aerosol_volume_fraction,
        "activity_aerosol_volume_fraction",
    ) / positive(
        baseline_aerosol_volume_fraction,
        "baseline_aerosol_volume_fraction",
    )
