"""Integrate per-susceptible airborne hazard over a scenario."""

from __future__ import annotations

import math

from .._validation import non_negative, probability


def indoor_reproductive_number(
    susceptible_count: float,
    transmission_rate_s: float,
    exposure_time_s: float,
) -> float:
    """Return Rin = Ns * beta_a * tau for a constant hazard."""

    return (
        non_negative(susceptible_count, "susceptible_count")
        * non_negative(transmission_rate_s, "transmission_rate_s")
        * non_negative(exposure_time_s, "exposure_time_s")
    )


def indoor_reproductive_number_from_population_reproductive_number(
    population_reproductive_number: float,
    indoor_attribution_fraction: float = 1.0,
) -> float:
    """Return the indoor share of a population reproductive number."""

    return non_negative(
        population_reproductive_number,
        "population_reproductive_number",
    ) * probability(
        indoor_attribution_fraction,
        "indoor_attribution_fraction",
    )


def first_transmission_probability(indoor_reproductive_number: float) -> float:
    """Return the Poisson probability of at least one transmission."""

    return -math.expm1(
        -non_negative(
            indoor_reproductive_number,
            "indoor_reproductive_number",
        )
    )
