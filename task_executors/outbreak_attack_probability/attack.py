"""Convert admitted outbreak counts into calibration targets."""

from __future__ import annotations

from .._validation import non_negative, positive


def secondary_case_count(
    confirmed_secondary_case_count: float,
    probable_secondary_case_count: float = 0.0,
) -> float:
    """Return total observed secondary cases."""

    return non_negative(
        confirmed_secondary_case_count,
        "confirmed_secondary_case_count",
    ) + non_negative(
        probable_secondary_case_count,
        "probable_secondary_case_count",
    )


def secondary_case_count_from_cumulative(
    final_infected_count: float,
    initial_infected_count: float = 1.0,
) -> float:
    """Return secondary cases from cumulative infected counts."""

    final_infected = non_negative(
        final_infected_count,
        "final_infected_count",
    )
    initial_infected = positive(
        initial_infected_count,
        "initial_infected_count",
    )
    if final_infected < initial_infected:
        raise ValueError("final_infected_count must be >= initial_infected_count")
    return final_infected - initial_infected


def susceptible_count(
    attendee_count: float,
    index_case_count: float = 1.0,
) -> float:
    """Return exposed people who were not index cases."""

    susceptible = non_negative(attendee_count, "attendee_count") - positive(
        index_case_count,
        "index_case_count",
    )
    if susceptible <= 0.0:
        raise ValueError("attendee_count must exceed index_case_count")
    return susceptible


def secondary_attack_probability(
    secondary_case_count: float,
    susceptible_count: float,
) -> float:
    """Return observed secondary cases per susceptible exposed person."""

    cases = non_negative(secondary_case_count, "secondary_case_count")
    susceptible = positive(susceptible_count, "susceptible_count")
    if cases > susceptible:
        raise ValueError("secondary_case_count must be <= susceptible_count")
    return cases / susceptible


def observed_indoor_reproductive_number(
    secondary_case_count: float,
    index_case_count: float = 1.0,
) -> float:
    """Return observed secondary cases per index case."""

    return non_negative(secondary_case_count, "secondary_case_count") / positive(
        index_case_count,
        "index_case_count",
    )
