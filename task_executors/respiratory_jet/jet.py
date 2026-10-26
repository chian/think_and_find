"""Crude short-range plume correction for local respiratory jets."""

from __future__ import annotations

import math

from .._validation import non_negative, positive, probability


def jet_dilution_ratio(
    mouth_area_m2: float,
    entrainment_coefficient: float,
    neighbor_distance_m: float,
) -> float:
    """Return Cj(x) / C0 = sqrt(Am) / (alpha_t * x)."""

    return math.sqrt(positive(mouth_area_m2, "mouth_area_m2")) / (
        positive(entrainment_coefficient, "entrainment_coefficient")
        * positive(neighbor_distance_m, "neighbor_distance_m")
    )


def respiratory_jet_multiplier(
    plume_encounter_probability: float,
    mouth_area_m2: float,
    entrainment_coefficient: float,
    neighbor_distance_m: float,
    well_mixed_dilution_factor: float,
    susceptible_count: float,
) -> float:
    """Return the multiplier added to long-range airborne risk."""

    return (
        probability(plume_encounter_probability, "plume_encounter_probability")
        * jet_dilution_ratio(
            mouth_area_m2,
            entrainment_coefficient,
            neighbor_distance_m,
        )
        / (
            positive(well_mixed_dilution_factor, "well_mixed_dilution_factor")
            * positive(susceptible_count, "susceptible_count")
        )
    )


def jet_corrected_reproductive_number(
    indoor_reproductive_number: float,
    jet_multiplier: float,
) -> float:
    """Return the long-range risk multiplied by 1 + jet_multiplier."""

    return non_negative(
        indoor_reproductive_number, "indoor_reproductive_number"
    ) * (1.0 + non_negative(jet_multiplier, "jet_multiplier"))

