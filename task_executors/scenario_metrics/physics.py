"""Scalar extensions of Bazant and Bush (2021), with explicit boundaries."""

from __future__ import annotations

import math

from .._validation import non_negative, positive, probability


def ratio(numerator: float, denominator: float) -> float:
    """A zero hazard gives an unbounded limit, including at zero tolerance."""
    return numerator / denominator if denominator else math.inf


def buildup_time(time_s: float, removal_rate_s: float) -> float:
    """Integral of 1-exp(-lambda*t), stable close to a clean start."""
    t = non_negative(time_s, "time_s")
    rate = non_negative(removal_rate_s, "removal_rate_s")
    x = rate * t
    if x < 1e-4:
        return t * x * (0.5 + x * (-1 / 6 + x * (1 / 24 - x / 120)))
    return t + math.expm1(-x) / rate


def integrated_concentration(
    time_s: float, emission_rate_quanta_s: float, room_volume_m3: float,
    removal_rate_s: float, initial_concentration_quanta_m3: float,
) -> float:
    """Exact integral of the constant-source, single-bin Eq. 1 solution."""
    t = non_negative(time_s, "time_s")
    source = non_negative(emission_rate_quanta_s, "emission_rate_quanta_s")
    volume = positive(room_volume_m3, "room_volume_m3")
    rate = non_negative(removal_rate_s, "removal_rate_s")
    initial = non_negative(initial_concentration_quanta_m3, "initial_concentration_quanta_m3")
    if rate == 0:
        return initial * t + source / volume * t * t / 2
    return initial * (-math.expm1(-rate * t)) / rate + source / (volume * rate) * buildup_time(t, rate)


def transient_concentration(
    time_s: float, emission_rate_quanta_s: float, room_volume_m3: float,
    removal_rate_s: float, initial_concentration_quanta_m3: float,
) -> float:
    t = non_negative(time_s, "time_s")
    source = non_negative(emission_rate_quanta_s, "emission_rate_quanta_s")
    volume = positive(room_volume_m3, "room_volume_m3")
    rate = non_negative(removal_rate_s, "removal_rate_s")
    initial = non_negative(initial_concentration_quanta_m3, "initial_concentration_quanta_m3")
    if rate == 0:
        return initial + source * t / volume
    return initial * math.exp(-rate * t) + source / (volume * rate) * (-math.expm1(-rate * t))


def exact_time_bound(
    risk_tolerance: float, susceptible_count: float, breathing_flow_m3_s: float,
    mask_penetration: float, relative_transmissibility: float,
    emission_rate_quanta_s: float, room_volume_m3: float, removal_rate_s: float,
    initial_concentration_quanta_m3: float,
) -> float:
    """Invert Ns times integrated hazard, not the SI interpolation formula."""
    epsilon = non_negative(risk_tolerance, "risk_tolerance")
    factor = (non_negative(susceptible_count, "susceptible_count")
              * positive(breathing_flow_m3_s, "breathing_flow_m3_s")
              * probability(mask_penetration, "mask_penetration")
              * non_negative(relative_transmissibility, "relative_transmissibility"))
    source = non_negative(emission_rate_quanta_s, "emission_rate_quanta_s")
    initial = non_negative(initial_concentration_quanta_m3, "initial_concentration_quanta_m3")
    rate = non_negative(removal_rate_s, "removal_rate_s")
    positive(room_volume_m3, "room_volume_m3")
    if factor == 0 or (source == 0 and initial == 0):
        return math.inf
    if epsilon == 0:
        return 0.0
    if source == 0 and rate > 0 and factor * initial / rate <= epsilon:
        return math.inf
    def risk(t):
        return factor * integrated_concentration(t, source, room_volume_m3, rate, initial)
    high = 1.0
    for _ in range(1024):
        if risk(high) >= epsilon:
            break
        high *= 2
        if not math.isfinite(high):
            raise ValueError("Time bound exceeds representable range")
    else:
        raise ValueError("Could not bracket the time bound")
    low = 0.0
    for _ in range(160):
        mid = (low + high) / 2
        if risk(mid) <= epsilon:
            low = mid
        else:
            high = mid
    return low


def critical_distance(mouth_area_m2: float, entrainment_coefficient: float,
                      well_mixed_dilution_factor: float) -> float:
    """Jet/background concentration crossover; not a safe separation."""
    return math.sqrt(positive(mouth_area_m2, "mouth_area_m2")) / (
        positive(entrainment_coefficient, "entrainment_coefficient")
        * positive(well_mixed_dilution_factor, "well_mixed_dilution_factor"))


def threshold_distance(risk_tolerance: float, indoor_reproductive_number: float,
                       susceptible_count: float, plume_encounter_probability: float,
                       critical_distance_m: float) -> float:
    """Eq. 7 boundary: infinity means no finite distance meets the budget."""
    epsilon = non_negative(risk_tolerance, "risk_tolerance")
    risk = non_negative(indoor_reproductive_number, "indoor_reproductive_number")
    count = positive(susceptible_count, "susceptible_count")
    plume = probability(plume_encounter_probability, "plume_encounter_probability")
    distance = positive(critical_distance_m, "critical_distance_m")
    if risk > epsilon:
        return math.inf
    if risk == 0 or plume == 0:
        return 0.0
    if risk == epsilon:
        return math.inf
    return risk * plume * distance / (count * (epsilon - risk))


def well_mixed_distance(risk_tolerance: float, indoor_reproductive_number: float) -> float:
    """Spatially uniform risk: no positive minimum, or no separation remedy."""
    epsilon = non_negative(risk_tolerance, "risk_tolerance")
    risk = non_negative(indoor_reproductive_number, "indoor_reproductive_number")
    return 0.0 if risk <= epsilon else math.inf


def critical_radius(air_exchange_rate_s: float, room_height_m: float,
                    air_dynamic_viscosity_pa_s: float, particle_density_kg_m3: float,
                    air_density_kg_m3: float, gravity_m_s2: float) -> float:
    delta = particle_density_kg_m3 - air_density_kg_m3
    return math.sqrt(9 * non_negative(air_exchange_rate_s, "air_exchange_rate_s")
                     * positive(room_height_m, "room_height_m")
                     * positive(air_dynamic_viscosity_pa_s, "air_dynamic_viscosity_pa_s")
                     / (2 * positive(gravity_m_s2, "gravity_m_s2") * positive(delta, "density_difference")))


def clearance_time(removal_rate_s: float, clearance_fraction: float) -> float:
    non_negative(removal_rate_s, "removal_rate_s")
    fraction = probability(clearance_fraction, "clearance_fraction")
    if fraction == 0:
        return 0.0
    if fraction == 1 or removal_rate_s == 0:
        return math.inf
    return -math.log1p(-fraction) / positive(removal_rate_s, "removal_rate_s")


def maximum_total_occupancy(susceptible_bound: float) -> float:
    return math.inf if math.isinf(susceptible_bound) else math.floor(susceptible_bound) + 1
