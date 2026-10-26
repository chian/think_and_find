"""Room-scale loss rates for airborne infectious material."""

from __future__ import annotations

from .._validation import non_negative, positive, probability

GRAVITY_M_S2 = 9.80665
WATER_DENSITY_KG_M3 = 997.0
AIR_DENSITY_KG_M3 = 1.204
AIR_DYNAMIC_VISCOSITY_PA_S = 1.825e-5


def air_exchange_rate(
    outdoor_air_flow_m3_s: float,
    room_volume_m3: float,
) -> float:
    """Return the outdoor-air exchange rate, Q / V, in 1/s."""

    return non_negative(outdoor_air_flow_m3_s, "outdoor_air_flow_m3_s") / positive(
        room_volume_m3, "room_volume_m3"
    )


def filtration_loss_rate(
    recirculation_flow_m3_s: float,
    filter_efficiency: float,
    room_volume_m3: float,
) -> float:
    """Return the filtered recirculation loss rate, pf * Qr / V, in 1/s."""

    return (
        probability(filter_efficiency, "filter_efficiency")
        * non_negative(recirculation_flow_m3_s, "recirculation_flow_m3_s")
        / positive(room_volume_m3, "room_volume_m3")
    )


def stokes_settling_speed(
    radius_m: float,
    *,
    particle_density_kg_m3: float = WATER_DENSITY_KG_M3,
    air_density_kg_m3: float = AIR_DENSITY_KG_M3,
    air_dynamic_viscosity_pa_s: float = AIR_DYNAMIC_VISCOSITY_PA_S,
    gravity_m_s2: float = GRAVITY_M_S2,
) -> float:
    """Return Stokes settling speed, 2 * delta_rho * g * r**2 / (9 * mu)."""

    radius = positive(radius_m, "radius_m")
    particle_density = positive(
        particle_density_kg_m3, "particle_density_kg_m3"
    )
    air_density = non_negative(air_density_kg_m3, "air_density_kg_m3")
    density_difference = particle_density - air_density
    if density_difference <= 0.0:
        raise ValueError("particle_density_kg_m3 must exceed air_density_kg_m3")
    return (
        2.0
        * density_difference
        * positive(gravity_m_s2, "gravity_m_s2")
        * radius**2
        / (9.0 * positive(air_dynamic_viscosity_pa_s, "air_dynamic_viscosity_pa_s"))
    )


def sedimentation_rate(settling_speed_m_s: float, room_height_m: float) -> float:
    """Return the well-mixed sedimentation loss rate, vs / H, in 1/s."""

    return non_negative(settling_speed_m_s, "settling_speed_m_s") / positive(
        room_height_m, "room_height_m"
    )


def air_purification_rate(
    air_exchange_rate_s: float,
    filtration_rate_s: float = 0.0,
    deactivation_rate_s: float = 0.0,
) -> float:
    """Return removal excluding gravitational settling."""

    return (
        non_negative(air_exchange_rate_s, "air_exchange_rate_s")
        + non_negative(filtration_rate_s, "filtration_rate_s")
        + non_negative(deactivation_rate_s, "deactivation_rate_s")
    )


def total_removal_rate(
    air_exchange_rate_s: float,
    filtration_rate_s: float = 0.0,
    sedimentation_rate_s: float = 0.0,
    deactivation_rate_s: float = 0.0,
) -> float:
    """Return ventilation + filtration + settling + deactivation loss."""

    return (
        air_purification_rate(
            air_exchange_rate_s,
            filtration_rate_s,
            deactivation_rate_s,
        )
        + non_negative(sedimentation_rate_s, "sedimentation_rate_s")
    )


def removal_capacity(
    room_volume_m3: float,
    air_purification_rate_s: float,
    settling_speed_m_s: float = 0.0,
    floor_area_m2: float = 0.0,
) -> float:
    """Return clean-air capacity, lambda_c * V + vs * A."""

    return (
        positive(room_volume_m3, "room_volume_m3")
        * non_negative(air_purification_rate_s, "air_purification_rate_s")
        + non_negative(settling_speed_m_s, "settling_speed_m_s")
        * non_negative(floor_area_m2, "floor_area_m2")
    )


def removal_capacity_from_total_loss_rate(
    room_volume_m3: float,
    total_removal_rate_s: float,
) -> float:
    """Return V * lambda when the reported rate already includes all losses."""

    return positive(room_volume_m3, "room_volume_m3") * non_negative(
        total_removal_rate_s, "total_removal_rate_s",
    )
