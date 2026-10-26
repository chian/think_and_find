"""Small unit conversions needed before SI calculation leaves."""

from __future__ import annotations

from .._validation import non_negative

SECONDS_PER_HOUR = 3_600.0
SECONDS_PER_MINUTE = 60.0
HOURS_PER_DAY = 24.0
SECONDS_PER_DAY = SECONDS_PER_HOUR * HOURS_PER_DAY
LITERS_PER_CUBIC_METER = 1_000.0
METERS_PER_FOOT = 0.3048


def per_hour_to_per_second(rate_h: float) -> float:
    """Convert a rate from 1/h to 1/s."""

    return non_negative(rate_h, "rate_h") / SECONDS_PER_HOUR


def per_second_to_per_hour(rate_s: float) -> float:
    """Convert a rate from 1/s to 1/h."""

    return non_negative(rate_s, "rate_s") * SECONDS_PER_HOUR


def per_day_to_per_second(rate_d: float) -> float:
    """Convert a rate from 1/day to 1/s."""

    return non_negative(rate_d, "rate_d") / SECONDS_PER_DAY


def per_second_to_per_day(rate_s: float) -> float:
    """Convert a rate from 1/s to 1/day."""

    return non_negative(rate_s, "rate_s") * SECONDS_PER_DAY


def hours_to_seconds(hours: float) -> float:
    """Convert hours to seconds."""

    return non_negative(hours, "hours") * SECONDS_PER_HOUR


def minutes_to_seconds(minutes: float) -> float:
    """Convert minutes to seconds."""

    return non_negative(minutes, "minutes") * SECONDS_PER_MINUTE


def days_to_seconds(days: float) -> float:
    """Convert days to seconds."""

    return non_negative(days, "days") * SECONDS_PER_DAY


def cubic_meters_per_hour_to_cubic_meters_per_second(flow_m3_h: float) -> float:
    """Convert a volumetric flow from m3/h to m3/s."""

    return non_negative(flow_m3_h, "flow_m3_h") / SECONDS_PER_HOUR


def cubic_meters_per_minute_to_cubic_meters_per_second(flow_m3_min: float) -> float:
    """Convert a volumetric flow from m3/min to m3/s."""

    return non_negative(flow_m3_min, "flow_m3_min") / SECONDS_PER_MINUTE


def cubic_feet_per_minute_to_cubic_meters_per_second(flow_cfm: float) -> float:
    """Convert actual volumetric flow using the exact international foot."""

    return non_negative(flow_cfm, "flow_cfm") * METERS_PER_FOOT**3 / SECONDS_PER_MINUTE


def cubic_meters_per_second_to_cubic_meters_per_hour(flow_m3_s: float) -> float:
    """Convert a volumetric flow from m3/s to m3/h."""

    return non_negative(flow_m3_s, "flow_m3_s") * SECONDS_PER_HOUR


def liters_per_second_to_cubic_meters_per_second(flow_l_s: float) -> float:
    """Convert a volumetric flow from L/s to m3/s."""

    return non_negative(flow_l_s, "flow_l_s") / LITERS_PER_CUBIC_METER


def cubic_meters_per_second_to_liters_per_second(flow_m3_s: float) -> float:
    """Convert a volumetric flow from m3/s to L/s."""

    return non_negative(flow_m3_s, "flow_m3_s") * LITERS_PER_CUBIC_METER


def meters_per_hour_to_meters_per_second(speed_m_h: float) -> float:
    """Convert a speed from m/h to m/s."""

    return non_negative(speed_m_h, "speed_m_h") / SECONDS_PER_HOUR


def meters_per_second_to_meters_per_hour(speed_m_s: float) -> float:
    """Convert a speed from m/s to m/h."""

    return non_negative(speed_m_s, "speed_m_s") * SECONDS_PER_HOUR
