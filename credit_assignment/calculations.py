"""One-shot credit wrappers for the current deterministic calculations."""

from __future__ import annotations

from typing import Any, Mapping

from episodic_workflow.task_executors.cumulative_exposure_bound import (
    conservative_cumulative_exposure_bound,
    steady_state_cumulative_exposure_bound,
    steady_state_susceptible_bound,
    steady_state_time_bound,
    transient_cumulative_exposure_bound,
    transient_susceptible_bound,
    transient_time_bound,
)
from episodic_workflow.task_executors.effective_removal_rate import (
    air_exchange_rate,
    air_purification_rate,
    filtration_loss_rate,
    removal_capacity,
    removal_capacity_from_total_loss_rate,
    sedimentation_rate,
    stokes_settling_speed,
    total_removal_rate,
)
from episodic_workflow.task_executors.indoor_reproductive_number import (
    first_transmission_probability,
    indoor_reproductive_number,
    indoor_reproductive_number_from_population_reproductive_number,
)
from episodic_workflow.task_executors.inhalation_hazard import (
    inhalation_transmission_rate,
    steady_state_inhalation_transmission_rate,
)
from episodic_workflow.task_executors.outbreak_attack_probability import (
    observed_indoor_reproductive_number,
    secondary_attack_probability,
    secondary_case_count,
    secondary_case_count_from_cumulative,
    susceptible_count,
)
from episodic_workflow.task_executors.quanta_concentration import (
    quanta_concentration_from_aggregate_reproductive_number,
    quanta_concentration_from_emission_rate,
    quanta_concentration_from_attack_probability,
    quanta_concentration_from_population_transmission_rate,
    quanta_concentration_from_reproductive_number,
    quanta_concentration_from_slow_incubation_attack_probability,
    quanta_emission_rate,
    rescaled_quanta_concentration,
)
from episodic_workflow.task_executors.relative_risk_index import relative_risk_index
from episodic_workflow.task_executors.respiratory_jet import (
    jet_corrected_reproductive_number,
    jet_dilution_ratio,
    respiratory_jet_multiplier,
)
from episodic_workflow.task_executors.room_geometry import (
    room_height,
    room_volume,
)
from episodic_workflow.task_executors.unit_conversion import (
    cubic_feet_per_minute_to_cubic_meters_per_second,
    cubic_meters_per_minute_to_cubic_meters_per_second,
    cubic_meters_per_hour_to_cubic_meters_per_second,
    cubic_meters_per_second_to_cubic_meters_per_hour,
    cubic_meters_per_second_to_liters_per_second,
    days_to_seconds,
    hours_to_seconds,
    liters_per_second_to_cubic_meters_per_second,
    meters_per_hour_to_meters_per_second,
    meters_per_second_to_meters_per_hour,
    minutes_to_seconds,
    per_day_to_per_second,
    per_hour_to_per_second,
    per_second_to_per_day,
    per_second_to_per_hour,
)
from episodic_workflow.task_executors.well_mixed_concentration import (
    concentration_at_time,
    dilution_factor,
    steady_state_concentration,
)

from .simple import (
    CalculationReport,
    CalculationTask,
    calculation_task,
    run_calculation,
)


CALCULATION_TASKS: dict[str, CalculationTask] = {
    "removal_capacity_from_total_loss_rate": calculation_task(
        "removal_capacity_from_total_loss_rate",
        "removal_capacity_m3_s",
        removal_capacity_from_total_loss_rate,
    ),
    "cubic_feet_per_minute_to_cubic_meters_per_second": calculation_task(
        "cubic_feet_per_minute_to_cubic_meters_per_second",
        "flow_m3_s",
        cubic_feet_per_minute_to_cubic_meters_per_second,
    ),
    "cubic_meters_per_minute_to_cubic_meters_per_second": calculation_task(
        "cubic_meters_per_minute_to_cubic_meters_per_second",
        "flow_m3_s",
        cubic_meters_per_minute_to_cubic_meters_per_second,
    ),
    "air_exchange_rate": calculation_task(
        "air_exchange_rate",
        "air_exchange_rate_s",
        air_exchange_rate,
    ),
    "filtration_loss_rate": calculation_task(
        "filtration_loss_rate",
        "filtration_rate_s",
        filtration_loss_rate,
    ),
    "stokes_settling_speed": calculation_task(
        "stokes_settling_speed",
        "settling_speed_m_s",
        stokes_settling_speed,
    ),
    "sedimentation_rate": calculation_task(
        "sedimentation_rate",
        "sedimentation_rate_s",
        sedimentation_rate,
    ),
    "air_purification_rate": calculation_task(
        "air_purification_rate",
        "air_purification_rate_s",
        air_purification_rate,
    ),
    "total_removal_rate": calculation_task(
        "total_removal_rate",
        "removal_rate_s",
        total_removal_rate,
    ),
    "removal_capacity": calculation_task(
        "removal_capacity",
        "removal_capacity_m3_s",
        removal_capacity,
    ),
    "steady_state_concentration": calculation_task(
        "steady_state_concentration",
        "steady_state_concentration_quanta_m3",
        steady_state_concentration,
    ),
    "concentration_at_time": calculation_task(
        "concentration_at_time",
        "concentration_quanta_m3",
        concentration_at_time,
    ),
    "dilution_factor": calculation_task(
        "dilution_factor",
        "well_mixed_dilution_factor",
        dilution_factor,
    ),
    "inhalation_transmission_rate": calculation_task(
        "inhalation_transmission_rate",
        "transmission_rate_s",
        inhalation_transmission_rate,
    ),
    "steady_state_inhalation_transmission_rate": calculation_task(
        "steady_state_inhalation_transmission_rate",
        "transmission_rate_s",
        steady_state_inhalation_transmission_rate,
    ),
    "indoor_reproductive_number": calculation_task(
        "indoor_reproductive_number",
        "indoor_reproductive_number",
        indoor_reproductive_number,
    ),
    "indoor_reproductive_number_from_population_reproductive_number": (
        calculation_task(
            "indoor_reproductive_number_from_population_reproductive_number",
            "indoor_reproductive_number",
            indoor_reproductive_number_from_population_reproductive_number,
        )
    ),
    "first_transmission_probability": calculation_task(
        "first_transmission_probability",
        "first_transmission_probability",
        first_transmission_probability,
        max_output=1.0,
    ),
    "steady_state_cumulative_exposure_bound": calculation_task(
        "steady_state_cumulative_exposure_bound",
        "cumulative_exposure_bound_person_s",
        steady_state_cumulative_exposure_bound,
        allow_infinite_output=True,
    ),
    "conservative_cumulative_exposure_bound": calculation_task(
        "conservative_cumulative_exposure_bound",
        "cumulative_exposure_bound_person_s",
        conservative_cumulative_exposure_bound,
        allow_infinite_output=True,
    ),
    "transient_cumulative_exposure_bound": calculation_task(
        "transient_cumulative_exposure_bound",
        "cumulative_exposure_bound_person_s",
        transient_cumulative_exposure_bound,
        allow_infinite_output=True,
    ),
    "steady_state_time_bound": calculation_task(
        "steady_state_time_bound",
        "exposure_time_bound_s",
        steady_state_time_bound,
        allow_infinite_output=True,
    ),
    "transient_time_bound": calculation_task(
        "transient_time_bound",
        "exposure_time_bound_s",
        transient_time_bound,
        allow_infinite_output=True,
    ),
    "steady_state_susceptible_bound": calculation_task(
        "steady_state_susceptible_bound",
        "susceptible_bound",
        steady_state_susceptible_bound,
        allow_infinite_output=True,
    ),
    "transient_susceptible_bound": calculation_task(
        "transient_susceptible_bound",
        "susceptible_bound",
        transient_susceptible_bound,
        allow_infinite_output=True,
    ),
    "quanta_concentration_from_reproductive_number": calculation_task(
        "quanta_concentration_from_reproductive_number",
        "quanta_concentration_exhaled_m3",
        quanta_concentration_from_reproductive_number,
    ),
    "quanta_concentration_from_aggregate_reproductive_number": calculation_task(
        "quanta_concentration_from_aggregate_reproductive_number",
        "quanta_concentration_exhaled_m3",
        quanta_concentration_from_aggregate_reproductive_number,
    ),
    "quanta_concentration_from_attack_probability": calculation_task(
        "quanta_concentration_from_attack_probability",
        "quanta_concentration_exhaled_m3",
        quanta_concentration_from_attack_probability,
    ),
    "quanta_concentration_from_slow_incubation_attack_probability": calculation_task(
        "quanta_concentration_from_slow_incubation_attack_probability",
        "quanta_concentration_exhaled_m3",
        quanta_concentration_from_slow_incubation_attack_probability,
    ),
    "quanta_concentration_from_population_transmission_rate": calculation_task(
        "quanta_concentration_from_population_transmission_rate",
        "quanta_concentration_exhaled_m3",
        quanta_concentration_from_population_transmission_rate,
    ),
    "quanta_emission_rate": calculation_task(
        "quanta_emission_rate",
        "quanta_emission_rate_s",
        quanta_emission_rate,
    ),
    "quanta_concentration_from_emission_rate": calculation_task(
        "quanta_concentration_from_emission_rate",
        "quanta_concentration_exhaled_m3",
        quanta_concentration_from_emission_rate,
    ),
    "rescaled_quanta_concentration": calculation_task(
        "rescaled_quanta_concentration",
        "quanta_concentration_exhaled_m3",
        rescaled_quanta_concentration,
    ),
    "jet_dilution_ratio": calculation_task(
        "jet_dilution_ratio",
        "jet_dilution_ratio",
        jet_dilution_ratio,
    ),
    "respiratory_jet_multiplier": calculation_task(
        "respiratory_jet_multiplier",
        "jet_multiplier",
        respiratory_jet_multiplier,
    ),
    "jet_corrected_reproductive_number": calculation_task(
        "jet_corrected_reproductive_number",
        "indoor_reproductive_number",
        jet_corrected_reproductive_number,
    ),
    "relative_risk_index": calculation_task(
        "relative_risk_index",
        "relative_risk_index",
        relative_risk_index,
    ),
    "secondary_case_count": calculation_task(
        "secondary_case_count",
        "secondary_case_count",
        secondary_case_count,
    ),
    "secondary_case_count_from_cumulative": calculation_task(
        "secondary_case_count_from_cumulative",
        "secondary_case_count",
        secondary_case_count_from_cumulative,
    ),
    "susceptible_count": calculation_task(
        "susceptible_count",
        "susceptible_count",
        susceptible_count,
    ),
    "secondary_attack_probability": calculation_task(
        "secondary_attack_probability",
        "attack_probability",
        secondary_attack_probability,
        max_output=1.0,
    ),
    "observed_indoor_reproductive_number": calculation_task(
        "observed_indoor_reproductive_number",
        "indoor_reproductive_number",
        observed_indoor_reproductive_number,
    ),
    "room_volume": calculation_task(
        "room_volume",
        "room_volume_m3",
        room_volume,
    ),
    "room_height": calculation_task(
        "room_height",
        "room_height_m",
        room_height,
    ),
    "per_hour_to_per_second": calculation_task(
        "per_hour_to_per_second",
        "rate_s",
        per_hour_to_per_second,
    ),
    "per_second_to_per_hour": calculation_task(
        "per_second_to_per_hour",
        "rate_h",
        per_second_to_per_hour,
    ),
    "per_day_to_per_second": calculation_task(
        "per_day_to_per_second",
        "rate_s",
        per_day_to_per_second,
    ),
    "per_second_to_per_day": calculation_task(
        "per_second_to_per_day",
        "rate_d",
        per_second_to_per_day,
    ),
    "hours_to_seconds": calculation_task(
        "hours_to_seconds",
        "seconds",
        hours_to_seconds,
    ),
    "days_to_seconds": calculation_task(
        "days_to_seconds",
        "seconds",
        days_to_seconds,
    ),
    "minutes_to_seconds": calculation_task(
        "minutes_to_seconds",
        "seconds",
        minutes_to_seconds,
    ),
    "cubic_meters_per_hour_to_cubic_meters_per_second": calculation_task(
        "cubic_meters_per_hour_to_cubic_meters_per_second",
        "flow_m3_s",
        cubic_meters_per_hour_to_cubic_meters_per_second,
    ),
    "cubic_meters_per_second_to_cubic_meters_per_hour": calculation_task(
        "cubic_meters_per_second_to_cubic_meters_per_hour",
        "flow_m3_h",
        cubic_meters_per_second_to_cubic_meters_per_hour,
    ),
    "liters_per_second_to_cubic_meters_per_second": calculation_task(
        "liters_per_second_to_cubic_meters_per_second",
        "flow_m3_s",
        liters_per_second_to_cubic_meters_per_second,
    ),
    "cubic_meters_per_second_to_liters_per_second": calculation_task(
        "cubic_meters_per_second_to_liters_per_second",
        "flow_l_s",
        cubic_meters_per_second_to_liters_per_second,
    ),
    "meters_per_hour_to_meters_per_second": calculation_task(
        "meters_per_hour_to_meters_per_second",
        "speed_m_s",
        meters_per_hour_to_meters_per_second,
    ),
    "meters_per_second_to_meters_per_hour": calculation_task(
        "meters_per_second_to_meters_per_hour",
        "speed_m_h",
        meters_per_second_to_meters_per_hour,
    ),
}


def run_named_calculation(
    name: str,
    inputs: Mapping[str, Any],
) -> CalculationReport:
    """Run a registered calculation once."""

    try:
        task = CALCULATION_TASKS[name]
    except KeyError as exc:
        raise KeyError(f"unknown calculation task: {name}") from exc
    return run_calculation(task, inputs)
