"""One conditional R(t, x), its time inverse, and its distance inverse."""

from dataclasses import dataclass
import math

from .._validation import non_negative, positive, probability
from .physics import integrated_concentration


MODEL_VERSION = "background-plus-local-v1"
EXPOSURE_OUTPUTS = {
    "allowable_room_time_s": ("solve_time", "steady"),
    "transient_allowable_room_time_s": ("solve_time", "transient"),
    "safe_distance_m": ("solve_distance", "steady"),
    "transient_safe_distance_m": ("solve_distance", "transient"),
    "scenario_reproductive_number": ("evaluate_R_in", "steady"),
    "transient_scenario_reproductive_number": ("evaluate_R_in", "transient"),
}
FIXED_FIELDS = {
    "solve_time": ("neighbor_distance_m",),
    "solve_distance": ("exposure_time_s",),
    "evaluate_R_in": ("exposure_time_s", "neighbor_distance_m"),
}
BACKGROUND_INPUTS = (
    "susceptible_count", "breathing_flow_m3_s", "mask_penetration",
    "relative_transmissibility", "emission_rate_quanta_s", "room_volume_m3", "removal_rate_s",
)
LOCAL_INPUTS = (
    "quanta_concentration_exhaled_m3", "plume_encounter_probability",
    "mouth_area_m2", "entrainment_coefficient",
)


def model_inputs(background, near_field):
    return (BACKGROUND_INPUTS
            + (("initial_concentration_quanta_m3",) if background == "transient" else ())
            + (LOCAL_INPUTS if near_field == "jet" else ()))


@dataclass(frozen=True)
class ExposureModel:
    background: str
    near_field: str
    risk_factor: float
    emission_rate: float
    volume: float
    removal_rate: float
    initial_concentration: float
    local_coefficient: float

    def background_risk(self, time_s):
        time_s = non_negative(time_s, "exposure_time_s")
        if self.risk_factor == 0 or time_s == 0:
            return 0.0
        if self.background == "steady":
            if self.emission_rate == 0:
                return 0.0
            return self.risk_factor * self.emission_rate / (self.volume * self.removal_rate) * time_s
        return self.risk_factor * integrated_concentration(
            time_s, self.emission_rate, self.volume, self.removal_rate, self.initial_concentration)

    def risk(self, time_s, distance_m):
        time_s = non_negative(time_s, "exposure_time_s")
        distance_m = positive(distance_m, "neighbor_distance_m")
        return self.background_risk(time_s) + self.local_coefficient * time_s / distance_m


def exposure_model(values, *, background, near_field):
    if background not in {"steady", "transient"} or near_field not in {"jet", "uniform"}:
        raise ValueError("Unknown exposure model")
    count = non_negative(values["susceptible_count"], "susceptible_count")
    breathing = positive(values["breathing_flow_m3_s"], "breathing_flow_m3_s")
    penetration = probability(values["mask_penetration"], "mask_penetration")
    susceptibility = non_negative(values["relative_transmissibility"], "relative_transmissibility")
    source = non_negative(values["emission_rate_quanta_s"], "emission_rate_quanta_s")
    volume = positive(values["room_volume_m3"], "room_volume_m3")
    removal = non_negative(values["removal_rate_s"], "removal_rate_s")
    if background == "steady" and source > 0 and removal == 0:
        raise ValueError("A continuing source with zero removal has no finite steady concentration")
    initial = (non_negative(values["initial_concentration_quanta_m3"], "initial_concentration_quanta_m3")
               if background == "transient" else 0.0)
    local = 0.0
    if near_field == "jet":
        concentration = non_negative(values["quanta_concentration_exhaled_m3"], "quanta_concentration_exhaled_m3")
        encounter = probability(values["plume_encounter_probability"], "plume_encounter_probability")
        area = positive(values["mouth_area_m2"], "mouth_area_m2")
        entrainment = positive(values["entrainment_coefficient"], "entrainment_coefficient")
        # Eq. 7's additive term after cancelling Ns and the background dilution.
        local = breathing * concentration * susceptibility * encounter * math.sqrt(area) / entrainment if count else 0.0
    return ExposureModel(background, near_field, count * breathing * penetration * susceptibility,
                         source, volume, removal, initial, local)


def evaluate_R_in(model, fixed_time, fixed_distance):
    return model.risk(fixed_time, fixed_distance)


def solve_time(model, fixed_distance, risk_threshold):
    """Invert precisely the forward model, retaining the supplied separation."""
    threshold = non_negative(risk_threshold, "risk_tolerance")
    unit_risk = model.risk(1.0, fixed_distance)
    if model.background == "steady":
        return threshold / unit_risk if unit_risk else math.inf
    if model.local_coefficient == 0:
        if model.risk_factor == 0 or (model.emission_rate == 0 and model.initial_concentration == 0):
            return math.inf
        if model.emission_rate == 0 and model.removal_rate > 0:
            eventual = model.risk_factor * model.initial_concentration / model.removal_rate
            if eventual <= threshold:
                return math.inf
    if threshold == 0:
        return 0.0
    high = 1.0
    for _ in range(1024):
        if model.risk(high, fixed_distance) >= threshold:
            break
        high *= 2
        if not math.isfinite(high):
            raise ValueError("Time bound exceeds representable range")
    else:
        raise ValueError("Could not bracket the time bound")
    low = 0.0
    for _ in range(160):
        mid = (low + high) / 2
        if model.risk(mid, fixed_distance) <= threshold:
            low = mid
        else:
            high = mid
    return low


def solve_distance(model, fixed_time, risk_threshold):
    """Invert the same R(t, x); zero/inf are explicit model boundary cases."""
    time_s = non_negative(fixed_time, "exposure_time_s")
    threshold = non_negative(risk_threshold, "risk_tolerance")
    background = model.background_risk(time_s)
    local = model.local_coefficient * time_s
    if background > threshold:
        return math.inf
    if local == 0:
        return 0.0
    if background == threshold:
        return math.inf
    return local / (threshold - background)


def calculate_exposure(values, *, operation, background, near_field):
    model = exposure_model(values, background=background, near_field=near_field)
    if operation == "solve_time":
        return solve_time(model, values["neighbor_distance_m"], values["risk_tolerance"])
    if operation == "solve_distance":
        return solve_distance(model, values["exposure_time_s"], values["risk_tolerance"])
    if operation == "evaluate_R_in":
        return evaluate_R_in(model, values["exposure_time_s"], values["neighbor_distance_m"])
    raise ValueError("Unknown exposure operation")
