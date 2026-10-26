"""Equation/reference checks, not independent empirical validation.

Scenario inputs: Bazant & Bush, PNAS 2021, Fig. 3 and adjacent discussion,
https://doi.org/10.1073/pnas.2018995118 . Air/water constants below are explicit
room-temperature approximations, not additional measurements from the paper.
"""

import math
import unittest

from episodic_workflow.task_executors.scenario_metrics.evaluate import evaluate_parameters


PHYSICAL = {
    "particle_density_kg_m3": 997, "air_density_kg_m3": 1.204,
    "air_dynamic_viscosity_pa_s": 1.825e-5, "gravity_m_s2": 9.80665,
}
CLASSROOM = {
    **PHYSICAL, "floor_area_m2": 83.6, "room_volume_m3": 301,
    "susceptible_count": 19, "occupant_count": 20,
    "breathing_flow_m3_s": .5/3600, "quanta_concentration_exhaled_m3": 30,
    "relative_transmissibility": .25, "risk_tolerance": .1,
    "initial_concentration_quanta_m3": 0, "exposure_time_s": 3600,
    "recirculation_flow_m3_s": 0, "filter_efficiency": 0,
    "deactivation_rate_s": .3/3600, "effective_radius_m": .5e-6,
}
CONDITIONS = {name: "Explicit reference-reproduction assumption" for name in (
    "well_mixed", "single_infectious_source", "uniform_breathing_and_masks", "stokes_particles")}


class ReferenceChecks(unittest.TestCase):
    def classroom(self, ach, pm):
        return evaluate_parameters({**CLASSROOM, "air_exchange_rate_s": ach/3600, "mask_penetration": pm},
            CONDITIONS, ("exposure_time_bound_s", "transient_exposure_time_bound_s", "transient_interpolation_time_bound_s"),
            subject="reference classroom", label="Fig. 3 reference parameterization")["metrics"]

    def test_rounded_classroom_reference_times_and_interpolation_difference(self):
        # The low-ACH, short-time interpolation is NOT exact integrated hazard.
        for ach, mask, reported_hours in ((.34, 1, 1.2), (8, 1, 7.2), (.34, .3, 8), (8, .3, 80)):
            with self.subTest(ach=ach, mask=mask):
                rows = self.classroom(ach, mask)
                interpolated = rows["transient_interpolation_time_bound_s"]["value"]/3600
                self.assertLess(abs(interpolated/reported_hours-1), .07)
        rows = self.classroom(.34, 1)
        exact = rows["transient_exposure_time_bound_s"]["value"]/3600
        self.assertAlmostEqual(exact, 1.5192148612124925)
        self.assertGreater(exact/ (rows["transient_interpolation_time_bound_s"]["value"]/3600), 1.2)

    def test_nursing_home_mechanical_steady_bound(self):
        params = {**CLASSROOM, "floor_area_m2": 22.3, "room_volume_m3": 53.5,
                  "susceptible_count": 2, "occupant_count": 3, "relative_transmissibility": 1,
                  "risk_tolerance": .01, "mask_penetration": 1, "air_exchange_rate_s": 8/3600}
        result = evaluate_parameters(params, CONDITIONS, ("exposure_time_bound_s",), subject="nursing room", label="Reference")
        minutes = result["metrics"]["exposure_time_bound_s"]["value"]/60
        self.assertLess(abs(minutes/18-1), .04)

    def test_critical_radius_and_evaporation_examples(self):
        for ach, expected_um in ((.35, 1.3), (6, 5.5)):
            result = evaluate_parameters({**PHYSICAL, "air_exchange_rate_s": ach/3600, "room_height_m": 2.1},
                CONDITIONS, ("critical_aerosol_radius_m",), subject="reference", label="Critical size")
            radius = result["metrics"]["critical_aerosol_radius_m"]["value"] * 1e6
            self.assertLess(abs(radius/expected_um-1), .03)
        for radius, expected_s in ((.5e-6, .0012), (50e-6, 12)):
            result = evaluate_parameters({"initial_radius_m": radius, "relative_humidity": .5,
                "evaporation_coefficient_m2_s": 4.2e-10}, {"equilibrium_evaporation": "25 degrees C, 50 percent RH"},
                ("droplet_evaporation_time_s",), subject="reference", label="Evaporation")
            self.assertLess(abs(result["metrics"]["droplet_evaporation_time_s"]["value"]/expected_s-1), .01)

    def test_mask_squared_and_ten_virion_characteristic_dose(self):
        bare, masked = self.classroom(8, 1), self.classroom(8, .3)
        self.assertAlmostEqual(masked["exposure_time_bound_s"]["value"]/bare["exposure_time_bound_s"]["value"], 1/.3**2)
        result = evaluate_parameters({"microscopic_infectivity_per_virion": .1},
            {"uniform_virion_infectivity": "Illustrative infectivity coefficient; not RNA copies"},
            ("infectious_dose_virions",), subject="reference", label="Characteristic dose")
        self.assertEqual(result["metrics"]["infectious_dose_virions"]["value"], 10)
        self.assertAlmostEqual(1-math.exp(-.1*10), .6321205588)
