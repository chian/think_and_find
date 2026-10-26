# Quanta Concentration

## Calculation

Solve the steady indoor-risk equation backward to infer the exhaled infection
quanta concentration, `Cq`, from a summarized outbreak. Helpers start from an
indoor expected-transmission count with a separate susceptible count, an
already aggregated indoor expected-transmission count, a per-susceptible attack
probability, a slow-incubation attack model, or a fitted population-scale
transmission rate.

This package also contains two simpler `Cq` transforms: converting between
exhaled quanta concentration and quanta emission rate, and rescaling a baseline
`Cq` by the ratio of infectious aerosol volume fractions for two respiratory
activities.

## Inputs

- `indoor_reproductive_number`: expected transmissions in the calibration
  event, either before or after susceptible-count aggregation.
- `attack_probability`: observed or inferred per-susceptible probability of
  infection.
- `population_transmission_rate_s`: a fitted `N * beta` transmission rate,
  in `1/s`.
- `removal_capacity_m3_s`: clean-air removal capacity for the event room.
- `susceptible_count`: number of susceptible exposed people when using the
  reproductive-number form.
- `total_occupant_count`: total exposed population used to normalize a fitted
  population-scale rate.
- `initial_infected_count`: infected occupants present at the beginning of a
  slow-incubation fit.
- `exposure_time_s`: event duration.
- `breathing_flow_m3_s`: breathing flow for the infector and susceptible,
  currently assumed equal.
- `mask_penetration`: shared exhale and inhale mask penetration.
- `relative_transmissibility`: strain or population multiplier to divide out
  of the inferred `Cq`.
- `quanta_concentration_exhaled_m3`: `Cq`, in quanta/m3.
- `quanta_emission_rate_s`: exhaled infection quanta per second.
- `activity_aerosol_volume_fraction`: integrated aerosol volume fraction for
  the target respiratory activity.
- `baseline_aerosol_volume_fraction`: integrated aerosol volume fraction for
  the activity used to anchor the baseline `Cq`.

## Outputs

- `quanta_concentration_from_reproductive_number`: `Cq`, in quanta/m3.
- `quanta_concentration_from_aggregate_reproductive_number`: `Cq`, in
  quanta/m3.
- `quanta_concentration_from_attack_probability`: `Cq`, in quanta/m3.
- `quanta_concentration_from_slow_incubation_attack_probability`: `Cq`, in
  quanta/m3.
- `quanta_concentration_from_population_transmission_rate`: `Cq`, in
  quanta/m3.
- `quanta_emission_rate`: exhaled infection quanta per second.
- `quanta_concentration_from_emission_rate`: `Cq`, in quanta/m3.
- `rescaled_quanta_concentration`: activity-rescaled `Cq`, in quanta/m3.

