# Cumulative Exposure Bound

## Calculation

Invert airborne-transmission equations into maximum occupancy-time bounds.

The steady bound returns the maximum value for `(N - 1) * tau`. The
conservative bound uses ventilation only and returns a looser form for
`N * tau`. The transient helpers apply a finite-relaxation correction to the
same steady bound so a parent can ask for the maximum exposure time for a fixed
susceptible count, the maximum susceptible count for a fixed exposure time, or
the transiently corrected cumulative person-time bound itself.

## Inputs

- `risk_tolerance`: acceptable expected transmissions for the scenario.
- `removal_capacity_m3_s`: clean-air removal capacity from
  `effective_removal_rate`.
- `air_exchange_rate_s`: outdoor air exchange rate for the conservative bound.
- `room_volume_m3`: room volume for the conservative bound.
- `breathing_flow_m3_s`: breathing flow for the infector and susceptible,
  currently assumed equal.
- `quanta_concentration_exhaled_m3`: exhaled infection quanta concentration.
- `mask_penetration`: mask penetration for exhale and inhale, currently
  assumed equal.
- `relative_transmissibility`: disease, strain, or population multiplier.
- `steady_state_bound_person_s`: the steady person-time bound.
- `cumulative_exposure_bound_person_s`: a steady or transient person-time
  bound.
- `concentration_relaxation_rate_s`: total aerosol concentration relaxation
  rate, including ventilation, filtration, sedimentation, and deactivation.
- `exposure_time_s`: candidate exposure duration.
- `susceptible_count`: susceptible occupants sharing the room with an infector.

## Outputs

- `steady_state_cumulative_exposure_bound`: person-seconds for
  `(N - 1) * tau`.
- `conservative_cumulative_exposure_bound`: person-seconds for `N * tau`.
- `transient_cumulative_exposure_bound`: transiently corrected
  person-seconds for `Ns * tau`.
- `steady_state_time_bound`: maximum exposure time in seconds.
- `transient_time_bound`: transient maximum exposure time in seconds.
- `steady_state_susceptible_bound`: maximum susceptible occupants for an
  exposure time.
- `transient_susceptible_bound`: transient maximum susceptible occupants for an
  exposure time.

