# Relative Risk Index

## Calculation

Compute a coarse relative behavioral risk index for comparing events. This
strips out an explicit policy tolerance and susceptibility multiplier and uses
the ventilation-only room denominator.

## Inputs

- `occupant_count`: number of people represented by the event score.
- `exposure_time_s`: event duration.
- `quanta_concentration_exhaled_m3`: exhaled infection quanta concentration.
- `breathing_flow_m3_s`: breathing flow for the infector and susceptible,
  currently assumed equal.
- `mask_penetration`: shared exhale and inhale mask penetration.
- `air_exchange_rate_s`: outdoor air exchange rate.
- `room_volume_m3`: room volume.

## Outputs

- `relative_risk_index`: dimensionless event-risk index.

