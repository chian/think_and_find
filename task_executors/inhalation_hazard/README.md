# Inhalation Hazard

## Calculation

Convert airborne quanta concentration into a per-susceptible infection hazard.
The direct function consumes a room-air concentration; the steady function uses
the steady well-mixed simplification with exhaled quanta concentration
and a clean-air removal capacity.

## Inputs

- `airborne_quanta_concentration_m3`: infection quanta per cubic meter in room
  air.
- `quanta_concentration_exhaled_m3`: infection quanta per cubic meter of
  exhaled air.
- `breathing_flow_m3_s`: susceptible breathing flow; in the steady helper, the
  infector and susceptible breathing flows are currently assumed equal.
- `removal_capacity_m3_s`: clean-air removal capacity from
  `effective_removal_rate`.
- `inhalation_mask_penetration`: mask penetration on inhaled air.
- `mask_penetration`: shared exhale and inhale mask penetration for the steady
  helper.
- `relative_transmissibility`: disease, strain, or population multiplier.

## Outputs

- `inhalation_transmission_rate`: hazard in 1/s from an ambient concentration.
- `steady_state_inhalation_transmission_rate`: steady hazard in 1/s from
  exhaled infectiousness and room removal.

