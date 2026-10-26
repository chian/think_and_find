# Effective Removal Rate

## Calculation

Compute the room processes that remove suspended infectious material:
ventilation, filtered recirculation, gravitational settling, and biological
deactivation. The folder also contains the clean-air capacity term used by the
steady cumulative exposure bound.

## Inputs

- `outdoor_air_flow_m3_s`: outdoor ventilation flow into the room.
- `recirculation_flow_m3_s`: recirculating filtered flow.
- `filter_efficiency`: radius-specific probability that the filter removes a
  particle from the recirculating flow.
- `room_volume_m3`: well-mixed room volume.
- `radius_m`: particle radius for Stokes settling.
- `particle_density_kg_m3`, `air_density_kg_m3`, `air_dynamic_viscosity_pa_s`:
  Stokes settling parameters.
- `room_height_m`: room height for converting settling speed into a loss rate.
- `deactivation_rate_s`: pathogen deactivation rate in aerosols.
- `floor_area_m2`: settling area for the removal-capacity term.
- `total_removal_rate_s`: reported combined pathogen loss rate, in 1/s.

## Outputs

- `air_exchange_rate`: outdoor-air loss rate, in 1/s.
- `filtration_loss_rate`: filtered-recirculation loss rate, in 1/s.
- `stokes_settling_speed`: particle settling speed, in m/s.
- `sedimentation_rate`: gravitational settling loss rate, in 1/s.
- `air_purification_rate`: ventilation, filtration, and deactivation loss rate,
  excluding settling.
- `total_removal_rate`: full size-specific removal rate, including settling.
- `removal_capacity`: clean-air capacity, in m3/s.
- `removal_capacity_from_total_loss_rate`: volume times an already combined
  loss rate, in m3/s; component losses must not be added again.
