# Respiratory Jet

## Calculation

Estimate the short-range plume correction from a local respiratory jet. The
calculation compares a nearest-neighbor respiratory plume with the diluted
well-mixed background and multiplies the long-range indoor reproductive number
by `1 + jet_multiplier`.

## Inputs

- `mouth_area_m2`: effective mouth area at the jet source.
- `entrainment_coefficient`: turbulent jet entrainment coefficient.
- `neighbor_distance_m`: distance from the emitting mouth to the susceptible
  neighbor.
- `plume_encounter_probability`: fraction or probability of the neighbor
  sitting in the plume.
- `well_mixed_dilution_factor`: room dilution factor from
  `well_mixed_concentration`.
- `susceptible_count`: people sharing the long-range airborne risk.
- `indoor_reproductive_number`: long-range expected transmissions before the
  short-range correction.

## Outputs

- `jet_dilution_ratio`: respiratory-jet concentration divided by exhaled
  concentration at the neighbor distance.
- `respiratory_jet_multiplier`: extra local-plume risk factor.
- `jet_corrected_reproductive_number`: corrected expected transmission count.

