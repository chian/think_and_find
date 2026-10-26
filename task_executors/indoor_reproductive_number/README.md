# Indoor Reproductive Number

## Calculation

Integrate a constant per-susceptible airborne hazard over an exposure duration
and susceptible count. The result is the expected number of airborne
transmissions attributable to one infected person in the room.

## Inputs

- `susceptible_count`: people who can be infected by the index case.
- `transmission_rate_s`: per-susceptible hazard from `inhalation_hazard`.
- `exposure_time_s`: exposure duration.
- `population_reproductive_number`: population-scale reproductive number to
  attribute to indoor transmission.
- `indoor_attribution_fraction`: fraction of a population-scale reproductive
  number assigned to indoor airborne transmission.

## Outputs

- `indoor_reproductive_number`: expected transmission count.
- `first_transmission_probability`: Poisson probability of at least one
  transmission for a given expected count.

