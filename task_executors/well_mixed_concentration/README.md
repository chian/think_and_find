# Well-Mixed Concentration

## Calculation

Solve the linear well-mixed mass-balance equation for one constant source and
one removal rate. This folder covers both the steady concentration and the
transient concentration curve used while a room approaches steady state.

## Inputs

- `emission_rate_quanta_s`: infectious quanta emitted into the room per second.
- `room_volume_m3`: well-mixed room volume.
- `removal_rate_s`: total removal rate from `effective_removal_rate`.
- `time_s`: elapsed exposure time for the transient function.
- `initial_concentration_quanta_m3`: initial concentration in the room.
- `breathing_flow_m3_s`: infected breathing flow for the dilution factor.

## Outputs

- `steady_state_concentration`: steady well-mixed concentration, in quanta/m3.
- `concentration_at_time`: transient well-mixed concentration, in quanta/m3.
- `dilution_factor`: ratio of well-mixed room concentration to unfiltered
  exhaled concentration.
