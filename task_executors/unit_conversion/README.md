# Unit Conversion

## Calculation

Normalize literature values into the SI units expected by the deterministic
airborne-transmission kernels. Sources often report rates per hour and
durations in hours or days, while the one-shot functions in this tree use
seconds at their boundaries.

## Inputs

- `rate_h`: a rate in `1/h`.
- `rate_d`: a rate in `1/day`.
- `rate_s`: a rate in `1/s`.
- `hours`: a duration in hours.
- `minutes`: a duration in minutes.
- `days`: a duration in days.
- `flow_m3_h`: volumetric flow in `m3/h`.
- `flow_m3_s`: volumetric flow in `m3/s`.
- `flow_l_s`: volumetric flow in `L/s`.
- `speed_m_h`: speed in `m/h`.
- `speed_m_s`: speed in `m/s`.

## Outputs

- `per_hour_to_per_second`: a rate in `1/s`.
- `per_second_to_per_hour`: a rate in `1/h`.
- `per_day_to_per_second`: a rate in `1/s`.
- `per_second_to_per_day`: a rate in `1/day`.
- `hours_to_seconds`: a duration in seconds.
- `minutes_to_seconds`: a duration in seconds.
- `days_to_seconds`: a duration in seconds.
- `cubic_meters_per_hour_to_cubic_meters_per_second`: a flow in `m3/s`.
- `cubic_meters_per_second_to_cubic_meters_per_hour`: a flow in `m3/h`.
- `liters_per_second_to_cubic_meters_per_second`: a flow in `m3/s`.
- `cubic_meters_per_second_to_liters_per_second`: a flow in `L/s`.
- `meters_per_hour_to_meters_per_second`: a speed in `m/s`.
- `meters_per_second_to_meters_per_hour`: a speed in `m/h`.
