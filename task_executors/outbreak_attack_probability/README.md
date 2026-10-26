# Outbreak Attack Probability

## Calculation

Normalize an admitted outbreak record into the observed quantities used by Cq
calibration: secondary cases, susceptible count, attack probability, and an
observed indoor reproductive number.

## Inputs

- `confirmed_secondary_case_count`: secondary cases confirmed by the event's
  accepted case definition.
- `probable_secondary_case_count`: optional clinically probable secondary
  cases.
- `initial_infected_count`: infected source cases at the start of an event.
- `final_infected_count`: cumulative infected people at the end of an event.
- `attendee_count`: exposed people present at the calibration event.
- `index_case_count`: infected source cases already present at the event.
- `secondary_case_count`: total secondary infections credited for the event.
- `susceptible_count`: exposed non-index people.

## Outputs

- `secondary_case_count`: confirmed plus probable secondary cases, or final
  infected cases minus initial infected cases.
- `susceptible_count`: attendees minus index cases.
- `secondary_attack_probability`: observed secondary infections per
  susceptible person.
- `observed_indoor_reproductive_number`: observed secondary infections per
  index case.

## Scoring and failure modes

The searched `build_attack_probability` episode asks the LLM for one of four
supported source shapes:

- `attendee_count`, `index_case_count`, and `secondary_case_count`
- `attendee_count`, `index_case_count`, and a reported `attack_probability`
- `susceptible_count`, `index_case_count`, and `secondary_case_count`
- `susceptible_count`, `index_case_count`, and a reported `attack_probability`

Direct `susceptible_count` values are accepted only when the source reports the
exposed non-index people. Count values must remain whole raw counts after
normalization. Reported attack probabilities may be raw fractions or percents,
but the normalized fraction must match the raw value. Unsupported evidence,
unsupported units, fractional counts, invalid probabilities, and inconsistent
normalized values fail closed before the deterministic susceptible-count and
secondary-attack leaves run.
