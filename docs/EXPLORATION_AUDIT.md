# Auditing exploration credit from a discovery run

The credit rule rewards changes in coherent argument materials, not searches,
citations, or step count by themselves. Its case coordinate is bounded local
derivation progress plus a larger complete-witness component. The existing
attainment archive counts a material's dominated volume once per conceptual
theme. `learning_feedback_allocated` then assigns the resulting method gain to
the reasoning cycle and distinct cycles that supplied newly used evidence.
Those shares train the action forecast; they do not add method credit.

## What the first long attempt actually showed

The 2026-10-26 v11.5 Argo attempt completed four cycles before prompt overflow
ended the Episode after about 25 minutes. It acquired 102 source claims from
four sources, proposed six cases across four themes, and reviewed three. The
root chose general discovery in every cycle. Method gains were `0.39344`,
`0.01788`, `0.00425`, and `0`. The first cycle's two credited materials
contributed `0.25815` and `0.13530`; later gains were much smaller. Every
positive learning share went to its own cycle. There was no observed delayed
share to an earlier evidence-gathering cycle and no root choice to refine an
existing inquiry.

The large first gain lifted general discovery's next-action score to `0.1133`
for cycle two, versus `0.0205` for the strongest untried case. At cycle four,
the scores were `0.0703` and `0.0227`. This is evidence that credit influenced
action choice, but it does not establish that the influence was desirable.
The early run did not last long enough to show whether declining general
discovery gains would eventually redirect work toward an existing inference
gap. Repeated prompt-bound failures also interfered with theory measurement.

## How to assess the day-long run

Follow each consequential argument change through these logged records:

1. `source_retrieved`, `reading_completed`, and source claim IDs show what was
   acquired and when. Count distinct sources and conceptual themes; repeated
   queries or more claims from one source are not in themselves exploration.
2. Case review materials show the ordered premise path, each step's warrant,
   the attempted inference translation, and the fixed-category rival. Check
   whether a new step changes a conclusion or exposes a real bridge gap.
3. `goal_committed` and `discovery_cycle_completed` show actual method gain.
   Compare gains with argument progress and material identity, not citation
   count. Repeated assessments of the same material should yield no new gain.
4. `learning_feedback_allocated` shows which earlier evidence-origin cycles
   received retrospective shares. A later, warranted revision using earlier
   evidence should produce a delayed share without duplicating method credit.
5. `acquisition_scheduled` records every action's expected yield, exploration
   bonus, and final score. Compare the chosen action with the best alternative
   and inspect whether the next search or refinement addresses the measured
   bottleneck. There is no required discover/refine sequence.
6. Track `operation_failed`, message sizes, and completion reason. A run that
   ends early or repeatedly cannot measure long-horizon behavior.

A good result would show later source work improving a specific multi-step
argument, the improvement producing one material gain, retrospective feedback
for genuinely useful earlier exploration, and a subsequent question or action
that changes accordingly. A high first-cycle reward followed only by related
general searches and no refinement would be a warning, even if the total
credit rises. Scientific success requires a warranted reversal that survives
the fixed-category rival; a near miss or high epistemic quality alone is not
that result.
