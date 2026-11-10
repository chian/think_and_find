# Episode contribution and numerical continuation

A numerical projection measures what actual work contributes under the episode's
assigned scope and result contract. Define the meaning and operands of each
component before using it to steer continuation. Scoring definitions, functions
that populate them and the episode's substantive instructions have separate
responsibilities.

The episode chooses work from its available alternatives. The configured local
yield estimator/controller observes completed own units and governs numerical
continuation. Its observations do not prescribe which action to choose.

Root alone uses signed investigation-outcome value-of-computation
forecasts. The reusable statistical function is `value_of_computation.py`; the
application binding lives in `case_reports/research/computation_yield.py`. All
other episodes, including Discovery and Scientific, retain `joint_preferential_discovery_v1`. See
[VOC measurement, continuation and limitations](../docs/VALUE_OF_COMPUTATION.md).

Record outcomes before projecting credit. One completed child supplies one
parent-level observation through its return. Internal child calls do not create
extra parent turns, and child scores are not summed into parent progress.
Repeated facts, report construction and shared-state refreshes do not manufacture
new contributions.

A returned failed unit is still an observed attempt with zero realized yield.
`ResearchController.observe` sends that empty numerical sample to the estimator
while retaining the result's `evaluated=False`, failure count and diagnostics.
Scientific validity and numerical observation availability are distinct. Do not
map a completed failed operation to a missing incidence observation: a history
containing only such failures would never acquire enough observed samples to
yield. Child failures use the same return and parent credit path as other child
outcomes; internal retries do not become extra parent observations. Trace events
separate `evaluated` from `yield_observed` and record the controller's reason.

Keep investigation progress distinct from support for a particular conclusion.
Useful negative findings and corrections can advance the investigation while
weakening that conclusion. Numerical yielding does not establish scientific
completion, correctness or exhaustive search.

This directory contains inherited numerical utilities, including `research.py`
and `attainment.py`. Their availability does not select a causal-research scoring
policy. The removed binding's weights, gates, scheduling and attribution rules
are not current instructions. Resolve the scientific scoring requirements from
the user's original messages and inspect the actual episode trace with them.

`causal_sheet.py` compiles importance from changed circumstance mass divided by
declared total mass, retaining unresolved results. Calculation completion earns
0 points; outcomes inform support and causal-change assessments. Ordinal grades
and their numerical anchors are removed.
The operands and current limitations are documented in
[`CAUSAL_SCORING_SHEET.md`](../docs/CAUSAL_SCORING_SHEET.md). These are sheet
calculations for the existing Episodes, not a new binding or controller.

`causal_support.py` supplies identified support units and graduated task-comparison
credit linked to exact G and original expected-generalizability-bounds versions.
Support keeps hypothesis and observation counts visible; it is not certainty.
It stays separate from applicability differences and task-error improvements,
including useful extensions outside original bounds.
Scientific definitions/arguments remain separate from host arithmetic.
