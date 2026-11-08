# Executable causal scoring sheets

The application investigates candidate causal inferences G, their expected
generalizability bounds B_G, theories that depend on them, and useful changes
when those expectations break. Models supply scientific formalizations and
source extractions. Calculators derive numerical quantities from explicit
operands. A model-issued valid/invalid verdict never supplies or withdraws credit.

## Existing mechanisms retained

`CausalSheet` retains versioned records, exact historical snapshots, optimistic
merges and dependency invalidation. Expectations, failures and revisions remain
separate records. Later revisions cannot erase an earlier failure. The existing
exact equation, deterministic acyclic SCM and discrete-simulation evaluators are
retained. The support ledger retains observation deduplication, original-G
identity and post-break revaluation of an existing change.

Dependencies on an append-only theory reconstruction resolve the exact recorded
fragment snapshot. Adding components advances the reconstruction version without
invalidating earlier assessments or withdrawing their support/change credit.
The new material remains available for further scientific comparisons. Update
admission still rejects stale writes; version advancement does not establish a
scientific failure or change to G.

Calculation receipts now reproduce their declared computation on admission.
Digest or shape validation alone does not establish that a result was calculated.
Support/change records require their declared executable operands.

## Dependence and importance

An immutable `importance` protocol declares its application/G realization,
question, baseline model, focal alteration, answer expression, circumstances,
positive weights and typed-answer comparator. `evaluate` executes selected
predeclared circumstances. Results retain exact rational or Boolean answers.

Importance is changed circumstance mass divided by total declared circumstance
mass. Unknown and failed cases give explicit lower/upper bounds, not invented
zero effects. A fully evaluated cancelling dependency gives measured zero.
Finite enumeration and sampled frequency remain different quantities; neither
establishes continuous-space coverage. The evaluator supports explicit acyclic
static systems and finite deterministic synchronous dynamics, not arbitrary
implicit solvers or external proof assistants.

`dependence_result` adds exact G/application/protocol bindings and changed-case
witnesses to the existing importance calculation. This is a computed consequence
of the declared formal realization, not certification of its empirical adequacy.

## Expected generalizability bounds

`expectation` stores the expected bounds, basis, grounding and rationale.
`expectation_break` preserves the failed expectation and triggering snapshots.
A revision records its prior version, causes, proposed rule, retained/lost
applications and new predictions. Literature, reasoning across literature, or
explicitly provisional judgment can ground proposed bounds. No confidence
number is assigned to that judgment.

An `expectation_test` binds one exact expectation to an importance protocol:

- `scope` computes membership from the declared circumstance inputs.
- `target` specifies the expected exact answer or reported interval.
- `claim_relation` formalizes G as a Boolean predicate over paired answers and
  the circumstance inputs.
- `claim_intervention` gives the shared input, node or edge operation those
  paired answers must realize. A grader executes that operation and compares
  it with the stored altered answer.

Empty or unrepresentable relations/interventions remain unresolved. One exact
B_G/protocol cannot be relabeled under another key to change its scope, target
or original-G predicate. Revising the scientific expectation requires a recorded
revision. The test reports membership, expected/broken/unresolved mass and
per-case relation results. It does not grade how confidently the bounds were proposed.

## Support coupled to B_G

`expectation_support` names exact G and B_G snapshots. A `support_assessment`
links that claim, an application, an importance protocol and an expectation test.
All must bind the same G version, bounds and formal comparison.

A supporting hypothesis requires an executed in-scope witness satisfying the
original G relation and expected behavior, whose answer changes under the
specified alteration. A written dependency link alone does not establish this.
A selected observation requires a source identity, quantity, units, numeric
conditions and reported value/uncertainty, plus its declared circumstance ID.
The host compares the calculated prediction with those actual operands.
Free-text conditions or unsupported uncertainty remain unresolved. Unreported
uncertainty permits only explicitly qualified literal reported-value agreement;
it is not converted into a known error interval.

Aligned observations can support a dependent application even when the altered
prediction also aligns. Discrimination is reported separately. A hypothesis is
not multiplied by contexts or coefficients under the same canonical theory.
Formal aliases additionally share an identity. This grouping is an explicit
operational identity convention, not a solution to all scientific semantic equivalence.
Source-established observation identities prevent copied citations, extraction
renaming and metadata fields from becoming new facts.

`support_contribution.reasoning_key` references the exact key of its public
`reasoning` record. Create that record earlier in the same batch or select an
accessible admitted record; its `steps` and `conclusion` contain the argument
text. `expectation_support.argument` contains prose. Missing, inaccessible or
wrong-kind reasoning references return a field-specific repair diagnostic through
the existing episode feedback.

`support_contribution.checks` selects these executed assessments. The ledger
reports hypothesis and observation components separately and per exact B_G and G.
The current exposed aggregation convention is one unit per admitted hypothesis
group plus one per admitted observation. Equal relative weights are an
implementation convention, not a user-specified scientific weighting law.

## Repairs and useful changes

`repair_assessment` compares a complete candidate model and declared alternatives
against the original expectation test. It also binds the revised expectation's
test. Original circumstances, weights, answer meaning and expected targets remain
fixed. Shrinking the new scope reports excluded failures rather than repairing them.
Added equations, dependencies and branches are reported as structural changes;
those counts do not classify a repair as trivial or magical.

The same original-G predicate and intervention are executed on source, candidate
and alternative models. A missing dependency that improves predictions while
preserving G is not a demonstrated change to G. Changing equation text or a G
version number alone does not establish a reversal.

Per-case outputs distinguish restored failures, retained/lost successes,
remaining failures, excluded failures and unresolved work. A restored case
qualifies for scoped change credit only when original G holds, the candidate
changes that relation, and the compared G-preserving alternatives do not restore
that case. Empty or unfinished comparisons leave necessity unresolved. This is
necessity within declared alternatives, not exclusion of every conceivable repair.
Construction and declared test cases are separate. Declaring a test partition
does not establish empirical independence or novel corroboration.

`causal_change.assessment` links the executed comparison to the same recorded
break. The exposed scaling convention multiplies support for original G by the
qualifying restored mass divided by original expected-scope mass. Raw importance
and restored-target fraction remain distinct reported quantities. Later support
for original G revalues the same change even after G and B_G have been revised.
A subsequently demonstrated preserving repair can retract that scoped change
contribution while leaving independent support for original G intact.

## Pragmatic transfer failures and current scoring limits

The current scientific target is a demonstrated failure of a specified present
representation to transfer usefully, together with evidence explaining why.
Universal irreducibility and fundamental contradiction are not required. Tasks
may differ in question, observable, organism, scale or grain. Establish their
connection through G and preserve each task's native meanings and success criteria.
A numeric comparison within one task still requires a justified mapping and
fixed operands. The repair adapter's common-case restrictions protect its test;
they are not universal admission criteria for scientific candidates.

The existing records can describe this broader target without a new workflow:

- `application.context` identifies the task and conditions; `inference_use`
  describes how it uses, attempts to carry over, omits or transforms G.
- `application.direction=possible_transfer_failure` retains a candidate without
  asserting an opposite-sign effect. This label does not grant credit or establish
  a scientific conclusion.
- `comparison.mapping` connects the tasks; `conflict` records the attempted
  transfer and its consequences; `repair_search_bounds` scopes examined alternatives.
- `expectation_break` preserves the expected transfer and triggering evidence.
  Public `reasoning` records distinguish demonstrated causes from proposed
  explanations and unresolved alternatives.
- `repair` records construction versus test evidence and the costs of a concrete
  adjustment. Expected-bounds revisions retain original successes and losses.

The executable `repair_assessment` is a restricted adapter, not a complete
decision procedure for pragmatic failure. It currently has three limitations:

1. Change credit requires the original executable G predicate to become false
   while a failed prediction is restored. Useful omission or aggregation can
   escape that predicate even when its benefit is demonstrated.
2. Any fitting G-preserving alternative defeats its scoped necessity test.
   Information requirements, fitting assumptions and costs on other tasks do
   not enter that decision. `repair.costs` remains a public scientific argument,
   not an executable usefulness measurement.
3. An explained failure without a successful replacement earns no change credit.
   Its absence of credit must not be reported as absence of a scientific result.

The five score components and continuation rules are unchanged. This is a real
incentive limitation: broader prompt guidance does not make the scorer reward
every newly relevant finding. Receipt qualifications and prompts expose that
limit; they do not fix it by treating a model's usefulness verdict as an operand.

### Proposed numerical extension, not implemented

Retain the existing G identities, expected generalizability bounds, evidence,
versioned failures and repair history. Extend executable comparison to separate
task-specific evaluations. Each task would declare its intended prediction or
explanation, available inputs, and a justified success requirement before
comparing representations. Where measurable, use native prediction errors or
reported intervals, information requirements with an actual availability or
resource constraint, and retained or lost successes. Do not equate a coefficient
count with information cost or invent an arbitrary usefulness threshold.

Evaluate carrying G unchanged, concrete proposed repairs and a revised or reduced
representation on the tasks for which each is claimed to work. Each task keeps
its own observable, denominator and justified mappings; a more accurate result
on one task cannot silently erase a failure on another. Keep construction fit,
independent predictive performance, untested alternatives and unmeasured costs
separate. A demonstrated current failure and the warrant for its explanation
must be distinguishable from later success in replacing the representation.

The resulting claim is limited to the examined transfers or repairs under those
requirements. A successful useful repair can defeat that scoped failure claim;
a fit that violates a demonstrated task requirement need not do so. Unknown
costs or unsupported diagnoses remain unknown. The criteria for crediting these
results, including non-numeric explanations, still require design and testing;
no new scalar score or prose validity gate has been introduced.

## Episode and message integration

Scientific parent and Reasoning can record these operands and select existing
calculators for importance, expected generalizability bounds, support and repair
comparisons. Executing a calculation earns 0 points. Its findings enter the
support or causal-change score through the corresponding scientific assessment.
Literature roles project scientific contributions through evidence provenance.
Character counts, passages, rows, source counts and model opinions earn no points.
For a fixed observation need the parent can commission an existing
support_assessment template with literature inputs. Readers match source-native
quantity, units and numeric circumstances and execute that fixed comparison,
so actual aligned support can contribute before the literature episode ends.
No source is scored merely because it was read. Broad exploratory findings
without such a formalization await downstream assessment. Closed episode controllers are not reopened retroactively; the global ledger
still revalues support and changes as later scientific operands arrive.

Model handoffs contain bounded summaries and exact record references. Full
per-case results stay with the calculators. `calculate` with `calculation_offset`
and `calculation_limit` retrieves detail pages. Source measurement payloads are
referenced instead of copied recursively into parent reports.
