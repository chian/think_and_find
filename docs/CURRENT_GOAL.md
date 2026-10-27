# Generic research task and scope

## Objective

Adapt the copied episodic workflow to investigate a user-defined question through
both evidence acquisition and reasoning. The system should maintain a developing
answer, identify unresolved inferences, acquire information that bears on them,
and revise its conclusions when the evidence warrants it.

The deliverable is a reasoned answer with traceable evidence and explicit limits.
A source inventory or a populated table may support that answer, but does not
establish that the requested reasoning has been completed.

## Present boundary

The migration preserved the existing executable implementation and rewrote its
design documentation. Its domain CLI path, records, calculation catalog and
credit policy remain available. A subsequent task-specific reasoning prototype
adds the causal-reversal path and its separate credit policy. Generic reasoning
operations and task-neutral records remain design work.
The copied repository is now published separately. Generic reasoning records
remain future work.

## Task contract

A generic task should declare its question, requested outputs, relevant units of
analysis, supplied premises, and criteria for what constitutes a useful result.
Where the user requests a collection, its target count and diversity dimensions
belong to this contract. Do not turn one task's criteria into universal defaults.

Unknown distinctions can be investigated and revised. A task contract cannot
assume that every scientifically relevant context can be listed in advance.
Changes to categories, comparisons or scope need an explicit recorded reason;
they must not silently weaken the user's requested result.

## Evidence and inference

Preserve these distinctions in state and reporting:

- Retrieved source text and exact quotations, with source/version/span identity.
- Source-reported claims and the context in which the source makes them.
- User-supplied premises and definitions.
- Inferences that combine or interpret evidence.
- Hypotheses, contrary evidence and unresolved questions.
- Conclusions and the task criteria they satisfy, fail, or leave unresolved.

An inference need not appear verbatim in a source. Its premises must remain
traceable, and its status must not be relabeled as an extracted observation.
Exact quotation checks establish faithful transcription, not the validity of an
argument. A numerical calculation establishes a consequence of its inputs, not
the truth of its assumptions.

## Reasoning during acquisition

Reasoning should determine which uncertainty matters next and which evidence
could discriminate between competing interpretations. That information need
must reach the strategy, search, source-selection and reading operations.
Returned evidence and failed attempts should update the parent investigation.

For a comparative task, finding two claims is only the beginning. The investigation
may need to examine whether their actors, interventions, outcomes, time horizons
and categories are comparable. The comparison can produce a new question rather
than a binary verdict. These dimensions are an example of a task-specific
reasoning contract, not an exhaustive definition of sufficient context.

## Runtime and progress

The operator's current requirement is **no scientific gates**. Represent warrant,
comparability, uncertainty and task fit through continuous measurements. Continue
according to numerically measured improvement; apply collection filters only
afterward. A duplicate conclusion, a fresh evaluator rating or restoration of
previously credited work earns no additional credit. Current scientific answers
and historical credit memory remain distinct.

Compose the existing Episode runtime and acquisition bindings. Retain the
separation between semantic work and numerical continuation. A model may propose
queries, interpretations and structured judgments; continuation and credit use
declared numerical rules over recorded results.

The inherited calculation weights are not a generic reasoning policy. Define
future progress from the declared task and its actual results, with explicit
rules for duplication, revision and unresolved work. Do not award completion
merely for calls, sources, filled fields or a repeated judgment. Stopping and
success remain distinct outcomes.

## Current discovery implementation: 2026-10-27

The operator clarified that a task-specific question must discover its own cases.
The fixed three-case pilot was an investigation/calibration fixture, not the
requested search space. Its historical record is in
[CREDIT_EXPERIMENT.md](CREDIT_EXPERIMENT.md).

The active binding starts with a biological causal-generalization question and
an empty collection. Each inquiry centers on a particular source-grounded
causal inference and its proposed domain. Source reading supplies candidate
rules; reviews build explicit derivations of distinct theories it supports,
formal additions, an independently warranted target theory, and repair tests.
The warranted domain decides which organisms and scales are relevant. Numerical
attention balances general discovery and existing inquiries. Continuous quality
and fit contribute separate credit coordinates; scientific eligibility never
closes an episode. Collection filtering happens only after numerical stopping.

[GENERALIZABLE_INFERENCE_CREDIT.md](GENERALIZABLE_INFERENCE_CREDIT.md) defines
the active proof obligations and credit objective. Ten strong examples are an
empirical result to assess from actual sources, not an initial condition or a
consequence of program exit. Generic task-neutral interfaces remain separate work.
