# Episode ownership

## Existing implementation

The copied application composes its work through the local `method_loop` runtime:

```text
[E] evaluate_airborne_safety_guideline
    +-- [T] calculation tools
    +-- [E] investigate
            +-- [E] strategy
                    +-- [E] search
                            +-- [E] page
                                    +-- [E] source_table
                                    +-- [E] report
                                    +-- [E] lexical_probe
```

This is the current domain application, not the public API of a generic system.
`case_reports/research/engine.py` owns runtime composition;
`case_reports/research/workflow.py` owns the domain root and investigation;
`case_reports/research/acquisition.py` owns strategy, search, page and reading
sources. `configured_providers.py` supplies the configured clients.

## Generic design

The current task-specific reasoning prototype adds a separate composition in
`reversal_task.py`, using the same Episode class:

```text
discovery > cycle > find > page > report / lexical / table
                  > synthesize > acquired claims and search hypotheses
                  > think > construct or challenge
```

The discovery controller observes complete acquisition-and-reasoning cycles.
It begins with no cases. Exact source spans ground new candidate inquiries;
bounded synthesis can infer comparisons across acquired sources. The query
episode receives the selected scope, recent measured query feedback and the
focused inquiry's next information need when applicable. Numerical
attention selects general discovery or refinement. The sole continuation signal
is predicted marginal attained credit; fit and epistemic quality are separate
continuous coordinates. Scientific labels and agreement cannot close an inquiry.
See [the current discovery plan](DISCOVERY_PLAN.md). This binding preserves the
retained domain tree and does not complete the generic design below.

Retain the acquisition hierarchy and give its parent a task-neutral question and
result contract. Replace the assumption that the requested answer consists of
named numerical outputs. Existing calculation tools can remain callable when
arithmetic helps answer a task; they should not determine the shape of every task.

The parent investigation owns a developing argument: claims, evidence, inferred
relationships, alternatives and open questions. It selects substantive reasoning
work or further acquisition as needed. Whether this can reuse existing operation
contracts or needs a new tool contract must be established before adding another
Episode type. Do not duplicate the runtime to express that choice.

| Owner | Generic responsibility |
| --- | --- |
| Task root | Requested answer, task criteria, scope, overall progress and final result |
| Investigation | Candidate explanations, comparisons, judgments and unresolved inferences |
| Strategy | An approach to resolving a particular uncertainty |
| Search | Literal query execution and candidate-page selection |
| Page | Retrieved document/version and selection of a reading method |
| Source table | Addressed rows and table context relevant to the delegated question |
| Report | Document-order reading windows with retained context |
| Lexical probe | Targeted ranking and reading of source-local spans |
| Tool | A calculation or another bounded substantive operation with a declared result |

These are responsibilities, not a mandatory sequence of actions.

## Parent-child communication

Each parent supplies a specific information need, necessary context and expected
structured result. A query string is search-engine input; the information question
is a separate field. The causal-discovery binding now returns compact counts and
termination status between episodes. Evidence, inferences and unresolved questions
remain in research state for operation-specific selection at the next model call.

The existing return contract has `parameter_results`, `reviews`, `calculations`,
`reading_coverage`, `operations` and `completion`. Preserve it while copying;
replace or generalize domain fields when the generic implementation is built.
The new design must carry reasoning results to the next decision, rather than
only storing them in an audit that the parent never sees.

## State and revision

Source records and inferred conclusions need different identities. A conclusion
can depend on multiple sources and on explicit interpretations. New evidence
may revise the interpretation without changing the source text. Preserve the
prior conclusion and its basis when that happens; report which result is current.

The existing hypothesis, assessment and dependency mechanisms are candidates for
reuse. Their domain-specific semantics and scoring are not automatically suitable
for arbitrary judgments. Inspect them before deciding what to replace.

## Continuation and completion

Use the existing Episode/controller boundary. Numerical yield measures progress
under the configured policy; it does not prove literature exhaustion or argument
validity. The task's result contract defines completion separately. No new credit
weights, confidence cutoffs or stop rules are selected by this documentation.

Reading ownership, coverage and transport details are in [SEARCH.md](SEARCH.md).
Checkpoint ownership is in [CHECKPOINTS.md](CHECKPOINTS.md).
