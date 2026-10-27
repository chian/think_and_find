# First improvement: formalize a causal reversal

## The object the research must produce

The objective is a **reversal witness**, not two related causal-sounding
paragraphs. It consists of two warranted causal constructions, a particular
inference made in one, an attempted translation of that inference into the
other, and a consequential demonstration that the translation fails. Each
construction must remain useful for its own biological question. The research
system may assemble, challenge, or revise these pieces in any order. This is a
standard for the result and its credit, not a prescribed sequence of Episodes.

Represent each construction as

```text
F = (Q, O, B, I, Y, R, S)
```

where `Q` is the biological question; `O` defines the actors and categories at
the chosen grain; `B` is the system boundary; `I` defines meaningful
interventions; `Y` defines outcomes; `R` contains causal relations; and `S`
states scope. These are question-relative choices, not names for universal
objects. A construction contains an argument path from source-supported
premises through explicit inferential steps to a causal conclusion. Each step
records its premises, conclusion, inferential rule or biological mechanism,
scope, and evidence or contrary evidence. A source may warrant a premise
without itself making the entire inference.

For a proposed case, construct `F_A` and `F_B` for the same biological process
but with distinct useful questions.
Then state a partial, typed translation `T` between their actors, interventions,
outcomes, and relations. Name the exact inference `p_A` that is warranted in
`F_A`, and the bridge premise that would license carrying `p_A` through `T`.
The temptation to carry it must be intelligible; otherwise there is no
substantive failed inference to explain.

The witness identifies a consequential mismatch:

```text
T(infer_A(premises_A))  !=  infer_B(T(premises_A))
```

The expression is a question to investigate, not an automatic proof.
`T` may be partial, and a term can lose its meaning at the second grain. In
that case, the account must show the concrete prediction, intervention choice,
or causal role that would be mistaken by treating the map as valid. A bare
statement that “scales differ” or that `T` is undefined is insufficient. The
evidence must warrant both local constructions and the failure account.

## The near-miss challenge

Before claiming a reversal, try a fixed-category explanation of the same
observations. For example, `Y = f(X, C)` can encode an effect that changes
with a cofactor `C`; adding `C` or a missing dependency changes graph details
while preserving what `X`, `Y`, and their interventions mean. That is a useful
ordinary causal refinement, not the target pluralism. Different measurements
or outcomes under unchanged categories also do not establish a second useful
construction.

The account must state why this kind of reconciliation does or does not work.
If both inferences fit one ordinary graph with stable variable and intervention
meanings, record a near miss. A stronger witness shows which actor, relation,
boundary, or intervention meaning has to be reconstructed to answer the second
question, and why the transported inference would mislead. This does not claim
that no sufficiently large graph could describe the full biology. It locates
the failure of a specific, plausible translation between two explanatory
constructions.

Make the rival account explicit. Let `H0` keep one vocabulary of actors,
variables, interventions, and outcomes, while allowing added context variables
and dependencies. Ask whether `F_A` and `F_B` are simply two restrictions of
`H0`. If so, the apparent reversal is ordinary conditioning or model repair.
If not, name the first term whose meaning cannot be held fixed and give the
observation or intervention comparison that establishes the difference. A
generic assertion that no single graph works is not a valid answer: only a
specified `H0` can be challenged by evidence.

## Multi-step reasoning as an objective and credit rule

The argument may take as many steps as the scientific problem requires. A
longer chain earns nothing merely for being longer. A step matters when it
adds a warranted premise, derives a consequence that was unavailable before,
exposes a bridge assumption, or changes the result of the near-miss challenge.
The result must show the path from evidence to each conclusion and then to the
failed transfer; it cannot skip the causal work with an overall similarity
rating.

Keep one method-credit calculation over coherent, evidence-versioned argument
materials. Design its continuous measures around:

- warrant and inferential completeness of each construction;
- precision and plausibility of the attempted translation;
- evidence for the consequential failure of its bridge premise;
- resistance to the fixed-category cofactor/dependency explanation; and
- distinct pragmatic use of both constructions.

The implemented starting rule scores each frame by its declared conclusion
path, taking the weakest warranted step on that path as its strength
(`D_A`, `D_B`). Let `W` be the weakest of the plausible translation, the
evidenced failure, the answer to the specified `H0` challenge, and the two
pragmatic-use claims. Then a provisional task coordinate is

```text
argument_progress = 0.1 max(D_A, D_B) + 0.1 min(D_A, D_B)
                  + 0.8 min(D_A, D_B, W)
```

All inputs lie in `[0, 1]`. This gives bounded credit for building one or both
local arguments, while the larger share requires the complete witness. A
longer chain cannot improve its score simply by adding steps. The formula is
a task coordinate in the existing single coherent-material credit
calculation. It is not a second controller or a rule that dictates the order
of research. Constructed positive, partial, irrelevant-step and near-miss
arguments check its intended ordering; the numeric anchors still require
scientific calibration.
When the structured rival account says a fixed-category model reconciles the
findings, the completed-witness term is zero regardless of a model's numeric
rating. This leaves bounded credit for useful local derivations.

## Implementation status

`causal_argument.py` validates source references, step identities, premise
order, conclusion pointers, translation pointers, and the explicit rival
account. `reversal_task.py` stores full reviewed arguments and step-warrant
ratings, computes the formula above, and uses its result as one case-credit
coordinate. The model judges biological warrant; the validator checks
structure and provenance. Cross-case theory receives at most two selected
case arguments as compact conclusion dependency paths, with source claims
selected under a separate 8,000-character budget. Identity and synthesis
calls select whole relevant inquiry records under 12,000 and 10,000 character
catalog budgets; omitted IDs are logged. Prompt rendering rejects messages
above 48,000 characters. The return schema and full case argument are not yet
bounded as an addressable delta protocol. Omitted identity anchors may hide
semantic duplicates, so the selected views need a later recall audit. Those
are the next communication changes; the numbered build steps below remain
the wider plan.

The 2026-10-26 v11.5 Argo run completed two discovery cycles before an
operator stop. It acquired 16 source claims and reviewed two cases; both
specified fixed-category reconciliations and therefore received zero witness
strength and only local-derivation progress (0.16 and 0.17). No model or
operation failed in those two cycles. The largest theory measurement message
was 43,766 characters under the 48,000-character Episode limit. These are
workflow and credit-behavior checks, not positive biological validation. The
root's positive-prior exact-zero stop remains incapable of finite automatic
closure, so the run produced no final collection report.

An attempted day-long v11.5 run ended after four cycles and about 25 minutes
because growing identity and theory messages repeatedly exceeded the explicit
bound. It acquired 102 source claims and reviewed three cases before ending;
the root's numerical stop was not reached. v11.6 adds the bounded catalog
views and compact prior theory account. A fresh day-long run is in progress;
the early failure is retained as an operational diagnostic, not evidence that
exploration credit is well calibrated.

The replacement v11.6 run also ended after four cycles. It crossed three
biological systems and assigned some delayed learning feedback, but did not
establish a completed reversal. Its fourth search result contained a
120,623-character description that overflowed source selection. v11.7 now
shows a bounded description preview and preserves the full search result in
the event log. The [behavior trace](BEHAVIOR_TRACE_2026-10-27.md) examines
how the formalism affected those searches and credits.

Credit only a material change to the argument, once per semantic identity.
Repeated ratings, more source quotations for the same step, and cosmetic
rewrites add no attainment. Partial, warranted constructions can receive
bounded progress so the system has reason to build them; the largest gain
belongs to a demonstrated, nontrivial reversal. A careful finding that a
candidate is a near miss changes the research state and future information
needs, but is not reported as a positive case. The numerical mapping must be
checked against these ordering constraints before use: with equal local
warrants, a completed witness outranks a partial chain or cofactor near miss;
adding an irrelevant step does not improve either;
aliases cannot multiply credit. These checks can be constructed from the
formal definitions without importing failed historical candidates as models
of success. No scientific score acts as an admission gate during research.

## How Episodes communicate the argument

The full argument graph, source spans, and revisions live in durable state.
An Episode receives a bounded, addressable view of the part relevant to its
current question. That view names the active inference path, the bridge
premise or competing explanation at issue, the evidence IDs, and the finding
that would change the account. The Episode chooses its own next research
action; the message does not command a fixed workflow.

A child returns a bounded **argument delta**: new or revised premises and
inference links, exact source IDs, a changed or still unsupported conclusion,
the status of the fixed-category alternative, and any new information need.
The parent incorporates the delta into durable state and can pass a selected
gap to another child. Cross-case reasoning uses the same mechanism to return
a proposed generalization and a discriminating information need to broad
discovery. A failed search reports what was tried and does not masquerade as
contrary biological evidence.

Start with a 12,000-character bound on the argument view sent to a child and
a 4,000-character bound on its returned delta. Source-reading text has its
own existing window and is accounted for separately. Select whole records by
their dependency on the current inference and by relevance to the recipient's
question. Send stable IDs for omitted records so an Episode can request them;
do not paste the entire case or alias catalog or truncate an argument mid-step.
If the needed dependency set cannot fit, split the question or fetch by ID
instead of silently dropping a premise. Log message size, selected IDs,
omitted dependencies, and which returned delta changed the parent's next
question. The initial bounds are explicit resource limits to measure and
revise, not claims that this amount of context is scientifically sufficient.

## Build and verify

1. Write the reversal-witness schema and work through abstract positive and
   near-miss constructions by hand. Specify the evidence and inference links
   required for each claim and the exact fixed-category challenge. This is the
   first deliverable and is defined independently of past candidate cases.
2. Implement versioned argument nodes, inference links, translations, bridge
   premises, and deltas in research state. Keep reported source claims distinct
   from inferred conclusions. Validate IDs and dependency structure
   mechanically; causal warrant remains a scientific judgment.
3. Revise the task objective and continuous credit so new warranted inferential
   work can improve an inquiry without rewarding step count or near misses as
   completed cases. Check the ordering constraints above with constructed
   arguments before applying the rule to research output.
4. Replace oversized prompts with the bounded Episode views and deltas above.
   Verify that a missing premise can be fetched by ID and that the parent's
   subsequent question reflects the child's actual inference result. Keep
   source and argument detail in durable state rather than in every message.
5. Run a registered, end-to-end discovery experiment. Assess the produced
   derivations and near-miss challenges, the effect of argument deltas on later
   searches, prompt bounds, source coverage, and target attainment. Revise
   attention and resumption mechanics in response to observed failures. The
   experiment starts from the question, without a preset case list.
