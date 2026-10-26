# Discovery from the question

## Permanent target: pragmatic causal pluralism

The target is the verbatim `question`, `candidate_fit_rubric`, and
`theory_fit_rubric` recorded in the v8 registration at
`~/think-and-find-runs/discovery-v8-20261023T181055Z/registration.json`.
The v9 registration contains those same values, as well as the same quality
rubric, target count, credit columns, candidate-origin rule, and bounds. Its
`question` SHA-256 is
`8806a49f4e0c9ac88cedcee809e6609f01455d880930b13f62e127df9a9bd641`
in both registrations. Version numbers identify implementation and evaluator
revisions; they do not rename or replace this target.

## V10 implementation correction: linked causal inferences across sources

The operator stopped v9 after 49 completed cycles because its Arabidopsis root
account passed the numerical screen even though lineage traced descent while
the proposed ancestry-to-fate causal rule was not independently warranted.
V9 could combine papers, but it did not make each causal inference and its own
source warrant the object of search and credit. Its run remains preserved under
`~/think-and-find-runs/discovery-v9-20261023T190300Z/`.

V10 keeps the permanent pragmatic causal pluralism question, ten-case reporting
milestone, no scientific gates, and literal-zero root continuation. Candidate
and review records now contain a linked inference pair: one causal inference
and warrant refs per frame, the proposed translation, and its causal tension.
An unresolved side has an empty warrant list and remains a provisional case.
Focused search numerically selects the weakest warrant or mapping coordinate
and searches for independent evidence, potentially from another paper.
The task-fit vector adds continuous `inference_reversal` credit; it approaches
zero when the first frame never licensed the translated prediction or a later
experiment merely invalidates a prior hypothesis. The other warrant and
mapping dimensions refer to this same linked pair. Post-run numerical
selection still occurs only after the Episode stops.

## V9 implementation correction: separate case and theme identities

Version v8 exposed an identity error after nine completed cycles. A biofilm
detachment candidate was assigned to the Peyer's-patch T-cell inquiry because
both can be described as leaving a compartment. The biofilm review then replaced
the T-cell answer, and its credit appeared under the T-cell case. The v8 run is
preserved at `~/think-and-find-runs/discovery-v8-20261023T181055Z/` as a failed
verification; its reported credit and case count do not establish discovery
quality.

Version v9 gives every occurrence two addresses. The biological inquiry
identity groups evidence only for the same biological process and explanatory
question pair. The conceptual theme identity groups the abstract category and
relation translation problem across distinct biological cases. Each case keeps
its own source claims, review, and materials. The root score sums one dominated
union hypervolume per theme, over all materials of cases in that theme, plus
one theory volume. Identical attainment in another biological case adds no
credit; a distinct, improved account can extend that theme's volume. The
post-run selection reports at most one representative per theme. There is no
scientific admission gate. The root continues until predicted yield is exactly
zero under its literal-zero rule.

The strongest v8 source-supported observation was a developmental switch in
mouse epidermal cell competition from neighbor killing and engulfment to
upward differentiation flux. It demonstrates changed removal relations, but
it may remain a developmental-stage contrast rather than the deeper
question-indexed pluralism sought here. Mammary transplantation versus
ordinary maintenance remains promising with its mechanism unresolved.

## V8 implementation and verification history

The task seeks two warranted causal descriptions of a biological process whose
entities, categories, and relations are built at grains suited to different
questions. A consequential inference can fail when carried from one description
to the other because the category mapping, intervention meaning, collective
actor, or relation changes. Evidence supports each description; a contrary
measurement by itself is not the sought phenomenon. A routine moderator with
fixed categories is a weak case. No single grain is declared fundamentally
wrong merely because another question needs another grain.

The active candidate record names each frame's question and claim, compares
its causal categories and relations, and states the inference-translation
problem. Task-fit ratings measure warrant for each frame, the category-relation
break, and why both grains are pragmatically useful. Epistemic quality remains
separate. Local credit uses the raw seven-coordinate dominated volume, with
zero as the empty baseline; root credit sums distinct inquiry volumes and one
cross-case account volume. This is a continuous score with no scientific
admission gate. Negative and weak cases remain stored even when their task
volume is low or zero. The cross-case account's current conclusion, mapping,
and next information need now enter the next general-search message as
decision feedback. The literal-zero root continuation rule is unchanged.

The compatibility launcher still names this task `causal-reversals`. The v6
run was stopped after the operator clarified that the target is perspective
dependent causal description, not observed sign changes. Earlier sections of
this document record the superseded v1–v6 design and run outcomes; their
task definitions and credit formula do not govern v8.

The v7 verification run under
`~/think-and-find-runs/discovery-v7-20261023T180145Z/` completed two cycles
before interruption. Its first search asked whether lineage ancestry and
spatial organization of a regenerating tissue require different causal
categories. The first reviewed candidate explicitly left the proposed
translation failure unresolved and scored 0.38 on category-relation break.
Root gain was 0.157816, much less than the weak v6 candidates received under
the lifted coordinates. The next search received a category-mapping gap, but
two 180-character decision fields were cut mid-phrase, and general discovery
remained in the same biological system. Version v8 keeps the v7 scientific
target and raw-volume credit, raises the response contract's text capacity,
instructs complete one-sentence abstract decision fields, and directs general
discovery to search another system or scale. A focused inquiry remains the
route for returning to the known case.

The v8 run under
`~/think-and-find-runs/discovery-v8-20261023T181055Z/` began with
a completed cycle that reviewed a mammary-tissue comparison between
single-cell transplantation potential and lineage-specific maintenance in an
undamaged gland. The account keeps the causal mapping unresolved and records
the source's warning that transplantation can alter cell properties and
disaggregate connected epithelial sheets. Its first root gain was 0.357529.
The theory returned three complete, one-sentence decision fields without
truncation; the next general-discovery query moved to a microbial biofilm
system and asked whether the causal actor is a producing cell or a neighborhood
that retains and uses the shared product. These are live checks of prompt
direction and message passing, not scientific validation of either case.

## Correction to the prior scope

The operator requested discovery of biological cases, not further investigation
of a list supplied by the implementation. The fixed PROBES root was a calibration
fixture promoted into the production search space. Its local checks remain
evidence about those mechanisms, not verification of discovery. This design
replaces that root in the authorized think_and_find copy. Original repositories
and prior experiment artifacts remain unchanged.

## Episode composition

Start with the task question and an empty collection. Do not preload biological
examples, sources, mechanism categories or candidate names.

```text
discovery
  > cycle
      > find > page > report / lexical / table
      > synthesize
      > think
      > theory
```

Each root observation integrates acquisition, bounded discovery synthesis, an
inquiry assessment when an eligible inquiry exists, and a cross-case reasoning
attempt once source claims exist. A source-reading operation extracts exact spans and proposes candidate
causal comparisons linked to those spans. Candidates exist only after retrieval
and reading. Source records and provisional interpretations remain distinct.
Discovery synthesis compares newly acquired claims with relevant older records,
with a 24,000-character context bound. The next query receives the task question,
selected scope, recent query feedback, and a focused inquiry's next information
need when one is selected. Synthesized learning and the full candidate catalog
remain in stored research state; they are not copied into query feedback.
The root can explore generally or revisit an
existing inquiry. Reusing the Episode class avoids a second orchestration loop.

## Decision messages

The stored request is JSON. `reversal_prompts.py` keeps the standing research
task and each operation's full instruction text in the prompt template. It
selects decision-relevant fields from the stored request into a separate,
formulaic iteration message, then renders the full prompt. Stable inquiry,
evidence, material, page, span and table identifiers
become call-local labels in the rendered prompt and response schema; the
provider maps returned labels back to stored identifiers. Source versions,
revision hashes and storage records remain outside model messages.

The scientific instruction now frames discovery as a candidate inference
projected from a warranted base relation through an explicit, possibly partial
alignment into a target frame. The prompt distinguishes that projection from the
target finding and asks which transfer assumption fails. It includes schematic
sign, role, collective-scale, near-miss and no-effect examples with placeholders,
not named biological cases. The six comparison parts organize provisional work;
unknown parts remain unresolved, and they are not admission gates. Extraction,
synthesis, search, identity resolution, case review and measurement each receive
stage-specific instructions for using this representation. Routing calls retain
their narrower acquisition instructions.

The query operation receives the standing task in its instruction template;
its changing message contains the selected scope, bounded recent query feedback,
and a focused inquiry's next information need. Its response records a causal
search hypothesis and discriminating evidence alongside one literal query and
one short information need. Child-to-parent
reports contain counts of new inquiries, source claims, reasoning attempts and
readings, plus termination kind and consumed units. Full child results remain in
the audit. The completed cycle report supplies the counts used in the next query
message. Page and reading reports supply compact outcome updates to subsequent
source and reading selections. The response schema is a provider-side output
contract and is not printed into the scientific prompt. The v5 live run
completed two cycles through these messages; longer-run behavior remains open.

Candidate identity describes the specific causal mechanism and reasoning defect.
New organism names or another instance of the same argument do not mint a new
conceptual case. All occurrences and contrary evidence remain recorded under
that inquiry. A dedicated semantic identity operation resolves source-linked
proposals against an identity catalog containing all prior alias occurrences.
It assigns an address to every proposal; it does not assess eligibility. The
discovery model's proposed identity and the resolver's identity are both retained.
Mechanisms and descriptions are learned from sources, not selected
from a preset taxonomy. Identity judgments remain a calibration limitation.

## Numerical attention and continuation

Choose general discovery or an existing inquiry by an explicit upper-confidence
score: that action's smoothed expected normalized attributed gain plus
0.01 * sqrt(log(completed_cycles + 2)/(action_visits + 1)). Ties are stable.
Every action starts with expected yield 0.01. A newly credited material's gain
is allocated once over the distinct cycles that first acquired its newly used
evidence and the cycle that reasoned it through. Equal shares across these
eligible cycles are learning feedback, not extra method credit or proof that any
one query was necessary. Citation counts do not increase shares. Repeated facts,
materials and semantic paraphrases add no gain. If only old evidence supports a
new interpretation, the reasoning cycle gets the share. After delayed shares,
each action's estimate is recomputed in visit order as 0.75 * prior estimate +
0.25 * that visit's total attributed gain. The exploration bonus schedules
work but is not included in the root stop statistic. These scores never admit
or reject a scientific claim. Models propose query strings and measurements,
not controller verdicts.

Focused acquisition reviews its selected inquiry. General acquisition chooses
only from inquiries created during that cycle or receiving newly acquired facts.
Within that set, numerical priority favors unreviewed candidates and then
observed improvement. This prevents an unrelated backlog review from rewarding
the current acquisition action. One call
develops the account and a separate reasoning call measures prior and proposed
accounts against the same evidence. Construct and challenge alternate on visits.

The root stops after at least four integrated observations only when the maximum
expected normalized attributed gain across general discovery and
every known inquiry is exactly zero. Retrospective shares and the current
observation are incorporated before this verdict. An as-yet-unvisited inquiry retains its
positive prior and must receive attention before the root can stop. Child
acquisition keeps the existing four-observation tolerance 0.001. There is no
scientific eligibility exit, case quota stop, iteration/time/spend cap or
list-exhaustion completion. With a positive prior and this exponential update,
the estimate remains positive after any finite sequence of zero gains. Thus the
literal-zero rule has no finite numerical exit under the present estimator; an
explicit operator stop or an operational failure can still end a run. A future
yield estimator able to justify an exact zero would be needed for automatic
closure without changing the operator's stop condition.

## Credit and collection projection

Use seven continuous coordinates: the three epistemic-quality dimensions and
the four task-fit dimensions already measured by the binding. Keep them separate
coordinates; do not multiply epistemic quality by suitability. A supported
negative correction can open new volume even if fit falls. A new relevant causal
case can improve fit and conceptual coverage. No minimum score admits a result.

Keep the first measured coherent vector per material identity. All attempts stay
in the audit. Exact repetitions, semantic aliases and fresh ratings of known
material cannot redraw credit. Current report preference still uses the three
epistemic coordinates independently of historical paid attainment.

The target defaults to 10 and is a reporting milestone only. For each distinct
conceptual inquiry i, keep a local archive of whole canonical answer vectors.
Its local volume V_i is the union dominated hypervolume of those vectors and
the empty baseline point, after mapping each quality coordinate q to
(1 + q) / 2. The collection score is:

```text
G = sum_i (V_i - 0.5^7) + 1[theory archive exists] * (V_theory - 0.5^7)
```

A conceptual biological inquiry contributes one term regardless of its number of source
occurrences. The baseline subtraction prevents an empty local archive from
earning credit. The theory term is zero until one source-grounded cross-case
account is measured; it is one evolving inquiry, never another biological case
or another point toward the target of ten. It uses three epistemic and four
theory-fit coordinates. The sum has no capacity and may exceed one; it is a total credit
score, not a completion fraction. Credit is the marginal increase in G. A new
canonical answer under an existing inquiry can add only the volume that its
whole vector newly covers. Exact repetitions and aliases retain their first
measurements and add nothing. Selecting a different current report does not
reset or repeat archived credit.

Within-inquiry alternatives remain coherent; no coordinate maxima combine
incompatible explanations. The retained local union-volume algorithm chooses
the axis with the fewest distinct boundaries and uses a direct two-dimensional
area sweep. The root sums one local score per inquiry with compensated
summation. It need not build portfolios across inquiries. The material registry
retains all identities even when numerical dominance prunes volume computation.

Conceptual diversity is expressed through source-grounded inquiry identity and
the one-slot-per-inquiry rule. This is a provisional semantic treatment of
diversity, not an established universal taxonomy or calibrated distance metric.
Record the proposed distinction and identity rationale for inspection.

### Calibration limitation

The 0.5 offset and seven equal coordinate treatments are provisional preferences,
not a neutral or empirically calibrated utility. One inquiry scored 0.8 on all
seven dimensions contributes 0.4704844 above baseline. An inquiry perfect on
six dimensions but scoring zero on actual reversal contributes 0.4921875.
Thus a strong negative account can earn more credit than a broadly good reversal
candidate. Useful supported negative conclusions should earn progress, but this
trade-off needs calibration against explicit preferences. Summing local volumes
also makes erroneous conceptual identity splits more costly. The new total
score is not a percentage of task completion.

## Implementation sequence and verification

1. Replace the fixed root and state projection; preserve source provenance,
   retained reading context, all reading modes and response repairs.
2. Wire the existing CLI to question-driven discovery, with an optional request
   file and target count. Do not introduce a replacement launcher.
3. Register and execute a full live discovery experiment from an empty collection.
   Use the actual configured providers, with unchanged models and effort, and
   let numerical rules end it. Record initial emptiness, broad first query,
   retrieval-before-candidate provenance, search feedback, identities, coherent
   credit, stop statistics and actual source coverage. No mocked provider or
   recorded-source replay is used.
4. Inspect outcomes separately from interpretation and scope. Verify that
   source-discovered candidates actually enter the evolving collection and
   influence later queries. Do not infer ten good examples from program exit.
5. Verify duplicate accounting and state restoration with live operations over
   the discovered state; keep these checks distinct from discovery performance.
   Complete static compilation, entry-point and documentation checks.

Success is a working source-driven discovery loop with inspectable numerical
control. Ten strong, conceptually diverse cases is a reporting milestone;
additional useful finds remain valuable. Scientific quality is an empirical
result to assess from the run, not an implementation assumption.

## Execution record

The discovery root, source-linked candidate creation, numerical scheduling,
seven-coordinate credit and existing CLI wiring are implemented.
All 159 Python files compile and CLI help loads. These are static outcomes,
not verification of discovery performance.

`discovery-live-20261022T230650708904Z` was the first registered live experiment, under
~/think-and-find-runs/. It starts with an empty collection and the default
question, target 10, with no preset cases or source URLs. The registration and
six exact implementation snapshots precede acquisition. Numerical continuation
ended it. Models and reasoning effort remained unchanged. Its actual outcomes
are recorded below, including the failed discovery criterion.

### First outcome and revision

The run ended naturally after seven integrated cycles, with root gains
0.003363014767, 0.005932511997, 0.000893141307, 0, 0, 0, 0.000032785253.
It discovered and reviewed two inquiries; neither passed the post-run fit screen.
The minimum-three discovery check failed. There were no failed model responses.
Numerical execution succeeded, but this is insufficient discovery performance.

Inspection located two missing parts of the intended reasoning task. Candidate
creation saw one source at a time; the acquired global evidence had no dedicated
discovery synthesis step. General acquisition could also hand off after merely
collecting background facts, before a causal candidate appeared.

The revised implementation adds one bounded synthesis Episode between acquisition
and inquiry assessment. It receives newly acquired facts and relevant older
facts from the durable registry (24,000 characters of context), and can propose
comparisons across them. Exact evidence IDs support each proposal; the source
need not state the combined inference verbatim. The earlier version passed its
learning and mechanistic hypotheses to query generation. The current message
boundary retains those proposals in research state and sends only selected
decision feedback to query generation. These proposals receive no automatic credit.

General source reading hands off on an actual provisional candidate event;
background facts remain available to subsequent windows and synthesis. Source
exhaustion and numerical child exits remain valid, with no scientific score
prerequisite. Inquiry-focused acquisition can still hand off on new evidence.
The root's four-observation tolerance and requested target remain unchanged.
The first run stays immutable and is not used as a replay input to the next run.

### Synthesis experiment

`discovery-synthesis-20261022T232235914998Z` starts empty under credit version
question-discovery-v2 and state schema 7. It uses the same question, target,
models, effort and stopping thresholds. Its registration and six implementation
snapshots precede live acquisition. It was subsequently stopped on the operator's
instruction, as recorded below. The verification included multiple-source
synthesis, source-grounded identities and learning supplied to later searches.
Its separate live duplicate/restoration follow-up was registered but not executed.

### Operator-authorized stop and repair

The synthesis experiment was stopped on the operator's explicit instruction
at 2026-10-23 01:05 UTC, after 16 complete cycles, 26 candidate inquiries and 16
reviews. This was a failed verification, not numerical convergence. Its original
state, raw operations and six implementation snapshots remain preserved.

Two gene-loss comparisons occupied separate conceptual slots even though a
recorded rationale identified them as the same category of reasoning problem.
Both appeared in 53 of 230 feasible frontier portfolios. Earlier alias occurrences
were omitted from later discovery catalogs. The dedicated resolver and complete
alias catalog address that observed information and identity failure; their
semantic judgments still require live verification.

The sixteenth credit operation took about 892 seconds. Pure numerical analysis
of its recorded ratings with the repaired algorithm computes the complete
portfolio and final volume in about 0.51 seconds, agreeing within 2.23e-16.
This verifies that numeric state, not general scientific performance. The old
live duplicate follow-up was not executed because discovery did not close
successfully; it is not reported as passed.

`discovery-repair-20261023T011040567459Z` was a fresh, empty v3 run under schema 8.
It retained the same question, target, models, effort and stopping thresholds.
It included identity resolution, numeric repairs, acquisition/review attribution,
optimistic unvisited-action scheduling and removal of duplicate query context.
Its registration also declared 12 subsequent live calls for duplicate and
state-boundary restoration controls. The registered run ended naturally after
22 complete cycles: 9 source-discovered inquiries, 9 reviewed, 144 exact physical
facts, 38 candidate batches each resolved by the identity operation, and 11
repeat reviews retaining existing credited material. The four final normalized
gains were zero. The complete report and a separate source-linked summary retain
all inquiries. None passed the post-run four-coordinate fit screen. This is not
the requested ten strong examples.

The full live duplicate controls passed: 12 model calls, 8 zero-credit
observations, 2 exact state restorations, and 0 error in the recomputed root
score. The largest recorded live portfolio calculation took 0.00313 seconds.
Four incomplete provider responses were rejected and recorded; the pipeline
recovered. Its source readers examined about 12.2 percent of the retrieved
snapshot characters, so source and literature exhaustion are not established.

The numerical stop needs a narrower interpretation than “the method has stopped
getting better.” Cycles 19–22 revisited two biofilm inquiries; general discovery
last ran in cycle 15 and still had historical mean normalized credit 0.01726.
The registered four-cycle stop rule fired, but did not re-sample general discovery
before doing so. Root volume 0.26048, or 0.25465 after empty-baseline
normalization, is also not a fraction of the scientific target achieved. The
seven-coordinate preferences remain uncalibrated; high-quality negative
accounts can earn substantial credit while the actual-reversal coordinate is
weak. The experiment artifacts under ~/think-and-find-runs/ record the outcome,
scope analysis, provider failures, and controls without altering the original
repositories or this run’s registered source.

### Open-ended root revision

After reviewing the v3 stop and collection score, the operator set two new
requirements: collect as many useful, distinct finds as the search can yield,
and base root credit on the sum of each find's local volume. Version v4 and
state schema 9 implement the additive score and the action-level expected-yield
controller described above. Ten remains a reporting milestone. Changing the
old four-cycle tolerance alone would not have changed the v3 exit: all four
final gains were exactly zero, even though general discovery had last earned
positive credit. Version v4 compiles and the existing launcher loads. A pure
numerical check gave 2.9765625 for three distinct perfect inquiries and zero
extra credit for an exact repeat; state restoration matched under schema 9.
Four zero-gain focused visits after a productive general search left the root
running because that general action retained positive expected yield. The
declared positive-prior exponential estimate remained positive after 10,000
zero observations in a numerical check. These checks establish arithmetic and
the literal stop comparison. Finite convergence, forecast calibration, and live
discovery performance remain unverified. The v3 run and its
scientific shortfall are historical observations, not live validation of v4.

### Iterative reasoning and delayed learning feedback

Version v5 and state schema 10 added one separately measured cross-case account.
Its rubric asks for cross-case grounding, explicit transfer formulation,
collective and actor-category reasoning, and discriminating research use;
epistemic quality remains separately measured. The account cites acquired
source claims and reviewed cases. It is one inquiry regardless of revisions,
and does not add a biological case to the target count. Material identity is
resolved against earlier cross-case accounts before new local volume is paid.

Source facts retain the ID of the cycle that first acquired them. Each new
credit-bearing material records the evidence refs it uses for the first time
within its inquiry. Its local gain is divided among the distinct first-origin
cycles for those refs and the current reasoning cycle. A cross-case account
also follows the origin cycles of the particular case materials it cites; an
earlier case can enable a newly revised comparison even when its source claim
had already appeared in a prior account. A source cited many
times gets one eligible cycle; a source already used by an earlier credited
material in that inquiry does not receive another retrospective share. The
shares of each event sum to its one method gain. The controller recomputes
action forecasts from the updated visit history. This is an explicit,
auditable eligibility heuristic for scheduling, not a causal estimate of each
action's necessity or a second route to collection credit.

The first v5 live run completed two cycles before it was interrupted as a
failed verification. Cycle 2 revised the theory while citing a case material
developed in cycle 1, yet its entire normalized gain 0.403862 went to cycle 2.
The state, event log, and registered v5 implementation remain preserved under
`~/think-and-find-runs/discovery-v5-20261023T180000Z/`. Version v6 and state
schema 11 add case-material origin lineage to the feedback allocation. The
literal-zero root rule, positive prior, no scientific admission gates, and
existing launcher remain unchanged. This is scheduling feedback, not a second
scientific score.

The fresh v6 run is active under
`~/think-and-find-runs/discovery-v6-20261023T172115Z/`. After two complete
cycles it had two source-discovered, reviewed case inquiries and two measured
versions of the one cross-case account. Cycle 2 added normalized method gain
0.368922. The allocation assigned 0.349460 to cycle 2 and 0.019462 to cycle 1
because the revised cross-case account cited the case material first developed
there. Across both completed cycles, normalized method gains, attributed visit
gains, and action-credit totals all equaled 1.217139 within floating precision.
This verifies the requested bookkeeping on a live revision. It does not
establish that the cases are strong reversals, that the theory is scientifically
correct, or that equal-share eligibility predicts future search yield well.
