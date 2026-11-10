# Root, Discovery and Scientific implementation plan

User-authorized implementation, 2026-11-08 UTC. This replaces the former
Discovery-root / Scientific / search-enabled reconstruction organization.

## Episode ownership

```text
Root [VOC]
├── Discovery [regular request-relative yield]
│   └── Search
│       └── Page
│           ├── Table
│           ├── Report → Extraction Repair
│           └── Lexical Probe → Extraction Repair
└── Scientific [regular scientific yield]
    ├── Theory Reconstruction → Theory Format Repair
    ├── Theory Repair → Theory Format Repair
    └── Theory Comparison
```

Every episode iterates through the common Episode implementation. Root selects
between acquiring information and working scientifically on available material;
there is no mandatory Discovery/Scientific order. Scientific and Discovery are
siblings and exchange commissions and information through Root. Acquisition
children retain extraction credit; their scores are not summed into parent credit.
No episode deadline, usage budget or action allowance is introduced. The existing
whole-workload CPU/RAM boundary and request-timeout policy remain unchanged.

## Implementation sequence

1. Define closed, versioned scientific request and return records. A discovery
   goal identifies its question, supplied scientific quantities/Gs, acceptable
   answer structure, source requirements and individually identifiable criteria.
   Root forwards Scientific's exact goals to Discovery. Findings retain source
   locators and the criteria they address. Returns include qualified partial
   answers and explicitly unresolved scientific questions.
2. Implement executable request-relative verification and credit. Scientific
   supplies closed native or scientific requirements over admitted source claims
   and formal scientific objects, not model-assigned numerical grades. A native
   match certifies only its literal predicate. Scientific requirements can use
   source predicates as retrieval candidates, but fulfillment requires linked
   source-grounded typed structure or the specified executed formal check. Validation,
   identity checks, supported expression equivalence and executed causal
   comparisons can establish particular criteria. Ambiguous scientific use of G
   is unresolved and returns through Root for Scientific assessment. A mere
   occurrence of a name does not establish that another theory depends on G.
   Verified criterion/finding identities earn credit once; subsequent receipts
   can resolve an earlier pending finding without duplicating credit.
3. Replace the root and acquisition wiring. Root owns VOC and chooses Discovery
   or Scientific with selected shared records. Discovery owns targeted query
   planning. Search/Page/readers retain their established acquisition machinery,
   source-relative extraction and ordinary yield estimators. They return evidence
   rather than independently conducting Scientific's theory reconstruction.
4. Give Scientific three concrete iterative children. Reconstruction defines
   theories and their Gs from supplied evidence. Repair develops and examines
   modifications, including G-preserving repairs. Comparison relates accounts,
   evaluates transfer and records useful alterations, retained successes and
   failures. Their missing-information requests return to Scientific and Root.
5. Implement scientific-object lookup and formal tools for Scientific and all
   three children. Lookup retrieves exact theory/G/bounds/comparison/repair
   records, not a hidden literature-search loop. Lean accepts selected definitions,
   assumptions and an equivalence statement plus a proposed proof/tactics; the
   actual Lean kernel checks the result. An unsuccessful attempt is unresolved,
   not proof of inequivalence. SCM/equation/simulation requests execute selected
   models and interventions and return reproducible results and limitations.
   Tool receipts identify their exact inputs and return to the invoking episode's
   next decision. Tools are chosen for scientific questions, not mandatory stages.
6. Bind Root VOC to completed child outcomes. Scientific earns one deduplicated
   progress unit for the first source-grounded public examination of a previously
   unexamined G, independent of whether the result supports, challenges or leaves
   it unresolved. The record identifies exact G/theory snapshots and a source-
   linked argument; a tool call or an episode visit alone cannot earn this point.
   Existing support and useful-change credit remain substantive calculated
   results. Visiting an episode or calling a tool is not itself an assessment.
   Keep discovery fulfillment, first examination and substantive support/change
   components visible and do not add descendant action counts as observations.
7. Validate focused production paths, not an exhaustive test suite: sibling
   routing, selected/versioned inputs, criteria/credit deduplication, a definite
   match, a supported negative finding, unresolved theory-use feedback, actual
   Lean equivalence and failure, SCM interventions, and scientific tool access
   for all four scientific episodes. Reuse retained run material where useful.
   Run one bounded-machine workload at a time through this repo's launcher.
8. Start a fresh normal workflow with the generic objective, no supplied topic
   and no resumed scientific state. Monitor actual routing, communications,
   assessment behavior and scientific results. Report implementation and observed
   workflow validation separately.

## Explicit exclusions

No compatibility aliases or old episode execution paths. No extra report composer,
review-before-credit gate, model-authored pass/fail score, invented certainty scale,
or compulsory Lean/SCM sequence. Shared scientific data and calculations remain
reusable across the three scientific responsibilities.

## Retained implementation verification

Implementation is in the production Root/Discovery/Scientific path. The previous
normal workflow was explicitly replaced for this work and fully exited before
verification. All execution below used this repository's launcher and unchanged
one-CPU, 5-GiB, no-swap resource boundary.

* `runs/lean_install_20261108_01`: actual Lean 4.35.0 and selected Mathlib modules
  installed beneath the run directory; complete workload exit recorded.
* `runs/scientific_tools_verify_20261108_03`: seven focused real-tool checks passed,
  including actual kernel equivalence, unsuccessful proof remaining unresolved,
  placeholder rejection, SCM interventions and Mathlib real-number equivalence.
  Two earlier verifier attempts exposed and led to fixes of receipt admission
  and result-table integration; they are retained as failed evidence.
* `runs/episode_organization_verify_20261108_04`: nine focused production checks
  passed for hierarchy/controllers, actual tool decision schemas in all four
  scientific episodes, sibling request/source handoffs, positive and supported
  negative findings, deduplication, pending scientific feedback resolved through
  SCM, examination credit and format-repair commission construction. The added
  distinct-theory case fulfills both requested criteria for declared different
  accounts and excludes same-theory aliases from distinctness credit. Reusing a
  native claim under a narrower goal does not mint another global information point.
  The failed live goal is rejected before acquisition, native operand fields and
  local type references are validated, and actual Lexical/Extraction Repair
  reads exercise the corrected acquisition-only request path.
* `runs/failure_yield_verify_20261108_01`: failed outcomes still count as zero-yield
  observations, reach parents, naturally yield, and allow transient recovery.
* `runs/episode_organization_verify_20261108_05`: all ten focused checks passed.
  The added retained-live observation regression rejects document/study IDs and
  identities shared by different quantities or conditions, routes all rejected
  entries through ordinary Extraction Repair, and retains six separate repaired
  observations. Copies of one observation still deduplicate; values, units and
  uncertainty do not define identity. Complete workload exit was recorded.
* `runs/episode_organization_verify_20261108_07`: all twelve focused checks passed
  after source-occurrence identity was added. The actual call 8/27 replay gives
  the three repeated observations zero new credit; call 27's two first typed
  explanatory claims earn two extraction units, and call 28 retains four units
  for new source content. Three-step quote-boundary drift, distinct quantities
  and conditions, and ordinary sidecar-only ambiguity repair also pass. Accepted
  scientific fields remain unchanged. The preceding `_06` attempt exposed an
  invalid cross-fragment fixture quote, now corrected without relaxing validation.
  Both workloads fully exited; these remain scripted contract checks.
* `runs/episode_organization_verify_20261108_08`: all twelve checks passed after
  resolving quotes inside the selected source unit and supplied span. Actual
  call 34's identical caption clauses now resolve separately by unit ID; repeated
  quotes within one unit, overlapping repeated substrings and cross-unit quotes
  remain rejected. Existing source deduplication and repair cases still pass.
  Complete workload exit is recorded.
* `runs/episode_organization_verify_20261108_09`: all thirteen focused checks
  passed. Actual Search credit/controller tests cover unrelated inherited
  evidence, selected and unselected sources, duplicates, newly extracted
  information at an already-known URL, type mismatches and empty/invalid results.
  All three semantic criterion kinds retain only preliminary source credit;
  Discovery/global fulfillment stays zero until linked executed Scientific work.
  Actual SCM execution then resolves the dependence criterion through a saved
  check receipt. Complete workload exit is recorded.

Fixture choices demonstrate contracts and credit behavior, not autonomous research.
The first fresh normal workflow, `runs/normal_live_20261108_01`, independently
selected an initial information goal and routed Root → Discovery → Search → Page.
It exposed two real integration gaps: criteria used custom-answer fields instead
of native source-claim fields, and the passage reader still referenced the old
reconstruction commission. It exited with a retained driver error before any
scientific outcome. Those live inputs are now passing regression cases; a corrected
fresh workflow remains to be observed before recording scientific routing/results
as validated.

The second fresh workflow, `runs/normal_live_20261108_02`, reached actual passage
extraction without the first run's reader error. It generated valid native-field
criteria and moved among sources. A retained six-observation extraction exposed
paper URLs being reused as observation IDs, collapsing distinct measurements.
The driver was interrupted to correct that credit-integrity bug; the supervisor
recorded complete descendant exit. The regression above covers the actual input
and ordinary repair path. Neither live attempt had yet reached Scientific, so
autonomous sibling handoffs and scientific results remain unvalidated.

`runs/normal_live_20261108_03` was the fresh normal workflow after those fixes,
with no topic argument and no resumed state. Root selected nutrient enrichment
in plankton predator–prey systems, commissioned Discovery and reached real
source extraction. The first lexical reader yielded naturally after zero-yield
passages; Page then selected an unexamined model-development read. Retained
outputs include experimental outcomes, qualified model mismatches and native
equations with symbol definitions. It also exposed a remaining credit-identity
defect: calls 8 and 27 returned three observations at the same document version
and physical source positions, but changed free-text locators produced new IDs.
The run was deliberately interrupted for this fix; complete descendant exit is
recorded. These observations validate acquisition routing only; no autonomous
return through Root to Scientific had occurred.

The source-identity correction replaces free locator identity with host-defined
document-wide statement units and validated exact quantity/condition/statement
anchors. Source positions identify occurrences, not scientific truth or universal
semantic equivalence. Distinct quantities and conditions must remain distinct;
ambiguous anchors use ordinary targeted Extraction Repair. Table-cell identities
remain unchanged. The focused replay above now passes. A fresh normal workflow
using that source contract remains to be observed through the Scientific handoff.

`runs/normal_live_20261108_04` was launched fresh with the new source contract,
without a supplied topic or resumed state. Root selected enrichment and
predator–prey stability and commissioned Discovery. Actual source extraction
accepted unit-labeled anchors. Calls 7/8 and 9/10 exercised quote ambiguity and
targeted repair: each correction changed only the rejected quote, after which
the pending claim earned its extraction unit. Later, call 34 exposed a resolver
bug: identical clauses in two distinct host units were rejected as ambiguous
across their shared source span, despite the supplied unit ID. Repairs alternated
between the repeated clause and a wider cross-unit quote. The run was stopped
for the scoped resolver correction above and fully exited. It had not returned
through Root to Scientific; useful G alterations remain unobserved.

`runs/normal_live_20261108_05` is the fresh normal run after the unit-scoped
resolver fix, without a topic argument or resumed state. Its own service and
Think and Find working directory are confirmed, with the unchanged one-CPU,
5-GiB, no-swap resource boundary. Root created an enrichment/predator–prey
information goal. Autonomous Scientific handoff and scientific results remain
under observation.

Follow-up code review confirmed a Search-credit scope bug for later commissions:
inherited unrelated native evidence could pre-fill preliminary semantic slots
before a new Search did any work. The correction restricts operands to this
Search's selected sources, subtracts stable native witness IDs present at its
start, then projects the remaining operands into saturated criterion slots.
Scientific fulfillment, Discovery credit and global credit are unchanged. The
correction and focused regression pass in `_09` above. `_05` used its
already-loaded Search implementation and had no inherited native evidence in
its initial Search. It was deliberately stopped to verify the ready correction
and replace the old-code workflow; complete descendant exit is recorded. Its
first two Page jobs returned to Search and a third source supplied a published
prey-evolution refinement, but no return through Root to Scientific had occurred.
A fresh run using the correction remains to be monitored.

`runs/normal_live_20261108_06` is that fresh current-code run: no topic argument
or resumed state, correct Think and Find working directory, and unchanged
CPU/RAM/swap boundary. Root independently commissioned a quantity-based versus
stoichiometric enrichment comparison. Through call 10, acquisition and targeted
quote repair have executed without a driver error; no autonomous Scientific
commission or examination has occurred yet.

A fresh read-only completion audit confirmed the implemented topology, all four
scientific tool dispatch paths, retained thirteen-check organization verification,
seven real-tool checks and failure-yield evidence. It found no additional concrete
regression. Completion remains unproven until the current workflow returns source
findings through Root and independently chooses Scientific work, whose actual
decisions, scientific results and credit must then be inspected. Optional tools
and children must not be forced into a prescribed itinerary to manufacture that
observation. Missing-information feedback should be inspected if it arises.

Further `_06` observations: the first reader returned at call 22; focused
reexaminations at calls 24 and 26 added qualified source information. Page
naturally yielded at call 27, and Search received one completed source job with
two request-relative information units. A call-28 finding submission reused the
goal's key and was rejected by immutable-record validation; Search acknowledged
that no finding had been admitted and continued to a second source. That
source's call-31 extraction succeeded without diagnostics. No autonomous return
through Discovery/Root to Scientific has occurred yet.

Through call 59, the second source's reader has retained 85 local extraction
units. Valid rereads and noncontributing passages earn zero; new source
qualifications can still add credit. The extracted material distinguishes
reduced oscillation amplitude from disappearance of a limit cycle and separates
model predictions from cited empirical associations. These are acquisition
observations, not yet a reconstructed G, an executed comparison or scientific
credit. The workflow remains active while its ordinary yield estimator governs
the reader's continuation.

Call 63 extracted a particularly relevant reduction example from figure D1:
the simplified two-dimensional model assumes infinitely fast nutrient
mineralization and algal uptake, whereas the full model retains nutrient
recycling through sediment. The source reports similar behavior for nonsinking
algae, with different ranges of stabilization and the enrichment effect. This
is a source-grounded candidate for reconstructing a reduction's present scope
and failure mechanism; it has not yet been formalized or scored as a G alteration.

At call 68, the second reader returned to Page after exhausting its selected
passages, with 121 extraction units and no unresolved extraction errors. Page
received one completed child observation. At call 69, Page selected the native
parameter table to obtain definitions, units and defaults, explicitly separating
model parameters from empirical observations. Search has not yet received this
Page's completed result.

The table returned at call 73 with `ended_by=yield_stop` and
`controller_reason=local_saturation`: its first query earned 102 cell units,
two repeated queries earned zero, and recall earned zero. Page received the
completed table once, with gain 102; the cells were not treated as separate
parent observations or global scientific credit. The next focused, single-span
reading returned four additional extraction units at call 75.

At call 77, a final focused reading returned 12 units and Page naturally yielded
with `controller_reason=local_saturation`, retaining 239 extraction units.
Search received this completed Page once and gained zero additional request
units: its two preliminary information slots were already filled. The new
source evidence was retained despite the zero request-credit delta. Search is
continuing; no admitted Discovery finding or autonomous Scientific work has yet
been observed.

Call 78 admitted the fresh immutable finding
`enrichment_food_quantity_quality_source_comparison_v2` and its host check.
Three broad native-source criteria were fulfilled by five unique source
witnesses, earning five global Discovery-fulfillment units without duplicating
equation witnesses shared by two criteria. This checks retained source types;
G registration, examination, support and alteration remained zero. The third
source reader rejected attribution of related-paper listings to the target
article and reported unavailable primary text. At call 89, Search naturally
yielded and Discovery received one completed query result worth three fulfilled
criterion slots. Discovery's call-90 follow-up seeks the primary experiment
behind a secondary claim about delayed Daphnia density buildup, explicitly
separating transient response from population stability. The workflow remains
active; return through Root to autonomous Scientific work is still unverified.

In the second Search, a newly extracted native operand at a previously known
URL earned one preliminary request unit at call 99. Fresh partial findings at
calls 100 and 131 were admitted with all three source criteria unfulfilled and
zero additional global credit: citation resolution was not promoted into
primary experimental evidence. Readers separated two Urabe papers, local
bibliographic year suffixes, host-study results and unavailable target text.
An archive error page was reported as an access failure with no scientific
claims. Search naturally yielded at call 138; Discovery received one completed
query result with zero new fulfillment credit. Its call-139 third Search seeks
Rosenzweig's original 1971 enrichment intervention and stability argument.
Root has not yet received this first Discovery commission.

The third Search first recovered the original 1971 publisher abstract and its
full-text access limitation; an ambiguous exact quote was repaired at call 143.
An alternative representation then supplied original article-body text and a
figure caption at call 153, including the explicit link from enrichment to
increased K and the graphical isocline argument. Damaged OCR derivative notation
and unavailable figure images remain qualified rather than silently repaired
into source equations. This improves the source basis for subsequent Scientific
reconstruction, which has still not been commissioned in this live workflow.

Call 179 exposed a concrete verifier-contract failure. The original-only finding
explicitly did not establish a nutrient-dependent food-quality equation, but the
two differently worded equation criteria used identical broad source schemas.
The host marked both fulfilled by the native equation V = J. This was not a
duplicated numerical witness, but an unjustified scientific fulfillment label.
The run was deliberately interrupted; its retained resource receipt confirms
driver exit -2 and complete descendant exit. No autonomous Scientific work had
yet occurred.

The correction replaces the unscoped `{id, question, check}` criterion with a
closed `{id, requirement}` contract. Native requirements contain only a literal
source predicate and yield `native_match` or `native_no_match`, explicitly scoped
to `native_fields_only`. Scientific requirements declare their scientific
question, an optional native candidate filter and a formal predicate. A new
`theory_structure` predicate checks a typed component and its transitive reference
closure from a linked theory fragment, grounded in the finding's exact selected
source evidence. Missing matching structure stays pending. Lean equivalence,
executed causal dependence and distinct-theory checks retain their separate
roles. Host receipts retain the authoritative requirement, checked scope and
exact component/source versions. This verifies representation properties and
provenance; it does not prove source fidelity or empirical truth. The old
criterion format is rejected rather than adapted. Aggregate exposed labels use
`discovery_progress` and `discovery_request_gain`; nested receipts separately
report native information and scientific fulfillment.

`runs/episode_organization_verify_20261108_12` passed all fourteen focused
checks, return code zero and complete descendant exit confirmed. This includes
the actual call-179 replay: V = J remains a native match, while the scientific
food-quality requirement stays pending. A matching source-grounded typed
structure resolves it; an unrelated component's source citation does not.
Repeated fragment aliases add no credit, and native-only progress is not labeled
scientific fulfillment. The first attempt at this new check (`_10`) rejected an
authored equation fixture lacking symbol definitions; that fixture was corrected
before `_11` and the final exposed-label verification `_12` passed. These are
production-contract checks, not autonomous scientific behavior. A fresh normal
workflow remains to be monitored through Root to Scientific.

`runs/normal_live_20261108_07` is now running fresh with the corrected requirement
contract, no supplied topic and no resumed state. Its actual service is active
with supervisor PID 3988249 in the Think and Find checkout; the retained resource
receipt confirms one CPU, 5 GiB RAM and no swap. Its first Root decision is in
flight. Autonomous Scientific routing, returned theory work and live credit
behavior remain under observation; the implementation goal stays active.

The new Root chose enzyme activity versus steady-state pathway flux. Its first
goal contains two explicitly native requirements (equation shape and numerical
observation value), with interpretation deferred to Scientific. The first reader
returned 16 extraction units without errors. It retained a qualitative,
condition-dependent overexpression result and possible regulatory compensation,
without inventing numerical zeroes or claiming a recovered equation. At call 13,
Page naturally yielded; Search received zero request credit because neither
declared native operand type was supplied. Source information remains retained.
No Scientific handoff has yet occurred in `_07`.

The second Page returned navigation and an unresolved PDF lead. One final recall
at call 20 used an unavailable record; the host rejected it, passed the diagnostic
to Search and retained the accepted result while the iterator naturally yielded.
No driver failure occurred. A later source's broad readers returned zero, after
which Page used an inventory preview to target a previously unread discussion
of fivefold PFK overexpression. Call 33 extracted that reported numerical
magnitude while leaving flux response qualitative and distinguishing expression
from measured activity. Image-only equations were not reconstructed. At call 43,
Page yielded with 29 extraction units; Search received one preliminary native
operand unit. No finding had yet been submitted, so global progress remained
zero. Search's call-44 continuation seeks the same primary experiment's full
text and paired numerical values. Scientific routing remains unverified.

Subsequent `_07` acquisition retained the primary paper's DOI/access limits and
gave its alternative Page zero request credit at call 53. A further source
supplied protein-depletion and growth observations rather than direct glycolytic
flux. Through call 67 its reader retained 75 extraction units without new
diagnostics, keeping extract-based enzyme assays, growth measurements, stated
predictions and tentative regulatory explanations distinct. Search still had
one preliminary native operand slot and no submitted finding/global credit.
The actual service remains active; Root-to-Scientific routing is still pending.

At calls 72/73, one within-unit quote ambiguity was withheld and then resolved
by ordinary targeted repair. The reader returned at call 80 after exhausting
its selected passages, with 100 extraction units and no unresolved errors; Page
received one child observation. Its focused call-82 reread added two units.
The retained source packet explicitly separates extract-based active-enzyme
assays, growth/recovery outcomes, the authors' growth-as-flux interpretation,
and independently measured glycolytic flux. Scientific fulfillment has not been
claimed from these source-type matches; no Scientific commission has yet run.

At call 86, Page naturally yielded with 112 extraction units and returned once
to Search with zero additional request credit: its numerical-operand slot was
already filled. The next source's calls 89/90 completed extraction without
diagnostics. Its reported enzyme-activity and yield values retain missing
control/mutant row assignments; yield is not treated as a flux rate. These
remain qualified source operands, not a validated activity-to-flux relationship.
The workflow is still active and no finding receipt or Scientific handoff has
yet occurred.

The later reader returned to Page at call 108 with 139 extraction units and no
errors. Its source methods distinguish measured extracellular rates from
constrained model estimates of intracellular flux, and preserve damaged assay
units and missing figure-row assignments. Page's focused rereads then returned
through Search at call 116 with one new preliminary operand unit (Search score
two). The actual native equation is a phosphorylation-normalization relation,
not the still-missing kinetic or pathway-control equation. These two native type
slots therefore do not certify the broader scientific request. A subsequent
navigation/provenance-only Page returned zero credit at call 122; Search chose
another source. The workflow remains active without a submitted finding or
Scientific commission.

The first Search returned at call 135 after its call-134 partial finding was
rejected for putting raw evidence in `scientific_links`. No finding or global
credit was admitted. The exact failed payload and all 102 record updates were
returned to Discovery, including the cited prose passage and its exact version;
no source loss was demonstrated and no interface change was made for this model
action error. Discovery's call-136 query then targeted kinetic and control
equations. The new methodological Report returned at call 140 with 24 extraction
units, including explicit kinetic equations, quasi-steady-state conditions,
enzyme conservation and control-coefficient definitions. Source notation issues
and missing held-fixed conditions remain explicit. Scientific has still not
been commissioned in this normal workflow.

The methodological Page naturally yielded at call 152 with 53 extraction
units. One exact-quote issue at calls 148/149 was repaired successfully with
zero new credit for already known information. Call 153 then admitted the fresh
finding `enzyme_activity_to_pathway_flux_methodological_packet` and its host
receipt without diagnostics. Seven distinct equation witnesses earn seven
native-information units, with `native_match` / `native_fields_only` for the
equation operand; the numerical-observation operand is `native_no_match`.
`scientific_fulfillment_total` and all G/examination/support/change components
remain zero. Search's next source targets a primary glucose-transport study to
resolve activity-to-parameter correspondence and measured activity–flux pairs.
This is successful live native verification, not yet Scientific validation.

The primary transport study supplies conditional control estimates with standard
errors and a source-reported kinetic-parameter correction that improves a model
prediction. It distinguishes recalculated transport from unmodified measured
rates and does not by itself map oxygen flux to glycolytic flux. At calls
158/159, an `operational_definition` equation entry lacking symbol definitions
was rejected, then explicitly removed (`replacement: null`); no mathematical
definition was certified by that correction. The source record and rejection
diagnostic remained available to Page. Subsequent reading continues without a
new error, while Scientific has still not been commissioned.

At calls 171/172, three ambiguous observation-condition quotes were corrected
using the exact full table row; the host admitted those three withheld source
observations. The table child then returned at call 181: its first query earned
15 cell units, two repeated queries and recall earned zero, and it naturally
yielded. Page received this table once, with gain 15 (source total 120), while
global progress stayed at seven native-information units. This validates the
live leaf/table credit path without asserting scientific fulfillment. A focused
Page reading remains active.

Page returned the primary-study work at call 187 after naturally yielding, with
132 extraction units and one new preliminary numerical-information unit for
Search. Call 188 admitted
`enzyme_activity_to_pathway_flux_combined_packet` and its host receipt. Both
generic native criteria are `native_match` / `native_fields_only`. Seven earlier
native units plus one new equation and three new observation witnesses give
eleven native-information units; the shared equation witness was not counted
again. Scientific fulfillment and G/examination/support/change remain zero.
Search chose another source, whose call-191 extraction completed successfully.
No Scientific commission has yet occurred.

That methodological Page naturally yielded at call 234 with 108 local extraction
units. Its readers retained the distinction between independently changed model
parameters and resulting steady-state rates, source qualifications on transfer
from first-order to Michaelis–Menten kinetics, and unavailable equation images.
The call-204 quote ambiguity was corrected at 205. A later recall used a shortened
nonexistent record key; the failure was returned and subsequent recalls completed.
Page returned once to Search with zero additional request credit because its two
native slots were already filled. Global native information remained eleven and
Scientific fulfillment remained zero. The service was confirmed active after
this return; Root-to-Scientific routing is still unverified.

Search returned to Discovery at call 235. Its final submission incorrectly linked
another discovery finding as a scientific result and was rejected. Discovery
recalled the existing packet and repaired the submission at call 237 using empty
`scientific_links`; the finding and host check were admitted without diagnostics.
One new equation witness raised native information to twelve, while Discovery's
already-earned request score remained two and scientific fulfillment remained
zero. The subsequent query targets the original transport study's unresolved
Figure 5 series, activity calculation and printed units. This demonstrates a
specific follow-up search and parent-level recovery from the rejected submission;
it does not yet demonstrate the Root-to-Scientific handoff.

A live identity audit of call 274 compared Page's exact local-yield identities
before and after that reader. The repeated apparent-Km equation and 21 μM
inhibitor observation were already present and earned zero. The 18-item delta
was eight Figure 1 kinetic observations and ten first-counted prose occurrences;
no identity was removed. Four prose occurrences had previously appeared on a
mixed native/prose row without separate prose credit. These counts therefore
describe extraction accounting, not eighteen wholly unseen scientific facts.
Global native information remained twelve and scientific fulfillment zero.

Live calls 307 and 309 failed with transport connection resets. Each failure was
recorded as an observed zero-yield outcome, and the following repair calls 308
and 310 completed without new diagnostics or additional credit. Reader 79 then
yielded through local saturation, returning its four earlier accepted extraction
units and both original diagnostics to Page. The service remained active. This
confirms live request-failure recovery and ordinary episode continuation without
turning a request failure into an episode deadline or losing accepted work.

Call 396 exposed a further identity defect: it returned the same nine Figure 1
and inhibitor-constant measurements as call 274, at the same statement units and
quantity mentions, but changed descriptive condition-anchor sets. Hashing those
sets minted nine new observation identities. The actual Page return at call 399
therefore earned nine duplicate units, increasing its score from 201 to 210.
The run was stopped for this confirmed credit defect; its receipt records complete
descendant exit, and the owning service and slice were confirmed inactive/empty.
No Scientific commission had occurred.

The correction gives observations a required exact result-occurrence anchor,
alongside the quantity mention. Condition descriptions stay attached and validated
but do not change observation or claim identity. Separate equal-valued results
retain their distinct source positions. The old observation-anchor shape is
rejected rather than adapted.

`runs/episode_organization_verify_20261108_13` passed all sixteen focused checks
with return code zero and complete descendant exit. The retained 274/396 replay
keeps its nine numerical operands unchanged: first extraction earns child/Page
9, Search/Discovery 1 and global native information 9; the condition-drift reread
earns zero at every local level and leaves global native information at 9.
Claim condition drift also earns no new identity. Equal-valued result occurrences
at separate positions remain distinct. Literal mismatch reaches ordinary repair,
which corrects only the unaccepted observation; external quantity headings,
Unicode-minus numeric literals, Boolean/text literals and partial-number rejection
also pass. These remain scripted verification cases. A fresh normal workflow is
still required to observe the corrected protocol and autonomous Scientific routing.

The corrected fresh normal workflow is `runs/normal_live_20261108_08`. Its
launcher receipt and actual service confirm this checkout as the working
directory, the unchanged one-CPU/5-GiB/no-swap resource scope, and no topic or
resume argument. Root created `enzyme_turnover_mean_and_fluctuations` and
commissioned Discovery, which selected its first search. Live result-anchor
admission was verified at call 6: Report accepted seven records and seventeen
local extraction units without diagnostics. Both the exact qualitative
turnover-cluster phrase and the numeric 465 source value used the required
result-occurrence anchor. The latter retained its qualification as a reported
background enzyme property, not a new assay measurement. Global credit remained
zero. Autonomous Scientific routing remains unverified.

Calls 7/8 exercised ordinary repair in the live corrected protocol. Numeric
`43000` did not match the supported numeric-literal parser for source `43,000`;
repair retained that exact source form as text rather than certifying a numeric
conversion. An ambiguous quantity quotation was corrected to exact `with w`.
Both previously withheld operands were admitted, with two extraction units and
no new diagnostics. The broader reading had already retained nineteen units.
This validates live targeted repair and retention of unsupported literal forms;
it does not establish scientific fulfillment or general numeric-format support.

At call 23, the first `_08` Page yielded through local saturation with 106 local
extraction units and returned 48 updates to Search. Search received two native
request-slot units. The preceding reread at call 21 returned four child units but
zero Page units because its source identities were already known. The only
diagnostics carried upward were the two call-7 issues repaired at call 8. No
finding receipt or Scientific commission had yet run; global credit remained zero.

The read-only completion audit found no additional concrete defect in sibling
routing, Root-only VOC, typed verifier scopes, scientific tool access, version
grounding or parent observations. It confirmed the sixteen retained `_13`
checks and the seven real-tool checks. Autonomous Scientific consumption and
return remain separate, pending execution evidence.

At call 25, two supplementary-source retrieval failures returned zero gain and
their diagnostics. Retrieval then recovered without intervention. Call 27
admitted the first supplement window without diagnostics, retaining the displayed
DOI discrepancy, distinct preparation/measurement/simulation conditions and
ambiguous printed units. The reader continued to its next window. These are
source-extraction results; no global scientific credit had been awarded.

At call 49, the supplement Page yielded through its estimator and returned to
Search. Its source assessment distinguishes preparation from measurement
conditions, Poisson intensity fits from turnover waiting-time laws, and the
tested ten-conformer simulation's autocorrelation fit from its mismatch in trace
appearance. It explicitly leaves the full assay recipe, measurement temperature,
RGP expansion, DOI reconciliation and unavailable figure/rate-table operands
unresolved. Search retained its two native request slots and gained zero from
this additional Page. Retrieval and repaired-anchor diagnostics remained in the
return. This validates a qualified second Page return, not Scientific fulfillment.

Search returned to Discovery at call 51 with retained source evidence and rejected
finding submissions: call 50 reused the immutable goal key; call 51 used a new
key but incorrectly put raw evidence into `scientific_links`. Both failures
earned zero and reached the parent. Discovery's call-52 assessment explicitly
states that no fulfillment receipt exists and starts a concentrated follow-up
for missing waiting-time assumptions and serial-correlation measurements.
No finding or scientific fulfillment has yet been admitted in this run.

The follow-up interface audit confirmed that neither fresh-key ownership nor
`scientific_links` target kinds were described in the runtime finding schema.
The shared response contract now documents an unused finding key, the excluded
source/discovery record kinds, native selector fields and valid empty scientific
links. Only schema descriptions changed; validation and credit remain unchanged.
`git diff --check` passed. The active `_08` process retains its already-loaded
prompts, so its subsequent behavior cannot establish the effect of this wording.

At call 74, the concentrated follow-up reader returned 76 local extraction units
to Page. Its retained packet includes the kinetic scheme, reported mean-rate
consequence, waiting-time/randomness/serial-memory tests and explicit unresolved
equation typography and cycle-independence premises. Source-anchor repairs and
their diagnostics reached Page. These counts are local source accounting; no
finding receipt or Scientific commission had yet been admitted.

The follow-up Page yielded at call 85 and returned two preliminary native slots
to Search. Call 86 again reused the immutable goal key and was rejected. Search
corrected the key at call 87 to `english_turnover_statistics_source_packet_v2`,
with empty scientific links; the finding and host receipt were admitted without
diagnostics. Five source-claim and four numerical-observation witnesses produced
nine native-information units. Both criteria report `native_match` with scope
`native_fields_only`; scientific fulfillment, G examination, support and useful
change remain zero. Search has not yet returned this follow-up to Discovery.

The alternative-copy reader yielded at call 107 with 29 local units after seven
zero-credit windows. Later calls 120/121 exposed a reader-channel omission:
call 120's four units mixed two genuine Methods-erratum units with two unrelated
listing units; call 121's three units were entirely unrelated listing/abstract
warnings. Native claims and observations were empty. These warnings belonged in
reading memory, not credited findings. The source interleaves foreign excerpts
before their identifying listings, and a references cutoff would remove the
appended erratum, so no unsupported blanket text cutoff was added.

The reader prompt now explicitly routes bounded absence/status reports,
uninspected-source warnings and unrelated citation/recommendation leads into
reading memory. It preserves actual negative scientific results, definitions,
assumptions, limitations and requested own-work provenance. Credit, validation,
source offsets and repair/yield behavior are unchanged; existing malformed
semantic payloads remain structurally valid. `git diff --check` passed. This is
a prompt clarification, not proof of improved adherence: the active `_08`
process still uses its earlier loaded prompt.

At call 158, the alternative-copy Page yielded through `local_saturation` and
returned to Search with 101 locally counted units, including the misrouted
warnings described above. Its assessment explicitly leaves independent image
verification and measurement-linked supplementary inspection unestablished.
It distinguishes regeneration from renewal, PEG molecular weight from
concentration and control conditions from acquisition conditions. Search gained
zero additional request credit; global native information remained nine and
scientific fulfillment remained zero. The repaired quote and failed-recall
diagnostics reached Search. No numerical-controller change was made.

At call 169, a later theoretical-revisit Page and Search yielded to Discovery.
The Kundu/Saha/Gangopadhyay abstract (DOI `10.1021/acs.jpcb.1c04299`) reports
retained waiting-time/mean fits but a failed constant-randomness prediction,
an off-pathway repair with qualified agreement and a continuous reaction-diffusion
alternative. These are author-reported claims from an abstract; equations,
full-text verification and the unnamed parent study's identity remain missing.
Discovery received the packet and diagnostics, earning two local request units;
global native information stayed nine and scientific fulfillment stayed zero.
Its next query targets publisher-linked original displays and methods, explicitly
recognizing that repeating OCR cannot resolve the equation discrepancy. Root
has not yet received the completed Discovery commission.

At call 187, the next verification Page yielded with no additional Search
request credit. Its reading correctly kept a bibliography title about room
temperature out of the acquisition-temperature answer. Call 188 admitted
`english_original_typography_unresolved_packet`: equation `native_match`,
numerical observation `native_no_match`, both with `native_fields_only` scope.
One first-credited source claim raised global native information to ten while
Search's own verification-request score remained zero. Scientific fulfillment
remained zero, and the original typography stayed explicitly unresolved.

By call 198, another original-article copy supplied an explicit acquisition-
medium statement, including phosphate buffer at pH 7.5, distinct from purification
and preparation conditions. This is a genuine addition to the earlier source
packet's missing-methods assessment. The reader keeps those conditions separate
from measured outcomes and preserves the damaged temperature typography. It
explicitly reports text-layer access only: original equation images and direct
supplementary contents have not been inspected. Scientific work remains pending.

The methods reader returned 83 local units at call 206; its rejected memory
reference was repaired at 205 without changing accepted extraction. Its Page
yielded at 210 and gave Search one preliminary native request slot. Subsequent
Pages added no Search request credit. At call 275, Search yielded through
`local_saturation`, returning its one-slot result and repaired-memory diagnostic
to Discovery. Discovery gained zero, retaining its two already-earned request
units. Global native information remained ten, with scientific fulfillment and
G/examination/support/change still zero. Root's Scientific handoff remains pending.

Call 276 admitted `english_mean_rate_and_turnover_comparison_bounded_packet`
without diagnostics. Its two criteria are native matches only. Three first-
credited source witnesses raised global native information to thirteen while
Discovery's local request score remained two and its action gained zero.
Scientific fulfillment and G/examination/support/change remained zero. Discovery
then selected another concentrated query for the original waiting-time formula.

A read-only audit of calls 295/296/300/302/311/313/314/315 found no false
prose novelty. Native claim/observation maps are empty in those submissions.
At 302, the child counted two units but Page counted one: the large-k3 source
unit at 3877–4005 was already counted at 300, while the Lineweaver unit at
4006–4103 was new. The earlier record's locator ended at 3974, outside this
child's baseline scope; parent deduplication still excluded it. Calls 309/310
and the repeated catalytic-site-independence unit also earned zero. These are
first-counted source occurrences, not independently new scientific conclusions.

At call 324, that Page yielded and returned zero Search request credit. Call
325 admitted `english_equation3_visual_access_unresolved_and_reset_bounds` with
the source's maximum-velocity claim and no numerical observations or scientific
links. Its receipt reports equation `native_match`, numeric `native_no_match`,
both `native_fields_only`. Global native information rose to fourteen; Search's
request score and all scientific components remained zero. Visual verification
and the reset premise remain explicitly unresolved.

At call 338, another Search returned zero additional request credit. Discovery's
call-339 bounded finding was admitted without diagnostics, then its estimator
yielded. Root received one completed Discovery outcome with gain fourteen and
the accumulated diagnostics; descendant actions were not counted as Root turns.
At call 340, Root autonomously commissioned Scientific with exact finding/source
versions. It requested a conditional stochastic reconstruction, explicit reset
premises and separate mean/marginal/serial/trace tests, while prohibiting treating
an independent derivation as authentication of damaged source typography.

At call 341, Scientific selected the real scientific-object lookup tool for
existing theory/G/bounds/check records. The action completed without diagnostics
and earned zero credit. This establishes the live Root-to-Scientific handoff and
tool selection. An actual scientific reconstruction/assessment and its return to
Root were still pending at this point; no G/examination/support/change result
had yet been claimed.

At call 342, Scientific admitted eight records without diagnostics: a precisely
scoped conditional G, explicit statistical-reset premises, one theory, public
derivation, provisional bounds, two applications and their comparison. These
earned four registration units. The comparison preserves mean-rate agreement
while identifying reported marginal/serial failures of the examined stochastic
completion, rather than rejecting the mean-rate equation or all reduction.

Call 344's attempted examination was rejected because its argument did not bind
the exact G and theoretical inputs. Call 345 corrected those references and
earned one first-G-examination unit. Call 347 executed the equation-based
binding-block protocol in two declared conditional cases. Its result is model
dependence under those inputs, not an empirical alignment or alteration result.
Call 348 retrieved the executed results, then Scientific yielded to Root with
gain five and its retained diagnostic. Global progress was nineteen: fourteen
native-information units, four registrations and one examination; support and
useful-change components remained zero.

At calls 349/350, Root used Scientific's qualified return to create a typed
follow-up request for serial-data and finite-state-simulation operands and
commissioned Discovery again. This is observed scientific feedback through Root;
it is not a claim that Scientific itself authored that goal. Unchanged forwarding
of Scientific-authored criteria is separately covered by the retained production
communication check.

## Final engineering status

Implementation and required validation are complete. The retained `_13` workflow
verification passed sixteen focused cases, `_03` real-tool verification passed
seven Lean/Mathlib/SCM checks, and failure-yield checks passed. The fresh normal
`_08` workflow autonomously completed Discovery → Root → Scientific → Root and
began a feedback-directed Discovery commission. At that initial report, the
standing workflow remained active under its own continuation controllers and
unchanged machine bounds; its later exit and replacement are recorded below.

The live derivation is not kernel verified. No empirical support or useful G
alteration is claimed from the conditional binding-block evaluation. Autonomous
selection of all three scientific children, a live Lean call and Scientific-
authored goal forwarding were not observed in this first pass; their production
paths and contracts have the retained focused validation above. The two
postlaunch prompt clarifications have not been live-tested for adherence because
`_08` retains its original loaded prompts. These are limits on the observations,
not claims of additional scientific completion.

## Credit and scientific return correction, 2026-11-09 UTC

The completed `_08` trace subsequently reached call 1008. Scientific authored
the typed `enzyme_mean_flux_case_alignment_operands` goal at call 457 and Root
forwarded it unchanged at call 459. Its criterion nevertheless asked only whether
`value` was numeric. At call 516 an explicitly excluded saturation-fit parameter
therefore earned a native request slot despite the answer reporting the required
finite-concentration endpoints unresolved. This was a defective executable
commission, not evidence that the requested measurements had been located.

Observation-value commissions now constrain one complete native observation:
kind, quantity, conditions, value and units together. The commissioning parent
defines quantity/condition vocabulary and explicit allowed kinds/units. The
validator rejects scalar value-only goals, missing circumstances and merely
optional condition selectors. Named local JSON Schema types remain supported.
Native observations declare measurement, fitted parameter, simulation input or
output, prediction, reported property or unknown provenance. Readers retain
exact source anchors; the kind does not mint another observation identity.
Unresolved findings may return no native witness and earn zero witness credit.
Both Search and Discovery use the compound predicate on the same observation;
the check certifies typed native correspondence, not empirical truth.

The old run also autonomously selected Theory Reconstruction at calls 1004–1008.
Its return crashed because it emitted `discovery_goals` absent from the parent's
declared report fields. Reconstruction, Theory Repair and Comparison now declare
that field. Format Repair retains its actual narrower return. Exact report-field
validation remains enabled. The old receipt records driver exit 2 and complete
descendant exit; the window's cgroup was empty before verification and restart.

`runs/credit_handoff_verify_20261109_02` passed all seventeen focused production
checks and exited with no descendants. The retained call-457 goal is rejected;
the excluded 730 saturation fit earns zero; wrong quantity, conditions, units or
kind fail; constraints cannot combine separate observations; a measured endpoint
matches; an empty unresolved return earns zero. Actual reports from all scientific
children pass their declared parent contracts. These are scripted verification
cases, separate from autonomous workflow evidence.

The replacement is `runs/normal_live_20261109_02`, launched fresh with the normal
generic objective, no supplied topic and no resume argument. Its own Think and
Find service is active, with this checkout as working directory and unchanged
one-CPU/5-GiB/no-swap bounds. No episode duration or usage budget was introduced.
The new run's scientific results and autonomous use of the repaired interfaces
remain to be observed.

The brief `_01` replacement was deliberately stopped after noticing that the
new commission instructions used a topic-specific measurement example. That
example is now topic-neutral. Its complete descendant exit was verified before
starting `_02`; `_02` carries no scientific state from either preceding run.
In `_02`, Root's first accepted goal is `epidemic_heterogeneity_transfer`; its
next decision commissions Discovery, which begins choosing a query. This confirms
fresh autonomous topic selection and initial routing, not downstream credit or
scientific success.

## Graduated task credit, 2026-11-10 UTC

The user authorized trialing graduated applicability-difference and alteration
credit, pushing the repository, then replacing the normal workflow. The previous
`normal_live_20261109_02` was deliberately stopped for replacement; its receipt
records complete descendant exit and the owned cgroup was empty before any
verification workflow. Its model-written enzyme timing certificates were not
treated as independently executed results.

`causal_change` now declares one original/candidate/reference task comparison
with exact scientific definitions, original expected bounds, models, accuracy
requirements and circumstances. The old credit format and support multiplier
are removed. `compare_tasks` executes explicit equations or continuous ODE event
times, retains reproducible `task_result` records and supplies actual outcomes
to Scientific's next decision. Scientific and each of its scientific children
have the same primitive tool. Root receives the compiled numerical contribution.

An identified failed original task earns one deduplicated difference unit. A
candidate's fractional error reduction earns additional gradual credit, even
when its result still misses the task tolerance. Original scope membership,
retained/lost successes and missing early crossings remain visible. Original-G
support, full repair, a reversal and unsuccessful competing alternatives are not
prerequisites. Repeated cases, aliases, numerical refinements and alternative
labels do not mint another reward. Public reasoning and empirical/source
qualifications remain distinct from executable numerical results.

SciPy 1.18.1 and NumPy 2.5.4 were installed under
`runs/task_comparison_deps_20261110_01/dependencies` through this checkout's own
launcher. `requirements-scientific.txt` declares the tested SciPy version. The
physical-time interval of a numerical experiment is not an episode runtime
budget. No research deadline, token/turn/action allowance or alternative stopping
gate was introduced; whole-workload CPU/RAM/swap bounds are unchanged.

`runs/task_credit_verify_20261110_04` passes nine focused production checks,
including real numerical enzyme comparisons, partial/broader credit, zero
original support, duplicate/precision protection, outside-scope extension,
rationalization failure, early-event nonrepresentation, forged-result rejection
and actual Scientific record admission and Root credit. The four product cases
retain A, restore B and D, and improve C without passing its 5% tolerance. This
earns three difference units and about 2.7061 improvement units. These are
conditional numerical model results, not empirical validation or rigorous ODE
error enclosures. `runs/episode_organization_verify_20261110_01` passes all
seventeen shared production checks, including access to the new tool in all four
scientific episodes. Both workflows exited completely before release work.
