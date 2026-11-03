# Episode ownership

Every Episode owns a specific set of iterative decisions. Its instructions,
credit projection and message contract belong to that responsibility. There is
no generic Investigation episode, search-enabled flag, or parent/child role flag.

```text
Discovery root: which fresh scientific attempt should run next? ↻
└── Scientific attempt: which scientific uncertainty deserves work next? ↻
    ├── Search Strategy: which query/approach should address this information need? ↻
    │   └── Search: which source from ONE query should be examined next? ↻
    │       └── Page: what should be read or scientifically assessed in this source next? ↻
    │           ├── Report: what does the next document-order window establish? ↻
    │           ├── Lexical reading: what does the next ranked passage establish? ↻
    │           └── Table: which row query should resolve the remaining question? ↻
    └── Reasoning: which derivation, comparison, countermodel or check resolves this question? ↻
```

Edges declare allowed commissions, not a sequence. The discovery root retains
the shared scientific sheet and starts another attempt after one yields. Each
attempt has its own estimator and may repeatedly choose either scientific branch.
The root's own threshold may be zero so it remains open across attempts; this
does not disable an attempt's estimator. An attempt's starting question is a
seed for the generic objective, not a separate task mode or a prescribed case.
Reasoning has fixed sheet operations: record,
inspect, calculate and evaluate. First-principles deductions, repair arguments
and critiques are logical assignments with those same decisions;
no assignment enables search or changes the episode's action set. Calculators
remain tools. The common `method_loop` owns iteration and numerical continuation.
The scoped input and historical snapshot records are already supplied in each
Reasoning decision. The episode retains every tool choice; feedback from its
completed actions tells it whether inspection exposed anything new.

Search Strategy returns source findings, exact source/record references, their
bearing on the assigned question, unresolved reasoning questions, searches and
coverage. The parent uses that information to formulate a specific Reasoning
commission. Reasoning returns public arguments, premises, conclusions,
limitations, critiques, calculated sheet changes and unresolved
premises. Full derivations remain addressable by record key and version.
Neither return contains a recursive transcript or invokes a report composer.

Every decision response also replaces that episode's working assessment of its
completed work. Each concrete source declares its own assessment questions:
Page answers the source question and reports assessment status; Search compares
sources from one query; Search Strategy answers the delegated information need
and identifies remaining reasoning questions; Reasoning states its argument,
premises and limitations; Scientific states its causal assessment and remaining
uncertainties. Readers update their source-grounded reading memory during
extraction and select prior evidence still needed for their answer.

The next decision receives the current assessment and results not yet incorporated
into it. A child report is delivered once after a successful assessment update;
it is not also rebuilt from the entire ledger under `own_results`. If an update
is malformed, the previous assessment and unreviewed results remain available;
that transport failure does not invalidate scientific operands or gate credit.
The controller may yield immediately after any unit. The outward return therefore
includes both the owned assessment and actual final results not yet reviewed.
Final child outcomes are flattened into factual packets, not recursively nested
episode histories. There is no separate final report-composition call.

Each return includes an immutable `details_ref`. `recall` is a read-only primitive
selected in the same episode loop: it retrieves a job's record/receipt index,
an exact scientific record version, or an actual grading receipt. Search and
Search Strategy have this primitive as well as scientific episodes. Retrieval
earns zero points. Complete records and receipts remain in the ledger; prompts
contain the episode's chosen assessment and relevant results. Assessment and
reference selection has no character or item quota. Score components are calculated
from the complete scientific ledger, independently of summary selection.

Readers return explicit source-linked grading needs and a concise synthesis to Page.
Page can select record, inspect, evaluate, calculate or a fixed observation comparison
in its existing loop, alongside further reading. Report, Lexical reading and Table
cannot invoke graders or register scientific operands. Grading flags earn zero points
and neither require an action order nor certify a finding. Page tool outcomes record
actual calculated results, exact evidence/record references, score-component changes
and unresolved diagnostics. Search and Search Strategy receive relevant needs and actual outcomes
in the owned return and unreviewed final results. Their next decisions can use the result
without waiting for the scientific parent. Repeated actions remain available.

Search returns its literal query and examined sources. Page returns its source
version and remaining source coverage. Report returns document position and
unresolved links. Lexical reading returns query/ranking position and unresolved
links. Table returns row-query outcomes and remaining rows. Each also returns
selected findings, coverage and failures relevant to its parent's next decision.
The commissioned G and expected generalizability bounds travel as explicitly
selected records or historical snapshots, not copied parent context.

The discovery root deduplicates global points across attempts. A scientific
attempt projects only point changes since its starting sheet. The five numeric
score components are G registration (+1), expected generalizability bounds
registration (+1), theory/context application registration (+1), support for G
(+0.1 per qualifying support unit), and causal change (support for original G
multiplied by the qualifying restored original-scope fraction). Every component
is shown separately; the total is their sum. Repeated scored identities add 0.
Extracting a source claim or observation, writing a support argument, and
executing calculations each earn 0 points. Calculation results are scientific
inputs to support and change assessments; neither the shared score nor a
Reasoning episode rewards the fraction of calculations completed.
The calculators establish consequences within the supplied formal models and
source mappings; model verdicts have no numerical effect.

A sourced theoretical statement can be recorded as `source_formalization`, linked
to its evidence and one G importance protocol. It uses that protocol's method
(equations, acyclic SCM or discrete simulation), named circumstance inputs and
output quantity/units. The support assessment compares the executed source and
G models, including their declared common G alteration, within the tested
expected generalizability bounds. Conflicting or unresolved cases remain in the
calculation. A matching source formalization can support a dependent hypothesis;
an unformalized theoretical statement cannot. This comparison checks consequences
of authored formalizations, not fidelity of the source-to-model derivation or
agreement over unexamined circumstances. Empirical observations are kept as
typed measured targets rather than manufactured causal models.

Literature roles project scientific contributions attributable to their returned
evidence, with scope determined by their actual responsibility:

| Episode | Attribution scope |
| --- | --- |
| Search Strategy | Evidence returned across its commissioned searches |
| Search | Sources selected from its single query |
| Page | Its source |
| Report | Its document-order reading scope |
| Lexical reading | Its selected source-local ranked passages |
| Table | Its table rows |

Each subtracts inherited scientific points before attributing new gain. Query
activity, source count, passage count, characters and row count earn no points.
Reading coverage remains available to guide further choices. For a fixed
observation need, the parent can commission a support_assessment template.
Page can select that template and source evidence to match native quantity, units
and numeric conditions, execute the fixed case and compile actual support. Readers
only extract and flag; no automatic reader assessment remains. Page can also record
a new source-grounded G, expected generalizability bounds or application and formal
comparison operands. Questions it cannot settle return upward with the actual outcomes. Later assessments can revalue the global ledger; closed child
controllers are not retroactively reopened. No separate learning or retrospective
credit redistribution algorithm is implemented.

One completed own unit supplies one observation at that scope. Child turns do
not become parent turns. The parent recomputes its own credit from returned state;
it never sums child scores. Shared refresh creates no extra observation.
Numerical yielding, physical exhaustion, execution failure and scientific
completion remain distinct. Reasoning does not auto-return after one changed row.

Bindings live in `workflow.py`, `acquisition.py`, `episode_contracts.py` and
`episode_credit.py` under `case_reports/research`. `engine.py` shares transport,
ledger merging, trace handling and the existing Episode loop, not role selection.
