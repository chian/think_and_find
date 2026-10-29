# Episode ownership

Every Episode owns a specific set of iterative decisions. Its instructions,
credit projection and message contract belong to that responsibility. There is
no generic Investigation episode, search-enabled flag, or parent/child role flag.

```text
Scientific parent: which scientific uncertainty deserves work next? ↻
├── Search Strategy: which query/approach should address this information need? ↻
│   └── Search: which source from ONE query should be examined next? ↻
│       └── Page: where/how should this source be examined next? ↻
│           ├── Report: what does the next document-order window establish? ↻
│           ├── Lexical reading: what does the next ranked passage establish? ↻
│           └── Table: which row query should resolve the remaining question? ↻
└── Reasoning: which derivation, comparison, countermodel or check resolves this question? ↻
```

Edges declare allowed commissions, not a sequence. The scientific parent may
repeatedly choose either branch. Reasoning has fixed sheet operations: record,
inspect, calculate and evaluate. First-principles deductions, repair arguments
and critiques are logical assignments with those same decisions;
no assignment enables search or changes the episode's action set. Calculators
remain tools. The common `method_loop` owns iteration and numerical continuation.

Search Strategy returns source findings, exact source/record references, their
bearing on the assigned question, unresolved reasoning questions, searches and
coverage. The parent uses that information to formulate a specific Reasoning
commission. Reasoning returns public arguments, premises, conclusions,
limitations, critiques, calculated sheet changes and unresolved
premises. Full derivations remain addressable by record key and version.
Neither return contains a recursive transcript or invokes a report composer.

Search returns its literal query and examined sources. Page returns its source
version and remaining source coverage. Report returns document position and
unresolved links. Lexical reading returns query/ranking position and unresolved
links. Table returns row-query outcomes and remaining rows. Each also returns
bounded findings, coverage and failures relevant to its parent's next decision.
The commissioned G and expected generalizability bounds travel as explicitly
selected records or historical snapshots, not copied parent context.

The scientific parent retains the executed support-for-G and scoped
proportional-change score. Reasoning observes changes in that scientific credit
and coverage of explicitly assigned importance, bounds, support or repair
calculations. A resolved negative result can complete an assigned calculation.
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
Readers match returned source-native quantity, units and numeric conditions,
execute the matching fixed case and compile actual support during acquisition.
This is deterministic calculation, with no hidden agent or numerical grade.
Broad findings without a commissioned formalization await the parent's assessment. Later assessments can revalue the global ledger; closed child
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
