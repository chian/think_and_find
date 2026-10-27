# Think and Find

A standalone starting point for episodic research that uses evidence and reasoning
together to answer a question. Reasoning should identify what to search for next;
new evidence should be able to change the reasoning and the eventual conclusion.

This repository began as a source copy of `episodic_workflow`, without
credentials, environments, caches, run outputs, or historical experiment reports.
The original domain application remains available. A subsequent task-specific
pragmatic causal-pluralism pilot adds a research binding and extends the existing
launcher and provider composition. The generic task interface and judgment
records are still work to implement. Renaming the directory does not implement
that transition.

## Start here

- [Task and scope](docs/CURRENT_GOAL.md)
- [Episode ownership](docs/RESEARCH_EPISODES.md)
- [Reasoning and judgment design](docs/REASONING.md)
- [Argument progress and Episode communication plan](docs/ARGUMENT_PROGRESS_PLAN.md)
- [Exploration credit audit](docs/EXPLORATION_AUDIT.md)
- [Cycle-by-cycle behavior trace](docs/BEHAVIOR_TRACE_2026-10-27.md)
- [Generalizable inference credit](docs/GENERALIZABLE_INFERENCE_CREDIT.md)
- [Historical same-system rule](docs/SAME_SYSTEM_SCOPE.md)
- [Credit repair plan and live checks](docs/ITERATIVE_CREDIT_PLAN.md)
- [Question-driven discovery and current verification](docs/DISCOVERY_PLAN.md)
- [Search and whole-document reading](docs/SEARCH.md)
- [Implementation work](docs/IMPLEMENTATION_CHECKLIST.md)
- [Copy and verification](docs/WORKFLOW_VERIFICATION.md)
- [Checkpoint boundaries](docs/CHECKPOINTS.md)

## Preserved foundation

The copy includes the local `method_loop` runtime, acquisition and reading
bindings, provider clients, evidence records, calculation tools, checkpointing,
source tests and maintenance scripts. Python runtime imports use only the local
source tree and standard library. No sibling repository is required.

Search includes literal web queries, candidate relevance assessment and selection,
full-body retrieval, document-order report windows, source-local lexical ranking,
and table queries. Whole-document reading is a distinct iterative path, not a
synonym for selecting a few ranked passages. See [the search inventory](docs/SEARCH.md).

The existing workflow also has hypothesis, evidence-review and contextual
assessment operations. The generic design must examine and reuse these where
appropriate; it should not assume the only existing reasoning is arithmetic.

## Providers and local configuration

The copied provider composition retains Firecrawl Search/Scrape, TypeSafe Jev
relevance assessment, and both configured model transports: the local Argo shim
and Codex-login OAuth. The template defaults to the Argo shim at
`http://127.0.0.1:12329/v1`; start it with
`/home/chia/.local/bin/argo-shim --no-auth` before running a workflow. Endpoint
and model settings remain configurable. The client sends the Argo account name
in the request body and no local authentication header. The shim handles its
own upstream headers. Only the template [.env.example](.env.example) is
included. Shell configuration or a new local `.env` supplies credentials when a
run is separately requested.

The current configuration names retain the `EPISODIC_` prefix. The internal Python
package name remains `episodic_workflow` so the copied absolute imports and existing
entry point continue to work. These are implementation compatibility details,
not a requirement for the future public API.

| Configuration | Purpose |
| --- | --- |
| `FIRECRAWL_API_KEY`, `FIRECRAWL_BASE_URL` | Web search and document retrieval |
| `TYPESAFE_API_KEY`, `TYPESAFE_ENDPOINT`, `TYPESAFE_MODEL` | Relevance assessment |
| `EPISODIC_LLM_PROVIDER` | Existing `codex_oauth` or `argo` transport |
| `EPISODIC_LLM_MODEL`, `EPISODIC_LLM_FAST_MODEL` | Configured reasoning and fast tiers |
| `EPISODIC_LLM_REASONING_EFFORT`, `EPISODIC_LLM_FAST_REASONING_EFFORT` | Transport-supported reasoning effort |
| `EPISODIC_LLM_BASE_URL`, `EPISODIC_LLM_USER` | Optional Argo endpoint configuration |
| `EPISODIC_REQUEST_TIMEOUT_S` | Provider timeout |

## Existing entry point

The biological discovery path starts from the question and an empty
collection. It seeks source-grounded causal inferences with stated domains,
the distinct theories they support, and warranted targets where transport and
plausible repairs lose useful predictions. Cases arise during source reading and synthesis;
the developing cross-case account now informs later general searches. The root
alternates general discovery and inquiry refinement
through numerical scheduling and uses literal zero expected marginal yield as
its stop condition. Its current positive-prior exponential estimate remains
positive after any finite sequence of misses, so automatic numerical closure
requires a different justified yield estimator. It has no
scientific admission gates. See [the discovery design and live registration](docs/DISCOVERY_PLAN.md).
This task-specific implementation does not establish that ten strong cases have
been collected or complete the generic interface.

Earlier fixed-case calibration exposed admission, reassessment and duplicate
credit defects. Its [repair record](docs/ITERATIVE_CREDIT_PLAN.md) is retained as
historical evidence; it must not be mistaken for discovery performance.

From this directory, inspect the copied launcher without starting research:

```sh
python3 -I -S -B . --help
```

The retained domain path accepts a disease and scenarios. The biological frame
path retains the compatibility flag `--task causal-reversals`; an optional `--request-file` supplies its
question, and `--target-count` sets its reporting milestone (default 10).
The root sums credit across distinct conceptual themes without a collection capacity;
biological cases in one theme keep separate evidence and answers, and their
attainment occupies one theme-level dominated volume;
the milestone is not a stopping rule. This remains a
biological task-specific rubric, not a task-neutral interface. The existing
launcher serves both paths.

Case reviews now store an explicit generalizable rule and its proposed domain,
distinct supported theories with formal additions, an independent target
theory, and concrete repair tests. The rule's warranted domain determines
whether cross-organism tests are relevant; an organism change alone earns no
failure credit. Each inference step names its premises, so a weak required step
limits its conclusion. Case credit includes bounded credit for distinct
supported theories and a larger, evidence-bottlenecked failure bonus when
repairs lose reusable predictive or explanatory value.
Collection-account credit derives from the progress of its cited reviewed
cases: an account can summarize partial progress, but it cannot earn reversal
credit when those cases have no reversal witness. Reviewer prose ratings remain
visible without becoming independent evidence of progress.
The research process chooses what to investigate next; there is no required
number or order of steps. Full arguments remain in durable state, while
cross-case reasoning receives compact dependency paths. See
[the inference credit design](docs/GENERALIZABLE_INFERENCE_CREDIT.md).

### Causal inference, theory, and scope

Keep three objects distinct. A **named biological causal inference** `G` is a
scoped claim about how a change in a biological cause affects a biological
outcome, with the relevant mechanism or relations stated. A **hypothesis or
theory** `H` combines `G` with other causal links or candidate mechanisms to
explain a larger phenomenon. A **theory prediction** states what `H` predicts
and what part of that prediction depends on `G`. There is no required number
of links or fixed order for constructing `H`.

Evidence that `G` still works in another organism, setting, or scale is
**scope evidence**. A warranted similarity deserves positive credit because
it establishes more of `G`'s generalizability, even when it does not create a
new theory prediction. A warranted reversal deserves more credit when `G`
loses useful predictive or explanatory work for a target theory and plausible
repairs fail to preserve its earlier uses. A change of organism or model size
alone is neither a reversal nor a reason to discard valid scope evidence.

A model or algorithm can help formulate and test `G`, but changing an input to
an algorithm and observing a different computed output is not, by itself, a
biological causal inference. The claim must say what would change in the
biological system and why the model's variables and operations warrant that
claim. In the current cFBA record, applying the same optimization method to a
genome-scale network is genuine evidence about the stated `G`'s scope and
deserves positive scope credit. The problem is that this `G` describes an
algorithm's predictive capability rather than a biological causal inference;
the scale result alone does not show that a biological intervention causes the
predicted community outcome. Fix the choice and warrant of `G`, not the credit
for warranted similarity.

Discovery therefore searches for biological causes, contrasts, and outcomes
in living systems, including laboratory and engineered systems. Modeling papers
remain useful when they connect their variables and predictions to those
systems. A result confined to the behavior of a solver is a method fact or
search lead, not the causal phenomenon the task seeks.

Model requests now have explicit size bounds. Identity and synthesis calls
select relevant whole inquiry records under separate budgets, and theory
measurement receives a compact prior account. Full case histories and theory
arguments still need more general addressable retrieval as they grow. The
current positive-prior root stop rule also needs repair for finite automatic
completion.
The calculation kernels and their references remain available as optional tools
for the eventual generic workflow.

## Layout

| Location | Responsibility |
| --- | --- |
| `method_loop/` | Episode execution, identity, contracts, archive and reconstruction |
| `case_reports/research/` | Existing research state, hypotheses, acquisition, reading and tools |
| `case_reports/` | Retained domain composition and source-admission helpers |
| `credit_assignment/` | Existing progress and numerical yield policies |
| `task_executors/` | HTTP/model/search adapters and deterministic tools |
| `configured_providers.py` | Provider composition and model routing |
| `reversal_prompts.py` | Decision-specific field selection, temporary reference labels and rendered model instructions |
| `causal_argument.py` | Causal argument validation, compact handoffs and derived progress |
| `checkpoint.py`, `environment.py`, `cli.py`, `__main__.py` | Persistence, local configuration and existing entry point |
| `tests/`, `scripts/` | Copied source checks and maintenance utilities |
| `scenarios/` | Example input used by the retained domain workflow |

The included tests check source behavior and the new argument-credit ordering
rules. Live discovery runs are recorded separately and do not establish a
scientifically validated reversal or a generic reasoning interface.
