# Scientific tool execution

Scientific, Theory Reconstruction, Theory Repair and Theory Comparison all expose
the same `scientific_tool` action. Each episode chooses a tool for its own
scientific question. The call is one ordinary selected unit in that episode's
iterator; it is not a child episode, a hidden agent loop or a required stage.

```text
Scientific or its commissioned scientific child
  → select exact theory/G/bounds/comparison records
  → formulate the lookup, equivalence or intervention question
  → execute the selected tool
  ← host-generated receipt with exact inputs, result and limitations
  → use that result in the next reconstruction, repair or comparison decision
  → return findings and specific missing-information goals to the parent
```

Reconstruction can check whether two extracted forms express the same
relationship. Repair can check whether its proposed change preserves a G and
calculate affected outcomes. Comparison can check an asserted correspondence
between accounts and compare their intervention consequences. Scientific can
make a focused call directly or commission a child for an iterative investigation.
The same scientific objects identify the task, tool inputs and returned result.

## Common action contract

`case_reports/research/scientific_tools.py:tool_schema` supplies closed JSON Schema
branches. Each request has exactly these fields:

| Field | Meaning |
|---|---|
| `action` | `scientific_tool` |
| `name` | `lookup`, `lean_equivalence`, `causal_calculation` or `compare_tasks` |
| `question` | The scientific question this call addresses |
| `inputs` | Exact `{key, version}` scientific-record references |
| `arguments` | The selected tool's concrete operands, described below |
| `rationale` | Why the episode selected this call |

The ordinary episode decision also includes its owned working assessment.
Formal calls use commissioned or selected scientific objects. Lookup is the
explicit operation for selecting an additional scientific object from the shared
store. A malformed action returns normal validator/iterator feedback; it does
not authorize an unsupported scientific conclusion.

## Scientific-record lookup

`lookup.arguments` contains:

- `keys`: exact `{key, version}` references to retrieve;
- `kinds`: scientific record kinds to select;
- `query`: a case-insensitive substring query over scientific keys and content.

All three fields are present; unused arrays and the query may be empty. The
supported store includes theories, Gs, expected generalizability bounds,
applications, comparisons, repairs, scientific reasoning, calculation protocols
and results, discovery goals/findings/checks and formal-tool receipts. Raw
`evidence` records are excluded: this is not source-text retrieval or literature
search.

The receipt contains matched identities. Current matches become the invoking
episode's selected scientific keys; historical matches become its selected
snapshots. The next ordinary decision receives those selected records. The
receipt does not copy entire reconstruction graphs into another record.

## Lean equivalence

`lean_equivalence.arguments` contains:

| Field | Type and interpretation |
|---|---|
| `relation` | `equal` for equality of expressions or `iff` for equivalence of propositions |
| `parameters` | List of `{name, type, source, selector}` formal parameters |
| `assumptions` | List of `{name, proposition, source, selector}` explicit hypotheses |
| `left`, `right` | The two Lean terms/propositions to compare |
| `proof` | Proposed ordinary tactic proof, without a surrounding declaration |

Each `source` is an exact scientific input reference. Its `selector` is a
nonempty list of keys or numeric list-index strings resolving inside that
record's `data`. The receipt retains the selected scientific field alongside
the Lean parameter or assumption. Parameter names are explicit distinct
identifiers; accepted scalar/function types use `Nat`, `Int`, `Rat`, `Bool`,
`Prop` and `Real` or their supported notation. Real-number/library tactics
require the Mathlib-enabled backend.

The adapter, rather than the model, constructs the declaration:

```lean
theorem scientific_equivalence
    (parameters_and_source_linked_hypotheses) :
    (left) = (right) := by
  proposed_tactic_proof
```

For `iff`, the generated target uses `↔` instead of `=`. For example, selected
real-valued quantities `x` and `y` could be used to ask whether `2 * x + y` equals
`y + x * 2`, with `ring` as the proposed tactic. The scientific reason for asking
must come from the reconstruction, repair or comparison; the algebraic example
does not itself establish a biological or causal finding.

The generated file is executed by the actual Lean executable. The adapter checks
the process result and the axiom report for the generated theorem. It rejects
`sorry`/`admit`, additional declarations or axioms, arbitrary metaprogramming and
unchecked compiler-evaluation shortcuts. The requested equivalence cannot itself
be supplied as an assumption. Permitted foundational axioms are `propext`,
`Quot.sound` and `Classical.choice`.

The result is `verified_equivalent`, `unresolved`, `unavailable` or `error`.
A failed proof is unresolved; it is not a proof of inequivalence. Unsupported
formal types/tactics or an unavailable library are reported explicitly.

The scope of success is the exact encoded statement under its explicit
assumptions. Lean does not establish that the encoding faithfully captures a
paper, that two prose theories are universally equivalent, or that the physical
premises hold. Those source-to-formal bindings are retained scientific
interpretations, available to subsequent reconstruction and comparison.

## SCM, equations and discrete simulation

`causal_calculation.arguments` contains:

- `assessment`: an admitted `importance` protocol key;
- `case_ids`: the declared circumstances to execute;
- `evaluation_context`: a scoped historical support/change context, or `""`.

The request's exact inputs must include that protocol. Its existing closed
protocol specifies the method (`scm`, `equations` or `simulation`), named input
values, structural equations, focal relationship, replacement, requested answer
and circumstance measure. The tool executes the original and altered models for
each requested circumstance through the production evaluator and retains the
ordinary `importance_result` records.

The current SCM evaluator is deterministic and acyclic. It does not silently
solve cyclic feedback, implicit equilibria or a continuous-time causal process.
Discrete simulation uses the explicitly declared update equations and initial
state. Failed/unsupported cases remain in the returned results with their actual
errors. Status is `computed`, `partial` or `unresolved` according to the returned
case results; execution success is not empirical validation.

## Task comparison and gradual credit

`compare_tasks.arguments` contains `assessment` (a declared `causal_change` key)
and `case_ids` (the selected predeclared circumstances). Exact inputs include the
comparison record. Its public argument connects G, the original expected
generalizability bounds and the scientific sources of all three models.

The comparison fixes original/candidate/reference models, task quantity and
units, absolute/relative accuracy requirements, cases, original scope membership
and limitations. Explicit equation models supply `equations` and `answer`.
ODE models supply derivative `equations`, `initial`, a directional `event`,
physical experiment `time` start/end expressions and `solver` rtol/atol. The
closed expression language supports arithmetic, comparisons and common scalar
functions such as `sqrt`, `log`, `exp` and `tanh`; it executes no supplied code.

SciPy DOP853 integrates ODE events twice at increasing precision. Refinement
differences are numerical uncertainty estimates, not rigorous enclosures.
Physical-time experiment intervals describe the scientific model, not episode
runtime or stopping policy. Initial states beyond a requested crossing remain
explicitly unrepresented. Missing events within the interval are unresolved.

The host retains reproducible `task_result` records and calculates applicability
differences and gradual error improvements. It reports retained successes,
restorations, improvements still missing tolerance, losses and nonrepresentations.
Extensions outside original bounds can earn credit; original-G support, a full
repair and failed competing alternatives are not prerequisites. See the exact
formula and deduplication in [the scoring specification](CAUSAL_SCORING_SHEET.md#graduated-task-differences-and-useful-alterations).

Install `requirements-scientific.txt` in the workflow's Python environment for
ODE comparisons. Missing dependencies return `unavailable` to the invoking
episode. Dependency installation and scientific artifacts belong under `runs/`.

## Receipts and their use

Valid tool calls return a host-generated `tool_receipt` record with fields:

```text
tool, question, inputs, request, status, result, artifacts, limitations
```

Receipts are not model-authored scientific records. They retain exact scientific
snapshots and expose the result through the invoking episode's normal return and
next-decision feedback. Each generated receipt becomes a selected scientific
record, so the next decision receives its contents through the ordinary selected
sheet inputs. It is not inserted into the separate physical-metric calculation
table. Repeating an identical result reuses its semantic receipt identity. A tool
invocation alone supplies no scientific support/change credit.

For Lean, retained data include the exact proposition, source bindings, generated
source SHA-256, backend configuration, process return code and actual axiom
report. Receipt validation reconstructs and checks that source hash and statement
without rerunning Lean. For causal calculations, validation checks the returned
cases against their retained execution snapshots.

Artifacts are written beneath the launcher-owned run directory:

```text
runs/<run>/scientific-tools/<invocation>/
  request.json
  result.json
  Equivalence.lean    # Lean calls
  lean.stdout        # Executed Lean calls
  lean.stderr        # Executed Lean calls
```

## Backend installation and configuration

The adapter reads `THINK_AND_FIND_LEAN_CONFIG`, if supplied, or
`runs/scientific_tools/lean.json`. Otherwise it can use `lean` on PATH with `Std`.
The configuration fields are `executable`, `imports`, `lean_path`, and optional
`version` and `toolchain_origin`. Configured imports are explicit `Std`/`Mathlib`
modules. The installed binary, dependency sources and caches remain under
`runs/`, not the user's global Lean/cache directories.

After confirming this window's previous complete workload has exited, install
the pinned official Lean toolchain and selected Mathlib modules in a fresh run:

```sh
task_run="runs/lean-install-$(date -u +%Y%m%dT%H%M%SZ)"
bash tools/one_workflow.sh "$task_run" -- \
  bash tools/install_scientific_lean.sh "$task_run" --with-mathlib
```

The installer verifies the Lean archive digest and writes the local backend
configuration. It does not create scientific results or resume an old research
case. Focused verification, after the installation workload fully exits, is:

```sh
task_run="runs/scientific-tools-verification-$(date -u +%Y%m%dT%H%M%SZ)"
bash tools/one_workflow.sh "$task_run" -- \
  /usr/bin/python3 -B scripts/verify_scientific_tools.py --run-dir "$task_run"
```

The tools introduce no elapsed-time, token, action-count or heartbeat quota.
Their processes and descendants run inside the existing one-CPU, 5-GiB-RAM,
no-swap workload boundary. Every returned outcome enters the invoking episode's
ordinary iterator; formal-tool failure does not terminate that episode.

Lean's proof-term/kernel mechanism is described in the
[Lean tactic-proof reference](https://lean-lang.org/doc/reference/latest/Tactic-Proofs/).
The pinned toolchain comes from the
[Lean 4.35.0 release](https://github.com/leanprover/lean4/releases/tag/v4.35.0),
with the corresponding Mathlib tag and retained installation commit.
