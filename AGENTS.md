# Episode ownership

The user's original instructions and later corrections govern this work.

Run Think and Find from this repository's root using its own
`tools/one_workflow.sh` for every local Python command. In this workspace the
launcher is `/home/chia/repos/think_and_find/tools/one_workflow.sh`; its resource
supervisor is this repository's `tools/workflow_resources.py`. This application
has its own launcher and no longer launches through `gene_annotation_episode`.
Do not reuse that repository's launcher, working directory, supervisor or run
directory, even when restarting a historical job that used them. Disposable
output belongs beneath this checkout's `runs/`. Verify the launched service is
named `think-and-find-workflow-<window-key>.service` and its working directory is
this checkout before reporting a successful start.

For a fresh job, select a new directory beneath this checkout's `runs/`, pass
that directory's checkpoint/trace/report/audit paths, and omit `--resume`.
Use the stable `CODEX_THREAD_ID`; confirm the previous complete workload
has exited before launching another in this window. Keep one CPU, 5 GiB RAM and
no swap across the complete workload, without process/thread count ceilings.

Keep instructions local to their subject. On 2026-11-04 the user distinguished
machine resource bounds from episode stopping budgets: no episode elapsed-time,
token, turn, action-count or cumulative-usage budgets; retain sane compute bounds.
The environment constrains CPU and RAM across the complete workload.
An environmental CPU bound does not authorize any additional application budget.
Do not add episode deadline/usage-budget knobs or enforcement, including a
runtime-maximum setting with an unlimited value. API calls have their own request
timeouts; LLM calls use 300 seconds. A failed or timed-out call returns its error
to the episode. It does not terminate the episode. The yield estimator owns
episode continuation. Retain per-window ownership.

The current scientific target is a demonstrated failure of a specified causal
representation to transfer usefully between explanatory tasks, and why it fails
under the conditions examined. Do not require fundamental incompatibility or
proof that every reduction must fail. Tasks can differ in question, observable,
organism, scale or grain; justify their connection through G. Distinguish an
observed failure, an explanation of it and the consequences of examined repairs.
A useful omission or reformulation need not reverse a sign. Preserve successful
repairs and explained failures even without a replacement. The current numerical
change calculator covers a narrower relation-change/restored-prediction result;
its zero score is not a refutation of other pragmatic failures. See the current
user clarification in docs/CURRENT_GOAL.md and limits in docs/CAUSAL_SCORING_SHEET.md.

Use the user's term "expected generalizability bounds" exactly. Distinguish those
bounds from the circumstance domain and measure of a particular calculation.
Their purpose is to expose and explain broken expectations, not establish an
absolutely correct domain. Ground expectations in literature or reasoning across
literature when possible; the user explicitly permits provisional judgment in
proposing and revising these bounds when that grounding is unavailable. This
exception does not permit subjective numerical grading elsewhere. Revise bounds
as new hypotheses emerge, retaining what was expected, what broke it, what
changed, and why. Distinguish better definitions or a missing dependency within
a general rule set from a magic fudge factor or special rule outside that set.
Do not narrow expected bounds silently after finding a failure. Model-authored valid/invalid verdicts must never gate credit,
refute operands, or stand in for executable verification. Numerical scores
must be calculated from explicit scientific operands, not assigned ordinal
ratings or verbal anchors. "Distinct theoretical uses" is not a user-defined
scoring dimension. Label assistant proposals as proposals, not requirements.

Instructions to the coding agent to read methodological literature or design the
scoring rubric are development instructions, not runtime research assignments.
Translate that work into explicit criteria and contracts. Do not copy the
development reading list, conversation, or design tasks into a research request or
child commission. Runtime requests investigate the user's natural-phenomenon
causal inferences; methodological references are not prescribed search targets.

The application implements one generic task type: discovering and assessing
examples of pragmatic causal pluralism across natural phenomena. Individual
candidate inferences are research objects, not alternative task modes and not
a predetermined single case for the entire application.

Every episode owns a specific set of iterative decisions, with its own complete
instructions, numerical credit projection, input scope and return contract.
Never switch one generic episode between scientific parent, search and reasoning
roles. The scientific parent commissions Theory Reconstruction or Reasoning.
Theory Reconstruction owns a typed connected theory fragment and can commission
Search Strategy, a focused source reread, Reasoning, or investigation of a
recorded substantive gap. Search Strategy translates acquired source information
into that same vocabulary. Typed-output repair and substantive-gap investigation
have separate child episodes; the latter can acquire additional literature.
There is no generic Investigation wrapper and no search-enabled switch. Sharing the
Episode loop and primitive sheet/calculation tools does not share decision ownership.

Every Episode is a decision point that iterates. Identify its recurring
question, available choices, received information and how outcomes can change
its next choice. Parent-child relationships constrain scope and available
choices. Runtime order emerges from episode decisions.

Use the existing common Episode machinery. Adapt this one research application
within its episode structure. Do not create an alternative task path, task
selector, parallel binding or fixed acquisition/reasoning itinerary. Primitive
operations remain tools selected within an episode. Do not wrap a single model
call in an episode merely to give it a stage name, or hide an independent agent
loop inside a tool.

Parents commission a specific job, scope, constraints and typed return form.
Children own their repeated choices. Every call receives complete instructions
for its actual job and concise steering selected from relevant outcomes.
Parent histories, private reasoning and recursive traces are not child input.
Child returns contain the requested information with qualifications and
provenance; the parent uses it in its next decision. No extra report composer.

The episode selects work. The configured numerical controller owns continuation.
Record actual outcomes before projecting their contribution. One completed own
unit produces one local observation. A child's internal turns do not become
parent turns, and child scores are not summed as parent progress. Distinguish
investigation progress, support for a particular conclusion and completion.
Useful refutations and corrections can advance the investigation.

Theory exploration distinguishes a substantive new theory from additional
contexts or evidence for an existing theory. Retain possible causal reversals
without prematurely declaring them established. Children may perform reasoning
without search. Public derivations, their premises and independent verification
when commissioned are scientific sheet inputs. Scoring sheets do not prescribe
the episode hierarchy or a fixed sequence of scientific actions.

Use the user's recovered original scientific instructions when defining the
causal-inference task. Do not reconstruct them from removed implementations or
substitute a repository summary for their authority. Scoring rules and graders
need their own meanings and operands; they do not prescribe the research order.

Read docs/RESEARCH_EPISODES.md and docs/REASONING.md. Inspect each affected
producer and consumer through decisions, returned information, state, credit
and continuation. Documentation is not proof of implemented behavior. Do not
run unit tests or research jobs unless the user requests them.
