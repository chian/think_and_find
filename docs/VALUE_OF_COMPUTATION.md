# Scientific value of computation

DiscoverySource and ScientificSource use `ScientificVOCController`. All other
episode responsibilities retain `ResearchController` and nano-graphrag's
`joint_preferential_discovery_v1`, including Reasoning and every literature,
reconstruction, gap and repair episode. Scientific continuation actually obeys
VOC; there is no rarefaction fallback in that controller.

`credit_assignment/value_of_computation.py` is an unchanged copy of
`chian/millenial_biology` at commit
`fac93f19efa38dbb3c0605c2c08c470466a56c28`. The source receipt next to it records
the SHA-256. The reusable module supplies posterior-predictive adaptive
finite-lookahead forecasts. The application-specific state, measurement,
transition model, feedback and continuation binding is in
`case_reports/research/computation_yield.py`.

## Measurements and ownership

The objective is the existing **sum of five deduplicated scientific score
components**, compiled by `compile_episode_points`. It is never the best score
among candidates. Forecasts compare terminal scientific stocks, so revisiting a
credited artifact cannot harvest its score again. Zero and negative changes
remain observations. The scientific attempt's displayed score is the signed
change from its initial stock; Discovery displays the global stock.

The binding measures the full sheet directly. It does not infer signed changes
from the older positive-only attempt-credit projection, invent an intermediate
learning reward, or use a model's numerical judgment as credit. Admitted records
and checked calculation outputs remain state information even at zero credit.
Admission establishes structure and provenance, not scientific truth.

One completed own action creates one observation, attributed to the selected
action and its actual unit label. One completed child creates one parent
observation. Child internal units and accumulated scores are never added to
parent observations. Completed failures retain their failure status and
diagnostics while contributing their measured realized change, normally zero.
An unmodeled forecast is different: it remains `None` and adds no observation.

Before an action, the controller freezes its starting state and measured stock.
Shared refresh updates visible stock but neither changes that baseline nor
creates a transition. The completed parent observes the whole signed change
once, even if a child made it visible earlier. Corrections remain in the same
history rather than resetting or discarding negative outcomes.

History retains produced and newly exposed record versions, checked-calculation
identities, supplied inputs, diagnostics, action identities and score changes.
`depends_on` links supplied record/check references to the actions that exposed
them. These are provenance/availability links, not proof of scientific causality.
Zero-credit preparation can lead to a state in which a follow-up has measured
value; the adaptive forecast recognizes that path without awarding extra points.

## Forecasts and continuation

At each next decision, the correct parent's `value_of_computation` input includes
per-action immediate, continuation and total expected gains, predictive outcome
dispersion, improvement probability, evidence IDs, unmodeled choices, model
assumptions and its own concise signed credit history. This is decision input,
not just an audit log. The existing scientific and discovery choices and their
contracts are unchanged. Discovery forecast labels distinguish fresh starts
from revisits selected through its existing question/input contract.
When forecasts use observations from earlier attempts of the same responsibility,
`forecast_evidence` supplies their relevant signed action histories alongside the
evidence IDs. Earlier attempts do not become additional current-parent turns.

Scientific yields only when all examined decisions in the forecast are modeled
and no available action has positive expected gain within the lookahead.
Unknown choices or follow-ups keep it open. This is a conditional model decision,
not a claim of scientific completion or exhaustive search. Discovery's zero
root threshold keeps the standing investigation open even when all modeled
gains are zero. Forecast depth (currently three adaptive steps) never limits
actual execution; it is not a turn or action allowance.

## Assumptions and limitations

Models start without invented priors. Exact-state observations take precedence.
Otherwise the binding pools transitions within the same responsibility, available
action set, admitted record kinds/currentness, selected input kinds, theory
component/gap kinds, exact checked-calculation output, and nonzero scientific
component flags. Actual state identities also retain exact record versions,
commission, selections, snapshots, owned assessment, prior discovery starts and
recalled/calculated information.

Structural transfer assumes exchangeable action outcomes under those features
and transfers **signed additive score changes**. It does not establish that two
scientific contents, questions or candidates are interchangeable. A transferred
outcome predicting a negative total is unknown rather than clipped or discarded
from a renormalized distribution. Action families are coarse: differing record
targets or reasoning commissions can behave differently within a family.

Predictions use only observed outcome support: the zero-pseudocount categorical
posterior-predictive limit. Hypothetical observations update branch-local counts,
never actual history. Unseen successes and failures are not ruled out. Outcome
standard deviation is not a confidence interval on the mean. No calibration on
live scientific research is claimed by the deterministic checks.

Cold-start choices and new structural/checked states remain unknown, so an
attempt may continue indefinitely if it keeps selecting a known action while
leaving alternatives unmodeled. Exact checked-output matching can also make
transfer sparse. Models are shared between attempts of the same responsibility
within a run; Discovery and Scientific have separate models. A new run starts
with no learned transitions. Checkpoint reconstruction replays actual completed
units; changed implementation does not silently resume an old checkpoint.

No finite-target model is included. A request to repair something is not a
machine-checked completion obligation.

## Deterministic verification

```sh
task_run="runs/voc-verification-$(date -u +%Y%m%dT%H%M%SZ)"
bash tools/one_workflow.sh "$task_run" -- python3 -B \
  scripts/verify_computation_yield.py --run-dir "$task_run"
```

This uses finite fixtures and the production Episode, source, sheet, controller
and parent-input paths, without provider calls or a full research investigation.
`scripts/verify_failed_yield.py` separately verifies the retained child failure
behavior. The run directory retains named check results and the actual trace.
