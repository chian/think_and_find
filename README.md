# Think and Find

This repository implements one generic research task type using iterative Episodes:
discovering and assessing examples of pragmatic causal pluralism in natural phenomena.
Every Episode owns a recurring decision within an assigned question. It chooses
work, receives actual outcomes and revises its next choice. The user's original
scientific instructions define the requested result.

The scientific target is a demonstrated failure to carry a useful causal
inference G into another explanatory task, and an account of why that particular
transfer or repair fails. Questions, scales and observables may differ when their
connection through G is justified. The task does not require proving fundamental
incompatibility or the failure of all future reduction. The
[current change score covers only part of this target](docs/CAUSAL_SCORING_SHEET.md#pragmatic-transfer-failures-and-current-scoring-limits).

The implementation foundation is the existing research composition in
`case_reports/research/` and the shared `method_loop` runtime. The discarded
parallel causal-discovery implementation is deleted. The launcher now runs this causal-inference research application. The scientific
parent and Reasoning episodes assess the global scientific sheet. Theory
Reconstruction builds a typed connected account through Search Strategy, Search,
Page and source-reading episodes, with separate output and substantive-gap repairs.

## Episode design

- [Episode ownership and recurring decisions](docs/RESEARCH_EPISODES.md)
- [Reasoning, task instructions and steering](docs/REASONING.md)
- [Task authority](docs/CURRENT_GOAL.md)
- [Search and reading responsibilities](docs/SEARCH.md)
- [Typed theory reconstruction and repair](docs/THEORY_RECONSTRUCTION.md)
- [Durable state and reconstruction](docs/CHECKPOINTS.md)
- [Credit and numerical continuation](credit_assignment/README.md)
- [Executable causal scoring sheet draft](docs/CAUSAL_SCORING_SHEET.md)

The episode hierarchy narrows each child's decision problem. The parent defines
its assignment and requested information; the child iterates and returns that
information. The parent then chooses useful work from the updated state. A
primitive search, extraction, calculation or reasoning operation is an available
action within this structure.

## Source layout

| Location | Responsibility |
| --- | --- |
| `method_loop/` | Shared iteration, identity, contracts, trace and reconstruction |
| `case_reports/research/` | Existing scientific decisions, research state and acquisition episodes |
| `credit_assignment/` | Numerical projections and continuation mechanisms |
| `task_executors/` | Provider adapters and deterministic tools |
| `configured_providers.py` | Provider composition and model routing |
| `checkpoint.py`, `environment.py`, `cli.py` | Persistence, configuration and launcher |

Python imports retain the package name `episodic_workflow`. Source runs locally;
external repositories are design references, not runtime dependencies. Provider
configuration is described in `.env.example`. Credentials and research artifacts
belong outside tracked documentation.

## Run

```sh
task_run="runs/discovery-$(date -u +%Y%m%dT%H%M%SZ)"
bash tools/one_workflow.sh "$task_run" -- python3 -B __main__.py --progress \
  --checkpoint-path "$task_run/checkpoint" \
  --trace-path "$task_run/trace.jsonl" \
  --report-path "$task_run/report.json" \
  --audit-path "$task_run/audit.json"
```

Use this checkout's launcher for local Python commands. It owns one workload per
stable `CODEX_THREAD_ID`, with one CPU, 5 GiB RAM and no swap across the complete
workload. Services are named `think-and-find-workflow-…`; locks, logs, temporary
files, caches and resource receipts stay beneath this checkout's `runs/`.
It retains ownership until descendants have exited. Machine resource bounds are
separate from the application's episode continuation decisions.

Use `--root-yield-threshold 0` for a standing investigation. Discovery remains
open regardless of its VOC forecast. Scientific attempts use VOC continuation;
all other episodes retain their existing numerical yield estimators. See
[VOC measurements and limitations](docs/VALUE_OF_COMPUTATION.md).
Provider or input failures remain operational failures. Checkpoints retain the threshold.

The default objective is defined in `case_reports/research/task.py`. Optional
request text or `--request-file` can supply a scientific scope. Multiple candidate
inferences can emerge, each with its own theory applications and scoring
subsheet. There is no task selector or disease/scenario argument.

Literature assigned to the coding agent informs the implemented formalism and
rubric. It is not a runtime reading list or a prescribed scientific example.
Provider credentials and model routing are read from this checkout's `.env` and shell
environment. `--checkpoint-path` selects a new durable run; `--resume` continues
that run under its recorded request and unchanged implementation. A numerical
yield is an execution outcome, not a claim of scientific completeness.

Set `EPISODIC_LLM_PROVIDER=codex_oauth` to use the current Codex CLI ChatGPT
login, or `EPISODIC_LLM_PROVIDER=argo` to use the local Argo shim. Select the provider explicitly. Both providers can retain their own
model settings in `.env`, as shown in `.env.example`, so switching needs only
the selector:

```sh
export EPISODIC_LLM_PROVIDER=codex_oauth
# Or: export EPISODIC_LLM_PROVIDER=argo
```

`EPISODIC_LLM_ARGO_*` and `EPISODIC_LLM_CODEX_OAUTH_*` configure each provider's
`MODEL`, `FAST_MODEL`, `REASONING_EFFORT` and `FAST_REASONING_EFFORT`. Shared
`EPISODIC_LLM_*` model/effort settings take precedence when present, preserving
existing configurations and allowing shell overrides. Provider selection does
not fall back automatically if a provider fails. Firecrawl and Typesafe keep
their separate configuration.

Codex OAuth reads the access token from the CLI's login directory (`CODEX_HOME`
or `~/.codex`); no OAuth token belongs in this checkout's `.env`. The existing
adapter leaves token refresh to the CLI and reports an error if its current
token is expired. Check the login with `codex login status`. Existing checkpoints
retain their recorded provider/model configuration; use a new checkpoint when
switching providers.
