# Think and Find

This repository implements one generic research task type using iterative Episodes:
discovering and assessing examples of pragmatic causal pluralism in natural phenomena.
Every Episode owns a recurring decision within an assigned question. It chooses
work, receives actual outcomes and revises its next choice. The user's original
scientific instructions define the requested result.

The implementation foundation is the existing research composition in
`case_reports/research/` and the shared `method_loop` runtime. The discarded
parallel causal-discovery implementation is deleted. The launcher now runs this causal-inference research application. The scientific
parent and investigations populate and calculate the global sheet; acquisition
uses the existing Search Strategy/Search/Page/reading hierarchy.

## Episode design

- [Episode ownership and recurring decisions](docs/RESEARCH_EPISODES.md)
- [Reasoning, task instructions and steering](docs/REASONING.md)
- [Task authority](docs/CURRENT_GOAL.md)
- [Search and reading responsibilities](docs/SEARCH.md)
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
python3 __main__.py --progress
```

Use `--root-yield-threshold 0` for a standing investigation. The estimator
yields only when nonnegative recent gain is strictly below its threshold, so
zero makes that cutoff unreachable. Child thresholds retain their normal values.
Provider or input failures remain operational failures. Checkpoints retain the threshold.

The default objective is defined in `case_reports/research/task.py`. Optional
request text or `--request-file` can supply a scientific scope. Multiple candidate
inferences can emerge, each with its own theory applications and scoring
subsheet. There is no task selector or disease/scenario argument.

Literature assigned to the coding agent informs the implemented formalism and
rubric. It is not a runtime reading list or a prescribed scientific example. Argo is the default model provider through the local shim. Codex OAuth requires
an explicit provider selection. Provider credentials and model routing are read from this checkout's `.env` and shell
environment. `--checkpoint-path` selects a new durable run; `--resume` continues
that run under its recorded request and compatible implementation. A numerical
yield is an execution outcome, not a claim of scientific completeness.
