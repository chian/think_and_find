# Copy and verification

## Scope

This workspace was copied from `/home/chia/repos/episodic_workflow` on 2026-10-21.
It includes the source worktree's two existing local modifications, in
`configured_providers.py` and `case_reports/research/scenario_sharing.py`.
The source checkout and `/home/chia/repos/nano-graphrag` are not edit targets.

The initial copy contains 183 files, including 157 Python files. The historical
method-loop audit, original scientific request and response-failure study were
excluded. Git metadata, local `.env`, virtual environments, caches and artifacts
were not selected. Two new generic design documents, `REASONING.md` and `SEARCH.md`,
are added to the copied documentation. Runtime Python is unchanged.

## Verification record

The final workspace contains 185 files. Nine copied documentation files were
rewritten and two design documents added. No Python implementation file changed.

| Outcome | What it means | Interpretation | Scope |
| --- | --- | --- | --- |
| No missing selected files; all 157 Python files match source SHA-256 hashes | Runtime, tests and maintenance code were preserved | The executable starting point is complete | In scope: migration |
| All 157 Python files compile in memory | No Python syntax errors were found | Static consistency, not behavioral validation | In scope: migration |
| All 101 runtime modules import under isolated Python with no site packages; all resolve inside this directory | No sibling repository or third-party package was required | The copied runtime is self-contained within this import coverage | In scope: migration |
| Strategy, search, page, report, lexical and source-table classes import; all copied Python hashes match | Every original acquisition implementation is present | Whole-document and targeted reading paths were retained; no live success is implied | In scope: migration |
| Existing launcher returns help successfully from `/tmp` under `python3 -I -S -B` | Its bootstrap works independently of the original checkout | No replacement launcher is needed for the retained CLI | In scope: migration |
| All 19 local links across 24 Markdown files resolve; the included example JSON parses | Documentation references and example structure are intact | Structural completeness, not validation of a research answer | In scope: migration |
| Source tracked-file hashes and its pre-existing worktree status are unchanged; nano-graphrag remains clean | Neither original checkout was edited | Copy isolation was preserved | In scope: migration |
| No Git worktree, credentials file, environments, caches or artifacts are present | The directory contains source and supporting files without repository history or runs | Ready for later repository initialization | In scope: migration |
| No generic reasoning run or live provider call was performed | Semantic judgment and provider integration were not exercised | No claim of generic task success or endpoint health | Out of scope: migration checks |

Checks used in-memory compilation, local import inspection, file hashes, link/JSON
validation and the help path. No test suite, calculation replay, research run or
provider request was executed. Historical source tests were copied but not run;
their names and assertions are not verification of the generic design.

Generic reasoning behavior, credit design and live provider integration remain
future work described in IMPLEMENTATION_CHECKLIST.md. Documentation changes alone
do not establish those capabilities.
