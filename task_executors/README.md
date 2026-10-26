# Provider adapters and optional tools

This package contains both acquisition adapters and deterministic domain tools.
All modules are retained so the copied workflow and its source checks keep their
local dependencies. Runtime code uses the Python standard library.

| Location | Role |
| --- | --- |
| `_http.py` | Shared HTTP requests and provider errors |
| `outbreak_search/` | Firecrawl Search/Scrape, Jev assessment, result normalization and legacy query helpers |
| `evidence_extraction/` | Model transports and extraction contracts |
| `scenario_metrics/` | Existing domain calculation catalog, dependencies and evaluation |
| Remaining tool directories | Numerical kernels and their API references |

The generic research design should retain search, retrieval and model transport
as capabilities. Existing numerical kernels can remain optional tools when a
question requires arithmetic. Their SI units, formulas and domain references
remain documented in the tool subdirectories; they do not define the result
contract for every future task.

Reasoning and judgment require their own explicit inputs, premises and result
semantics. They are future implementation work, not an automatic renaming of a
calculator. See [REASONING.md](../docs/REASONING.md) and
[the complete search inventory](../docs/SEARCH.md).
