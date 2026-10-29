# Provider adapters and optional tools

This package contains both acquisition adapters and deterministic domain tools.
The owning Episode chooses when a capability serves its assigned question. Runtime code uses the Python standard library.

| Location | Role |
| --- | --- |
| `_http.py` | Shared HTTP requests and provider errors |
| `outbreak_search/` | Firecrawl Search/Scrape, Jev assessment, result normalization and legacy query helpers |
| `evidence_extraction/` | Model transports and extraction contracts |
| `scenario_metrics/` | Existing domain calculation catalog, dependencies and evaluation |
| Remaining tool directories | Numerical kernels and their API references |

Search, retrieval and model transport are capabilities available to episode
decisions. Existing numerical kernels can remain optional tools when a
question requires arithmetic. Their SI units, formulas and domain references
remain documented in the tool subdirectories; they do not define the result
contract for every future task.

Reasoning and judgment require their own explicit inputs, premises and result
semantics. They are future implementation work, not an automatic renaming of a
calculator. See [REASONING.md](../docs/REASONING.md) and
[the complete search inventory](../docs/SEARCH.md).
