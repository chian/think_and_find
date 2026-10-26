# Search and whole-document reading

## Preserved acquisition paths

The migration copied the modules implementing these paths. The current discovery
binding reuses the same local Episode runtime, documents and provider adapters.
Page assessment now accepts task-specific relevance wording; its inherited
domain defaults remain available. The paths below remain present.

| Capability | Implementation | Boundary |
| --- | --- | --- |
| Strategy-generated web queries | `case_reports/research/acquisition.py::StrategySource` | Separates a literal query from its information objective |
| Web search | `task_executors/outbreak_search/firecrawl.py` | Configured Firecrawl Search endpoint, with optional inline scrape |
| Candidate relevance and page selection | `method_loop/episode_binding/web_search_binding.py` | Jev assessment and model selection among returned candidates |
| Explicit document retrieval | `firecrawl_retrieve_callable` in the Firecrawl module | Fetches source body through Scrape |
| Page reading choice | `acquisition.py::PageSource` | Chooses table, report or lexical reading for the delegated question |
| Whole-document path | `acquisition.py::ReportSource` | Successive document-order windows with retained context |
| Source-local lexical search | `acquisition.py::LexicalProbeSource`, `research/lexical.py` | BM25/context ranking over the selected source scope |
| Source-table queries | `acquisition.py::SourceTableSource` | Exact row addresses, literal whole-table queries, optional column |
| Source addresses and table parsing | `case_reports/research/documents.py` | Original character ranges, source hash and parsed Markdown tables |
| Coverage and parent returns | `case_reports/research/communication.py` | Reading ranges and structured returned information |

The legacy outbreak and assumption search compositions are also retained in
`case_reports/search_episodes.py`, `case_reports/assumption_search.py`, and their
local query/selection helpers. They use the same copied provider adapters.

## Whole-document means an iterative reading path

`ReportSource` starts from the source's unexamined ranges and advances through
successive bounded windows in document order. Reading state retains a concise
summary, unresolved links and exact-span anchors from earlier windows. This is
separate from lexical ranking and is not restricted to the opening preview.

The configured text-window size bounds one execution window, not the total
document available to the reader. Numerical continuation can still end a reading
Episode before the source is exhausted. Therefore, availability of whole-document
reading is not a guarantee that a particular run examined every character.
Report actual coverage and unexamined material.

A named section in a report question does not move its cursor. A targeted lexical
probe can locate relevant material later in the retrieved document. The source
may be revisited for a different question. Focused reexamination uses explicit
spans within the existing window bound and keeps source-version coverage distinct.

The document is the body returned by the retrieval provider. Neither successful
retrieval nor this local reading path proves that the provider returned every
page, supplement, figure or table of the publisher's original document.

## Table and lexical completeness

A table preview is not the table's entire query domain. Literal table queries
scan remaining rows, optionally within a selected column, and return bounded
sets of exact rows. Subsequent queries can continue through remaining matches.
Header context informs interpretation without fabricating a contiguous quote
from nonadjacent rows.

Lexical ranking retains the eligible spans, including the tail; it does not
itself decide when reading stops. `unexamined` and deliberate `reexamine` scopes
preserve the distinction between new material and a new interpretation of material
already supplied to extraction.

## Evidence boundary

Retain source/version/span identity, retrieval provenance, exact quotations,
reading attempts and failures. Working memory supplies context and is not itself
quoted evidence. A snippet or failed reading is not a substitute for examining
the relevant source body.

In the current discovery binding, query generation receives the question,
selected scope, recent measured query feedback and a focused inquiry's next
information need. Page reading receives its information need and a compact
source summary. Source extraction proposes causal comparisons tied to exact records;
candidate names are not seeded before acquisition. General discovery and inquiry
refinement share report, lexical and table modes. Retained source versions,
examined ranges and reading anchors survive later visits in the run. See
[DISCOVERY_PLAN.md](DISCOVERY_PLAN.md). Task-neutral extraction beyond this
biological binding remains future work.

## Configuration

Preserve the copied Firecrawl, Jev and model endpoints through
`configured_providers.py` and `.env.example`. No credentials, provider sessions,
source bodies or prior search results were copied. No live endpoint call was
needed to establish that the local implementations are present.
