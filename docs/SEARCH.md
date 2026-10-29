# Search and reading Episodes

The literature hierarchy delegates increasingly focused decision problems.
Strategy, Search, Page and each reader retain their own iteration and local
continuation. A literal provider query is separate from the information question
that gives the search its purpose.

| Episode | Recurring choice and relevant feedback |
| --- | --- |
| Strategy | Propose a useful query for the delegated uncertainty, using earlier searches and returned information. |
| Search | Select another returned source using the question, relevance and previous page outcomes. |
| Page | Select table investigation, ordered reading or targeted passage work using source structure, coverage and returned findings. |
| Table | Choose exact rows or a literal query over remaining rows, using prior query outcomes. |
| Ordered reading | Examine further document windows with retained context, unresolved links and exact anchors. |
| Targeted passage investigation | Examine source-local passages relevant to its question, retaining scope and coverage. |

The existing bindings are in `case_reports/research/acquisition.py`; document
addresses and tables are in `documents.py`, lexical ranking in `lexical.py`, and
coverage projections in `communication.py`. `SearchPageProposer` supplies page
assessment and selection. Firecrawl adapters provide search and retrieval.

A table preview does not restrict queries to those rows. The table episode can
query remaining rows repeatedly, including matches outside the preview. Each
selected reading window retains header context and exact source addresses.

Ordered reading advances through document windows. A window limit bounds one
input, not the whole document. Targeted passage work can examine material later
in the source or deliberately reexamine a passage for a changed question. Report
actual examined ranges and remaining material. Numerical yielding does not
establish whole-document coverage or absence of evidence.

Preserve source versions, exact quotations, addresses, retrieval provenance and
failed reading attempts. Reading memory supplies working context; it is not an
independent source. Retrieval success does not establish that the provider
returned every supplement, figure or table.

These are responsibilities and existing source locations. Their inherited
domain extraction contracts still require adaptation to the scientific task.
