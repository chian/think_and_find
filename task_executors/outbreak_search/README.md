# Outbreak Search

## Calculation

Formulate scoped search requests for outbreak records that can calibrate `Cq`.
The helpers do not own credentials. A child Episode can pass in a
search callable and an optional retrieval callable
for metadata-only hits before normalizing query/result bundles. The configured
Firecrawl factories use this repository's standard-library HTTP client and
route candidate selection through its local `SearchPageProposer`. The local
Jev client measures each candidate's relevance before the configured LLM
selects the next page. Long candidate text is evaluated in conservative UTF-8
byte windows; the maximum window probability is retained for each page.

## Inputs

- `disease_name`: disease or pathogen to search for.
- `setting`: optional setting hints such as `choir`, `bus`, `restaurant`,
  `classroom`, or `cruise ship`; callers may pass one string, a sequence of
  strings, or `None`.
- `search`: callable with the shape `search(query, limit)`.
- `retrieve`: optional callable with the shape `retrieve(candidate)`.
  The search boundary still calls it when a hit has only a search-engine
  `snippet`, `description`, or `summary`; only full inline text such as
  `markdown`, `text`, `passage`, `content`, `html`, or `abstract` suppresses
  explicit retrieval. Retrieved provider metadata is merged with search-result
  metadata so a fetched body cannot erase locators such as
  `metadata.sourceURL` or `metadata.DOI`; fetched top-level URL, DOI, title,
  and source-ref values fill gaps but cannot blank out search-result
  provenance.
- `limit`: provider result limit per query.

## Outputs

- `outbreak_search_specs`: targeted queries with a `purpose` label.
- `setting_hints_text`: normalized query suffix text for missing, singular, or
  sequence-valued setting hints.
- scoped search-result bundles keyed by the query that produced them.
- `candidate_sources_from_search_records`: deduplicated source candidates with
  source references, requested-disease scope, bibliographic metadata when
  present, full retrieved text, and the query that found them. Results without
  a returned source identity such as a `source_ref`, DOI, URL, link,
  `metadata.sourceURL`, or provider ID remain in the raw search records for
  repair context but are not promoted to usable LLM extraction candidates. A
  title alone is kept as bibliographic context and is not treated as a stable
  source identity. DOI and URL locators take priority over provider IDs so
  retrievable duplicate hits merge consistently.
  Search-result `snippet`, `description`, and `summary` values are kept as
  preview text, but they are not promoted to full extraction text. Malformed
  bundle wrappers, non-sequence `results` values, and malformed result entries
  are ignored at this normalization layer so a single bad record cannot block
  other already-normalized provider hits from becoming candidates.
- `firecrawl_search_callable`: configured Firecrawl adapter for Search
  requests mediated by this repository's search episode binding.
- `firecrawl_retrieve_callable`: configured Firecrawl adapter for Scrape
  requests against hits that were not scraped inline. Firecrawl inline Search
  scrapes and
  explicit Scrape retrievals tag returned text with `retrieval.provider`,
  `retrieval.mode`, and the fetched URL when available so candidate records
  preserve the concrete text-acquisition path passed to LLM extraction.

## Scoring and yield

The searched outbreak child scores normalized candidates for:

- scoped disease and setting queries
- at least one returned candidate
- source metadata plus retrievable text
- a retrieved page body that can be sent to LLM extraction
- coverage across multiple query purposes or sources

The method-loop search episode evaluates the cumulative score after each
scoped query and yields through the shared Cq rarefaction rule when all source
criteria pass or the observed score gain saturates.

## Failure modes

Search provider exceptions become `search_error` child errors. Firecrawl
adapter failures include missing API keys, invalid local request inputs, HTTP
errors with their status code, network failures, non-UTF-8 responses,
malformed JSON, non-object API responses, provider `success=false` messages,
and invalid search result shapes. Non-Firecrawl search providers use the same
normalizer and must return an iterable of result objects.

Retrieval provider exceptions become `retrieval_error` child errors tied to
the source ref, URL, DOI, or search result that could not be fetched, including
DOI and URL values nested under provider metadata. Partial retrieval errors are
kept in `provider_errors` when other retrieved sources satisfy the search
child's score; they block the child when they prevent enough full source text
from being gathered. Candidates without retrieved text can still be retained as
repair context, but they cannot become `EvidenceSource` records for LLM
extraction.
