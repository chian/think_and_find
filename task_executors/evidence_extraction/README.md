# Evidence Extraction

This folder holds the Argo-backed LLM extraction adapter used by the workflow.
The adapter calls this repository's `ArgoClient`; source text is converted into
JSON by that LLM and is then validated by `case_reports.run_llm_field_extraction`.

## Inputs

- an `ArgoClient` with an explicit endpoint, model, and optional credential
- the workflow extraction request, including `source.text`, `source.source_ref`,
  required fields, the field-specific instruction, and whether assumption
  admission requires a similarity basis

## Outputs

The adapter returns the model JSON object directly:

- `values`: numeric field records with raw and normalized values, units,
  `source_ref`, exact `evidence_span`, confidence, and notes.
- `similarity_basis`: present only for searched assumption admission.
- `similarity_evidence_span`: exact source text supporting the assumption
  similarity basis.

## Scoring and yield

The adapter itself does not assign credit. `run_llm_field_extraction` validates
the returned object, rejects fields whose confidence is below the configured
extraction threshold, and produces the scored `ExtractionReport`. A child
evidence episode yields once a candidate source passes or once every candidate
source has failed.

