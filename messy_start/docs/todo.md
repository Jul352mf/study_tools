# TODO

## High Priority
- Consolidate duplicate summarisation scripts into a single module.
- Parameterise `build_index.py` with CLI options for docs path, chunk size and overlap.
- Store Learning Units as JSON objects following `schema/learning_unit.json`.
- Implement a central config loader to share model and tokenizer settings.

## Medium Priority
- Convert scripts into a CLI (`python -m study_tools <command>`).
- Add automated tests for indexing and summarisation pipelines.
- Support per-model context windows and dynamic chunk sizes.
- Persist LU status and relations (prerequisites, duplicates) in a small database.

## Suggestions from audit
- Capture extra metadata such as difficulty or exam relevance.
- Allow user-defined categories when summarising.
- Modularise PDF ingestion to support other file types.
- Remove hard coded paths (`docs/`, `chroma/`) in favour of config defaults.
