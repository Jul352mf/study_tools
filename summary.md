# Repository Audit Summary

This repo hosts a prototype study bot. Most functionality is contained in `messy_start/`.

## Working Features
- PDF ingestion and vector index building with Chroma (`build_index.py`).
- Summarisation pipeline with caching and async support (`summarize_improved.py`).
- CLI chat interface and optional quiz mode (`chat.py`).
- Simple Anki deck generation from stored chunks (`flashcards.py`).

## Broken / WIP
- Multiple summarisation scripts with overlapping logic.
- Hard coded directories and magic numbers across the codebase.
- No unified configuration or modular package structure.
- Legacy scripts (`legacy/`) are unused but still present.

## Stubbed or Unused
- Flashcard generation works but lacks integration with Learning Units.
- LU management is planned but not implemented.

## Tech Debt
- Tight coupling to local paths (`docs`, `chroma`, `cache`).
- Model names and chunk sizes baked into scripts.
- No tests or continuous integration.
