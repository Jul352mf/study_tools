# TODO Backlog

## P0
- Centralised configuration loader (`utils.load_config`).
- Remove hard coded paths; read from `config.yaml`.
- Store PDFs in `Dev/data/` (optionally migrate to Git LFS later).

## P1
- OCR fallback and duplicate detection during ingestion.
- Implement KnowledgeNode graph with status counters.
- Tagging pipeline using local Mistral model.
- CLI commands via `python -m study_tools <command>`.

## P2
- Evaluation harness (ROUGE-L, entity overlap, manual rubric).
- Streamlit MVP for progress view.

## P3
- Difficulty-graded exam question generator (IRT).
- Anki `*.apkg` exporter with AnkiConnect.

## P4
- Visual progress dashboard and Obsidian vault export.
