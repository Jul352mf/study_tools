# Study Tools Dev Package

This `Dev/` directory houses the refactored implementation of the **Universal Study Tutor**. The old prototype remains in `messy_start/` for reference.
Course PDFs should be placed in `Dev/data/`, which is ignored by Git.

## Features
- Configurable PDF ingestion and chunking
- Async summarisation using local Mistral and OpenAI GPT‑4o
- CLI tools for building the index, chat, flashcards and maintenance
- Learning Unit JSON schema with status counters and categories
- Externalised configuration via `config.yaml`
- Course PDFs stored locally in `Dev/data/` (see `docs/MIGRATE_LARGE_FILES.md`)

## Quickstart
```bash
python -m pip install -r requirements.txt
python -m study_tools.build_index
python -m study_tools.summarize
python -m study_tools.cli_chat
```

See `docs/overview.md` for more details.
