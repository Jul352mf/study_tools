# Overview

The Dev package implements the second iteration of the study bot based on the **Hybrid‑Edge** architecture:

- **Local tagging** with Mistral‑7B‑Instruct classifies text chunks into categories.
- **GPT‑4o/4.1** performs heavy summarisation and tutoring logic.
- **SQLite** stores metadata and Learning Units. **Qdrant** provides vector search.
- Outputs are plain JSON which are rendered to Markdown files.

Course PDFs belong in `Dev/data/` and are not tracked in Git.

Scripts read defaults from `config.yaml` so chunk sizes and model names are easily changed.
