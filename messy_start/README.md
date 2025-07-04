# Study Tools

This repository contains an experimental study bot that ingests PDF course material, stores it in a vector database and generates language‑aware summaries, flashcards and chat‑style Q&A. The code base is in flux and will be refactored into a modular tutoring system.

## Current State

* `build_index.py` loads PDFs from `docs/` and creates a Chroma vector store.
* `summarize_improved.py` summarises every chunk with OpenAI models, caches the results and tree‑merges them into a `summary.md` / optional PDF.
* `chat.py` provides a CLI chat interface backed by the vector store.
* `flashcards.py` turns retrieved chunks into an Anki deck.
* Several legacy scripts exist and can be ignored during refactor.

## Setup

1. Install Python 3.11+.
2. `pip install llama-index-core llama-index-llms-openai chromadb tiktoken genanki tenacity tqdm pypandoc`
3. Export your OpenAI API key: `export OPENAI_API_KEY=sk-...`
4. Optionally set the model via environment variable or edit `config.json`.

## Usage

Build the index:

```bash
python build_index.py
```

Generate summaries (cached, async):

```bash
python summarize_improved.py --no-pdf
```

Chat with the material:

```bash
python chat.py
```

Create flashcards:

```bash
python flashcards.py
```

To ingest new PDFs, place them under `docs/` (or a sub‑folder per module) and rerun `build_index.py`.

## Architecture Overview

1. **Ingestor** – splits PDFs into overlapping text chunks and stores them in Chroma.
2. **Summariser** – summarises each chunk and reduces by module.
3. **Merger** – combines module summaries into a final exam guide.
4. **Chat/Q&A** – retrieves relevant chunks for user questions.
5. **Flashcard Builder** – converts chunk summaries into Anki cards.

The refactor aims to replace ad‑hoc scripts with reusable modules and a CLI entry point. JSON based "Learning Units" (see `agents.md`) will track progress and relations between pieces of knowledge.
