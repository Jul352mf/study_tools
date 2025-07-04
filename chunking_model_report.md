# Chunking & Model Configuration Audit

## Chunking
- `build_index.py` uses `SentenceSplitter` with `CHUNK_SIZE=1024` and `CHUNK_OVERLAP=128`.
- Legacy scripts use `chunk_size=512`.
- Summarisation batches (`summarize_improved.py`) group chunks until `CHUNK_GROUP_LIMIT=6000` tokens.
- `summarize_stable_3.py` and `summarize.py` split summaries if token count exceeds `MAX_TOKENS_PER_PROMPT=20000`.
- Chunk parameters are hardcoded; no CLI option to change per run.

## Model Configuration
- `config.json` defines `model_name` and `output_language`.
- Environment variable `MODEL_NAME` overrides the model.
- `summarize_improved.py` maintains a dictionary `CONTEXT_SIZES` mapping known OpenAI models to max token windows and derives `MAX_TOKENS_PER_PROMPT` from it.
- Other scripts assume a 20k token window regardless of model.

## Issues
- Different scripts rely on different chunk sizes and token limits.
- Context size lookup is incomplete; unknown models default to 8192 tokens.
- Hard coded paths (`docs`, `chroma`) prevent running multiple datasets in parallel.

Refactoring should introduce a central configuration module that exposes chunk sizes per model and allows overriding via CLI.
