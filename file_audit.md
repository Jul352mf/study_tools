# File Audit

| File | Purpose | Works | Issues | Suggestions |
|------|---------|-------|--------|-------------|
|`messy_start/build_index.py`|Create Chroma index from PDFs.|✅|Chunk size and paths hard coded. Overwrites existing index.|Expose docs path and chunk settings via CLI. Add safety check before deleting old index.|
|`messy_start/summarize_improved.py`|Async summarisation with caching and merging.|✅|Many magic numbers (rate limits, token limits). Config scattered.|Move constants to config, support per-model defaults.|
|`messy_start/summarize.py`|Older summariser; synchronous.|✅|Duplicated with improved version; lacks retries.|Deprecate in favour of the improved module.|
|`messy_start/summarize_stable_3.py`|Stable synchronous summariser.|✅|Token limit fixed at 20k; no async.|Merge features with `summarize_improved.py`.|
|`messy_start/flashcards.py`|Create Anki deck from chunks.|❓|Depends on `storage/` index that is not built by default.|Integrate with main index builder; handle Q/A parsing robustly.|
|`messy_start/chat.py`|CLI chat or quiz over vector store.|✅|Paths and system prompt hardcoded.|Wrap as module; allow custom prompts.|
|`messy_start/ingest.py`|Load docs and count tokens.|✅|Only prints stats; not integrated.|Merge into build step or remove.|
|`messy_start/reset.py`|Delete data folders for a clean start.|✅|Hard coded paths.|Parameterise paths; confirm before deleting.|
|`messy_start/utils.py`|Return tokenizer for configured model.|✅|Assumes presence of `config.json`; minimal error handling.|Use central config loader.|
|`messy_start/test.py`|Print debug info about index.|✅|Development helper; no args.|Remove or convert to unit test.|
|`messy_start/config.json`|Default model and output language.|✅|JSON only, not validated.|Extend to include chunk sizes and other defaults.|
|`messy_start/legacy/*`|Previous prototype scripts.|❌|Dead code duplicates functionality.|Archive or delete once refactor is complete.|
