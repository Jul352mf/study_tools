from pathlib import Path
import json, os

from llama_index.core import SimpleDirectoryReader
import tiktoken                                             # pip install tiktoken

# --- read config -----------------------------------------------------------
CFG_PATH  = Path("config.json")
MODEL     = os.getenv("MODEL_NAME")
if not MODEL:
    with CFG_PATH.open() as f:
        MODEL = json.load(f).get("model_name", "gpt-3.5-turbo")

# --- ingest ---------------------------------------------------------------
docs = SimpleDirectoryReader("docs").load_data()

# pick the correct tokenizer for that model
try:
    enc = tiktoken.encoding_for_model(MODEL)
except KeyError:                    # unknown model → fall back to cl100k_base
    enc = tiktoken.get_encoding("cl100k_base")

total_tokens = sum(len(enc.encode(d.text)) for d in docs)
print(f"Loaded {len(docs)} docs, {total_tokens:,} tokens (tokenizer: {enc.name})")