#!/usr/bin/env python
"""
End-to-end course-material summariser
------------------------------------
1. Load *all* nodes from an existing LlamaIndex + Chroma store.
2. Summarise each node (cached).
3. Batch-reduce those summaries (cached).
4. Incrementally merge batches to stay below token limits.
5. Write `summary.md`; optionally convert to `summary.pdf`.

Resumable at every stage: kill & rerun without losing progress.

Dependencies
------------
pip install llama-index-core llama-index-llms-openai \
            chromadb tqdm pydantic>2 \
            pypandoc tiktoken  # tiktoken only used for rough token counting
Install Pandoc from https://pandoc.org/install  (or skip PDF output).
"""

from __future__ import annotations
import argparse, asyncio, hashlib, json, os, sys
from pathlib import Path
from typing import List

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode
from tqdm import tqdm

# ── config ──────────────────────────────────────────────────────────────────
CACHE_DIR   = Path("cache")
PARTIAL_DIR = CACHE_DIR / "partials"
BATCH_DIR   = CACHE_DIR / "batches"
FINAL_MD    = Path("summary.md")
PDF_PATH    = Path("summary.pdf")

for d in (PARTIAL_DIR, BATCH_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL = os.getenv("MODEL_NAME") or json.load(open("config.json"))["model_name"]
llm   = OpenAI(model=MODEL)

CHUNK_SIZE  = 25         # partial summaries per batch
TOKEN_CAP   = 8_000      # soft cap for incremental merge

# ── helpers ────────────────────────────────────────────────────────────────
def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def save(path: Path, txt: str): path.write_text(txt, encoding="utf-8")
def load(path: Path) -> str:     return path.read_text(encoding="utf-8")

# ---------- enumerate ALL nodes, regardless of storage layout ------------
def get_all_nodes(index, vstore) -> List[TextNode]:
    """Return every stored TextNode, whether text is in the doc-store or Chroma."""
    # 1) try doc-store (newer/older attribute names)
    for attr in ("docs", "_docs"):
        store = getattr(index.docstore, attr, {})
        if store:
            return list(store.values())

    # 2) fall back to raw Chroma collection (store_text=True case)
    if isinstance(vstore, ChromaVectorStore):
        coll = vstore._collection
        res  = coll.get(include=["documents", "metadatas"])
        if res["documents"]:
            return [
                TextNode(text=doc or "", metadata=meta or {})
                for doc, meta in zip(res["documents"], res["metadatas"])
            ]
    raise RuntimeError("No nodes found – check that you built the index correctly.")

# ── argparse ───────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--force",      action="store_true", help="Ignore caches and recompute everything.")
ap.add_argument("--async",      dest="async_mode", action="store_true", help="Parallelise per-chunk calls.")
ap.add_argument("--no-pdf",     action="store_true", help="Skip PDF export.")
args = ap.parse_args()

# ── load index & nodes ─────────────────────────────────────────────────────
print("📚 Loading index…")
db   = chromadb.PersistentClient(path="chroma")
coll = db.get_or_create_collection("study")
vstore  = ChromaVectorStore(chroma_collection=coll)
storage = StorageContext.from_defaults(persist_dir="chroma", vector_store=vstore)
index   = load_index_from_storage(storage)

all_nodes = get_all_nodes(index, vstore)
print(f"Found {len(all_nodes)} chunks in vector DB.")

# ── map step (per-chunk summaries, cached) ─────────────────────────────────
async def summarise(node) -> str:
    fp = PARTIAL_DIR / f"{md5(node.text)}.md"
    if fp.exists() and not args.force:
        return load(fp)

    meta = node.metadata
    tag  = f"[{meta.get('module','unk')}/{meta.get('file_type','unk')}] " \
           f"{meta.get('file_name','unk')}, page {meta.get('page','?')}"

    prompt = (
        f"You are a study assistant. Summarise:\n📄 {tag}\n\n---\n"
        f"{node.text}\n---\nGive 2-3 bullet points."
    )

    if args.async_mode:
        resp = await llm.acomplete(prompt)   # async call
    else:
        resp = llm.complete(prompt)          # sync call – no await

    txt = resp.text.strip()
    doc = f"🔹 **{tag}**\n{txt}"
    save(fp, doc)
    return doc

print("📝 Generating (or loading) per-chunk summaries…")
if args.async_mode:
    partials = asyncio.run(asyncio.gather(*(summarise(n) for n in tqdm(all_nodes))))
else:
    partials = [asyncio.run(summarise(n)) for n in tqdm(all_nodes)]

# ── reduce step (topic grouping per batch, cached) ─────────────────────────
print("📦 Reducing batches…")
batches = [partials[i:i+CHUNK_SIZE] for i in range(0, len(partials), CHUNK_SIZE)]
batch_files = []
for i, chunk in enumerate(tqdm(batches)):
    bp = BATCH_DIR / f"batch_{i:03}.md"
    batch_files.append(bp)
    if bp.exists() and not args.force:
        continue
    prompt = "Organise by topic and keep source tags:\n\n" + "\n\n".join(chunk)
    save(bp, llm.complete(prompt).text.strip())

# ── incremental merge (token-safe) ─────────────────────────────────────────
if FINAL_MD.exists() and not args.force and FINAL_MD.stat().st_size > 0:
    final = load(FINAL_MD)
    print(f"📋 summary.md already exists with {len(final)} characters.")
    
else:
    print("🧠 Merging batches incrementally…")
    final = ""
    for bf in tqdm(sorted(batch_files), desc="🧠 Merging"):
        new = load(bf).strip()
        if not new:
            print(f"⚠️  Skipping empty batch file: {bf}")
            continue

        merge_prompt = f"""You are a study assistant.

Merge the new batch summary into the existing overview. Do NOT discard specific details, even if there is some overlap.

--- OVERVIEW SO FAR ---
{final or '⟪none yet⟫'}

--- NEW BATCH SUMMARY ---
{new}

Final result: A coherent study guide that includes everything above, without losing anything important.
"""
        try:
            result = llm.complete(merge_prompt).text.strip()
            if not result:
                print(f"⚠️  Empty response on merge of {bf}")
                continue
            final = result
        except Exception as e:
            print(f"❌ Error during merging {bf}: {e}")
            continue

    print(f"\n📋 Final summary length: {len(final)} characters")
    save(FINAL_MD, final)
    print("✅ summary.md written.")


# ── optional PDF export ────────────────────────────────────────────────────
if not args.no_pdf:
    try:
        import pypandoc
        pypandoc.get_pandoc_version()                       # ensure pandoc binary
        pypandoc.convert_file(FINAL_MD, "pdf",
                              outputfile=str(PDF_PATH),
                              extra_args=["--pdf-engine=xelatex"])
        print(f"📄 PDF saved → {PDF_PATH}")
    except (ImportError, OSError, RuntimeError) as e:
        print("⚠️  PDF export skipped –", e)

print("🎉 Done – open summary.md!")