"""
summarize.py  –  language-aware, structured, module-based summarizer

What it does:
1. Load all nodes from Chroma + LlamaIndex storage.
2. Summarize each node in its original language with structured bullets (cached).
3. Group summaries by `module` and reduce each module into its own summary.
4. Merge modules into a final topic-organized summary.
5. Track compression ratio.
6. Optionally convert to PDF.

Run with:
    python summarize.py --no-pdf

Install dependencies:
    pip install llama-index-core llama-index-llms-openai chromadb tqdm pypandoc tiktoken
"""

from __future__ import annotations
import argparse, asyncio, hashlib, json, os
from pathlib import Path
from typing import List, Dict
import tiktoken

import chromadb
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode
from tqdm import tqdm

# ── CONFIG ───────────────────────────────────────────────────────────────
CACHE_DIR   = Path("cache")
PARTIAL_DIR = CACHE_DIR / "partials"
MODULE_DIR  = CACHE_DIR / "modules"
FINAL_MD    = Path("summary.md")
PDF_PATH    = Path("summary.pdf")
MAX_TOKENS_PER_PROMPT = 20000  # threshold to split on

for d in (PARTIAL_DIR, MODULE_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODEL = os.getenv("MODEL_NAME") or json.load(open("config.json"))["model_name"]
llm   = OpenAI(model=MODEL)

CFG = json.load(open("config.json"))
OUTPUT_LANG = CFG.get("output_language", "English")

enc = tiktoken.encoding_for_model(MODEL)

# ── HELPERS ──────────────────────────────────────────────────────────────
def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def save(path: Path, txt: str): path.write_text(txt, encoding="utf-8")
def load(path: Path) -> str:     return path.read_text(encoding="utf-8")

# ── LOAD VECTOR DB ───────────────────────────────────────────────────────
print("📚 Loading vector store...")
db   = chromadb.PersistentClient(path="chroma")
coll = db.get_or_create_collection("study")
vstore = ChromaVectorStore(chroma_collection=coll)
storage = StorageContext.from_defaults(persist_dir="chroma", vector_store=vstore)
index   = load_index_from_storage(storage)

# ── GET ALL CHUNKS ───────────────────────────────────────────────────────
def get_all_nodes(index, vstore) -> List[TextNode]:
    for attr in ("docs", "_docs"):
        store = getattr(index.docstore, attr, {})
        if store:
            return list(store.values())
    if isinstance(vstore, ChromaVectorStore):
        coll = vstore._collection
        res  = coll.get(include=["documents", "metadatas"])
        return [TextNode(text=d or "", metadata=m or {}) for d, m in zip(res["documents"], res["metadatas"])]
    raise RuntimeError("❌ No nodes found.")

all_nodes = get_all_nodes(index, vstore)
print(f"Found {len(all_nodes)} chunks.")
original_char_count = sum(int(n.metadata.get("char_count", len(n.text))) for n in all_nodes)

# ── ARGPARSE ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--force",      action="store_true")
parser.add_argument("--no-pdf",     action="store_true")
args = parser.parse_args()

# ── PER-CHUNK SUMMARIES ──────────────────────────────────────────────────
async def summarise(node) -> str:
    fp = PARTIAL_DIR / f"{md5(node.text)}.md"
    if fp.exists() and not args.force:
        return load(fp)

    meta = node.metadata
    tag = f"[{meta.get('module','unk')}/{meta.get('file_type','unk')}] {meta.get('file_name','unk')}, page {meta.get('page','?')}"

    prompt = f"""
You are a genius private study assistant. 
Summarize the following content strictly in {OUTPUT_LANG}. Use {OUTPUT_LANG} for all headings, categories, and bullets.

Instructions:
- Be precise: No vague textbook phrases.
- Distinguish closely related concepts clearly (e.g., "Zuschlagskalkulation" vs. "differenzierende Zuschlagskalkulation").
- Highlight practical relevance where possible ("used in practice to...").
- Use mini-subheadings within categories if the bullets are many.
- Use memory hooks, mnemonics, or analogies to aid memorization.
- Write as many bullet points as needed to cover important ideas, but no more.
- If a category would be empty, still include at least one informative point for it.

Structure your output into exactly these sections:
- 🧠 Core Concepts
- 📐 Key Formulas / Definitions
- 🧩 Typical Problems & Approaches
- 📌 Must-Know Facts & Memory Hooks

📄 Source: {tag}

--- BEGIN TEXT ---
{node.text}
--- END TEXT ---

❗❗❗ Respond only in {OUTPUT_LANG} — every part of your output must be in {OUTPUT_LANG}, including headings and bullets. Do not respond in English.
"""
    txt = llm.complete(prompt).text.strip()
    result = f"🔹 **{tag}**\n{txt}"
    save(fp, result)
    return result

print("📝 Summarizing all chunks...")
partials: Dict[str, List[str]] = {}
for node in tqdm(all_nodes):
    mod = node.metadata.get("module", "unknown")
    summary = asyncio.run(summarise(node))
    partials.setdefault(mod, []).append(summary)

# ── REDUCE PER MODULE ────────────────────────────────────────────────────
print("📦 Reducing summaries by module...")
module_paths = []
for mod, summaries in tqdm(partials.items()):
    path = MODULE_DIR / f"{mod}.md"
    module_paths.append(path)
    if path.exists() and not args.force:
        continue
    # Split summaries into batches under token limit
    batches, current, count = [], [], 0
    for s in summaries:
        tok = len(enc.encode(s))
        if current and count + tok > MAX_TOKENS_PER_PROMPT:
            batches.append(current)
            current, count = [s], tok
        else:
            current.append(s)
            count += tok
    if current:
        batches.append(current)
    # Process each batch and combine results
    batch_results = []
    for batch in batches:
        prompt = f"""
You are a summarization expert.

Organize the following summaries into a clear topic structure in {OUTPUT_LANG}.
Use the four categories: 
- 🧠 Core Concepts
- 📐 Key Formulas / Definitions
- 🧩 Typical Problems & Approaches
- 📌 Must-Know Facts & Memory Hooks

Instructions:
- Deduplicate points without losing important distinctions.
- Combine similar ideas into clearer, more informative bullets.
- Preserve original source references (e.g., filenames, page numbers) where possible.
- Use markdown-friendly subheadings inside categories if helpful.
- Each category must contain at least one point, even if inferred.
- Make it exam-ready: include formulas, tricky definitions, and typical traps.
- Final output must be written in {OUTPUT_LANG} — do not switch languages.

Summaries:
{"".join(batch)}
"""
        result = llm.complete(prompt).text.strip()
        batch_results.append(result)
    module_summary = "\n\n".join(batch_results)
    save(path, module_summary)

# ── MERGE MODULE SUMMARIES ───────────────────────────────────────────────
print("🧠 Merging module summaries...")
final = ""
for path in tqdm(sorted(module_paths)):
    new = load(path).strip()
    if not new:
        print(f"⚠️  Empty: {path}")
        continue
    prompt = f"""
You are a highly precise study assistant.

Merge the following two summaries into a single coherent overview.
- Remove duplicate points without losing important nuances.
- If two bullets overlap, combine them into a richer, more complete point.
- Preserve structure, categories, and clarity.
- Wenn Informationen in mehreren Kategorien vorkommen, prüfe ob sie dort notwendig sind – wenn nicht, verweise lieber oder integriere sie dort, wo sie besser passen.
- Maintain full output in {OUTPUT_LANG}. Every heading and bullet must be in {OUTPUT_LANG}.

--- OVERVIEW SO FAR ---
{final or '⟪none yet⟫'}

--- NEW MODULE SUMMARY ---
{new}
"""
    final = llm.complete(prompt).text.strip()

summary_char_count = len(final)
compression = round(summary_char_count / original_char_count * 100, 1)
save(FINAL_MD, final)

print(f"✅ summary.md written – {summary_char_count:,} characters")
print(f"📉 Compression: from {original_char_count:,} to {summary_char_count:,} chars ({compression}%)")

# ── PDF EXPORT ───────────────────────────────────────────────────────────
if not args.no_pdf:
    try:
        import pypandoc
        pypandoc.get_pandoc_version()
        pypandoc.convert_file(FINAL_MD, "pdf", outputfile=str(PDF_PATH), extra_args=["--pdf-engine=xelatex"])
        print(f"📄 PDF saved → {PDF_PATH}")
    except Exception as e:
        print("⚠️  PDF export failed:", e)

print("🎉 Done!")