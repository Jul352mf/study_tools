from pathlib import Path
import argparse, asyncio, hashlib, json, os

import chromadb
import tiktoken  # pip install tiktoken
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import TextNode
from tqdm import tqdm
from typing import List, Dict

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

# Setup tokenizer for prompt token counting
enc = tiktoken.encoding_for_model(MODEL)

# ── HELPERS ──────────────────────────────────────────────────────────────
def md5(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def save(path: Path, txt: str):
    path.write_text(txt, encoding="utf-8")

def load(path: Path) -> str:
    return path.read_text(encoding="utf-8")

# ── LOAD VECTOR DB ───────────────────────────────────────────────────────
print("📚 Loading vector store...")
db   = chromadb.PersistentClient(path="chroma")
coll = db.get_or_create_collection("study")
vstore = ChromaVectorStore(chroma_collection=coll)
storage = StorageContext.from_defaults(persist_dir="chroma", vector_store=vstore)
index   = load_index_from_storage(storage)

# ── GET ALL CHUNKS ───────────────────────────────────────────────────────
def get_all_nodes(index, vstore):
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
    tag = f"[{meta.get('module','unk')}/{meta.get('file_type','unk')}] " \
          f"{meta.get('file_name','unk')}, page {meta.get('page','?')}"
    prompt = f"""
You are a genius private study assistant.
Summarize the following content strictly in {OUTPUT_LANG}. Use {OUTPUT_LANG} for all headings, categories, and bullets.

Instructions:
- Be precise: No vague textbook phrases.
- Distinguish closely related concepts clearly.
- Highlight practical relevance where possible.
- Use mini-subheadings within categories if the bullets are many.
- Use memory hooks, mnemonics, or analogies to aid memorization.
- Write as many bullet points as needed to cover important ideas, but no more.
- If a category would be empty, still include at least one informative point.

Structure output into these sections:
- 🧠 Core Concepts
- 📐 Key Formulas / Definitions
- 🧩 Typical Problems & Approaches
- 📌 Must-Know Facts & Memory Hooks

📄 Source: {tag}

--- BEGIN TEXT ---
{node.text}
--- END TEXT ---

Respond only in {OUTPUT_LANG}.
"""
    txt = llm.complete(prompt).text.strip()
    result = f"🔹 **{tag}**\n{txt}"
    save(fp, result)
    return result

print("📝 Summarizing all chunks...")
partials: Dict[str, list] = {}
for node in tqdm(all_nodes):
    mod = node.metadata.get("module", "unknown")
    summary = asyncio.run(summarise(node))
    partials.setdefault(mod, []).append(summary)

# ── REDUCE PER MODULE ───────────────────────────────────────────────────
print("📦 Reducing summaries by module...")
module_paths = []
for mod, summaries in tqdm(partials.items()):
    path = MODULE_DIR / f"{mod}.md"
    module_paths.append(path)
    if path.exists() and not args.force:
        continue
    # Token-aware batching
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
    # Process batches
    batch_results = []
    for batch in batches:
        prompt = f"""
You are a summarization expert.
Organize the following summaries in {OUTPUT_LANG} into a clear topic structure.
Use categories:
- 🧠 Core Concepts
- 📐 Key Formulas / Definitions
- 🧩 Typical Problems & Approaches
- 📌 Must-Know Facts & Memory Hooks

Instructions:
- Deduplicate points without losing important distinctions.
- Combine similar ideas into clearer, informative bullets.
- Preserve source references (filenames, page numbers).
- Use markdown-friendly subheadings if helpful.
- Include at least one point per category.
- Make it exam-ready: include formulas, tricky definitions, typical traps.

Summaries:
{"".join(batch)}
"""
        batch_results.append(llm.complete(prompt).text.strip())
    save(path, "\n\n".join(batch_results))

# ── TREE MERGE MODULE SUMMARIES ───────────────────────────────────────────
print("🧠 Tree-merging module summaries...")
summaries = [load(path) for path in sorted(module_paths)]
# Iteratively merge pairs
while len(summaries) > 1:
    next_round = []
    for i in range(0, len(summaries), 2):
        left = summaries[i]
        right = summaries[i+1] if i+1 < len(summaries) else None
        if right:
            prompt = f"""
You are a highly precise study assistant.
Merge these two summaries in {OUTPUT_LANG} into a single coherent overview.
- Remove duplicate points without losing nuances.
- Combine overlapping bullets into richer points.
- Preserve structure and clarity.
- Resolve cross-category duplicates by placing info in the best category.
- Output entirely in {OUTPUT_LANG}.

--- SUMMARY A ---
{left}

--- SUMMARY B ---
{right}
"""
            merged = llm.complete(prompt).text.strip()
        else:
            merged = left
        next_round.append(merged)
    summaries = next_round

final = summaries[0]
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
        pypandoc.convert_file(FINAL_MD, "pdf",
                              outputfile=str(PDF_PATH),
                              extra_args=["--pdf-engine=xelatex"])
        print(f"📄 PDF saved → {PDF_PATH}")
    except Exception as e:
        print("⚠️  PDF export failed:", e)

print("🎉 Done!")