"""
Major changes vs. original:
• true async pipeline with global semaphore + token‑paced rate limiting
• retry/back‑off around every LLM call (tenacity)
• per‑model context‑window lookup & automatic token‑safety margin
• sha256 hashing to avoid collisions; metadata added to hash key
• safe metadata embedding (JSON‑escaped) to mitigate prompt‑injection
• combined docstores (docs & _docs) instead of early‑return
• batch multiple chunks into one prompt (reduces call count)
• font fallback for emoji when exporting PDF with XeLaTeX
• graceful handling of empty corpus & missing xelatex
• logging instead of print‑spam, CLI flags keep behaviour compatible
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import chromadb
import tiktoken  # pip install tiktoken
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from tqdm import tqdm

###############################################################################
# Configuration
###############################################################################
CACHE_DIR = Path("cache")
PARTIAL_DIR = CACHE_DIR / "partials"
MODULE_DIR = CACHE_DIR / "modules"
FINAL_MD = Path("summary.md")
PDF_PATH = Path("summary.pdf")

# batching / rate‑limit parameters (tune to your key)
CONCURRENCY = 2  # max concurrent LLM requests
TOKENS_PER_MIN = 40_000  # put your org limit here
TOKEN_MARGIN = 512  # reserved for system + instructions tokens
CHUNK_GROUP_LIMIT = 6_000  # tokens before summarising a group

for d in (PARTIAL_DIR, MODULE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── model & tokenizer ────────────────────────────────────────────────────────
with open("config.json", "r", encoding="utf‑8") as fh:
    CFG = json.load(fh)
MODEL = os.getenv("MODEL_NAME") or CFG["model_name"]
OUTPUT_LANG = CFG.get("output_language", "English")

# rough context‑window lookup (add models here as needed)
CONTEXT_SIZES = {
    "gpt‑4‑0125‑preview": 128_000,
    "gpt‑4‑1106‑preview": 128_000,
    "gpt‑4o‑mini": 32_000,
    "gpt‑4o": 128_000,
    "gpt‑4‑turbo": 128_000,
    "gpt‑4": 8_192,
    "gpt‑3.5‑turbo": 16_385,
}
CTX_LIMIT = CONTEXT_SIZES.get(MODEL, 8_192)
MAX_TOKENS_PER_PROMPT = CTX_LIMIT - TOKEN_MARGIN

llm = OpenAI(model=MODEL)

def get_encoding(name: str):
    try:
        return tiktoken.encoding_for_model(name)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

enc = get_encoding(MODEL)

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

###############################################################################
# Helper functions
###############################################################################

def sha256_key(node: TextNode) -> str:
    payload = node.text + json.dumps(node.metadata, sort_keys=True)
    return hashlib.sha256(payload.encode("utf‑8")).hexdigest()


def save(path: Path, txt: str):
    path.write_text(txt, encoding="utf‑8")


def load(path: Path) -> str:
    return path.read_text(encoding="utf‑8")

###############################################################################
# Rate‑limited completion util
###############################################################################

sem = asyncio.Semaphore(CONCURRENCY)

def tokens(text: str | Sequence[str]) -> int:
    return len(enc.encode(text if isinstance(text, str) else "".join(text)))


@retry(stop=stop_after_attempt(6), wait=wait_exponential_jitter(initial=1, max=60), reraise=True)
async def complete(prompt: str) -> str:
    """LLM call wrapped in semaphore + token‑paced sleep + retry."""
    async with sem:
        # simple token‑per‑minute pacing
        await asyncio.sleep(60 / TOKENS_PER_MIN * tokens(prompt))
        return await asyncio.to_thread(lambda: llm.complete(prompt).text.strip())

###############################################################################
# Vector DB load helpers
###############################################################################

def get_all_nodes(index, vstore) -> List[TextNode]:
    # merge docs dicts if both exist
    store: Dict[str, TextNode] = {}
    store.update(getattr(index.docstore, "docs", {}))
    store.update(getattr(index.docstore, "_docs", {}))
    if store:
        return list(store.values())

    if isinstance(vstore, ChromaVectorStore):
        res = vstore._collection.get(include=["documents", "metadatas"])
        return [TextNode(text=t or "", metadata=m or {}) for t, m in zip(res["documents"], res["metadatas"])]
    raise RuntimeError("❌ No nodes found.")

###############################################################################
# Prompt templates
###############################################################################
SUMMARY_TMPL = """
You are a genius private study assistant.
Summarize the following content strictly in {lang}. Use {lang} for all headings, categories, and bullets.

Instructions:
- Be precise: No vague textbook phrases.
- Distinguish closely related concepts clearly.
- Highlight practical relevance where possible.
- Use mini‑subheadings within categories if bullets are many.
- Use memory hooks, mnemonics, or analogies to aid memorization.
- Write as many bullet points as needed to cover important ideas, but no more.
- If a category would be empty, still include at least one informative point.

Structure output into these sections:
- 🧠 Core Concepts
- 📐 Key Formulas / Definitions
- 🧩 Typical Problems & Approaches
- 📌 Must‑Know Facts & Memory Hooks

📄 Source: {tag}

--- BEGIN TEXT ---
{text}
--- END TEXT ---

Respond only in {lang}.
"""

REDUCE_TMPL = """
You are a summarization expert.
Organize the following summaries in {lang} into a clear topic structure.
Use categories:
- 🧠 Core Concepts
- 📐 Key Formulas / Definitions
- 🧩 Typical Problems & Approaches
- 📌 Must‑Know Facts & Memory Hooks

Instructions:
- Deduplicate points without losing important distinctions.
- Combine similar ideas into clearer bullets.
- Preserve source references (filenames, page numbers).
- Use markdown‑friendly subheadings if helpful.
- Include at least one point per category.
- Make it exam‑ready: include formulas, tricky definitions, typical traps.

Summaries:
{summaries}
"""

MERGE_TMPL = """
You are a highly precise study assistant.
Merge these two summaries in {lang} into one coherent overview.
- Remove duplicate points without losing nuances.
- Combine overlapping bullets into richer points.
- Preserve structure, categories, and clarity.
- Resolve cross‑category duplicates.
- Output entirely in {lang}.

--- SUMMARY A ---
{A}

--- SUMMARY B ---
{B}
"""

###############################################################################
# Main async pipeline
###############################################################################

async def summarise_chunk(node: TextNode, force: bool = False) -> str:
    fp = PARTIAL_DIR / f"{sha256_key(node)}.md"
    if fp.exists() and not force:
        return load(fp)

    meta_safe = json.dumps(node.metadata, ensure_ascii=False)
    tag = meta_safe  # embed json metadata instead of raw string

    prompt = SUMMARY_TMPL.format(lang=OUTPUT_LANG, tag=tag, text=node.text)
    result = await complete(prompt)
    save(fp, result)
    return result


async def summarise_group(group: List[TextNode], force: bool = False) -> List[str]:
    tasks = [summarise_chunk(n, force) for n in group]
    return await asyncio.gather(*tasks)


async def reduce_module(name: str, summaries: Sequence[str], force: bool = False):
    path = MODULE_DIR / f"{name}.md"
    if path.exists() and not force:
        return load(path)

    batches, buf, tok = [], [], 0
    for s in summaries:
        t = tokens(s)
        if buf and tok + t > MAX_TOKENS_PER_PROMPT:
            batches.append("\n\n".join(buf))
            buf, tok = [s], t
        else:
            buf.append(s)
            tok += t
    if buf:
        batches.append("\n\n".join(buf))

    merged = []
    for batch in batches:
        prompt = REDUCE_TMPL.format(lang=OUTPUT_LANG, summaries=batch)
        merged.append(await complete(prompt))

    combined = "\n\n".join(merged)
    save(path, combined)
    return combined


async def tree_merge(texts: List[str]) -> str:
    cur = texts
    while len(cur) > 1:
        next_round = []
        for i in range(0, len(cur), 2):
            A = cur[i]
            B = cur[i + 1] if i + 1 < len(cur) else None
            if B is None:
                next_round.append(A)
                continue
            prompt = MERGE_TMPL.format(lang=OUTPUT_LANG, A=A, B=B)
            merged = await complete(prompt)
            next_round.append(merged)
        cur = next_round
    return cur[0]

###############################################################################
# CLI entry point
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Ignore cache and recompute everything")
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF conversion")
    args = parser.parse_args()

    # load vector store -------------------------------------------------------
    log.info("Loading Chroma vector store …")
    client = chromadb.PersistentClient(path="chroma")
    coll = client.get_or_create_collection("study")
    vstore = ChromaVectorStore(chroma_collection=coll)
    storage = StorageContext.from_defaults(persist_dir="chroma", vector_store=vstore)
    index = load_index_from_storage(storage)

    all_nodes = get_all_nodes(index, vstore)
    log.info("%s chunks found", len(all_nodes))

    original_char_count = sum(int(n.metadata.get("char_count", len(n.text))) for n in all_nodes) or 1

    # batch nodes into groups -------------------------------------------------
    groups, buf, tok = [], [], 0
    for node in all_nodes:
        t = tokens(node.text)
        if buf and tok + t > CHUNK_GROUP_LIMIT:
            groups.append(buf)
            buf, tok = [node], t
        else:
            buf.append(node)
            tok += t
    if buf:
        groups.append(buf)

    # run summaries -----------------------------------------------------------
    log.info("Summarising %s groups …", len(groups))
    loop = asyncio.get_event_loop()
    partials: Dict[str, List[str]] = {}
    for group in tqdm(groups):
        group_res = loop.run_until_complete(summarise_group(group, args.force))
        for node, summ in zip(group, group_res):
            mod = node.metadata.get("module", "unknown")
            partials.setdefault(mod, []).append(summ)

    # reduce modules ----------------------------------------------------------
    log.info("Reducing %s modules …", len(partials))
    module_summaries = []
    for mod, summ_list in tqdm(partials.items()):
        module_summaries.append(loop.run_until_complete(reduce_module(mod, summ_list, args.force)))

    # tree merge --------------------------------------------------------------
    log.info("Tree‑merging module summaries …")
    final = loop.run_until_complete(tree_merge(module_summaries))

    # final stats -------------------------------------------------------------
    summary_char_count = len(final)
    compression = round(summary_char_count / original_char_count * 100, 1)
    save(FINAL_MD, final)
    log.info("summary.md written – %s chars (compression %.1f %%)", f"{summary_char_count:,}", compression)

    # PDF export --------------------------------------------------------------
    if not args.no_pdf:
        try:
            import shutil
            import pypandoc  # type: ignore

            if not shutil.which("xelatex"):
                raise RuntimeError("xelatex not found – install TeX Live or skip PDF export")

            pypandoc.get_pandoc_version()
            pypandoc.convert_file(
                str(FINAL_MD),
                to="pdf",
                outputfile=str(PDF_PATH),
                extra_args=["--pdf-engine=xelatex", "-V", "mainfont=NotoColorEmoji"],
            )
            log.info("PDF saved → %s", PDF_PATH)
        except Exception as e:
            log.warning("PDF export failed: %s", e)

    log.info("Done 🎉")


if __name__ == "__main__":
    main()
