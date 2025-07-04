"""Async summarisation pipeline."""

from __future__ import annotations

import asyncio
import json
import hashlib
from pathlib import Path
from typing import Sequence, List, Any

# optional dependencies are imported lazily

from .utils import load_config


async def _complete(llm: Any, prompt: str) -> str:
    from tenacity import retry, stop_after_attempt, wait_exponential_jitter

    @retry(stop=stop_after_attempt(6), wait=wait_exponential_jitter())
    async def _call() -> str:
        return await asyncio.to_thread(lambda: llm.complete(prompt).text.strip())

    return await _call()


def _sha(node: Any) -> str:
    payload = node.text + json.dumps(node.metadata, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarise_group(llm: Any, nodes: Sequence[Any], lang: str, cache_dir: Path) -> List[str]:
    results = []
    for node in nodes:
        key = _sha(node)
        fp = cache_dir / f"{key}.md"
        if fp.exists():
            results.append(fp.read_text(encoding="utf-8"))
            continue
        prompt = f"Summarise in {lang}:\n\n{node.text}"
        text = asyncio.run(_complete(llm, prompt))
        fp.write_text(text, encoding="utf-8")
        results.append(text)
    return results


def main():
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.core.schema import TextNode
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from tqdm import tqdm

    cfg = load_config()
    paths = cfg["paths"]
    cache_dir = Path(paths["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    llm = OpenAI(model=cfg["models"]["summarizer"])

    client = QdrantClient(path=paths["chroma_dir"])
    store = QdrantVectorStore(client, collection_name="study")
    storage = StorageContext.from_defaults(persist_dir=paths["chroma_dir"], vector_store=store)
    index = load_index_from_storage(storage)

    nodes = [n for n in index.docstore.docs.values() if isinstance(n, TextNode)]
    groups = [nodes[i:i+5] for i in range(0, len(nodes), 5)]

    summaries = []
    for g in tqdm(groups):
        summaries.extend(summarise_group(llm, g, "English", cache_dir))

    Path("summary.md").write_text("\n\n".join(summaries), encoding="utf-8")


if __name__ == "__main__":
    main()
