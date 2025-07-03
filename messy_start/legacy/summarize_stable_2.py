# Not really stable it runs in an overflow error from openai
#     prompt = f"""Organize the following study summaries by topic. Use clear section headings, and preserve source references:


# This script summarizes study materials stored in a Chroma vector database using OpenAI's language model.
# It retrieves chunks of text, summarizes each chunk, and then merges the summaries into a final overview. The final summary is saved to a text file.
# This script summarizes study materials stored in a Chroma vector database using OpenAI's language model.


import os, json, chromadb
from tqdm import tqdm
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore

# ---------- SETTINGS ----------
MODEL = os.getenv("MODEL_NAME") or json.load(open("config.json"))["model_name"]
CHUNK_LIMIT = 999
llm = OpenAI(model=MODEL)

# ---------- LOAD INDEX ----------
db = chromadb.PersistentClient(path="chroma")
collection = db.get_or_create_collection("study")
vstore = ChromaVectorStore(chroma_collection=collection)
storage = StorageContext.from_defaults(persist_dir="chroma", vector_store=vstore)
index = load_index_from_storage(storage)

# ---------- FETCH CHUNKS ----------
retriever = index.as_retriever(similarity_top_k=CHUNK_LIMIT)
results: list[NodeWithScore] = retriever.retrieve("Summarize all content.")
all_nodes = [r.node for r in results]

# ---------- MAP STEP ----------
print(f"📄 Summarizing {len(all_nodes)} chunks from vector DB...")

partial_summaries = []

for node in tqdm(all_nodes):
    meta = node.metadata
    file_name = meta.get("file_name", "unknown file")
    file_type = meta.get("file_type", "unknown type")
    module = meta.get("module", "unknown module")
    page = meta.get("page", "?")

    source_tag = f"[{module}/{file_type}] {file_name}, page {page}"

    prompt = f"""You are a helpful study assistant. Summarize the content below from:
📄 {source_tag}

Provide bullet points that capture key learnings, definitions, or facts a student should memorize.

---
{node.text}
---
"""
    response = llm.complete(prompt)
    partial_summaries.append(f"🔹 **{source_tag}**\n{response.text.strip()}")

# ---------- REDUCE STEP ----------
print("🧠 Fusing all summaries into a final overview...")

# Split partial_summaries into chunks of ~25 items
CHUNK_SIZE = 25
batches = [partial_summaries[i:i+CHUNK_SIZE] for i in range(0, len(partial_summaries), CHUNK_SIZE)]
batch_summaries = []

for i, chunk in enumerate(tqdm(batches, desc="📦 Reducing batches")):
    joined = "\n\n".join(chunk)
    prompt = f"""Organize the following study summaries by topic. Use clear section headings, and preserve source references:

{joined}
"""
    response = llm.complete(prompt)
    batch_summaries.append(response.text.strip())

# Final pass
print("🧠 Final merging pass...")

final_prompt = (
    "Merge the following topic-organized summaries into one clear, non-repetitive final overview:\n\n"
    + "\n\n".join(batch_summaries)
)
final_summary = llm.complete(final_prompt).text.strip()

# ---------- SAVE ----------
with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(final_summary)

print("✅ Summary saved to summary.txt with topic grouping.")
