import os, chromadb, json
from tqdm import tqdm
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore

# 🔧 Settings
MODEL = os.getenv("MODEL_NAME") or json.load(open("config.json"))["model_name"]
llm = OpenAI(model=MODEL)
CHUNK_LIMIT = 999  # Change this if you want to limit # of chunks

# 🧬 Connect to vector store
db = chromadb.PersistentClient(path="chroma")
collection = db.get_or_create_collection("study")
vstore = ChromaVectorStore(chroma_collection=collection)

# 📦 Load index
storage = StorageContext.from_defaults(persist_dir="chroma", vector_store=vstore)
index = load_index_from_storage(storage)
retriever = index.as_retriever(similarity_top_k=300)
results: list[NodeWithScore] = retriever.retrieve("Summarize all content.")
all_nodes = [r.node for r in results]

# 🧠 Step 1: Summarize each chunk
partial_summaries = []
print(f"Summarizing {len(all_nodes)} chunks...")

for node in tqdm(all_nodes[:CHUNK_LIMIT]):
    prompt = f"""Summarize the following study material in 2-3 clear bullet points. Focus only on the most important knowledge someone should memorize.

---
{node.text}
---
"""
    response = llm.complete(prompt)
    partial_summaries.append(response.text.strip())

# 📘 Step 2: Fuse into a final summary
print("Fusing all summaries into a final overview...")

final_prompt = (
    "You are an expert study coach. Summarize all the key takeaways below "
    "into a single, structured list of important learnings, formulas, and facts to remember:\n\n"
    + "\n\n".join(partial_summaries)
)

final_summary = llm.complete(final_prompt).text.strip()

# 💾 Save to file
with open("summary.txt", "w", encoding="utf-8") as f:
    f.write(final_summary)

print("✅ Summary saved to summary.txt")
