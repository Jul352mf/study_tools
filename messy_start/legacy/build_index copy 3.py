import os, shutil, chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

# ── SETTINGS ─────────────────────────────────────────────────────────────
DOCS_DIR    = Path("docs")
CHROMA_DIR  = Path("chroma")
CHUNK_SIZE  = 512
CHUNK_OVERLAP = 128

# ── CLEAN EXISTING CHROMA ────────────────────────────────────────────────
if CHROMA_DIR.exists():
    print(f"🧹 Removing existing vector store at {CHROMA_DIR}...")
    shutil.rmtree(CHROMA_DIR)

# ── LOAD DOCS ────────────────────────────────────────────────────────────
print("📂 Loading documents...")
docs = SimpleDirectoryReader(input_dir=DOCS_DIR, recursive=True).load_data()
print(f"📄 Loaded {len(docs)} document(s).")
for doc in docs:
    print("•", doc.metadata.get("file_path", "??"))

# ── SPLIT + ENRICH CHUNKS ────────────────────────────────────────────────
parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
all_nodes = []

for doc in docs:
    file_path = Path(doc.metadata.get("file_path", "unknown"))
    file_name = file_path.name
    file_type = file_name.split(".")[0].lower()
    parent_folder = file_path.parent.name

    chunks = parser.get_nodes_from_documents([doc])
    for c in chunks:
        c.metadata = {
            "page": doc.metadata.get("page_label", "?"),
            "file_name": file_name,
            "file_type": file_type,
            "module": parent_folder,
            "char_count": len(c.text),
        }
    all_nodes.extend(chunks)

print(f"✂️ Split into {len(all_nodes)} chunks.")
token_counts = [int(n.metadata.get("char_count", len(n.text))) for n in all_nodes]
print(f"📊 Avg chunk size: {sum(token_counts)//len(token_counts)} chars")


# ── BUILD VECTOR DB ──────────────────────────────────────────────────────
print("🧠 Building vector index...")
db         = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = db.get_or_create_collection("study")
vstore     = ChromaVectorStore(chroma_collection=collection)
storage    = StorageContext.from_defaults(vector_store=vstore)
index      = VectorStoreIndex(all_nodes, storage_context=storage)
storage.persist(persist_dir=str(CHROMA_DIR))

print(f"✅ Index built and stored in {CHROMA_DIR}/")
