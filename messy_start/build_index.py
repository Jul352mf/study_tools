import os, shutil, chromadb, fitz
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

# ── SETTINGS ─────────────────────────────────────────────────────────────
DOCS_DIR      = Path("docs")
CHROMA_DIR    = Path("chroma")
CHUNK_SIZE    = 1024
CHUNK_OVERLAP = 128
PAGES_PER_GROUP = 2
PAGE_OVERLAP     = 1

# ── CLEAN EXISTING CHROMA ────────────────────────────────────────────────
if CHROMA_DIR.exists():
    print(f"🧹 Removing existing vector store at {CHROMA_DIR}...")
    shutil.rmtree(CHROMA_DIR)

# ── LOAD & GROUP PDF PAGES ───────────────────────────────────────────────
def extract_text_by_grouped_pages(pdf_path, pages_per_group, overlap):
    doc = fitz.open(pdf_path)
    results = []
    for i in range(0, len(doc), pages_per_group - overlap):
        end = min(i + pages_per_group, len(doc))
        text = "\n\n".join(doc[pg].get_text() for pg in range(i, end))
        meta = {
            "file_path": str(pdf_path),
            "file_name": pdf_path.name,
            "file_type": pdf_path.suffix.lower().strip("."),
            "module": pdf_path.parent.name,
            "page_start": i + 1,
            "page_end": end,
            "page_range": f"{i + 1}–{end}",
        }
        results.append(Document(text=text, metadata=meta))
    return results

print("📂 Loading and grouping documents...")
all_docs = []
for file in DOCS_DIR.rglob("*.pdf"):
    all_docs.extend(extract_text_by_grouped_pages(file, PAGES_PER_GROUP, PAGE_OVERLAP))

print(f"📄 Loaded {len(all_docs)} page-groups from {len(list(DOCS_DIR.rglob('*.pdf')))} file(s).")

# ── CHUNK DOCS BY SENTENCES ──────────────────────────────────────────────
print("✂️ Splitting into nodes (sentence-aware)...")
parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
all_nodes = parser.get_nodes_from_documents(all_docs)

# ── ENRICH CHUNKS WITH METADATA ──────────────────────────────────────────
for node in all_nodes:
    node.metadata = {
        **node.metadata,
        "char_count": len(node.text),
    }

print(f"✅ Created {len(all_nodes)} chunks.")

# ── BUILD VECTOR DB ──────────────────────────────────────────────────────
print("🧠 Building vector index...")
db         = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = db.get_or_create_collection("study")
vstore     = ChromaVectorStore(chroma_collection=collection)
storage    = StorageContext.from_defaults(vector_store=vstore)
index      = VectorStoreIndex(all_nodes, storage_context=storage)
storage.persist(persist_dir=str(CHROMA_DIR))

print(f"✅ Index built and stored in {CHROMA_DIR}/")