import os, chromadb
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

# Clean existing vector store
if os.path.exists("chroma"):
    import shutil
    shutil.rmtree("chroma")

# Load from nested folder like docs/Module1/
DOCS_DIR = "docs"
docs = SimpleDirectoryReader(input_dir=DOCS_DIR, recursive=True).load_data()

# Setup parser
parser = SentenceSplitter(chunk_size=512, chunk_overlap=128)
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

# Chroma vector DB
db = chromadb.PersistentClient(path="chroma")
collection = db.get_or_create_collection("study")
vstore = ChromaVectorStore(chroma_collection=collection)

# Build index
storage_context = StorageContext.from_defaults(vector_store=vstore)
index = VectorStoreIndex(all_nodes, storage_context=storage_context)
storage_context.persist(persist_dir="chroma")

print(f"✅ Built index with {len(all_nodes)} chunks from folder → ./chroma")
