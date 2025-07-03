import os, chromadb
from llama_index.core import VectorStoreIndex, StorageContext, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

# 🔁 Reset Chroma (optional for dev)
if os.path.exists("chroma"):
    import shutil
    shutil.rmtree("chroma")

# 🧩 Load and chunk with overlap
docs = SimpleDirectoryReader("docs").load_data()
parser = SentenceSplitter(chunk_size=512, chunk_overlap=128)
nodes = parser.get_nodes_from_documents(docs)

# 🔗 Set up Chroma
db = chromadb.PersistentClient(path="chroma")
collection = db.get_or_create_collection("study")
vector_store = ChromaVectorStore(chroma_collection=collection)

# 💾 Build index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context=storage_context)
storage_context.persist(persist_dir="chroma")

print(f"✅ Built index with {len(nodes)} overlapping chunks → ./chroma")
