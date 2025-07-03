from llama_index.core import StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb, json, os

db   = chromadb.PersistentClient(path="chroma")
coll = db.get_or_create_collection("study")
index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="chroma",
                                 vector_store=ChromaVectorStore(coll)))
print("index type:", type(index))
print("index.ref_doc_info keys →", list(index.ref_doc_info.keys())[:5])