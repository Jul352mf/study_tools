"""PDF ingestion and vector index creation."""

from pathlib import Path
import shutil

# Heavy imports are done inside functions to allow importing this module without
# optional dependencies.

from .utils import load_config


def extract_pages(pdf_path: Path, pages_per_group: int, overlap: int):
    import fitz  # PyMuPDF
    from llama_index.core import Document
    doc = fitz.open(pdf_path)
    for i in range(0, len(doc), pages_per_group - overlap):
        end = min(i + pages_per_group, len(doc))
        text = "\n\n".join(doc[pg].get_text() for pg in range(i, end))
        meta = {
            "file_path": str(pdf_path),
            "file_name": pdf_path.name,
            "page_start": i + 1,
            "page_end": end,
        }
        yield Document(text=text, metadata=meta)


def main():
    from llama_index.core import VectorStoreIndex, StorageContext, Document
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    cfg = load_config()
    paths = cfg["paths"]
    docs_dir = Path(paths["docs_dir"])
    chroma_dir = Path(paths["chroma_dir"])
    chunk = cfg["chunking"]

    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)

    docs = []
    for pdf in docs_dir.rglob("*.pdf"):
        docs.extend(
            extract_pages(
                pdf,
                chunk["pages_per_group"],
                chunk["page_overlap"],
            )
        )

    splitter = SentenceSplitter(
        chunk_size=chunk["chunk_size"],
        chunk_overlap=chunk["chunk_overlap"],
    )
    nodes = splitter.get_nodes_from_documents(docs)

    client = QdrantClient(path=str(chroma_dir))
    store = QdrantVectorStore(client, collection_name="study")
    storage = StorageContext.from_defaults(vector_store=store)
    VectorStoreIndex(nodes, storage_context=storage)
    storage.persist(persist_dir=str(chroma_dir))


if __name__ == "__main__":
    main()
