"""Generate Anki deck from summaries."""

import uuid
from pathlib import Path

# heavy imports in main()

from .utils import load_config


def main():
    import genanki
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    cfg = load_config()
    chroma_path = cfg["paths"]["chroma_dir"]
    client = QdrantClient(path=chroma_path)
    store = QdrantVectorStore(client, collection_name="study")
    storage = StorageContext.from_defaults(persist_dir=chroma_path, vector_store=store)
    index = load_index_from_storage(storage)
    retriever = index.as_retriever(similarity_top_k=50)

    deck = genanki.Deck(uuid.uuid4().int >> 64, "Study-Bot Deck")
    for node in index.docstore.docs.values():
        qa = retriever.query(f"Turn this into Q&A flashcards:\n\n{node.text}").response
        for line in qa.splitlines():
            if "?" in line:
                q, a = line.split("?", 1)
                note = genanki.Note(model=genanki.BASIC_MODEL, fields=[q.strip()+"?", a.strip()])
                deck.add_note(note)

    genanki.Package(deck).write_to_file("study.apkg")
    print("study.apkg ready – import into Anki")


if __name__ == "__main__":
    main()
