"""CLI chat interface."""

import argparse
from pathlib import Path

# heavy imports done in main()

from .utils import load_config


def main():
    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    cfg = load_config()
    llm = OpenAI(model=cfg["models"]["summarizer"])
    chroma_path = cfg["paths"]["chroma_dir"]
    client = QdrantClient(path=chroma_path)
    store = QdrantVectorStore(client, collection_name="study")
    storage = StorageContext.from_defaults(persist_dir=chroma_path, vector_store=store)
    index = load_index_from_storage(storage)
    engine = index.as_chat_engine(chat_mode="condense_question", llm=llm, verbose=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*")
    args = parser.parse_args()

    if args.question:
        q = " ".join(args.question)
        print(engine.chat(q).response)
    else:
        print("Ask questions (blank to exit)")
        while True:
            q = input("? ")
            if not q.strip():
                break
            print(engine.chat(q).response)


if __name__ == "__main__":
    main()
