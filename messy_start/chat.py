import os
import json
from pathlib import Path
from llama_index.core import (
    StorageContext, VectorStoreIndex, load_index_from_storage
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.schema import NodeWithScore
import chromadb
import argparse
import datetime

# === Load config ===
CONFIG_PATH = "config.json"
MODEL = os.getenv("MODEL_NAME") or json.load(open(CONFIG_PATH))["model_name"]

# === Set up LLM ===
llm = OpenAI(model=MODEL)

# === Reconnect to the Chroma collection ===
CHROMA_PATH = "chroma"
COLLECTION_NAME = "study"
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=collection)

# === Load the storage context ===
storage = StorageContext.from_defaults(persist_dir=CHROMA_PATH, vector_store=vector_store)
index = load_index_from_storage(storage)

# === Set up chat engine with source tracking ===
chat_engine = index.as_chat_engine(
    chat_mode="condense_question",
    llm=llm,
    verbose=True,
    system_prompt="You are a helpful assistant. Always cite your sources by filename or page when possible."
)

# === CLI Arguments ===
parser = argparse.ArgumentParser(description="StudyBot CLI")
parser.add_argument("--export", type=str, help="Export last Q&A to a markdown file.")
parser.add_argument("--quiz", action="store_true", help="Start quiz mode instead of Q&A.")
args = parser.parse_args()

# === Quiz Mode ===
if args.quiz:
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.question_gen import QuizQuestionGenerator
    retriever = index.as_retriever(similarity_top_k=5)
    engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
    quiz = QuizQuestionGenerator.from_defaults(llm=llm)
    questions = quiz.generate_questions_from_nodes(retriever.retrieve(""))

    print("\n🧠 Answer the following questions (type to answer, enter to skip):")
    for q in questions:
        print("\n❓", q.text)
        input("Your answer: ")
        print("✅ Sample Answer:", q.answer)
    exit()

# === Simple Q&A CLI loop ===
print("\n🧠 Ask me anything about your study materials (blank line to exit):")
chat_log = []

while True:
    try:
        question = input("❓ ")
        if not question.strip():
            print("👋 Bye!")
            break
        response = chat_engine.chat(question)
        print(f"\n💬 {response.response}\n")
        sources = []
        if response.source_nodes:
            print("🔗 Sources:")
            for node in response.source_nodes:
                meta = node.metadata
                src = meta.get("file_path") or meta.get("file_name") or meta.get("source") or "<unknown>"
                print(" -", src)
                sources.append(src)
        chat_log.append((question, response.response, sources))
    except KeyboardInterrupt:
        print("\n👋 Interrupted. Bye!")
        break

# === Optional export ===
if args.export:
    output_path = Path(args.export)
    with output_path.open("w", encoding="utf-8") as f:
        for q, a, srcs in chat_log:
            f.write(f"## Q: {q}\n\n{a}\n\n")
            if srcs:
                f.write("**Sources:**\n")
                for s in srcs:
                    f.write(f"- {s}\n")
            f.write("\n---\n\n")
    print(f"\n📄 Exported to {output_path.resolve()}")