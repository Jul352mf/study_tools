from llama_index.core import load_index_from_storage
from llama_index.core import StorageContext
import genanki, uuid

index = load_index_from_storage(StorageContext.from_defaults(persist_dir="storage"))
retriever = index.as_retriever(similarity_top_k=50)

notes = []
for doc_id in index.docstore.docs:
    chunk = index.docstore.docs[doc_id].text
    qa    = retriever.query(f"Turn this into Q&A flashcards:\n\n{chunk}").response
    for line in qa.split("\n"):
        if "?" in line:
            q,a = line.split("?",1)
            notes.append(genanki.Note(model=genanki.BASIC_MODEL,
                        fields=[q.strip()+"?", a.strip()]))

deck = genanki.Deck(uuid.uuid4().int>>64, "Study-Bot Deck")
for n in notes: deck.add_note(n)
genanki.Package(deck).write_to_file("study.apkg")
print("study.apkg ready — import into Anki")