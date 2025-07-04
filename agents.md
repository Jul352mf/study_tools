# Agents

| Agent | Role |
|-------|------|
|Ingestor|Split PDFs into text chunks with metadata and store them in Chroma.|
|Summariser|Summarise each chunk using OpenAI and cache the result.|
|Reducer|Combine chunk summaries per module respecting token limits.|
|Merger|Tree-merge module summaries into a single overview.|
|Chat|Answer questions over the vector store with source citations.|
|FlashcardBuilder|Generate Anki-compatible cards from summaries.|
|LUManager|Manage Learning Units, their status and relationships (planned).|
