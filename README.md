## KnowledgeBot

You need Groq and Qdrant Account.

- **`pipeline`**: ingest and extract texts from PDF files (located under: `pipeline/pdf/`), chunks the text content, and adds to vector storage
- **`chatbot`**: provides an interactive RAG-based question-answering interface
- **`common`**: provides common python scripts for pipeline & chatbot
- **`n8n`**: contains n8n workflow

## Quick Start

1. Copy `.env.example` to `.env`
2. Set required env variables in `.env` file
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
   - `GROQ_API_KEY`
3. Run poetry `poetry install`
4. Create folder in `pipeline/pdf` and place PDF files under it
5. Run ingestion pipeline `poetry run ingest`
6. Start chatbot `poetry run chatbot`


## Configuration

### Env variables

| Variable                    | Required | Description                                                 |
|-----------------------------|----------|-------------------------------------------------------------|
| `QDRANT_URL`                | Yes      | Qdrant vector database URL                                  |
| `QDRANT_API_KEY`            | Yes      | Qdrant authentication API key                               |
| `GROQ_API_KEY`              | Yes      | Groq LLM API key for chat functionality                     |
| `QDRANT_COLLECTION`         | No       | Collection name in Qdrant (default: `knowledge_base`)       |
| `EMBEDDING_MODEL`           | No       | HuggingFace embedding model (default: `all-MiniLM-L6-v2`)   |
| `VECTOR_SIZE`               | No       | Embedding vector dimensions (default: `384`)                |
| `CHUNK_SIZE`                | No       | Text chunk size for document splitting (default: `400`)     |
| `CHUNK_OVERLAP`             | No       | Overlap between text chunks (default: `100`)                |
| `RETRIEVAL_TOP_K`           | No       | Number of documents to retrieve (default: `5`)              |
| `RETRIEVAL_SCORE_THRESHOLD` | No       | Minimum similarity score for retrieval (default: `0.5`)     |
| `LLM_MODEL_NAME`            | No       | Groq model to use (default: `openai/gpt-oss-120b`)          |
| `BOT_NAME`                  | No       | Chatbot display name (default: `Accurate Assistant`)        |
| `BOT_GREETING_MESSAGE`      | No       | Welcome message (default: `Hai, ada yang bisa saya bantu?`) |

### Prompt file

System prompt for the chatbot (loaded from `bot_prompt.md` file)

## Using different embedding model

Update EMBEDDING_MODEL and VECTOR_SIZE then re-run pipeline: `poetry run ingest`


