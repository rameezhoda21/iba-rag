# IBA RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for IBA student Q&A.

This project ingests handbook + website content, builds embeddings, stores vectors (Pinecone or FAISS), retrieves policy-relevant chunks, reranks them, and generates grounded answers through a FastAPI backend and a lightweight web UI.

## Features

- Section-aware chunking with policy metadata enrichment
- Multi-query retrieval:
  - original query
  - rewritten query
  - synonym-expanded query
- Hybrid retrieval:
  - dense vector similarity
  - BM25 lexical retrieval
- Intent-aware retrieval bias and reranking
- Consistency filter to avoid blending conflicting policy tracks
- Strict context filtering (top relevant chunks only)
- Source formatting:
  - website sources shown as links
  - documents shown as filename only
- FastAPI backend + frontend chat UI
- Regression check script for quality monitoring

## Project Structure

- app/: core RAG pipeline, retrieval, reranking, API
- scripts/: ingestion, indexing, run, and regression scripts
- data/raw/: raw handbook and website text files
- data/processed/: cleaned documents and chunks
- data/artifacts/: embeddings and index artifacts
- data/config/: synonym mappings
- frontend/: static web chat interface

## Requirements

- Python 3.11 recommended
- Git Bash (or equivalent shell on Windows)
- Pinecone account/key if using Pinecone
- LLM API key depending on selected provider (Hugging Face/OpenAI-compatible)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Environment Configuration

1. Copy `.env.example` to `.env`
2. Fill in required values:

- Vector DB:
  - `VECTOR_DB=pinecone` (or `faiss`)
  - `PINECONE_API_KEY` if using Pinecone
- LLM:
  - `LLM_PROVIDER=huggingface` or `openai`
  - If using OpenAI-compatible providers (for example Groq), set:
    - `LLM_BASE_URL`
    - `LLM_API_KEY`

Important retrieval settings are already included in `.env.example`.

## Data Preparation Workflow

Run from project root.

### 1) Prepare Documents (clean + chunk)

```bash
python scripts/prepare_documents.py
```

Outputs:
- `data/processed/documents.jsonl`
- `data/processed/chunks.jsonl`

### 2) Build Index

```bash
python scripts/build_index.py
```

For Pinecone, vectors are upserted to configured index.
For FAISS, local index files are written to `data/artifacts/`.

## Run the API + Frontend

```bash
python scripts/run_api.py
```

Default URLs:
- Frontend UI: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

If port 8000 is busy, script automatically picks another free port.

## API Usage

### POST /chat

Request body:

```json
{
  "message": "What is the minimum grade criteria for admission?"
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    "https://admissions.iba.edu.pk/admissionpolicy.php",
    "pa-2025-26.pdf"
  ]
}
```

## Frontend

Frontend is served by FastAPI from the root route (`/`).
No separate frontend server is required.

## Regression Checks

Run regression checks against a running API:

```bash
python scripts/run_regression_checks.py --api-url http://127.0.0.1:8000
```

Regression suite file:
- `scripts/regression_suite.json`

## Common Commands (Windows + Git Bash)

```bash
cd /c/Users/HP/My_Documents/chatbot/project
source /c/Users/HP/My_Documents/chatbot/.venv311/Scripts/activate
python -m pip install -r requirements.txt
python scripts/prepare_documents.py
python scripts/build_index.py
python scripts/run_api.py
```

## Troubleshooting

- `python: command not found`:
  - Ensure environment is activated correctly.
- `pyhton scripts/run_api.py` fails:
  - Use `python`, not `pyhton`.
- `ERR_ADDRESS_INVALID` for `0.0.0.0`:
  - Open `127.0.0.1` or `localhost` in browser.
- Pipeline init errors:
  - Check `.env` keys and installed dependencies.
- Poor retrieval quality after pipeline changes:
  - Re-run both:
    - `python scripts/prepare_documents.py`
    - `python scripts/build_index.py`

## Notes

- `.env` is ignored by git; keep secrets there.
- If chunking or metadata logic changes, rebuild processed data and index.
- Prefer testing with regression checks after major retrieval/prompt changes.
