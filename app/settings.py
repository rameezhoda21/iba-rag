from __future__ import annotations

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


class RuntimeSettings:
    # Model and infrastructure choices
    VECTOR_DB = "pinecone"
    EMBED_PROVIDER = "sentence-transformers"
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_PROVIDER = "openai"  # Groq uses OpenAI-compatible client
    LLM_MODEL = "llama-3.1-8b-instant"
    LLM_BASE_URL = "https://api.groq.com/openai/v1"

    # Pinecone non-secret settings
    PINECONE_INDEX_NAME = "iba-rag-chatbot"
    PINECONE_NAMESPACE = "iba"
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"

    # Retrieval and ranking behavior
    USE_RERANKER = True
    USE_HYBRID_SEARCH = True
    USE_SEMANTIC_SEARCH = True
    USE_QUERY_REWRITER = True
    USE_SYNONYM_EXPANSION = True

    RAG_TOP_K_PER_QUERY = 5
    RAG_TOP_K = 10
    RAG_TOP_N = 5
    MAX_CONTEXT_CHUNKS = 4
    MIN_CONTEXT_CHUNKS = 2
    RELEVANCE_RELATIVE_THRESHOLD = 0.65
    RERANK_ORIGINAL_WEIGHT = 0.6
    RERANK_REWRITTEN_WEIGHT = 0.4
    DEFAULT_AUDIENCE = "undergraduate"
    USE_CONSISTENCY_FILTER = False  # Disabled for broader answers, was filtering out too much context
    EVAL_GROUNDEDNESS_THRESHOLD = 0.35

    SYNONYM_MAP_PATH = str(BASE_DIR / "data" / "config" / "synonym_map.json")


class ApiServerSettings:
    API_RELOAD = False
    API_PORT = int(os.getenv("PORT", 8000))
    API_HOST = "0.0.0.0"
