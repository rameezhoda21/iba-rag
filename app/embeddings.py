from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

from app.rag_pipeline import Chunk


@dataclass
class EmbeddingConfig:
    provider: str = "sentence-transformers"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    api_key: str | None = None


class EmbeddingService:
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.provider = config.provider.lower()

        if self.provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "OpenAI provider selected, but 'openai' is not installed. "
                    "Install it with: pip install openai"
                ) from exc

            self.client = OpenAI(api_key=config.api_key)
            self.model = config.model_name or "text-embedding-3-small"
            self.st_model = None
        elif self.provider == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers provider selected, but dependency import failed. "
                    "Ensure sentence-transformers/torch are installed and compatible."
                ) from exc

            self.client = None
            self.model = config.model_name
            self.st_model = SentenceTransformer(self.model)
        else:
            raise ValueError("Unsupported embedding provider. Use 'openai' or 'sentence-transformers'.")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self.provider == "openai":
            response = self.client.embeddings.create(model=self.model, input=texts)
            vectors = [d.embedding for d in response.data]
            return np.array(vectors, dtype=np.float32)

        vectors = self.st_model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


def save_embeddings(embeddings: np.ndarray, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)


def load_embeddings(path: str | Path) -> np.ndarray:
    return np.load(Path(path))


def save_chunk_store(chunks: Iterable[Chunk], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)


def load_chunk_store(path: str | Path) -> List[Chunk]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**item) for item in data]
