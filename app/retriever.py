from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.rag_pipeline import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass
class VectorStoreConfig:
    provider: str = "faiss"
    index_path: str | Path | None = None
    metadata_path: str | Path | None = None
    pinecone_api_key: str | None = None
    pinecone_index_name: str | None = None
    pinecone_namespace: str = "default"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"


class FaissVectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.chunks: List[Chunk] = []

    def add(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Embedding dimensionality mismatch.")

        # Normalize vectors so inner product behaves like cosine similarity.
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 8) -> List[RetrievedChunk]:
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)

        scores, indices = self.index.search(query, top_k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[idx], score=float(score)))
        return results

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_file))
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> "FaissVectorStore":
        index = faiss.read_index(str(index_path))
        with Path(metadata_path).open("r", encoding="utf-8") as f:
            chunk_data = json.load(f)

        store = cls(embedding_dim=index.d)
        store.index = index
        store.chunks = [Chunk(**item) for item in chunk_data]
        return store


class PineconeVectorStore:
    def __init__(
        self,
        api_key: str,
        index_name: str,
        namespace: str = "default",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        try:
            from pinecone import Pinecone, ServerlessSpec  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Pinecone provider selected, but 'pinecone' is not installed. "
                "Install it with: pip install pinecone"
            ) from exc

        if not api_key or not index_name:
            raise ValueError("Pinecone API key and index name are required.")

        self.namespace = namespace
        self._pc = Pinecone(api_key=api_key)
        self._serverless_spec = ServerlessSpec(cloud=cloud, region=region)
        self._index_name = index_name
        self._index = None

    def ensure_index(self, embedding_dim: int) -> None:
        listed = self._pc.list_indexes()
        if hasattr(listed, "names"):
            existing = set(listed.names())
        else:
            existing = {item["name"] for item in listed}

        if self._index_name not in existing:
            self._pc.create_index(
                name=self._index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=self._serverless_spec,
            )
        self._index = self._pc.Index(self._index_name)

    def upsert(self, embeddings: np.ndarray, chunks: List[Chunk], batch_size: int = 100) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings and chunks length mismatch.")

        if self._index is None:
            self.ensure_index(embedding_dim=embeddings.shape[1])

        vectors = []
        for chunk, vector in zip(chunks, embeddings):
            md = dict(chunk.metadata)
            md["text"] = chunk.text
            md["document_id"] = chunk.document_id

            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "values": vector.tolist(),
                    "metadata": md,
                }
            )

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace=self.namespace)

    def search(self, query_embedding: np.ndarray, top_k: int = 8) -> List[RetrievedChunk]:
        if self._index is None:
            self._index = self._pc.Index(self._index_name)

        result = self._index.query(
            vector=query_embedding.astype(np.float32).tolist(),
            top_k=top_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        matches = getattr(result, "matches", None) or result.get("matches", [])
        retrieved: List[RetrievedChunk] = []

        for item in matches:
            metadata = dict(getattr(item, "metadata", None) or item.get("metadata", {}))
            score = float(getattr(item, "score", None) or item.get("score", 0.0))
            chunk_id = str(getattr(item, "id", None) or item.get("id", ""))
            text = str(metadata.pop("text", ""))
            document_id = str(metadata.get("document_id", ""))

            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=text,
                metadata={k: str(v) for k, v in metadata.items()},
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        return retrieved


def load_vector_store(config: VectorStoreConfig) -> FaissVectorStore | PineconeVectorStore:
    provider = config.provider.lower()

    if provider == "pinecone":
        return PineconeVectorStore(
            api_key=config.pinecone_api_key or "",
            index_name=config.pinecone_index_name or "",
            namespace=config.pinecone_namespace,
            cloud=config.pinecone_cloud,
            region=config.pinecone_region,
        )

    if provider == "faiss":
        if not config.index_path or not config.metadata_path:
            raise ValueError("FAISS provider requires index_path and metadata_path.")
        return FaissVectorStore.load(config.index_path, config.metadata_path)

    raise ValueError("Unsupported vector store provider. Use 'pinecone' or 'faiss'.")


class BM25Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._tokens = [self._tokenize(chunk.text) for chunk in chunks]
        self._model = BM25Okapi(self._tokens)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    def search(self, query: str, top_k: int = 8) -> List[RetrievedChunk]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self._model.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results: List[RetrievedChunk] = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[int(idx)], score=score))
        return results


def fuse_results_rrf(
    semantic_results: List[RetrievedChunk],
    lexical_results: List[RetrievedChunk],
    top_k: int = 8,
    rrf_k: int = 60,
) -> List[RetrievedChunk]:
    fused_scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}

    for rank, item in enumerate(semantic_results, start=1):
        cid = item.chunk.chunk_id
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        chunk_map[cid] = item.chunk

    for rank, item in enumerate(lexical_results, start=1):
        cid = item.chunk.chunk_id
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        chunk_map[cid] = item.chunk

    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [RetrievedChunk(chunk=chunk_map[cid], score=score) for cid, score in ranked]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, retrieved: List[RetrievedChunk], top_n: int = 5) -> List[RetrievedChunk]:
        if not retrieved:
            return []

        pairs = [[query, item.chunk.text] for item in retrieved]
        scores = self.model.predict(pairs)

        reranked = [
            RetrievedChunk(chunk=item.chunk, score=float(score))
            for item, score in zip(retrieved, scores)
        ]
        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:top_n]
