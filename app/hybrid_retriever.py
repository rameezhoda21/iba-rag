from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.embeddings import EmbeddingService
from app.intent_detector import IntentResult
from app.retriever import BM25Retriever, RetrievedChunk, fuse_results_rrf


@dataclass
class QueryRetrievalTrace:
    label: str
    query: str
    dense_hits: List[RetrievedChunk]
    lexical_hits: List[RetrievedChunk]
    fused_hits: List[RetrievedChunk]


@dataclass
class MultiQueryRetrievalResult:
    candidates: List[RetrievedChunk]
    traces: List[QueryRetrievalTrace]


class HybridRetriever:
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService],
        vector_store,
        bm25_retriever: Optional[BM25Retriever],
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever

    @staticmethod
    def _text_for_intent(item: RetrievedChunk) -> str:
        md = item.chunk.metadata
        return " ".join(
            [
                str(md.get("category", "")),
                str(md.get("title", "")),
                str(md.get("source", "")),
                item.chunk.text[:700],
            ]
        ).lower()

    @staticmethod
    def _intent_boost(item: RetrievedChunk, intent: IntentResult) -> float:
        if intent.intent == "general":
            return 0.0

        text = HybridRetriever._text_for_intent(item)
        boost = 0.0

        for token in intent.preferred_categories:
            if token and token.lower() in text:
                boost += 0.22

        for token in intent.negative_categories:
            if token and token.lower() in text:
                boost -= 0.18

        for term in intent.matched_terms:
            if term and term.lower() in text:
                boost += 0.05

        if intent.official_policy_term and intent.official_policy_term.lower() in text:
            boost += 0.25

        return boost

    @staticmethod
    def _apply_intent_bias(
        items: List[RetrievedChunk],
        intent: IntentResult,
        keep_k: int,
    ) -> List[RetrievedChunk]:
        if not items:
            return []
        if intent.intent == "general":
            return items[:keep_k]

        rescored: List[RetrievedChunk] = []
        for item in items:
            bonus = HybridRetriever._intent_boost(item, intent)
            rescored.append(RetrievedChunk(chunk=item.chunk, score=float(item.score + bonus)))

        rescored.sort(key=lambda x: x.score, reverse=True)

        filtered = []
        for item in rescored:
            # Remove strongly off-intent chunks.
            if item.score < -0.05:
                continue
            filtered.append(item)
            if len(filtered) >= keep_k:
                break

        return filtered

    @staticmethod
    def _rrf_merge(lists: List[List[RetrievedChunk]], top_k: int = 12, rrf_k: int = 60) -> List[RetrievedChunk]:
        if not lists:
            return []

        fused_scores: Dict[str, float] = {}
        chunk_lookup: Dict[str, RetrievedChunk] = {}

        for result_list in lists:
            for rank, item in enumerate(result_list, start=1):
                cid = item.chunk.chunk_id
                fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
                if cid not in chunk_lookup or item.score > chunk_lookup[cid].score:
                    chunk_lookup[cid] = item

        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [RetrievedChunk(chunk=chunk_lookup[cid].chunk, score=score) for cid, score in ranked]

    def retrieve_multi_query(
        self,
        query_variants: Dict[str, str],
        intent: IntentResult,
        top_k_per_query: int = 8,
        fused_top_k: int = 12,
    ) -> MultiQueryRetrievalResult:
        traces: List[QueryRetrievalTrace] = []
        fused_lists: List[List[RetrievedChunk]] = []

        for label, query in query_variants.items():
            dense_hits: List[RetrievedChunk] = []
            lexical_hits: List[RetrievedChunk] = []

            if self.embedding_service is not None and self.vector_store is not None:
                query_embedding = self.embedding_service.embed_query(query)
                dense_hits = self.vector_store.search(query_embedding, top_k=top_k_per_query)

            if self.bm25_retriever is not None:
                lexical_hits = self.bm25_retriever.search(query, top_k=top_k_per_query)

            if dense_hits and lexical_hits:
                fused_hits = fuse_results_rrf(
                    semantic_results=dense_hits,
                    lexical_results=lexical_hits,
                    top_k=top_k_per_query,
                )
            else:
                fused_hits = dense_hits or lexical_hits

            fused_hits = self._apply_intent_bias(
                items=fused_hits,
                intent=intent,
                keep_k=top_k_per_query,
            )

            traces.append(
                QueryRetrievalTrace(
                    label=label,
                    query=query,
                    dense_hits=dense_hits,
                    lexical_hits=lexical_hits,
                    fused_hits=fused_hits,
                )
            )
            fused_lists.append(fused_hits)

        candidates = self._rrf_merge(fused_lists, top_k=fused_top_k)
        candidates = self._apply_intent_bias(candidates, intent=intent, keep_k=fused_top_k)
        return MultiQueryRetrievalResult(candidates=candidates, traces=traces)
