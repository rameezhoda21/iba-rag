from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.embeddings import EmbeddingConfig, EmbeddingService
from app.generator import AnswerGenerator
from app.hybrid_retriever import HybridRetriever
from app.intent_detector import IntentDetector, IntentResult
from app.query_rewriter import QueryRewriter
from app.rag_pipeline import load_chunks_jsonl
from app.reranker import MultiQueryReranker
from app.retriever import (
    BM25Retriever,
    RetrievedChunk,
    VectorStoreConfig,
    load_vector_store,
)
from app.synonym_mapper import SynonymMapper


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

ARTIFACTS_DIR = BASE_DIR / "data" / "artifacts"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INDEX_PATH = ARTIFACTS_DIR / "faiss.index"
METADATA_PATH = ARTIFACTS_DIR / "chunks.json"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
FRONTEND_DIR = BASE_DIR / "frontend"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str
    sources: List[str]


class ChatPipeline:
    COMMON_QUERY_CORRECTIONS = {
        "polocy": "policy",
        "poilcy": "policy",
        "admisson": "admission",
        "scholership": "scholarship",
    }
    UG_KEYWORDS = (
        "undergraduate",
        "undergrad",
        "bs",
        "bba",
        "bachelor",
    )
    MS_KEYWORDS = (
        "ms",
        "m.s",
        "graduate",
        "masters",
        "master",
    )

    def __init__(self) -> None:
        vector_db = os.getenv("VECTOR_DB", "pinecone").lower()
        embed_provider = os.getenv("EMBED_PROVIDER", "sentence-transformers")
        embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        embed_api_key = os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY")
        llm_provider = os.getenv("LLM_PROVIDER", "huggingface")
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        llm_model = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        llm_base_url = os.getenv("LLM_BASE_URL")
        use_reranker = os.getenv("USE_RERANKER", "true").lower() == "true"
        use_hybrid_search = os.getenv("USE_HYBRID_SEARCH", "true").lower() == "true"
        use_semantic_search = os.getenv("USE_SEMANTIC_SEARCH", "true").lower() == "true"
        use_query_rewriter = os.getenv("USE_QUERY_REWRITER", "true").lower() == "true"
        use_synonym_expansion = os.getenv("USE_SYNONYM_EXPANSION", "true").lower() == "true"

        self.top_k_per_query = int(os.getenv("RAG_TOP_K_PER_QUERY", "8"))
        self.top_k_fused = int(os.getenv("RAG_TOP_K", "14"))
        self.top_n = int(os.getenv("RAG_TOP_N", "6"))
        self.max_context_chunks = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))
        self.min_context_chunks = int(os.getenv("MIN_CONTEXT_CHUNKS", "3"))
        self.relevance_relative_threshold = float(os.getenv("RELEVANCE_RELATIVE_THRESHOLD", "0.72"))
        self.rerank_original_weight = float(os.getenv("RERANK_ORIGINAL_WEIGHT", "0.6"))
        self.rerank_rewritten_weight = float(os.getenv("RERANK_REWRITTEN_WEIGHT", "0.4"))
        self.default_audience = os.getenv("DEFAULT_AUDIENCE", "undergraduate").lower().strip()
        self.use_consistency_filter = os.getenv("USE_CONSISTENCY_FILTER", "true").lower() == "true"

        synonym_map_default = BASE_DIR / "data" / "config" / "synonym_map.json"
        synonym_map_path = os.getenv("SYNONYM_MAP_PATH", str(synonym_map_default))

        self.query_rewriter = QueryRewriter() if use_query_rewriter else None
        self.synonym_mapper = SynonymMapper(mapping_path=synonym_map_path) if use_synonym_expansion else None
        self.intent_detector = IntentDetector()

        self.embedding_service = None
        if use_semantic_search:
            self.embedding_service = EmbeddingService(
                EmbeddingConfig(
                    provider=embed_provider,
                    model_name=embed_model,
                    api_key=embed_api_key,
                )
            )

        if vector_db == "faiss":
            if not INDEX_PATH.exists() or not METADATA_PATH.exists():
                raise FileNotFoundError(
                    "Missing FAISS artifacts. Run scripts/build_index.py first."
                )

        self.store = None
        if use_semantic_search:
            self.store = load_vector_store(
                VectorStoreConfig(
                    provider=vector_db,
                    index_path=INDEX_PATH,
                    metadata_path=METADATA_PATH,
                    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
                    pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "iba-rag-chatbot"),
                    pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "iba"),
                    pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws"),
                    pinecone_region=os.getenv("PINECONE_REGION", "us-east-1"),
                )
            )

        self.bm25_retriever = None
        if use_hybrid_search and CHUNKS_PATH.exists():
            chunks = load_chunks_jsonl(CHUNKS_PATH)
            if chunks:
                self.bm25_retriever = BM25Retriever(chunks)

        self.hybrid_retriever = HybridRetriever(
            embedding_service=self.embedding_service,
            vector_store=self.store,
            bm25_retriever=self.bm25_retriever,
        )
        self.reranker = MultiQueryReranker() if use_reranker else None
        self.generator = AnswerGenerator(
            provider=llm_provider,
            api_key=llm_api_key,
            model=llm_model,
            base_url=llm_base_url,
        )

    def _normalize_query(self, query: str) -> str:
        normalized = query
        for wrong, right in self.COMMON_QUERY_CORRECTIONS.items():
            normalized = re.sub(
                rf"\b{re.escape(wrong)}\b",
                right,
                normalized,
                flags=re.IGNORECASE,
            )
        return normalized.strip()

    @staticmethod
    def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(word)}\b", text) for word in keywords)

    def _is_ambiguous_fee_query(self, query: str) -> bool:
        q = query.lower()
        fee_like = self._contains_any(
            q,
            (
                "fee",
                "fees",
                "tuition",
                "credit hour",
                "per credit",
            ),
        )
        mentions_ug = self._contains_any(q, self.UG_KEYWORDS)
        mentions_ms = self._contains_any(q, self.MS_KEYWORDS)
        return fee_like and not (mentions_ug or mentions_ms)

    def _program_bucket(self, item_text: str) -> str:
        text = item_text.lower()
        if self._contains_any(text, self.UG_KEYWORDS):
            return "ug"
        if self._contains_any(text, self.MS_KEYWORDS):
            return "ms"
        return ""

    def _ensure_fee_program_coverage(
        self,
        query: str,
        ranked: List[RetrievedChunk],
        candidate_pool: List[RetrievedChunk],
        max_items: int,
    ) -> List[RetrievedChunk]:
        if not self._is_ambiguous_fee_query(query):
            return ranked

        selected = []
        seen = set()
        for item in ranked:
            cid = item.chunk.chunk_id
            if cid in seen:
                continue
            seen.add(cid)
            selected.append(item)

        def as_program_text(item) -> str:
            md = item.chunk.metadata
            return " ".join(
                [
                    str(md.get("title", "")),
                    str(md.get("source", "")),
                    str(item.chunk.text),
                ]
            )

        selected_buckets = {self._program_bucket(as_program_text(item)) for item in selected}
        selected_buckets.discard("")

        available_buckets = {self._program_bucket(as_program_text(item)) for item in candidate_pool}
        available_buckets.discard("")

        for required in ("ug", "ms"):
            if required in selected_buckets or required not in available_buckets:
                continue
            for item in candidate_pool:
                cid = item.chunk.chunk_id
                if cid in seen:
                    continue
                if self._program_bucket(as_program_text(item)) == required:
                    selected.append(item)
                    seen.add(cid)
                    selected_buckets.add(required)
                    break

        if len(selected) <= max_items:
            return selected

        must_keep_ids = set()
        for required in sorted(available_buckets):
            for item in selected:
                if self._program_bucket(as_program_text(item)) == required:
                    must_keep_ids.add(item.chunk.chunk_id)
                    break

        trimmed = []
        for item in selected:
            cid = item.chunk.chunk_id
            if cid in must_keep_ids and item not in trimmed:
                trimmed.append(item)

        for item in selected:
            if len(trimmed) >= max_items:
                break
            if item in trimmed:
                continue
            trimmed.append(item)

        return trimmed[:max_items]

    def _late_fee_mismatch_guard(
        self,
        query: str,
        retrieved: List[RetrievedChunk],
    ) -> ChatResponse | None:
        q = query.lower()

        is_late_fee_query = (
            "late" in q
            and self._contains_any(q, ("fee", "fees", "fine", "policy", "payment"))
        )
        query_mentions_library = self._contains_any(
            q,
            (
                "library",
                "book",
                "overdue",
                "reserve",
            ),
        )
        if not is_late_fee_query or query_mentions_library:
            return None

        library_terms = (
            "library",
            "overdue",
            "general stacks",
            "course reserves",
            "reference material",
            "borrow",
        )
        tuition_terms = (
            "tuition",
            "semester fee",
            "credit hour",
            "challan",
            "payment deadline",
            "unpaid",
        )

        def combined_text(item: RetrievedChunk) -> str:
            md = item.chunk.metadata
            return " ".join(
                [
                    str(md.get("title", "")),
                    str(md.get("source", "")),
                    str(item.chunk.text),
                ]
            ).lower()

        lib_count = 0
        tuition_count = 0
        for item in retrieved:
            text = combined_text(item)
            if self._contains_any(text, library_terms):
                lib_count += 1
            if self._contains_any(text, tuition_terms):
                tuition_count += 1

        if lib_count > 0 and tuition_count == 0:
            source_candidates = []
            for item in retrieved:
                md = item.chunk.metadata
                title = str(md.get("title", "")).strip()
                source = str(md.get("source", "")).strip()
                if title and source:
                    source_candidates.append(f"{title} ({source})")
                elif title:
                    source_candidates.append(title)
                elif source:
                    source_candidates.append(source)

            unique_sources = []
            seen = set()
            for source in source_candidates:
                key = source.lower()
                if key in seen:
                    continue
                seen.add(key)
                unique_sources.append(source)
                if len(unique_sources) >= 2:
                    break

            return ChatResponse(
                answer=(
                    "I could not find a tuition late-fee policy in the retrieved context. "
                    "The retrieved text appears to be library overdue fines. "
                    "If you want, ask specifically for either 'tuition late payment policy' "
                    "or 'library late return fines'."
                ),
                sources=unique_sources,
            )

        return None

    def _build_query_variants(self, original_query: str) -> tuple[Dict[str, str], List[str]]:
        rewritten_query = original_query
        if self.query_rewriter is not None:
            rewritten_query = self.query_rewriter.rewrite(original_query)

        expanded_query = rewritten_query
        applied_expansions: List[str] = []
        if self.synonym_mapper is not None:
            expanded_query, applied_expansions = self.synonym_mapper.expand_query(rewritten_query)

        variants: Dict[str, str] = {"original": original_query}
        if rewritten_query.strip().lower() != original_query.strip().lower():
            variants["rewritten"] = rewritten_query
        if expanded_query.strip().lower() not in {
            q.strip().lower() for q in variants.values()
        }:
            variants["expanded"] = expanded_query

        return variants, applied_expansions

    @staticmethod
    def _apply_relevance_filter(
        retrieved: List[RetrievedChunk],
        min_chunks: int,
        max_chunks: int,
        threshold_ratio: float,
    ) -> List[RetrievedChunk]:
        if not retrieved:
            return []

        max_chunks = max(1, max_chunks)
        min_chunks = max(1, min(min_chunks, max_chunks))
        top = retrieved[: max(max_chunks * 2, max_chunks)]
        top_score = float(top[0].score)

        if top_score <= 0:
            return top[:max_chunks]

        kept = [item for item in top if float(item.score) >= top_score * threshold_ratio]
        if len(kept) < min_chunks:
            kept = top[:min_chunks]

        return kept[:max_chunks]

    @staticmethod
    def _summarize_hits(items: List[RetrievedChunk], limit: int = 5) -> List[dict]:
        summary = []
        for item in items[:limit]:
            md = item.chunk.metadata
            summary.append(
                {
                    "chunk_id": item.chunk.chunk_id,
                    "title": md.get("title"),
                    "source": md.get("source"),
                    "policy_name": md.get("policy_name"),
                    "audience": md.get("audience"),
                    "program_type": md.get("program_type"),
                    "semester_type": md.get("semester_type"),
                    "score": round(float(item.score), 6),
                }
            )
        return summary

    @staticmethod
    def _query_scope_terms(query: str) -> Dict[str, str]:
        q = query.lower()

        audience = ""
        if re.search(r"\bundergraduate|undergrad|bba|bs|bachelor\b", q):
            audience = "undergraduate"
        elif re.search(r"\bgraduate|postgraduate|master|ms|mba|executive\b", q):
            audience = "graduate"

        student_type = ""
        if re.search(r"\bfull[-\s]?time\b", q):
            student_type = "full-time"
        elif re.search(r"\bpart[-\s]?time\b", q):
            student_type = "part-time"

        semester_type = ""
        if re.search(r"\bsummer\b", q):
            semester_type = "summer"
        elif re.search(r"\bspring\b", q) and re.search(r"\bfall\b", q):
            semester_type = "spring/fall"
        elif re.search(r"\bspring\b", q):
            semester_type = "spring"
        elif re.search(r"\bfall\b", q):
            semester_type = "fall"

        return {
            "audience": audience,
            "student_type": student_type,
            "semester_type": semester_type,
        }

    @staticmethod
    def _chunk_scope(item: RetrievedChunk) -> Dict[str, str]:
        md = item.chunk.metadata
        return {
            "policy_name": str(md.get("policy_name", "") or "").lower().strip(),
            "audience": str(md.get("audience", "") or "").lower().strip(),
            "program_type": str(md.get("program_type", "") or "").lower().strip(),
            "student_type": str(md.get("student_type", "") or "").lower().strip(),
            "semester_type": str(md.get("semester_type", "") or "").lower().strip(),
            "degree_level": str(md.get("degree_level", "") or "").lower().strip(),
        }

    def _consistency_group_key(self, item: RetrievedChunk) -> str:
        scope = self._chunk_scope(item)
        policy = scope["policy_name"] or "general-policy"
        audience = scope["audience"] or scope["degree_level"] or scope["program_type"] or "general-audience"
        student_type = scope["student_type"] or "general-student"
        semester_type = scope["semester_type"] or "general-semester"
        return "|".join([policy, audience, student_type, semester_type])

    def _group_matches_query_scope(self, item: RetrievedChunk, query_scope: Dict[str, str]) -> bool:
        scope = self._chunk_scope(item)
        for key in ("audience", "student_type", "semester_type"):
            required = query_scope.get(key, "")
            if not required:
                continue
            haystack = " ".join(
                [
                    scope.get(key, ""),
                    scope.get("program_type", ""),
                    scope.get("degree_level", ""),
                ]
            )
            if required not in haystack:
                return False
        return True

    def _apply_consistency_filter(
        self,
        query: str,
        retrieved: List[RetrievedChunk],
    ) -> Tuple[List[RetrievedChunk], List[dict], str]:
        if not retrieved or not self.use_consistency_filter:
            return retrieved, [], "disabled-or-empty"

        query_scope = self._query_scope_terms(query)
        groups: Dict[str, List[RetrievedChunk]] = {}
        for item in retrieved:
            key = self._consistency_group_key(item)
            groups.setdefault(key, []).append(item)

        group_stats = []
        for key, items in groups.items():
            score_sum = sum(float(x.score) for x in items)
            top_score = max(float(x.score) for x in items)
            representative = items[0]
            group_stats.append(
                {
                    "key": key,
                    "count": len(items),
                    "score_sum": score_sum,
                    "top_score": top_score,
                    "representative": representative,
                }
            )

        matched_groups = [
            g for g in group_stats if self._group_matches_query_scope(g["representative"], query_scope)
        ]

        selection_reason = "best-scoring-consistent-group"
        if matched_groups:
            group_stats = matched_groups
            selection_reason = "query-scoped-group"

        broad_query = not any(query_scope.values())
        if broad_query:
            for g in group_stats:
                rep_scope = self._chunk_scope(g["representative"])
                audience_blob = " ".join(
                    [
                        rep_scope.get("audience", ""),
                        rep_scope.get("degree_level", ""),
                        rep_scope.get("program_type", ""),
                    ]
                )
                if self.default_audience and self.default_audience in audience_blob:
                    g["score_sum"] += 0.2
                    selection_reason = "default-audience-group"

        group_stats.sort(key=lambda x: (x["score_sum"], x["top_score"], x["count"]), reverse=True)
        best_key = group_stats[0]["key"]
        kept = groups[best_key]
        kept.sort(key=lambda x: float(x.score), reverse=True)
        kept = kept[: self.max_context_chunks]

        debug_groups = []
        for g in group_stats:
            rep_scope = self._chunk_scope(g["representative"])
            debug_groups.append(
                {
                    "group_key": g["key"],
                    "count": g["count"],
                    "score_sum": round(g["score_sum"], 6),
                    "top_score": round(g["top_score"], 6),
                    "policy_name": rep_scope.get("policy_name", ""),
                    "audience": rep_scope.get("audience", ""),
                    "program_type": rep_scope.get("program_type", ""),
                    "student_type": rep_scope.get("student_type", ""),
                    "semester_type": rep_scope.get("semester_type", ""),
                    "selected": g["key"] == best_key,
                }
            )

        return kept, debug_groups, selection_reason

    def ask(self, query: str, top_k: int = 8, top_n: int = 5) -> ChatResponse:
        normalized_query = self._normalize_query(query)
        intent: IntentResult = self.intent_detector.detect(normalized_query)
        query_variants, applied_expansions = self._build_query_variants(normalized_query)

        if intent.official_policy_term:
            lowered_variants = {v.lower() for v in query_variants.values()}
            if intent.official_policy_term.lower() not in lowered_variants:
                query_variants["intent_policy"] = intent.official_policy_term

        multi_query_result = self.hybrid_retriever.retrieve_multi_query(
            query_variants=query_variants,
            intent=intent,
            top_k_per_query=self.top_k_per_query,
            fused_top_k=self.top_k_fused,
        )
        retrieved = multi_query_result.candidates
        candidate_pool = list(retrieved)

        rewritten_for_rerank = query_variants.get("rewritten", normalized_query)

        if self.reranker:
            retrieved = self.reranker.rerank(
                original_query=normalized_query,
                rewritten_query=rewritten_for_rerank,
                intent=intent,
                retrieved=retrieved,
                top_n=self.top_n,
                original_weight=self.rerank_original_weight,
                rewritten_weight=self.rerank_rewritten_weight,
            )
        else:
            retrieved = retrieved[: self.top_n]

        retrieved = self._ensure_fee_program_coverage(
            query=normalized_query,
            ranked=retrieved,
            candidate_pool=candidate_pool,
            max_items=self.top_n,
        )

        retrieved = self._apply_relevance_filter(
            retrieved=retrieved,
            min_chunks=self.min_context_chunks,
            max_chunks=self.max_context_chunks,
            threshold_ratio=self.relevance_relative_threshold,
        )

        retrieved, consistency_debug, consistency_reason = self._apply_consistency_filter(
            query=normalized_query,
            retrieved=retrieved,
        )

        mismatch_response = self._late_fee_mismatch_guard(normalized_query, retrieved)
        if mismatch_response is not None:
            logging.info("Query: %s", query)
            logging.info("Normalized query: %s", normalized_query)
            logging.info("Answer (guarded): %s", mismatch_response.answer)
            return mismatch_response

        generated = self.generator.generate(
            question=normalized_query,
            chunks=retrieved,
            rewritten_query=query_variants.get("rewritten"),
            expanded_query=query_variants.get("expanded"),
            intent_label=intent.intent,
            official_policy_term=intent.official_policy_term,
        )

        logging.info("Query: %s", query)
        logging.info("Normalized query: %s", normalized_query)
        logging.info(
            "Detected intent: %s",
            {
                "intent": intent.intent,
                "confidence": round(intent.confidence, 4),
                "matched_terms": intent.matched_terms,
                "preferred_categories": intent.preferred_categories,
                "negative_categories": intent.negative_categories,
                "official_policy_term": intent.official_policy_term,
            },
        )
        logging.info("Query variants: %s", query_variants)
        logging.info("Synonym expansions: %s", applied_expansions)
        logging.info("Consistency filter reason: %s", consistency_reason)
        logging.info("Consistency groups: %s", consistency_debug)
        logging.info(
            "Retrieval traces: %s",
            [
                {
                    "label": trace.label,
                    "query": trace.query,
                    "dense": self._summarize_hits(trace.dense_hits),
                    "lexical": self._summarize_hits(trace.lexical_hits),
                    "fused": self._summarize_hits(trace.fused_hits),
                }
                for trace in multi_query_result.traces
            ],
        )
        if self.reranker:
            logging.info(
                "Reranked top chunks: %s",
                [
                    {
                        "chunk_id": item.chunk_id,
                        "score_original": round(item.score_original, 6),
                        "score_rewritten": round(item.score_rewritten, 6),
                        "intent_bonus": round(item.intent_bonus, 6),
                        "score_final": round(item.final_score, 6),
                    }
                    for item in self.reranker.last_trace[: self.top_n]
                ],
            )
        logging.info(
            "Retrieved chunks: %s",
            [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "title": r.chunk.metadata.get("title"),
                    "source": r.chunk.metadata.get("source"),
                    "policy_name": r.chunk.metadata.get("policy_name"),
                    "audience": r.chunk.metadata.get("audience"),
                    "program_type": r.chunk.metadata.get("program_type"),
                    "student_type": r.chunk.metadata.get("student_type"),
                    "semester_type": r.chunk.metadata.get("semester_type"),
                    "score": r.score,
                }
                for r in retrieved
            ],
        )
        logging.info("Answer: %s", generated["answer"])

        return ChatResponse(answer=generated["answer"], sources=generated["sources"])


app = FastAPI(title="IBA RAG Chatbot API", version="0.1.0")
pipeline: ChatPipeline | None = None

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.on_event("startup")
def startup_event() -> None:
    global pipeline
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    try:
        pipeline = ChatPipeline()
    except Exception as exc:
        logging.exception("Failed to initialize chat pipeline: %s", exc)
        pipeline = None


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline is not initialized.")

    return pipeline.ask(request.message)


@app.get("/")
def home() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)
