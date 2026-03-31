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
from app.evaluator import ResponseEvaluator
from app.generator import AnswerGenerator
from app.hybrid_retriever import HybridRetriever
from app.intent_detector import IntentDetector, IntentResult
from app.query_entity_extractor import QueryEntities, QueryEntityExtractor
from app.query_rewriter import QueryRewriter
from app.rag_pipeline import load_chunks_jsonl
from app.reranker import MultiQueryReranker
from app.settings import RuntimeSettings
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
    ALL_PROGRAM_KEYS = (
        "undergraduate",
        "ms",
        "mba",
        "executive mba",
    )

    def __init__(self) -> None:
        vector_db = RuntimeSettings.VECTOR_DB.lower()
        embed_provider = RuntimeSettings.EMBED_PROVIDER
        embed_model = RuntimeSettings.EMBED_MODEL
        embed_api_key = os.getenv("HF_API_TOKEN")
        llm_provider = RuntimeSettings.LLM_PROVIDER
        llm_api_key = os.getenv("GROQ_API_KEY")
        llm_model = RuntimeSettings.LLM_MODEL
        llm_base_url = RuntimeSettings.LLM_BASE_URL
        use_reranker = RuntimeSettings.USE_RERANKER
        use_hybrid_search = RuntimeSettings.USE_HYBRID_SEARCH
        use_semantic_search = RuntimeSettings.USE_SEMANTIC_SEARCH
        use_query_rewriter = RuntimeSettings.USE_QUERY_REWRITER
        use_synonym_expansion = RuntimeSettings.USE_SYNONYM_EXPANSION

        self.top_k_per_query = RuntimeSettings.RAG_TOP_K_PER_QUERY
        self.top_k_fused = RuntimeSettings.RAG_TOP_K
        self.top_n = RuntimeSettings.RAG_TOP_N
        self.max_context_chunks = RuntimeSettings.MAX_CONTEXT_CHUNKS
        self.min_context_chunks = RuntimeSettings.MIN_CONTEXT_CHUNKS
        self.relevance_relative_threshold = RuntimeSettings.RELEVANCE_RELATIVE_THRESHOLD
        self.rerank_original_weight = RuntimeSettings.RERANK_ORIGINAL_WEIGHT
        self.rerank_rewritten_weight = RuntimeSettings.RERANK_REWRITTEN_WEIGHT
        self.default_audience = RuntimeSettings.DEFAULT_AUDIENCE.lower().strip()
        self.use_consistency_filter = RuntimeSettings.USE_CONSISTENCY_FILTER
        self.groundedness_threshold = RuntimeSettings.EVAL_GROUNDEDNESS_THRESHOLD

        synonym_map_path = RuntimeSettings.SYNONYM_MAP_PATH

        self.query_rewriter = QueryRewriter() if use_query_rewriter else None
        self.synonym_mapper = SynonymMapper(mapping_path=synonym_map_path) if use_synonym_expansion else None
        self.intent_detector = IntentDetector()
        self.entity_extractor = QueryEntityExtractor()

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
                    pinecone_index_name=RuntimeSettings.PINECONE_INDEX_NAME,
                    pinecone_namespace=RuntimeSettings.PINECONE_NAMESPACE,
                    pinecone_cloud=RuntimeSettings.PINECONE_CLOUD,
                    pinecone_region=RuntimeSettings.PINECONE_REGION,
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
        self.evaluator = ResponseEvaluator(groundedness_threshold=self.groundedness_threshold)

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

    @staticmethod
    def _item_text_blob(item: RetrievedChunk) -> str:
        md = item.chunk.metadata
        return " ".join(
            [
                str(md.get("title", "")),
                str(md.get("source", "")),
                str(md.get("policy_name", "")),
                str(md.get("program_type", "")),
                str(md.get("row_program_label", "")),
                item.chunk.text,
            ]
        ).lower()

    @staticmethod
    def _program_terms(program: str) -> tuple[str, ...]:
        if program == "undergraduate":
            return ("undergraduate", "undergrad", "bba", "bs", "bachelor")
        if program == "ms":
            return ("ms", "m.s", "master of science", "masters")
        if program == "mba":
            return ("mba",)
        if program == "executive mba":
            return ("executive mba", "emba")
        return (program,)

    def _chunk_matches_program(self, item: RetrievedChunk, program: str) -> bool:
        haystack = self._item_text_blob(item)
        return any(re.search(rf"\b{re.escape(token)}\b", haystack) for token in self._program_terms(program))

    @staticmethod
    def _chunk_metric_blob(item: RetrievedChunk) -> str:
        md = item.chunk.metadata
        return " ".join(
            [
                str(md.get("fee_type", "")),
                str(md.get("per_credit_hour_value", "")),
                str(md.get("semester_value", "")),
                item.chunk.text,
            ]
        ).lower()

    def _chunk_matches_metric(self, item: RetrievedChunk, metric: str) -> bool:
        if not metric:
            return True
        blob = self._chunk_metric_blob(item)
        if metric == "tuition fee per credit hour":
            return bool(re.search(r"tuition|per\s+credit|credit\s+hour", blob))
        return metric.lower() in blob

    def _score_fee_candidates(
        self,
        candidates: List[RetrievedChunk],
        target_program: str,
        entities: QueryEntities,
    ) -> tuple[List[RetrievedChunk], List[dict], List[dict]]:
        scored: List[RetrievedChunk] = []
        matched: List[dict] = []
        rejected: List[dict] = []

        for item in candidates:
            md = item.chunk.metadata
            score = float(item.score)
            reasons: List[str] = []
            text_blob = self._item_text_blob(item)
            source = str(md.get("source", "") or "").lower()

            target_match = self._chunk_matches_program(item, target_program)
            if target_match:
                score += 1.25
                reasons.append("target-program-match")

            if md.get("structured_record_type") == "fee_table_row":
                score += 0.55
                reasons.append("structured-row")

                row_label = str(md.get("row_program_label", "") or "").strip().lower()
                value_blob = str(md.get("per_credit_hour_value", "") or md.get("semester_value", "") or "")
                value_int = int(re.sub(r"[^0-9]", "", value_blob) or "0")

                if re.fullmatch(r"[-\d\s]+", row_label) or row_label in {"program", "programs", "around", "-"}:
                    score -= 1.35
                    reasons.append("low-quality-row-label")

                if entities.metric in {"tuition fee per credit hour", "fees", ""}:
                    if value_int and (value_int < 10000 or value_int > 100000):
                        score -= 1.1
                        reasons.append("unlikely-credit-hour-value")

            if "website_https_www_iba_edu_pk_fee_structure_php" in source:
                score += 0.35
                reasons.append("official-fee-structure-source")

            if self._chunk_matches_metric(item, entities.metric):
                score += 0.5
                reasons.append("metric-match")

            for program in self.ALL_PROGRAM_KEYS:
                if program == target_program:
                    continue
                if self._chunk_matches_program(item, program) and not target_match:
                    score -= 1.15
                    reasons.append(f"near-match-penalty:{program}")

            rescored = RetrievedChunk(chunk=item.chunk, score=score)
            scored.append(rescored)

            log_item = {
                "chunk_id": item.chunk.chunk_id,
                "source": md.get("source"),
                "program_type": md.get("program_type"),
                "fee_type": md.get("fee_type"),
                "score": round(score, 6),
                "reasons": reasons,
                "text_preview": text_blob[:140],
            }
            if target_match and score > 0:
                matched.append(log_item)
            elif any(reason.startswith("near-match-penalty") for reason in reasons):
                rejected.append(log_item)

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored, matched[:8], rejected[:8]

    def _retrieve_fee_entity_chunks(
        self,
        normalized_query: str,
        entities: QueryEntities,
        intent: IntentResult,
        query_variants: Dict[str, str],
    ) -> tuple[List[RetrievedChunk], dict]:
        selected: List[RetrievedChunk] = []
        seen = set()
        logs = {"by_program": {}}

        target_programs = entities.programs or ["undergraduate", "ms"]
        for program in target_programs:
            program_variants = {k: f"{v} {program}" for k, v in query_variants.items()}
            if entities.metric:
                program_variants["metric"] = f"{normalized_query} {entities.metric} {program}"

            entity_result = self.hybrid_retriever.retrieve_multi_query(
                query_variants=program_variants,
                intent=intent,
                top_k_per_query=max(self.top_k_per_query, 10),
                fused_top_k=max(self.top_k_fused, 16),
            )
            scored, matched_logs, rejected_logs = self._score_fee_candidates(
                candidates=entity_result.candidates,
                target_program=program,
                entities=entities,
            )

            per_program = []
            for item in scored:
                if len(per_program) >= 3:
                    break
                if item.score < -0.1:
                    continue
                if not self._chunk_matches_program(item, program):
                    continue
                if entities.metric and not self._chunk_matches_metric(item, entities.metric):
                    continue
                cid = item.chunk.chunk_id
                if cid in seen:
                    continue
                seen.add(cid)
                per_program.append(item)
                selected.append(item)

            logs["by_program"][program] = {
                "matched_entities": matched_logs,
                "rejected_near_matches": rejected_logs,
                "selected": self._summarize_hits(per_program, limit=3),
            }

        selected.sort(key=lambda x: x.score, reverse=True)
        return selected[: self.max_context_chunks], logs

    def _validate_fee_entity_coverage(
        self,
        chunks: List[RetrievedChunk],
        entities: QueryEntities,
    ) -> tuple[Dict[str, List[RetrievedChunk]], List[str]]:
        evidence_by_program: Dict[str, List[RetrievedChunk]] = {}
        missing: List[str] = []

        for program in entities.programs:
            matched = [
                item
                for item in chunks
                if self._chunk_matches_program(item, program)
                and self._chunk_matches_metric(item, entities.metric)
            ]
            if matched:
                evidence_by_program[program] = matched
            else:
                missing.append(program)

        return evidence_by_program, missing

    @staticmethod
    def _program_display_name(program: str) -> str:
        mapping = {
            "undergraduate": "Undergraduate",
            "ms": "MS",
            "mba": "MBA",
            "executive mba": "Executive MBA",
        }
        return mapping.get(program, program.title())

    @staticmethod
    def _format_currency(value: str) -> str:
        cleaned = re.sub(r"[^0-9,]", "", value)
        return f"PKR {cleaned}" if cleaned else value

    def _compose_hostel_answer(self, rows: List[RetrievedChunk]) -> str:
        lines = []
        for item in rows:
            md = item.chunk.metadata
            room_type = md.get("hostel_room_type")
            amount = md.get("hostel_amount")
            if room_type and amount:
                lines.append(f"  * {room_type.title()}: PKR {amount}")
                
        unique_lines = list(dict.fromkeys(lines))
        
        if not unique_lines:
            return "I found hostel fee structure information, but could not determine the exact amounts. Please check the website."
            
        ans = "Based on the official fee structure, here are the hostel charges (which apply uniformly across degree levels):\n\n"
        ans += "- **Undergraduate**:\n" + "\n".join(unique_lines) + "\n"
        ans += "- **Graduate**:\n" + "\n".join(unique_lines) + "\n"
        ans += "- **MS**:\n" + "\n".join(unique_lines) + "\n"
        
        return ans

    def _compose_attendance_answer(self, rows: List[RetrievedChunk]) -> str:
        # validate that we have valid allowed_absences
        lines = []
        for item in rows:
            md = item.chunk.metadata
            course_type = md.get("attendance_course_type")
            allowed = md.get("attendance_allowed_absences")
            if course_type and allowed:
                lines.append(f"- For **{course_type}** courses, you are allowed a maximum of **{allowed} absences**.")
        
        # Deduplicate
        unique_lines = list(dict.fromkeys(lines))
        
        if not unique_lines:
            return "I found attendance policy information, but could not determine the exact maximum absences. Please check the student handbook."
            
        ans = "Based on the IBA attendance requirement policy, the limits for allowed absences are:\n"
        ans += "\n".join(unique_lines)
        ans += "\n\n(Note: This specifies the maximum allowed absences, not the total sessions)."
        return ans

    def _compose_fee_answer(
        self,
        entities: QueryEntities,
        evidence_by_program: Dict[str, List[RetrievedChunk]],
        missing_programs: List[str],
    ) -> str:
        lines: List[str] = []

        for program in entities.programs:
            label = self._program_display_name(program)
            evidence = evidence_by_program.get(program, [])
            if not evidence:
                lines.append(f"- {label}: I could not find {entities.metric or 'fee'} information for this program in the current sources.")
                continue

            per_credit_values = sorted(
                {
                    self._format_currency(str(item.chunk.metadata.get("per_credit_hour_value", "") or ""))
                    for item in evidence
                    if str(item.chunk.metadata.get("per_credit_hour_value", "") or "").strip()
                }
            )
            semester_values = sorted(
                {
                    self._format_currency(str(item.chunk.metadata.get("semester_value", "") or ""))
                    for item in evidence
                    if str(item.chunk.metadata.get("semester_value", "") or "").strip()
                }
            )

            if per_credit_values:
                if len(per_credit_values) == 1:
                    lines.append(f"- {label}: Tuition fee per credit hour is {per_credit_values[0]}.")
                else:
                    lines.append(
                        f"- {label}: Tuition fee per credit hour varies by program, from {per_credit_values[0]} to {per_credit_values[-1]}."
                    )
                continue

            if semester_values:
                if len(semester_values) == 1:
                    lines.append(f"- {label}: Program fee is {semester_values[0]}.")
                else:
                    lines.append(f"- {label}: Program fee values include {', '.join(semester_values[:3])}.")
                continue

            top = evidence[0]
            lines.append(f"- {label}: {top.chunk.text[:130].strip()}")

        if missing_programs:
            joined = ", ".join(self._program_display_name(p) for p in missing_programs)
            lines.append(f"I found partial information; missing: {joined}.")

        return "\n".join(lines)

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
        entities = self.entity_extractor.extract(normalized_query)
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
        fee_route_logs = {}

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

        if entities.is_fee_question and entities.programs:
            fee_retrieved, fee_route_logs = self._retrieve_fee_entity_chunks(
                normalized_query=normalized_query,
                entities=entities,
                intent=intent,
                query_variants=query_variants,
            )
            if fee_retrieved:
                retrieved = fee_retrieved

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

        # Check for attendance policy explicit intent to route cleanly
        if intent.intent == "attendance" or "attendance" in (intent.official_policy_term or "") or "miss" in query.lower() or "absence" in query.lower():
            attendance_rows = [item for item in retrieved if item.chunk.metadata.get("structured_record_type") == "attendance_table_row"]
            if attendance_rows:
                answer = self._compose_attendance_answer(attendance_rows)
                sources = self.generator._fallback_sources(attendance_rows, limit=2)
                logging.info("Query: %s", query)
                logging.info("Answer (structured attendance route): %s", answer)
                return ChatResponse(answer=answer, sources=sources)

        # Check for hostel intent to strictly answer from hostel records, removing noise like transport
        if intent.intent == "hostel" or "hostel" in query.lower():
            hostel_rows = [item for item in retrieved if item.chunk.metadata.get("structured_record_type") == "hostel_fee_row"]
            if hostel_rows:
                answer = self._compose_hostel_answer(hostel_rows)
                sources = self.generator._fallback_sources(hostel_rows, limit=2)
                logging.info("Query: %s", query)
                logging.info("Answer (structured hostel route): %s", answer)
                return ChatResponse(answer=answer, sources=sources)

        mismatch_response = self._late_fee_mismatch_guard(normalized_query, retrieved)
        if mismatch_response is not None:
            logging.info("Query: %s", query)
            logging.info("Normalized query: %s", normalized_query)
            logging.info("Answer (guarded): %s", mismatch_response.answer)
            return mismatch_response

        evidence_by_program: Dict[str, List[RetrievedChunk]] = {}
        missing_programs: List[str] = []
        if entities.is_fee_question and entities.programs:
            evidence_by_program, missing_programs = self._validate_fee_entity_coverage(
                chunks=retrieved,
                entities=entities,
            )

            has_structured_rows = any(
                item.chunk.metadata.get("structured_record_type") == "fee_table_row"
                for items in evidence_by_program.values()
                for item in items
            )
            if has_structured_rows:
                answer = self._compose_fee_answer(
                    entities=entities,
                    evidence_by_program=evidence_by_program,
                    missing_programs=missing_programs,
                )
                sources = self.generator._fallback_sources(retrieved, limit=4)
                logging.info("Query: %s", query)
                logging.info("Normalized query: %s", normalized_query)
                logging.info(
                    "Extracted entities: %s",
                    {
                        "metric": entities.metric,
                        "programs": entities.programs,
                        "query_type": entities.query_type,
                        "is_fee_question": entities.is_fee_question,
                    },
                )
                logging.info("Fee route logs: %s", fee_route_logs)
                logging.info("Fee entity coverage: found=%s missing=%s", list(evidence_by_program.keys()), missing_programs)
                logging.info("Answer (structured fee route): %s", answer)
                return ChatResponse(answer=answer, sources=sources)

            if missing_programs and evidence_by_program:
                answer = self._compose_fee_answer(
                    entities=entities,
                    evidence_by_program=evidence_by_program,
                    missing_programs=missing_programs,
                )
                sources = self.generator._fallback_sources(retrieved, limit=4)
                logging.info("Answer (partial fee coverage): %s", answer)
                return ChatResponse(answer=answer, sources=sources)

        generated = self.generator.generate(
            question=normalized_query,
            chunks=retrieved,
            rewritten_query=query_variants.get("rewritten"),
            expanded_query=query_variants.get("expanded"),
            intent_label=intent.intent,
            official_policy_term=intent.official_policy_term,
            entity_constraints={
                "metric": entities.metric,
                "programs": entities.programs,
                "query_type": entities.query_type,
                "missing_programs": missing_programs,
            },
        )

        evaluation = self.evaluator.evaluate(
            question=normalized_query,
            answer=generated.get("answer", ""),
            chunks=retrieved,
        )
        if evaluation.should_abstain:
            generated["answer"] = (
                "I could not verify this answer from the retrieved policy context. "
                "Please rephrase your question with specific policy terms or program details."
            )
            if not generated.get("sources"):
                generated["sources"] = self.generator._fallback_sources(retrieved, limit=3)

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
        logging.info(
            "Extracted entities: %s",
            {
                "metric": entities.metric,
                "programs": entities.programs,
                "query_type": entities.query_type,
                "is_fee_question": entities.is_fee_question,
            },
        )
        if fee_route_logs:
            logging.info("Fee route logs: %s", fee_route_logs)
        if entities.programs:
            logging.info("Fee entity coverage: found=%s missing=%s", list(evidence_by_program.keys()), missing_programs)
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
            "Evaluation: %s",
            {
                "groundedness": round(evaluation.groundedness, 4),
                "keyword_overlap": round(evaluation.keyword_overlap, 4),
                "numeric_support": round(evaluation.numeric_support, 4),
                "issues": evaluation.issues,
                "should_abstain": evaluation.should_abstain,
            },
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
