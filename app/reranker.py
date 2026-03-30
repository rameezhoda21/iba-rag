from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.intent_detector import IntentResult
from app.retriever import RetrievedChunk


@dataclass
class RerankTraceItem:
    chunk_id: str
    score_original: float
    score_rewritten: float
    intent_bonus: float
    final_score: float


class MultiQueryReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.enabled = True
        self.model_name = model_name
        self.model = None
        self._last_trace: List[RerankTraceItem] = []

        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self.model = CrossEncoder(model_name)
        except Exception:
            self.enabled = False
            self.model = None

    @property
    def last_trace(self) -> List[RerankTraceItem]:
        return self._last_trace

    @staticmethod
    def _intent_bonus(item: RetrievedChunk, intent: IntentResult) -> float:
        if intent.intent == "general":
            return 0.0

        text = " ".join(
            [
                str(item.chunk.metadata.get("category", "")),
                str(item.chunk.metadata.get("title", "")),
                str(item.chunk.metadata.get("source", "")),
                item.chunk.text[:700],
            ]
        ).lower()

        bonus = 0.0
        for token in intent.preferred_categories:
            if token and token.lower() in text:
                bonus += 0.18
        for token in intent.negative_categories:
            if token and token.lower() in text:
                bonus -= 0.15
        if intent.official_policy_term and intent.official_policy_term.lower() in text:
            bonus += 0.2
        return bonus

    def rerank(
        self,
        original_query: str,
        rewritten_query: str,
        intent: IntentResult,
        retrieved: List[RetrievedChunk],
        top_n: int = 5,
        original_weight: float = 0.6,
        rewritten_weight: float = 0.4,
    ) -> List[RetrievedChunk]:
        self._last_trace = []

        if not retrieved:
            return []
        if not self.enabled or self.model is None:
            return retrieved[:top_n]

        original_pairs = [[original_query, item.chunk.text] for item in retrieved]
        original_scores = self.model.predict(original_pairs)

        use_rewritten = rewritten_query.strip().lower() != original_query.strip().lower()
        if use_rewritten:
            rewritten_pairs = [[rewritten_query, item.chunk.text] for item in retrieved]
            rewritten_scores = self.model.predict(rewritten_pairs)
        else:
            rewritten_scores = original_scores

        combined = []
        for item, score_o, score_r in zip(retrieved, original_scores, rewritten_scores):
            base = float(original_weight * float(score_o) + rewritten_weight * float(score_r))
            intent_bonus = self._intent_bonus(item, intent)
            final = float(base + intent_bonus)
            combined.append(RetrievedChunk(chunk=item.chunk, score=final))
            self._last_trace.append(
                RerankTraceItem(
                    chunk_id=item.chunk.chunk_id,
                    score_original=float(score_o),
                    score_rewritten=float(score_r),
                    intent_bonus=float(intent_bonus),
                    final_score=final,
                )
            )

        combined.sort(key=lambda x: x.score, reverse=True)
        self._last_trace.sort(key=lambda x: x.final_score, reverse=True)
        return combined[:top_n]
