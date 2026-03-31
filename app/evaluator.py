from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from app.retriever import RetrievedChunk


@dataclass
class EvaluationResult:
    groundedness: float
    keyword_overlap: float
    numeric_support: float
    issues: List[str]
    should_abstain: bool


class ResponseEvaluator:
    """Evaluate how well an answer is supported by retrieved evidence."""

    STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "with",
        "you",
        "your",
    }

    ABSTAIN_PATTERNS = (
        r"could not find",
        r"not present in the context",
        r"not available in the provided",
        r"insufficient information",
        r"missing from the provided",
    )

    def __init__(self, groundedness_threshold: float = 0.45):
        self.groundedness_threshold = groundedness_threshold

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        return re.findall(r"\b[a-z][a-z0-9\-]{3,}\b", text.lower())

    @staticmethod
    def _extract_numeric_tokens(text: str) -> List[str]:
        # Captures values like 12,500 or 12500 and preserves ordering for diagnostics.
        return re.findall(r"\b\d[\d,]*\b", text)

    def _is_abstention(self, answer: str) -> bool:
        lowered = answer.lower()
        return any(re.search(pattern, lowered) for pattern in self.ABSTAIN_PATTERNS)

    def evaluate(
        self,
        question: str,
        answer: str,
        chunks: List[RetrievedChunk],
    ) -> EvaluationResult:
        if not answer.strip():
            return EvaluationResult(
                groundedness=0.0,
                keyword_overlap=0.0,
                numeric_support=0.0,
                issues=["empty-answer"],
                should_abstain=True,
            )

        if self._is_abstention(answer):
            return EvaluationResult(
                groundedness=1.0,
                keyword_overlap=1.0,
                numeric_support=1.0,
                issues=[],
                should_abstain=False,
            )

        context = self._normalize("\n".join(item.chunk.text for item in chunks))
        question_terms = {
            token
            for token in self._extract_keywords(question)
            if token not in self.STOPWORDS
        }
        answer_terms = [
            token
            for token in self._extract_keywords(answer)
            if token not in self.STOPWORDS and token not in question_terms
        ]
        unique_answer_terms = sorted(set(answer_terms))

        if unique_answer_terms:
            matched_terms = [token for token in unique_answer_terms if token in context]
            keyword_overlap = len(matched_terms) / len(unique_answer_terms)
        else:
            keyword_overlap = 1.0

        answer_numbers = self._extract_numeric_tokens(answer)
        if answer_numbers:
            supported_numbers = [token for token in answer_numbers if token in context]
            numeric_support = len(supported_numbers) / len(answer_numbers)
        else:
            numeric_support = 1.0

        groundedness = float(0.65 * keyword_overlap + 0.35 * numeric_support)

        issues: List[str] = []
        if keyword_overlap < 0.45:
            issues.append("low-keyword-overlap")
        if numeric_support < 1.0:
            issues.append("unsupported-number")
        if not chunks:
            issues.append("no-retrieved-context")

        should_abstain = groundedness < self.groundedness_threshold

        return EvaluationResult(
            groundedness=groundedness,
            keyword_overlap=keyword_overlap,
            numeric_support=numeric_support,
            issues=issues,
            should_abstain=should_abstain,
        )
