from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


PROGRAM_ALIASES: Dict[str, tuple[str, ...]] = {
    "undergraduate": ("undergraduate", "undergrad", "bs", "bba", "bachelor"),
    "ms": ("ms", "m.s", "master of science", "masters", "master"),
    "mba": ("mba", "mba-morning", "mba morning"),
    "executive mba": ("executive mba", "emba", "executive-mba"),
}

METRIC_PATTERNS: Dict[str, List[str]] = {
    "tuition fee per credit hour": [
        r"tuition\s+fee\s+per\s+credit\s+hour",
        r"fee\s+per\s+credit\s+hour",
        r"per\s+credit\s+hour",
        r"per\s+credit",
        r"credit\s+hour\s+fee",
    ],
    "student activity charges": [
        r"student\s+activity\s+charges?",
        r"activity\s+charges?",
    ],
    "semester fee": [
        r"semester\s+fee",
        r"fee\s+per\s+semester",
    ],
    "program fee": [
        r"program\s+fee",
        r"total\s+program\s+fee",
    ],
}

FEE_KEYWORDS = (
    "fee",
    "fees",
    "tuition",
    "payment",
    "challan",
    "credit hour",
    "per credit",
)

POLICY_KEYWORDS = ("policy", "rule", "criteria", "requirement")
DEADLINE_KEYWORDS = ("deadline", "last date", "due date", "closing date")


@dataclass
class QueryEntities:
    metric: str
    programs: List[str]
    query_type: str
    is_fee_question: bool


class QueryEntityExtractor:
    @staticmethod
    def _contains(text: str, alias: str) -> bool:
        return bool(re.search(rf"\b{re.escape(alias)}\b", text, flags=re.IGNORECASE))

    def _extract_programs(self, query: str) -> List[str]:
        q = query.lower()
        programs: List[str] = []

        if any(self._contains(q, alias) for alias in PROGRAM_ALIASES["executive mba"]):
            programs.append("executive mba")

        for canonical in ("undergraduate", "ms", "mba"):
            if any(self._contains(q, alias) for alias in PROGRAM_ALIASES[canonical]):
                programs.append(canonical)

        return programs

    def _extract_metric(self, query: str) -> str:
        q = query.lower()
        for metric, patterns in METRIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, q, flags=re.IGNORECASE):
                    return metric

        if any(token in q for token in FEE_KEYWORDS):
            return "fees"
        if any(token in q for token in DEADLINE_KEYWORDS):
            return "deadline"
        if any(token in q for token in POLICY_KEYWORDS):
            return "policy"
        return ""

    @staticmethod
    def _extract_query_type(query: str, program_count: int, metric: str) -> str:
        q = query.lower()
        if program_count >= 2 or re.search(r"\b(compare|difference|versus|vs|both|between)\b", q):
            return "comparison"
        if any(token in q for token in DEADLINE_KEYWORDS):
            return "deadline"
        if any(token in q for token in POLICY_KEYWORDS):
            return "policy"
        if metric:
            return "lookup"
        return "general"

    def extract(self, query: str) -> QueryEntities:
        programs = self._extract_programs(query)
        metric = self._extract_metric(query)
        is_fee_question = any(token in query.lower() for token in FEE_KEYWORDS)

        # If a fee question is broad, retrieve explicit UG/MS evidence instead of mixing tracks.
        if is_fee_question and not programs:
            programs = ["undergraduate", "ms"]

        query_type = self._extract_query_type(query, len(programs), metric)
        return QueryEntities(
            metric=metric,
            programs=programs,
            query_type=query_type,
            is_fee_question=is_fee_question,
        )
