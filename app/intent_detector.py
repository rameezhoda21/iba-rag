from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class IntentResult:
    intent: str
    confidence: float
    matched_terms: List[str]
    preferred_categories: List[str]
    negative_categories: List[str]
    official_policy_term: str


INTENT_RULES: Dict[str, dict] = {
    "failure": {
        "keywords": [
            "fail",
            "failing",
            "failed",
            "not pass",
            "f grade",
            "kicked out",
            "dismiss",
            "probation",
            "course repeat",
        ],
        "preferred_categories": ["academic_policy", "examination", "grading", "promotion"],
        "negative_categories": ["attendance", "hostel", "transport"],
        "official_policy_term": "academic failure policy",
    },
    "attendance": {
        "keywords": [
            "attendance",
            "absence",
            "absences",
            "miss class",
            "miss classes",
            "skip class",
            "short attendance",
        ],
        "preferred_categories": ["attendance", "academic_policy"],
        "negative_categories": ["admission", "fees"],
        "official_policy_term": "attendance requirement policy",
    },
    "admission": {
        "keywords": [
            "admission",
            "eligibility",
            "minimum grade",
            "criteria",
            "get in",
            "entry requirement",
        ],
        "preferred_categories": ["admission", "eligibility", "academic_policy"],
        "negative_categories": ["attendance", "library"],
        "official_policy_term": "admission academic eligibility requirement",
    },
    "fees": {
        "keywords": [
            "fee",
            "fees",
            "tuition",
            "refund",
            "challan",
            "payment",
            "credit hour",
        ],
        "preferred_categories": ["fees", "finance", "accounts"],
        "negative_categories": ["attendance", "library"],
        "official_policy_term": "fees and refund policy",
    },
    "deadlines": {
        "keywords": [
            "deadline",
            "last date",
            "due date",
            "add drop",
            "withdraw",
            "late drop",
        ],
        "preferred_categories": ["academic_calendar", "registration", "deadlines"],
        "negative_categories": ["hostel", "transport"],
        "official_policy_term": "academic deadlines policy",
    },
}


class IntentDetector:
    def detect(self, query: str) -> IntentResult:
        q = query.lower().strip()
        if not q:
            return IntentResult(
                intent="general",
                confidence=0.0,
                matched_terms=[],
                preferred_categories=[],
                negative_categories=[],
                official_policy_term="",
            )

        best_intent = "general"
        best_score = 0.0
        best_matches: List[str] = []

        for intent_name, config in INTENT_RULES.items():
            matches: List[str] = []
            for term in config["keywords"]:
                if re.search(rf"\b{re.escape(term)}\b", q):
                    matches.append(term)

            if intent_name == "attendance":
                if ("miss" in q or "missed" in q) and re.search(r"\bclass(es)?\b", q):
                    matches.append("miss classes")

            score = float(len(matches))
            if score > best_score:
                best_score = score
                best_intent = intent_name
                best_matches = matches

        if best_intent == "general":
            return IntentResult(
                intent="general",
                confidence=0.0,
                matched_terms=[],
                preferred_categories=[],
                negative_categories=[],
                official_policy_term="",
            )

        cfg = INTENT_RULES[best_intent]
        confidence = min(1.0, best_score / max(2.0, len(cfg["keywords"]) / 2.0))
        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            matched_terms=best_matches,
            preferred_categories=list(cfg["preferred_categories"]),
            negative_categories=list(cfg["negative_categories"]),
            official_policy_term=str(cfg["official_policy_term"]),
        )
