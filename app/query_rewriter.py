from __future__ import annotations

import re


class QueryRewriter:
    """Rewrite student-style questions into policy-oriented institutional phrasing."""

    PHRASE_REWRITES = (
        (r"\bminimum\s+grade\s+criteria\b", "minimum academic eligibility requirement"),
        (r"\bwhat\s+marks\s+do\s+i\s+need\s+to\s+get\s+in\b", "admission academic eligibility requirements"),
        (r"\b(can\s+i\s+)?drop\s+a?\s*course\s+late\b", "late course withdrawal policy"),
        (r"\bfee\s+refund\b", "refund policy"),
        (r"\bfail\s+(a\s+)?(subject|course)\b", "failed course policy"),
        (r"\battendance\s+short(age)?\b", "attendance requirement policy"),
        (r"\bhostel\s+(fee|fees|charges)\s*(details|structure)?\b", "hostel fee structure"),
        (r"\bwhat\s+are\s+hostel\s+charges\b", "hostel fee structure"),
    )
    TERM_REWRITES = (
        (r"\bgrade\s+criteria\b", "academic eligibility requirement"),
        (r"\bmarks\b", "grades"),
        (r"\bget\s+in\b", "admission eligibility"),
        (r"\badmission\s+criteria\b", "academic eligibility requirements"),
    )
    def rewrite(self, query: str) -> str:
        rewritten = query.strip()
        if not rewritten:
            return rewritten

        for pattern, replacement in self.PHRASE_REWRITES:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

        for pattern, replacement in self.TERM_REWRITES:
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)

        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        return rewritten
