from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


DEFAULT_SYNONYMS: Dict[str, str] = {
    "grade criteria": "academic eligibility requirement",
    "minimum grade criteria": "minimum academic eligibility requirement",
    "marks": "grades",
    "get in": "admission eligibility",
    "fee refund": "refund policy",
    "fail a subject": "failed course policy",
    "fail a course": "failed course policy",
    "attendance shortage": "attendance requirement policy",
    "late drop": "late course withdrawal policy",
    "drop late": "late course withdrawal policy",
}


class SynonymMapper:
    """Expand student wording into official university policy terms."""

    def __init__(self, mapping_path: str | None = None):
        self.mapping = self._load_mapping(mapping_path)

    @staticmethod
    def _load_mapping(mapping_path: str | None) -> Dict[str, str]:
        if not mapping_path:
            return dict(DEFAULT_SYNONYMS)

        path = Path(mapping_path)
        if not path.exists():
            return dict(DEFAULT_SYNONYMS)

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                merged = dict(DEFAULT_SYNONYMS)
                for key, value in loaded.items():
                    merged[str(key).strip().lower()] = str(value).strip()
                return merged
        except Exception:
            pass

        return dict(DEFAULT_SYNONYMS)

    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        expanded = query.strip()
        applied: List[str] = []

        keys = sorted(self.mapping.keys(), key=len, reverse=True)
        for key in keys:
            value = self.mapping[key]
            pattern = rf"\b{re.escape(key)}\b"
            if re.search(pattern, expanded, flags=re.IGNORECASE):
                expanded = re.sub(pattern, value, expanded, flags=re.IGNORECASE)
                applied.append(f"{key} -> {value}")

        expanded = re.sub(r"\s+", " ", expanded).strip()
        return expanded, applied
