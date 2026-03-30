from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import ChatPipeline  # noqa: E402


SAMPLE_QUERIES = [
    "What is the add/drop deadline?",
    "How are semester fees paid and what are key deadlines?",
    "What is the policy for attendance and absences?",
    "Where can I find course registration information?",
]


def main() -> None:
    pipeline = ChatPipeline()

    for query in SAMPLE_QUERIES:
        print("=" * 80)
        print(f"Query: {query}")
        response = pipeline.ask(query)
        print(f"Answer: {response.answer}")
        print(f"Sources: {response.sources}")


if __name__ == "__main__":
    main()
