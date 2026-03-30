import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import ChatPipeline

queries = [
    "What is the add/drop deadline?",
    "How are semester fees paid and what are key deadlines?",
    "What is the policy for attendance and absences?",
    "Where can I find course registration information?",
]

pipeline = ChatPipeline()

for query in queries:
    result = pipeline.ask(query)
    print("Q:", query)
    print("A:", result.answer)
    print("S:", result.sources)
    print("-" * 80)
