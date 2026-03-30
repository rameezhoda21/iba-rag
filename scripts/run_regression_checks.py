from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import ChatPipeline  # noqa: E402

SUITE_PATH = PROJECT_ROOT / "scripts" / "regression_suite.json"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    reasons: List[str]
    answer: str
    sources: List[str]
    elapsed_seconds: float


def _contains_all(text: str, words: List[str]) -> bool:
    normalized = text.lower()
    return all(word.lower() in normalized for word in words)


def _contains_any(text: str, words: List[str]) -> bool:
    normalized = text.lower()
    return any(word.lower() in normalized for word in words)


def load_suite(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ask_via_api(api_url: str, question: str) -> Dict[str, Any]:
    payload = json.dumps({"message": question}).encode("utf-8")
    req = Request(
        api_url.rstrip("/") + "/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=90) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"API error {exc.code}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach API at {api_url}: {exc}") from exc


def evaluate_case(
    case: Dict[str, Any],
    pipeline: ChatPipeline | None = None,
    api_url: str | None = None,
) -> CaseResult:
    start = time.perf_counter()

    if api_url:
        payload = ask_via_api(api_url=api_url, question=case["question"])
        answer = str(payload.get("answer", "")).strip()
        sources = [str(s) for s in payload.get("sources", [])]
    elif pipeline is not None:
        response = pipeline.ask(case["question"])
        answer = (response.answer or "").strip()
        sources = list(response.sources or [])
    else:
        raise ValueError("Either pipeline or api_url must be provided.")

    elapsed = time.perf_counter() - start
    reasons: List[str] = []

    must_include_all = [str(x) for x in case.get("must_include_all", [])]
    must_include_any = [str(x) for x in case.get("must_include_any", [])]
    min_sources = int(case.get("min_sources", 0))

    if must_include_all and not _contains_all(answer, must_include_all):
        reasons.append(
            "Missing required terms (all): " + ", ".join(must_include_all)
        )

    if must_include_any and not _contains_any(answer, must_include_any):
        reasons.append(
            "Missing required terms (any): " + ", ".join(must_include_any)
        )

    if len(sources) < min_sources:
        reasons.append(f"Expected at least {min_sources} source(s), got {len(sources)}")

    return CaseResult(
        case_id=str(case.get("id", case["question"])),
        passed=len(reasons) == 0,
        reasons=reasons,
        answer=answer,
        sources=sources,
        elapsed_seconds=elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG regression checks.")
    parser.add_argument(
        "--api-url",
        type=str,
        default="",
        help="Run checks through a live API server, e.g. http://127.0.0.1:8000",
    )
    args = parser.parse_args()

    if not SUITE_PATH.exists():
        raise FileNotFoundError(f"Regression suite not found: {SUITE_PATH}")

    suite = load_suite(SUITE_PATH)
    api_url = args.api_url.strip() or None
    pipeline = None if api_url else ChatPipeline()

    results: List[CaseResult] = []
    for case in suite:
        result = evaluate_case(case=case, pipeline=pipeline, api_url=api_url)
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print("=" * 90)
        print(f"{status} | {result.case_id} | {result.elapsed_seconds:.2f}s")
        print(f"Q: {case['question']}")
        print(f"A: {result.answer}")
        print(f"S: {result.sources}")
        if result.reasons:
            for reason in result.reasons:
                print(f"- {reason}")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed

    print("\n" + "#" * 90)
    print(f"Summary: {passed} passed, {failed} failed, total {len(results)}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
