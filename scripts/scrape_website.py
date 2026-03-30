from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple

import requests
from bs4 import BeautifulSoup
from requests.exceptions import SSLError

requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "page"


def parse_url_entries(lines: Iterable[str]) -> list[Tuple[str, str, str]]:
    """
    Supported formats per line:
    1) URL
    2) title|url
    3) title|url|category
    """
    entries: list[Tuple[str, str, str]] = []

    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 1:
            url = parts[0]
            title = url
            category = "website"
        elif len(parts) == 2:
            title, url = parts
            category = "website"
        else:
            title, url, category = parts[0], parts[1], parts[2] or "website"

        entries.append((title, url, category))

    return entries


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "footer"]):
        tag.decompose()

    body = soup.body or soup
    text = body.get_text(separator="\n", strip=True)

    # Remove excessive whitespace and repeated empty lines.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_page(url: str, timeout: int = 20) -> str:
    headers = {
        "User-Agent": "IBA-RAG-Bot/1.0 (+educational-use)",
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text
    except SSLError:
        # Fallback for environments where CA bundles are misconfigured.
        response = requests.get(url, headers=headers, timeout=timeout, verify=False)
        response.raise_for_status()
        return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch website pages and save them as .txt files.")
    parser.add_argument(
        "--url-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "website" / "urls.txt",
        help="Path to URL list file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "website",
        help="Directory where fetched .txt files will be saved.",
    )
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout per URL in seconds.")

    args = parser.parse_args()

    if not args.url_file.exists():
        raise SystemExit(f"URL file not found: {args.url_file}")

    lines = args.url_file.read_text(encoding="utf-8").splitlines()
    entries = parse_url_entries(lines)
    if not entries:
        raise SystemExit("No valid URL entries found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0

    for title, url, category in entries:
        try:
            html = fetch_page(url, timeout=args.timeout)
            text = extract_main_text(html)
            if not text:
                raise ValueError("No text extracted from page")

            safe_title = slugify(title)
            safe_category = slugify(category)
            file_name = f"{safe_category}_{safe_title}.txt"
            out_path = args.output_dir / file_name
            out_path.write_text(text, encoding="utf-8")
            success += 1
            print(f"OK: {url} -> {out_path.name}")
        except Exception as exc:
            failed += 1
            print(f"FAIL: {url} ({exc})")

    print(f"Done. success={success}, failed={failed}")


if __name__ == "__main__":
    main()
