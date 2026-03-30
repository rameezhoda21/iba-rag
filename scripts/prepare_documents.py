from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag_pipeline import (  # noqa: E402
    chunk_documents,
    load_text_documents,
    save_chunks_jsonl,
    save_documents_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load, clean, and chunk university text documents.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Directory containing raw .txt and .pdf files from handbook and website pages.",
    )
    parser.add_argument(
        "--documents-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "documents.jsonl",
        help="Output path for cleaned documents.",
    )
    parser.add_argument(
        "--chunks-out",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chunks.jsonl",
        help="Output path for chunked documents.",
    )
    parser.add_argument("--chunk-size", type=int, default=400, help="Chunk size in words.")
    parser.add_argument("--overlap", type=int, default=80, help="Chunk overlap in words.")

    args = parser.parse_args()

    documents = load_text_documents(args.raw_dir)
    if not documents:
        raise SystemExit(f"No .txt or .pdf files found in {args.raw_dir}")

    chunks = chunk_documents(documents, chunk_size_words=args.chunk_size, overlap_words=args.overlap)

    save_documents_jsonl(documents, args.documents_out)
    save_chunks_jsonl(chunks, args.chunks_out)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Saved cleaned documents to: {args.documents_out}")
    print(f"Saved chunks to: {args.chunks_out}")


if __name__ == "__main__":
    main()
