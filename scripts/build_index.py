from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embeddings import EmbeddingConfig, EmbeddingService, save_embeddings  # noqa: E402
from app.rag_pipeline import load_chunks_jsonl  # noqa: E402
from app.retriever import FaissVectorStore, PineconeVectorStore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector index from preprocessed chunks.")
    parser.add_argument(
        "--chunks-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chunks.jsonl",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "artifacts" / "faiss.index",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "artifacts" / "chunks.json",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "artifacts" / "embeddings.npy",
    )
    parser.add_argument(
        "--embed-provider",
        type=str,
        default=os.getenv("EMBED_PROVIDER", "sentence-transformers"),
        choices=["sentence-transformers", "openai"],
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    parser.add_argument(
        "--vector-db",
        type=str,
        default=os.getenv("VECTOR_DB", "pinecone"),
        choices=["pinecone", "faiss"],
    )
    parser.add_argument(
        "--pinecone-index",
        type=str,
        default=os.getenv("PINECONE_INDEX_NAME", "iba-rag-chatbot"),
    )
    parser.add_argument(
        "--pinecone-namespace",
        type=str,
        default=os.getenv("PINECONE_NAMESPACE", "iba"),
    )
    parser.add_argument(
        "--pinecone-cloud",
        type=str,
        default=os.getenv("PINECONE_CLOUD", "aws"),
    )
    parser.add_argument(
        "--pinecone-region",
        type=str,
        default=os.getenv("PINECONE_REGION", "us-east-1"),
    )

    args = parser.parse_args()

    chunks = load_chunks_jsonl(args.chunks_file)
    if not chunks:
        raise SystemExit("No chunks found. Run scripts/prepare_documents.py first.")

    service = EmbeddingService(
        EmbeddingConfig(
            provider=args.embed_provider,
            model_name=args.embed_model,
            api_key=os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY"),
        )
    )

    texts = [chunk.text for chunk in chunks]
    embeddings = service.embed_texts(texts)
    save_embeddings(embeddings, args.embeddings_path)

    if args.vector_db == "pinecone":
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise SystemExit("PINECONE_API_KEY is required when vector-db is pinecone.")

        store = PineconeVectorStore(
            api_key=pinecone_key,
            index_name=args.pinecone_index,
            namespace=args.pinecone_namespace,
            cloud=args.pinecone_cloud,
            region=args.pinecone_region,
        )
        store.upsert(embeddings, chunks)
        print(f"Upserted vectors to Pinecone index: {args.pinecone_index}")
    else:
        store = FaissVectorStore(embedding_dim=embeddings.shape[1])
        store.add(embeddings, chunks)
        store.save(args.index_path, args.metadata_path)
        print(f"Saved FAISS index to: {args.index_path}")
        print(f"Saved chunk metadata to: {args.metadata_path}")

    print(f"Embedded {len(chunks)} chunks")
    print(f"Saved embeddings to: {args.embeddings_path}")


if __name__ == "__main__":
    main()
