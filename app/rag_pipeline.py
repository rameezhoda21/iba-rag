from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pypdf import PdfReader


@dataclass
class Document:
    id: str
    title: str
    source: str
    category: str
    text: str
    metadata: Dict[str, str]


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, str]


def clean_text(raw_text: str, repeated_header: str | None = None) -> str:
    """Normalize whitespace and remove repeated headers/empty lines."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    if repeated_header:
        pattern = re.escape(repeated_header.strip())
        text = re.sub(rf"(?im)^\s*{pattern}\s*$", "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_category(file_path: Path, root_dir: Path) -> str:
    relative = file_path.relative_to(root_dir)
    if len(relative.parts) > 1:
        return relative.parts[0].lower()
    return "general"


def _read_text_from_file(file_path: Path) -> str:
    if file_path.suffix.lower() == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore")

    if file_path.suffix.lower() == ".pdf":
        try:
            reader = PdfReader(str(file_path))
        except Exception:
            return ""

        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except BaseException:
                continue
        return "\n\n".join(pages)

    return ""


def load_text_documents(data_dir: str | Path) -> List[Document]:
    """Load all .txt and .pdf files from a directory tree and attach metadata."""
    root = Path(data_dir)
    files = sorted(
        [*root.rglob("*.txt"), *root.rglob("*.pdf")],
        key=lambda p: str(p).lower(),
    )
    documents: List[Document] = []

    for file_path in files:
        if file_path.name.lower() in {"urls.txt", "urls_remaining.txt", "urls_fee.txt"}:
            continue

        raw = _read_text_from_file(file_path)
        title = file_path.stem.replace("_", " ").replace("-", " ").strip().title()
        source = str(file_path.relative_to(root)).replace("\\", "/")
        category = infer_category(file_path, root)

        cleaned = clean_text(raw, repeated_header=title)
        if not cleaned:
            continue

        doc_id = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]

        documents.append(
            Document(
                id=doc_id,
                title=title,
                source=source,
                category=category,
                text=cleaned,
                metadata={
                    "title": title,
                    "source": source,
                    "category": category,
                },
            )
        )

    return documents


def save_documents_jsonl(documents: Iterable[Document], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")


def load_documents_jsonl(input_path: str | Path) -> List[Document]:
    input_file = Path(input_path)
    docs: List[Document] = []

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            docs.append(Document(**data))

    return docs


def chunk_text_words(text: str, chunk_size_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size_words - overlap_words)
    chunks: List[str] = []

    for start in range(0, len(words), step):
        end = start + chunk_size_words
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end >= len(words):
            break

    return chunks


def _looks_like_heading(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if len(text) > 120:
        return False
    if re.match(r"^(section|policy|rule|regulation|attendance|admission|withdrawal|fees?)\b", text, re.IGNORECASE):
        return True
    if re.match(r"^[A-Z][A-Z0-9\s\-/:()]{6,}$", text):
        return True
    if re.match(r"^\d+(\.\d+)*[\).\-:]?\s+", text):
        return True
    return False


def split_document_sections(text: str, default_heading: str) -> List[Tuple[str, str]]:
    """Split a document into heading-based sections to preserve policy context."""
    lines = [line.rstrip() for line in text.splitlines()]
    sections: List[Tuple[str, str]] = []

    current_heading = default_heading
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        section_text = "\n".join([ln for ln in buffer if ln.strip()]).strip()
        if section_text:
            sections.append((current_heading, section_text))
        buffer = []

    for line in lines:
        if _looks_like_heading(line):
            flush()
            current_heading = line.strip()[:140]
            continue
        buffer.append(line)

    flush()

    if not sections:
        return [(default_heading, text.strip())]
    return sections


def _contains_table_like_content(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    count = 0
    for ln in lines:
        if "|" in ln or "\t" in ln:
            count += 1
            continue
        if re.search(r"\b(spring|fall|summer|full-time|part-time|executive|undergraduate|graduate)\b", ln, re.IGNORECASE) and re.search(r"\d", ln):
            count += 1
    return count >= 2


def split_section_for_chunking(
    section_text: str,
    chunk_size_words: int,
    overlap_words: int,
) -> List[str]:
    """
    Keep policy rows together: table-like sections are chunked conservatively.
    Otherwise, use overlapping word chunks.
    """
    words = section_text.split()
    if not words:
        return []

    if len(words) <= chunk_size_words:
        return [section_text.strip()]

    if _contains_table_like_content(section_text):
        # Keep related rows together by splitting into larger paragraph blocks.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        blocks: List[str] = []
        current = ""
        current_words = 0
        for para in paragraphs:
            para_words = len(para.split())
            if current and current_words + para_words > max(chunk_size_words, 650):
                blocks.append(current.strip())
                current = para
                current_words = para_words
            else:
                current = (current + "\n\n" + para).strip() if current else para
                current_words += para_words
        if current:
            blocks.append(current.strip())
        return blocks if blocks else [section_text.strip()]

    return chunk_text_words(section_text, chunk_size_words, overlap_words)


def _extract_first_match(text: str, patterns: List[str]) -> str:
    low = text.lower()
    for pattern in patterns:
        m = re.search(pattern, low, flags=re.IGNORECASE)
        if m:
            return m.group(0).strip()
    return ""


def extract_policy_metadata(doc: Document, section_heading: str, chunk_text: str) -> Dict[str, str]:
    source_text = "\n".join([doc.title, section_heading, chunk_text])

    audience = _extract_first_match(
        source_text,
        [
            r"undergraduate",
            r"graduate",
            r"executive\s*mba",
            r"full[-\s]?time",
            r"part[-\s]?time",
            r"mba",
            r"ms",
            r"phd",
            r"all\s+students",
        ],
    )
    semester_type = _extract_first_match(
        source_text,
        [
            r"spring\s*/\s*fall",
            r"spring",
            r"fall",
            r"summer",
            r"semester",
        ],
    )
    degree_level = _extract_first_match(
        source_text,
        [
            r"undergraduate",
            r"graduate",
            r"postgraduate",
            r"executive",
        ],
    )
    program_type = _extract_first_match(
        source_text,
        [
            r"executive\s*mba",
            r"mba",
            r"bba",
            r"bs",
            r"ms",
            r"phd",
        ],
    )
    student_type = _extract_first_match(
        source_text,
        [
            r"full[-\s]?time",
            r"part[-\s]?time",
            r"regular",
            r"weekend",
        ],
    )

    policy_name = section_heading.strip() or doc.title
    if not policy_name:
        policy_name = "general policy"

    return {
        "policy_name": policy_name,
        "section_heading": section_heading.strip() or doc.title,
        "audience": audience or "",
        "program_type": program_type or "",
        "student_type": student_type or "",
        "semester_type": semester_type or "",
        "degree_level": degree_level or "",
        "has_table_like_content": "true" if _contains_table_like_content(chunk_text) else "false",
    }


def chunk_documents(
    documents: Iterable[Document],
    chunk_size_words: int = 400,
    overlap_words: int = 80,
) -> List[Chunk]:
    """Split documents by policy sections, then chunk while preserving structural context."""
    chunks: List[Chunk] = []

    for doc in documents:
        sections = split_document_sections(doc.text, default_heading=doc.title)

        chunk_counter = 0
        for section_idx, (heading, section_text) in enumerate(sections):
            section_chunks = split_section_for_chunking(
                section_text=section_text,
                chunk_size_words=chunk_size_words,
                overlap_words=overlap_words,
            )

            for local_idx, chunk_text in enumerate(section_chunks):
                raw_id = f"{doc.id}:{section_idx}:{local_idx}:{chunk_text[:100]}"
                chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]

                enriched = extract_policy_metadata(doc, heading, chunk_text)
                metadata = {
                    **doc.metadata,
                    "document_id": doc.id,
                    "chunk_index": str(chunk_counter),
                    "section_index": str(section_idx),
                    "section_chunk_index": str(local_idx),
                    "chunk_id": chunk_id,
                    **enriched,
                }
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=doc.id,
                        text=chunk_text,
                        metadata=metadata,
                    )
                )
                chunk_counter += 1

    return chunks


def save_chunks_jsonl(chunks: Iterable[Chunk], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks_jsonl(input_path: str | Path) -> List[Chunk]:
    input_file = Path(input_path)
    chunks: List[Chunk] = []

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(**data))

    return chunks
