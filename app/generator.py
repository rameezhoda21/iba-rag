from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

from huggingface_hub import InferenceClient

from app.retriever import RetrievedChunk


class AnswerGenerator:
    SYSTEM_INSTRUCTIONS = (
        "You are a helpful university assistant for IBA students. "
        "Answer using ONLY the provided context. "
        "Do not hallucinate or invent policies. "
        "Do not infer, assume, guess, or extrapolate missing details. "
        "Never write phrases like 'it can be inferred', 'may be applicable', or 'likely'. "
        "If the answer is not present in the context, clearly say you could not find it. "
        "For fee/tuition questions that do not specify program level, provide separate undergraduate and MS values if both are present in context. "
        "If retrieved context is about a different policy domain than the question, state the mismatch briefly and ask for clarification. "
        "Keep answers simple, concise, and student-friendly. "
        "Prefer 2-4 short sentences, or up to 3 short bullet points when needed. "
        "Do not dump raw text or long numeric lists unless directly asked. "
        "Name the official policy term clearly when user wording is informal. "
        "Do not combine conflicting details from different student groups/program tracks in one blended statement. "
        "If context spans multiple audiences, separate them clearly with labeled bullets or ask a brief clarification. "
        "Prefer one coherent policy track when user scope is ambiguous. "
        "For multi-program fee questions, answer entity-by-entity and never substitute one program for another. "
        "If a requested entity is missing, explicitly say that entity is missing. "
        "Use format like '- Undergraduate: ...' and '- MS: ...' when multiple entities are requested. "
        "Return valid JSON with keys: answer (string), sources (array of strings)."
    )

    def __init__(
        self,
        provider: str = "huggingface",
        api_key: str | None = None,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        base_url: str | None = None,
    ):
        self.provider = provider.lower()
        self.model = model

        if self.provider == "huggingface":
            # Token is optional for some free endpoints but recommended for stable limits.
            self.client = InferenceClient(token=api_key)
        elif self.provider == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "OpenAI provider selected, but 'openai' is not installed. "
                    "Install it with: pip install openai"
                ) from exc
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            raise ValueError("Unsupported LLM provider. Use 'huggingface' or 'openai'.")

    def _build_context(self, chunks: List[RetrievedChunk]) -> str:
        blocks = []
        for idx, item in enumerate(chunks, start=1):
            md = item.chunk.metadata
            blocks.append(
                "\n".join(
                    [
                        f"[Chunk {idx}]",
                        f"Title: {md.get('title', 'Unknown')}",
                        f"Source: {md.get('source', 'Unknown')}",
                        f"Category: {md.get('category', 'general')}",
                        f"Policy: {md.get('policy_name', '')}",
                        f"Audience: {md.get('audience', '')}",
                        f"Program Type: {md.get('program_type', '')}",
                        f"Student Type: {md.get('student_type', '')}",
                        f"Semester Type: {md.get('semester_type', '')}",
                        f"Text: {item.chunk.text}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _source_path_to_display(source: str) -> str:
        raw = str(source or "").strip().replace("\\", "/")
        if not raw:
            return ""

        if raw.startswith("http://") or raw.startswith("https://"):
            return raw

        filename = Path(raw).name
        stem = Path(filename).stem

        # Scraped website files are named like: website_https_admissions_iba_edu_pk_admissionpolicy_php.txt
        if stem.startswith("website_http_") or stem.startswith("website_https_"):
            slug = stem[len("website_") :]
            tokens = [t for t in slug.split("_") if t]
            if len(tokens) >= 2 and tokens[0] in {"http", "https"}:
                scheme = tokens[0]
                rest = tokens[1:]

                domain_end = -1
                generic_tlds = {"com", "org", "net", "edu", "gov", "io", "co", "info"}
                for idx, token in enumerate(rest):
                    # Prefer country-code endings (e.g., .pk, .uk) when present.
                    if idx >= 1 and token.isalpha() and len(token) == 2:
                        domain_end = idx
                        break

                if domain_end == -1:
                    for idx, token in enumerate(rest):
                        if idx >= 1 and token in generic_tlds:
                            domain_end = idx
                            break

                if domain_end == -1:
                    domain_end = min(1, len(rest) - 1)

                domain_tokens = rest[: domain_end + 1]
                consumed = domain_end + 1
                path_tokens = rest[consumed:]
                domain = ".".join(domain_tokens) if domain_tokens else ""

                path_slug = "_".join(path_tokens)
                for ext in ("php", "html", "htm", "aspx", "jsp"):
                    suffix = f"_{ext}"
                    if path_slug.endswith(suffix):
                        path_slug = path_slug[: -len(suffix)] + f".{ext}"
                        break

                if domain:
                    if path_slug:
                        return f"{scheme}://{domain}/{path_slug}"
                    return f"{scheme}://{domain}"

        # For local docs (e.g., handbook/pa-2025-26.pdf), show only filename.
        return filename or raw

    @staticmethod
    def _source_candidates(chunks: List[RetrievedChunk]) -> List[dict]:
        candidates = []
        for item in chunks:
            md = item.chunk.metadata
            source = str(md.get("source", "")).strip()
            title = str(md.get("title", "")).strip()
            display = AnswerGenerator._source_path_to_display(source)
            if not display:
                continue
            candidates.append(
                {
                    "display": display,
                    "source": source,
                    "title": title,
                }
            )
        return candidates

    @staticmethod
    def _sanitize_source_value(value: str, source_candidates: List[dict]) -> str:
        text = str(value or "").strip()
        if not text:
            return ""

        for candidate in source_candidates:
            raw_source = candidate["source"]
            title = candidate["title"]
            display = candidate["display"]
            if raw_source and raw_source.lower() in text.lower():
                return display
            if title and title.lower() in text.lower():
                return display

        # Handle values like "Title (path/to/file.pdf)".
        match = re.search(r"\(([^()]+)\)", text)
        if match:
            return AnswerGenerator._source_path_to_display(match.group(1))

        return AnswerGenerator._source_path_to_display(text)

    @staticmethod
    def _unique_keep_order(values: List[str], limit: int = 3) -> List[str]:
        seen = set()
        result: List[str] = []
        for value in values:
            key = value.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(value.strip())
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _fallback_sources(chunks: List[RetrievedChunk], limit: int = 3) -> List[str]:
        candidates = [c["display"] for c in AnswerGenerator._source_candidates(chunks)]
        return AnswerGenerator._unique_keep_order(candidates, limit=limit)

    def generate(
        self,
        question: str,
        chunks: List[RetrievedChunk],
        rewritten_query: str | None = None,
        expanded_query: str | None = None,
        intent_label: str | None = None,
        official_policy_term: str | None = None,
        entity_constraints: Dict[str, object] | None = None,
    ) -> dict:
        if not chunks:
            return {
                "answer": "I could not find this information in the provided university documents.",
                "sources": [],
            }

        context = self._build_context(chunks)
        query_details = [f"Original question: {question}"]
        if rewritten_query and rewritten_query.strip().lower() != question.strip().lower():
            query_details.append(f"Rewritten policy-oriented query: {rewritten_query}")
        if expanded_query and expanded_query.strip().lower() not in {
            question.strip().lower(),
            (rewritten_query or "").strip().lower(),
        }:
            query_details.append(f"Expanded terminology query: {expanded_query}")
        if intent_label:
            query_details.append(f"Detected intent: {intent_label}")
        if official_policy_term:
            query_details.append(f"Official policy term to use when relevant: {official_policy_term}")
        if entity_constraints:
            metric = str(entity_constraints.get("metric", "") or "").strip()
            programs = entity_constraints.get("programs", [])
            query_type = str(entity_constraints.get("query_type", "") or "").strip()
            missing_programs = entity_constraints.get("missing_programs", [])

            if metric:
                query_details.append(f"Requested metric: {metric}")
            if isinstance(programs, list) and programs:
                query_details.append(f"Requested programs/entities: {', '.join(str(p) for p in programs)}")
            if query_type:
                query_details.append(f"Query type: {query_type}")
            if isinstance(missing_programs, list) and missing_programs:
                query_details.append(f"Missing entities after retrieval validation: {', '.join(str(p) for p in missing_programs)}")

        user_prompt = (
            f"Query details:\n" + "\n".join(query_details) + "\n\n"
            f"Context:\n{context}\n\n"
            "Use only exact facts from context. If exact policy is missing, say so directly.\n"
            "When student wording differs from official wording, explicitly bridge both terms in the answer.\n"
            "Return a concise final answer, avoid irrelevant details, and mention the official policy term when available.\n"
            "If multiple audience/program tracks appear in context, do not merge their numbers blindly.\n"
            "Either present grouped bullets by audience labels or ask a brief clarification.\n"
            "If requested programs/entities are provided, answer strictly for them and do not substitute near matches.\n"
            "If one requested entity is not present, state exactly which one is missing.\n"
            "Respond in JSON only."
        )

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
        else:
            prompt = (
                f"System:\n{self.SYSTEM_INSTRUCTIONS}\n\n"
                f"User:\n{user_prompt}\n\n"
                "Assistant:\n"
            )
            try:
                content = self.client.text_generation(
                    prompt,
                    model=self.model,
                    max_new_tokens=500,
                    temperature=0.1,
                    do_sample=False,
                )
            except ValueError:
                try:
                    chat_response = self.client.chat_completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_INSTRUCTIONS},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.1,
                        max_tokens=500,
                    )
                    content = chat_response.choices[0].message.content or "{}"
                except Exception:
                    fallback_text = chunks[0].chunk.text[:500].strip()
                    return {
                        "answer": (
                            "I could not call the language model provider right now. "
                            "Here is the most relevant information I found: "
                            f"{fallback_text}"
                        ),
                        "sources": [chunks[0].chunk.metadata.get("title", "Unknown Source")],
                    }

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {
                "answer": content.strip(),
                "sources": [],
            }

        answer = str(parsed.get("answer", "")).strip()
        sources = parsed.get("sources", [])
        if not isinstance(sources, list):
            sources = []

        source_candidates = self._source_candidates(chunks)
        cleaned_sources = [
            self._sanitize_source_value(str(s), source_candidates)
            for s in sources
            if str(s).strip()
        ]
        cleaned_sources = [s for s in cleaned_sources if s]
        cleaned_sources = self._unique_keep_order(cleaned_sources, limit=3)

        if not cleaned_sources:
            cleaned_sources = self._fallback_sources(chunks, limit=3)

        return {
            "answer": answer or "I could not find a reliable answer in the provided context.",
            "sources": cleaned_sources,
        }
