"""
Phase 3: Two chunking strategies for comparison.

Strategy A — section-based: splits by markdown headings (H2/H3).
  Preserves semantic coherence; best for structured notes.

Strategy B — sliding window: fixed token window with overlap.
  Simpler; works even for unstructured prose.

Both strategies attach metadata to every chunk so OpenSearch can filter
by area, tags, heading path, and note path without loading the full note.
"""

import re
from dataclasses import dataclass, field

import tiktoken

from obsidian_search.parser import ParsedNote

HEADING_SPLIT_RE = re.compile(r"^(#{1,3}\s.+)$", re.MULTILINE)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_tokenizer.encode(text))


def _split_tokens(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = _tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = _tokenizer.decode(tokens[start:end])
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += max_tokens - overlap
    return chunks


@dataclass
class Chunk:
    chunk_id: str           # stable ID: note_hash + chunk_index
    note_path: str
    note_title: str
    heading_path: str       # e.g. "Introdução > Conceito A"
    text: str
    tags: list[str] = field(default_factory=list)
    area: str = ""
    maturidade: str = ""
    tecnologia: list[str] = field(default_factory=list)
    criado: str = ""
    strategy: str = ""      # "section" or "sliding"
    token_count: int = 0

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "note_path": self.note_path,
            "note_title": self.note_title,
            "heading_path": self.heading_path,
            "text": self.text,
            "tags": self.tags,
            "area": self.area,
            "maturidade": self.maturidade,
            "tecnologia": self.tecnologia,
            "criado": self.criado,
            "strategy": self.strategy,
            "token_count": self.token_count,
        }


def _base_metadata(note: ParsedNote) -> dict:
    return {
        "note_path": note.path,
        "note_title": note.title,
        "tags": note.tags,
        "area": note.area,
        "maturidade": note.maturidade,
        "tecnologia": note.tecnologia,
        "criado": note.criado,
    }


# --- Strategy A: section-based ---

def _extract_sections(content: str) -> list[tuple[str, str]]:
    """Return list of (heading_path, section_text) pairs."""
    parts = HEADING_SPLIT_RE.split(content)
    sections: list[tuple[str, str]] = []
    current_heading = "intro"
    current_body_parts: list[str] = []

    for part in parts:
        if HEADING_SPLIT_RE.match(part.strip()):
            if current_body_parts:
                body = "\n".join(current_body_parts).strip()
                if body:
                    sections.append((current_heading, body))
            current_heading = part.strip().lstrip("#").strip()
            current_body_parts = []
        else:
            current_body_parts.append(part)

    if current_body_parts:
        body = "\n".join(current_body_parts).strip()
        if body:
            sections.append((current_heading, body))

    return sections


def chunk_by_section(note: ParsedNote, max_tokens: int = 400, overlap: int = 60) -> list[Chunk]:
    meta = _base_metadata(note)
    chunks: list[Chunk] = []
    idx = 0

    for heading, body in _extract_sections(note.content):
        # If a section is larger than max_tokens, split it further
        if _token_count(body) > max_tokens:
            sub_texts = _split_tokens(body, max_tokens, overlap)
        else:
            sub_texts = [body]

        for sub in sub_texts:
            if not sub.strip():
                continue
            chunks.append(Chunk(
                chunk_id=f"{note.content_hash}-s-{idx:04d}",
                heading_path=heading,
                text=sub,
                strategy="section",
                token_count=_token_count(sub),
                **meta,
            ))
            idx += 1

    return chunks


# --- Strategy B: sliding window ---

def chunk_sliding_window(note: ParsedNote, max_tokens: int = 400, overlap: int = 60) -> list[Chunk]:
    meta = _base_metadata(note)
    texts = _split_tokens(note.content, max_tokens, overlap)
    return [
        Chunk(
            chunk_id=f"{note.content_hash}-w-{idx:04d}",
            heading_path="",
            text=text,
            strategy="sliding",
            token_count=_token_count(text),
            **meta,
        )
        for idx, text in enumerate(texts)
        if text.strip()
    ]


def chunk_note(note: ParsedNote, strategy: str = "section", max_tokens: int = 400, overlap: int = 60) -> list[Chunk]:
    if strategy == "section":
        return chunk_by_section(note, max_tokens, overlap)
    return chunk_sliding_window(note, max_tokens, overlap)
