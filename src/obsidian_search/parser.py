"""
Phase 1: Parse Obsidian markdown notes into structured documents.

Extracts frontmatter, headings, tags, wikilinks, and body content.
Computes a content hash per note to support incremental re-indexing.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import frontmatter
import structlog

log = structlog.get_logger()

WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
INLINE_TAG_RE = re.compile(r"(?<!\S)#([a-zA-Z][a-zA-Z0-9_/-]+)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class ParsedNote:
    path: str                        # relative to vault root
    title: str
    content: str                     # raw markdown body (no frontmatter)
    headings: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    area: str = ""
    maturidade: str = ""
    tecnologia: list[str] = field(default_factory=list)
    criado: str = ""
    content_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "title": self.title,
            "content": self.content,
            "headings": self.headings,
            "tags": self.tags,
            "wikilinks": self.wikilinks,
            "area": self.area,
            "maturidade": self.maturidade,
            "tecnologia": self.tecnologia,
            "criado": self.criado,
            "content_hash": self.content_hash,
        }


def _extract_headings(text: str) -> list[str]:
    return [m.group(2).strip() for m in HEADING_RE.finditer(text)]


def _extract_wikilinks(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(1).strip() for m in WIKILINK_RE.finditer(text)))


def _extract_inline_tags(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(1) for m in INLINE_TAG_RE.finditer(text)))


def _normalize_tags(raw: list | str | None) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, str):
        return [t.lstrip("#").strip() for t in raw.split() if t]
    return [str(t).lstrip("#").strip() for t in raw if t]


def _normalize_tecnologia(raw) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t) for t in raw]
    return [str(raw)]


def parse_note(file_path: Path, vault_root: Path) -> ParsedNote | None:
    try:
        post = frontmatter.load(str(file_path))
    except Exception as exc:
        log.warning("parse_error", file=str(file_path), error=str(exc))
        return None

    content = post.content or ""
    relative_path = str(file_path.relative_to(vault_root))
    title = post.metadata.get("title") or file_path.stem

    fm_tags = _normalize_tags(post.metadata.get("tags"))
    inline_tags = _extract_inline_tags(content)
    all_tags = list(dict.fromkeys(fm_tags + inline_tags))

    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    return ParsedNote(
        path=relative_path,
        title=title,
        content=content,
        headings=_extract_headings(content),
        tags=all_tags,
        wikilinks=_extract_wikilinks(content),
        area=str(post.metadata.get("area") or ""),
        maturidade=str(post.metadata.get("maturidade") or ""),
        tecnologia=_normalize_tecnologia(post.metadata.get("tecnologia")),
        criado=str(post.metadata.get("criado") or ""),
        content_hash=content_hash,
    )


def parse_vault(vault_root: Path, folders: list[str] | None = None) -> list[ParsedNote]:
    """Parse all .md notes in the vault (or a subset of folders)."""
    roots = [vault_root / f for f in folders] if folders else [vault_root]
    notes: list[ParsedNote] = []

    for root in roots:
        for md_file in sorted(root.rglob("*.md")):
            # skip system/template files
            if any(part.startswith(".") for part in md_file.parts):
                continue
            note = parse_note(md_file, vault_root)
            if note:
                notes.append(note)

    log.info("vault_parsed", total=len(notes))
    return notes
