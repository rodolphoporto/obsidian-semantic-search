"""
Phase 8: Incremental indexing — only re-index notes that changed.

Manifest: data/manifest.json stores {note_path: content_hash} for every
indexed note. On each sync run:

  NEW note     (path not in manifest)         → index chunks, add to manifest
  CHANGED note (hash differs from manifest)   → delete old chunks, index new, update manifest
  DELETED note (path in manifest but no file) → delete chunks, remove from manifest
  UNCHANGED    (same hash)                    → skip entirely

This makes repeated syncs cheap: only the delta is processed.
"""

import json
import structlog
from pathlib import Path

from opensearchpy import OpenSearch

from obsidian_search.chunker import chunk_note
from obsidian_search.config import settings
from obsidian_search.embedder import embed_chunks, load_cache
from obsidian_search.indexer import bulk_index, delete_by_note_path
from obsidian_search.parser import ParsedNote, parse_vault

log = structlog.get_logger()

MANIFEST_PATH = Path("data/manifest.json")


def load_manifest() -> dict[str, str]:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict[str, str]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def sync(
    client: OpenSearch,
    strategy: str = "section",
    with_knn: bool = True,
    dry_run: bool = False,
) -> dict:
    """
    Sync vault changes to OpenSearch incrementally.
    Returns a summary dict with counts of added/updated/deleted/skipped notes.
    """
    manifest = load_manifest()
    current_notes: list[ParsedNote] = parse_vault(settings.vault_path)
    current_map: dict[str, ParsedNote] = {n.path: n for n in current_notes}

    stats = {"added": 0, "updated": 0, "deleted": 0, "skipped": 0}
    new_manifest = {}

    # Detect deleted notes (in manifest but no longer in vault)
    for path, old_hash in manifest.items():
        if path not in current_map:
            log.info("note_deleted", path=path)
            if not dry_run:
                delete_by_note_path(client, path)
            stats["deleted"] += 1

    # Detect new and changed notes
    embeddings = load_cache() if with_knn else None

    for path, note in current_map.items():
        old_hash = manifest.get(path)

        if old_hash == note.content_hash:
            stats["skipped"] += 1
            new_manifest[path] = note.content_hash
            continue

        is_update = old_hash is not None
        action = "updated" if is_update else "added"

        log.info(f"note_{action}", path=path)

        if not dry_run:
            # Delete stale chunks before re-indexing
            if is_update:
                delete_by_note_path(client, path)

            chunks = chunk_note(
                note,
                strategy=strategy,
                max_tokens=settings.chunk_max_tokens,
                overlap=settings.chunk_overlap_tokens,
            )

            # Generate embeddings for new chunks not yet in cache
            if with_knn:
                embeddings = embed_chunks(chunks, force=False)

            bulk_index(client, chunks, embeddings=embeddings if with_knn else None)

        new_manifest[path] = note.content_hash
        stats[action] += 1

    if not dry_run:
        save_manifest(new_manifest)

    log.info("sync_complete", **stats, dry_run=dry_run)
    return stats
