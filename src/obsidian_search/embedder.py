"""
Phase 4: Generate and cache embeddings using a local multilingual model.

Model: intfloat/multilingual-e5-base
  - Multilingual, handles PT-BR well
  - 768 dimensions, ~1.1GB download (cached by sentence-transformers after first run)
  - Requires specific prefixes:
      "passage: " + text  →  for documents being indexed
      "query: "   + text  →  for search queries
    Skipping the prefix degrades recall significantly — this is a common E5 mistake.

Cache: embeddings stored locally keyed by chunk_id (which is content-hash based),
so unchanged chunks are never re-embedded on subsequent runs.
"""

import pickle
import structlog
from pathlib import Path

from sentence_transformers import SentenceTransformer

from obsidian_search.chunker import Chunk
from obsidian_search.config import settings

log = structlog.get_logger()

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info("loading_model", model=settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        log.info("model_loaded", dimensions=_model.get_sentence_embedding_dimension())
    return _model


def _cache_path() -> Path:
    path = Path(settings.embeddings_cache_dir) / "embeddings.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_cache() -> dict[str, list[float]]:
    path = _cache_path()
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache: dict[str, list[float]]) -> None:
    with open(_cache_path(), "wb") as f:
        pickle.dump(cache, f)
    log.info("cache_saved", total_entries=len(cache))


def embed_chunks(chunks: list[Chunk], force: bool = False) -> dict[str, list[float]]:
    """Embed chunks, skipping ones already in cache. Returns full cache."""
    cache = load_cache()

    to_embed = [c for c in chunks if force or c.chunk_id not in cache]

    if not to_embed:
        log.info("all_cached", total=len(chunks))
        return cache

    log.info("embedding_chunks", total=len(to_embed), cached=len(chunks) - len(to_embed))

    model = get_model()
    # "passage: " prefix is required by E5 for documents
    texts = ["passage: " + c.text for c in to_embed]

    embeddings = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity works correctly only on normalized vectors
    )

    for chunk, emb in zip(to_embed, embeddings):
        cache[chunk.chunk_id] = emb.tolist()

    save_cache(cache)
    return cache


def embed_query(query: str) -> list[float]:
    """Embed a search query. Must use 'query: ' prefix for E5 models."""
    model = get_model()
    embedding = model.encode(
        ["query: " + query],
        normalize_embeddings=True,
    )
    return embedding[0].tolist()
