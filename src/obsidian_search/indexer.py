"""
Phases 2 + 5: OpenSearch index with BM25 and KNN (HNSW).

Phase 2 — BM25 only: text fields with portuguese analyzer.
Phase 5 — adds knn_vector field with explicit HNSW configuration:

  engine: nmslib (CPU-native, best for single-node local setup)
  space_type: cosinesimil — correct because vectors are L2-normalized (norm=1.0)
  m: 16  — connections per node in the HNSW graph
           higher = better recall, more RAM (each node stores m*2 connections)
  ef_construction: 100 — graph quality during indexing
                         higher = better graph, slower indexing
  ef_search: 100 (index setting) — candidates evaluated per query
                                   higher = better recall, higher latency

Why these values:
  m=16 and ef_construction=100 are safe defaults for a ~5k document index.
  For production at 1M+ docs, m=32-64 and ef_construction=200-500 are common.
  ef_search can be tuned at query time without reindexing.
"""

import structlog
from opensearchpy import OpenSearch, helpers

from obsidian_search.chunker import Chunk
from obsidian_search.config import settings

log = structlog.get_logger()


def get_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_compress=True,
        use_ssl=False,
    )


def _base_settings() -> dict:
    return {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "portuguese_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "portuguese_stop", "portuguese_stem"],
                },
            },
            "filter": {
                "portuguese_stop": {"type": "stop", "stopwords": "_portuguese_"},
                "portuguese_stem": {"type": "stemmer", "language": "portuguese"},
            },
        },
    }


def _base_properties() -> dict:
    return {
        "chunk_id":     {"type": "keyword"},
        "note_path":    {"type": "keyword"},
        "note_title":   {"type": "text", "analyzer": "portuguese_analyzer",
                         "fields": {"keyword": {"type": "keyword"}}},
        "heading_path": {"type": "text", "analyzer": "portuguese_analyzer"},
        "text":         {"type": "text", "analyzer": "portuguese_analyzer"},
        "tags":         {"type": "keyword"},
        "area":         {"type": "keyword"},
        "maturidade":   {"type": "keyword"},
        "tecnologia":   {"type": "keyword"},
        "criado":       {"type": "keyword"},
        "strategy":     {"type": "keyword"},
        "token_count":  {"type": "integer"},
    }


BM25_MAPPING = {
    "settings": _base_settings(),
    "mappings": {"properties": _base_properties()},
}

KNN_MAPPING = {
    "settings": {
        **_base_settings(),
        "knn": True,
        "knn.algo_param.ef_search": settings.hnsw_ef_search,
    },
    "mappings": {
        "properties": {
            **_base_properties(),
            "embedding": {
                "type": "knn_vector",
                "dimension": 768,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "nmslib",
                    "parameters": {
                        "m": settings.hnsw_m,
                        "ef_construction": settings.hnsw_ef_construction,
                    },
                },
            },
        }
    },
}


def create_index(client: OpenSearch, recreate: bool = False, with_knn: bool = False) -> None:
    index = settings.opensearch_index
    exists = client.indices.exists(index=index)

    if exists and recreate:
        client.indices.delete(index=index)
        log.info("index_deleted", index=index)
        exists = False

    if not exists:
        mapping = KNN_MAPPING if with_knn else BM25_MAPPING
        client.indices.create(index=index, body=mapping)
        log.info("index_created", index=index, knn=with_knn)
    else:
        log.info("index_exists", index=index)


def _chunk_to_action(chunk: Chunk, embeddings: dict[str, list[float]] | None = None) -> dict:
    source = chunk.to_dict()
    if embeddings and chunk.chunk_id in embeddings:
        source["embedding"] = embeddings[chunk.chunk_id]
    return {
        "_index": settings.opensearch_index,
        "_id": chunk.chunk_id,
        "_source": source,
    }


def bulk_index(
    client: OpenSearch,
    chunks: list[Chunk],
    embeddings: dict[str, list[float]] | None = None,
) -> tuple[int, int]:
    """Bulk index chunks with optional embeddings. Returns (success_count, error_count)."""
    actions = [_chunk_to_action(c, embeddings) for c in chunks]
    success, errors = helpers.bulk(client, actions, raise_on_error=False, stats_only=False)
    error_count = len(errors) if isinstance(errors, list) else errors
    log.info("bulk_indexed", success=success, errors=error_count, with_embeddings=embeddings is not None)
    return success, error_count


def delete_by_note_path(client: OpenSearch, note_path: str) -> int:
    """Delete all chunks belonging to a note. Returns number of deleted docs."""
    body = {"query": {"term": {"note_path": note_path}}}
    resp = client.delete_by_query(index=settings.opensearch_index, body=body)
    deleted = resp.get("deleted", 0)
    log.info("chunks_deleted", note_path=note_path, deleted=deleted)
    return deleted


def search_bm25(client: OpenSearch, query: str, size: int = 10, filters: dict | None = None) -> list[dict]:
    """BM25 full-text search with portuguese analyzer and field boosting."""
    filter_clauses = [{"term": {k: v}} for k, v in (filters or {}).items()]
    body = {
        "query": {
            "bool": {
                "must": [{
                    "multi_match": {
                        "query": query,
                        "fields": ["note_title^3", "heading_path^2", "text"],
                        "type": "best_fields",
                        "analyzer": "portuguese_analyzer",
                    }
                }],
                "filter": filter_clauses,
            }
        },
        "size": size,
        "_source": {"excludes": ["embedding"]},
    }
    return client.search(index=settings.opensearch_index, body=body)["hits"]["hits"]


def search_knn(
    client: OpenSearch,
    query_vector: list[float],
    size: int = 10,
    filters: dict | None = None,
) -> list[dict]:
    """KNN vector search using HNSW graph. Returns nearest neighbours by cosine similarity."""
    filter_clauses = [{"term": {k: v}} for k, v in (filters or {}).items()]

    knn_clause: dict = {"vector": query_vector, "k": size}
    if filter_clauses:
        knn_clause["filter"] = {"bool": {"filter": filter_clauses}}

    body = {
        "query": {"knn": {"embedding": knn_clause}},
        "size": size,
        "_source": {"excludes": ["embedding"]},
    }
    return client.search(index=settings.opensearch_index, body=body)["hits"]["hits"]


def _reciprocal_rank_fusion(
    bm25_hits: list[dict],
    knn_hits: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Combine BM25 and KNN results via Reciprocal Rank Fusion.

    Why RRF instead of score sum:
      BM25 scores (~20-35) and KNN scores (0.0-1.0) live on different scales.
      Summing them directly lets BM25 dominate every time.
      RRF uses rank position — 1st place always contributes 1/(k+1) regardless
      of the raw score, making the fusion scale-invariant.

    k=60 is the standard constant from the original RRF paper (Cormack 2009).
    Higher k reduces the bonus for top-ranked documents.
    """
    scores: dict[str, float] = {}
    sources: dict[str, dict] = {}
    bm25_rank: dict[str, int] = {}
    knn_rank: dict[str, int] = {}

    for rank, hit in enumerate(bm25_hits, start=1):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        sources[doc_id] = hit["_source"]
        bm25_rank[doc_id] = rank

    for rank, hit in enumerate(knn_hits, start=1):
        doc_id = hit["_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        sources[doc_id] = hit["_source"]
        knn_rank[doc_id] = rank

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {
            "_id": doc_id,
            "_score": rrf_score,
            "_bm25_rank": bm25_rank.get(doc_id),
            "_knn_rank": knn_rank.get(doc_id),
            "_source": sources[doc_id],
        }
        for doc_id, rrf_score in ranked
    ]


def search_hybrid(
    client: OpenSearch,
    query: str,
    query_vector: list[float],
    size: int = 10,
    filters: dict | None = None,
    rrf_k: int = 60,
    candidate_multiplier: int = 3,
) -> list[dict]:
    """
    Hybrid search: BM25 + KNN fused with Reciprocal Rank Fusion.

    Fetches size * candidate_multiplier from each source before fusion
    to avoid missing relevant docs that ranked low in one method.
    """
    candidates = size * candidate_multiplier
    bm25_hits = search_bm25(client, query, size=candidates, filters=filters)
    knn_hits = search_knn(client, query_vector, size=candidates, filters=filters)
    fused = _reciprocal_rank_fusion(bm25_hits, knn_hits, k=rrf_k)
    return fused[:size]
