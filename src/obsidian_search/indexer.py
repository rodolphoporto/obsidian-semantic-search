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
