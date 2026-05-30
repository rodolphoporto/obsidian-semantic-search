"""
Phase 2: BM25 index — text + metadata only, no vectors yet.

Starting with BM25 forces understanding of index mapping and relevance
before adding the complexity of vector search.

Phase 5 will extend this module to add the knn_vector field.
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


# BM25-only mapping — knn_vector field added in Phase 5
BM25_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                # Portuguese analyzer: stemming + stopwords
                "portuguese_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "portuguese_stop", "portuguese_stem"],
                },
            },
            "filter": {
                "portuguese_stop": {
                    "type": "stop",
                    "stopwords": "_portuguese_",
                },
                "portuguese_stem": {
                    "type": "stemmer",
                    "language": "portuguese",
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "chunk_id":    {"type": "keyword"},
            "note_path":   {"type": "keyword"},
            "note_title":  {"type": "text", "analyzer": "portuguese_analyzer",
                            "fields": {"keyword": {"type": "keyword"}}},
            "heading_path":{"type": "text", "analyzer": "portuguese_analyzer"},
            "text":        {"type": "text", "analyzer": "portuguese_analyzer"},
            "tags":        {"type": "keyword"},
            "area":        {"type": "keyword"},
            "maturidade":  {"type": "keyword"},
            "tecnologia":  {"type": "keyword"},
            "criado":      {"type": "keyword"},
            "strategy":    {"type": "keyword"},
            "token_count": {"type": "integer"},
        }
    },
}


def create_index(client: OpenSearch, recreate: bool = False) -> None:
    index = settings.opensearch_index
    exists = client.indices.exists(index=index)

    if exists and recreate:
        client.indices.delete(index=index)
        log.info("index_deleted", index=index)
        exists = False

    if not exists:
        client.indices.create(index=index, body=BM25_MAPPING)
        log.info("index_created", index=index)
    else:
        log.info("index_exists", index=index)


def _chunk_to_action(chunk: Chunk) -> dict:
    return {
        "_index": settings.opensearch_index,
        "_id": chunk.chunk_id,
        "_source": chunk.to_dict(),
    }


def bulk_index(client: OpenSearch, chunks: list[Chunk]) -> tuple[int, int]:
    """Bulk index chunks. Returns (success_count, error_count)."""
    actions = [_chunk_to_action(c) for c in chunks]
    success, errors = helpers.bulk(client, actions, raise_on_error=False, stats_only=False)

    error_count = len(errors) if isinstance(errors, list) else errors
    log.info("bulk_indexed", success=success, errors=error_count)
    return success, error_count


def search_bm25(client: OpenSearch, query: str, size: int = 10, filters: dict | None = None) -> list[dict]:
    """BM25 full-text search across title, heading and text fields."""
    must = [
        {
            "multi_match": {
                "query": query,
                "fields": ["note_title^3", "heading_path^2", "text"],
                "type": "best_fields",
                "analyzer": "portuguese_analyzer",
            }
        }
    ]

    filter_clauses = []
    if filters:
        for field, value in filters.items():
            filter_clauses.append({"term": {field: value}})

    body = {
        "query": {"bool": {"must": must, "filter": filter_clauses}},
        "size": size,
        "_source": {"excludes": ["embedding"]},
    }

    response = client.search(index=settings.opensearch_index, body=body)
    return response["hits"]["hits"]
