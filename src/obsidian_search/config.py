from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    vault_path: Path = Path("/Users/esx/Documents/meu_obsidian")
    embeddings_cache_dir: Path = Path("data/embeddings_cache")

    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "obsidian-notes"

    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_batch_size: int = 32

    # HNSW parameters — tuned for recall vs latency tradeoff
    hnsw_m: int = 16               # connections per node (higher = better recall, more memory)
    hnsw_ef_construction: int = 100  # build quality (higher = better graph, slower indexing)
    hnsw_ef_search: int = 100        # search quality (higher = better recall, slower query)

    chunk_max_tokens: int = 400
    chunk_overlap_tokens: int = 60


settings = Settings()
