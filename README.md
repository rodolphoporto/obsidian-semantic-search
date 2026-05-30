# obsidian-semantic-search

Semantic search over Obsidian vault notes using Python, OpenSearch, and hybrid retrieval.

Built as a learning project covering the skills behind AI Platform Engineering:
chunking strategies, multilingual embeddings, OpenSearch KNN with HNSW, and hybrid search via Reciprocal Rank Fusion.

**Evaluation results (20 real queries, Hit@5):**

| Method | Hit@5 | MRR |
|--------|-------|-----|
| BM25 | 70% | 0.675 |
| KNN | 90% | 0.662 |
| **Hybrid** | **100%** | **0.756** |

---

## Architecture

```
vault/*.md
    ↓ parser        frontmatter, headings, tags, wikilinks, content hash
    ↓ chunker       section-based (H2/H3) or sliding window with overlap
    ↓ embedder      intfloat/multilingual-e5-base — 768 dimensions, PT-BR ready
    ↓ indexer       OpenSearch: BM25 fields + knn_vector (HNSW, nmslib, cosinesimil)
    ↓ searcher      BM25 · KNN · Hybrid (Reciprocal Rank Fusion)
    ↓ sync          incremental re-index via content-hash manifest
```

**Key decisions:**

- No LangChain — abstractions hide exactly what needs to be understood (chunking, index mapping, score fusion)
- BM25 implemented before vectors — forces understanding of relevance before adding embedding complexity
- `"passage: "` / `"query: "` prefixes on all E5 model calls — skipping them degrades recall significantly
- RRF instead of score sum — BM25 scores (~20–35) and KNN scores (0–1) are not directly comparable
- Content-hash manifest for incremental sync — unchanged notes are skipped entirely

---

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose

---

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/rodolphoporto/obsidian-semantic-search.git
cd obsidian-semantic-search
uv sync

# Start OpenSearch locally
docker compose up -d

# Configure your vault path (optional — defaults to ~/Documents/meu_obsidian)
cp .env.example .env   # then edit VAULT_PATH if needed
```

**First run — full pipeline:**

```bash
# 1. Generate embeddings (downloads ~1.1GB model on first run)
uv run obsidian-search embed

# 2. Create index with KNN and index all notes
uv run obsidian-search index --recreate --knn

# 3. Build the incremental sync manifest
uv run obsidian-search sync
```

---

## Usage

### Search

```bash
# BM25 — keyword search with portuguese stemmer
uv run obsidian-search search "circuit breaker retry"

# KNN — semantic vector search
uv run obsidian-search search-vec "como lidar com lentidão em chamadas de LLM"

# Hybrid — BM25 + KNN fused with Reciprocal Rank Fusion
uv run obsidian-search search-mix "estratégia de chunking para indexação vetorial"

# All methods accept --area to filter by vault area
uv run obsidian-search search-mix "flutter android" --area mobile
uv run obsidian-search search "jwt token" --area backend
```

**Hybrid output shows individual ranks for transparency:**

```
Hybrid results for "como configurar redis para não ter ponto de falha único"
 #   RRF       BM25↑  KNN↑  Note                          Heading
 1   0.01639   1      1     Redis Sentinel - Alta Disp…   O que é
 2   0.01613   2      2     Redis Sentinel - Alta Disp…   Arquitetura
```

`BM25↑` and `KNN↑` show the rank each method assigned. A `—` means that method did not return that document in its candidates.

### Inspect

```bash
# Inspect parsed note structure
uv run obsidian-search parse --folder 05-Knowledge --limit 5

# Inspect chunking output (section vs sliding window)
uv run obsidian-search chunks --folder 05-Knowledge --strategy section --limit 3
uv run obsidian-search chunks --folder 05-Knowledge --strategy sliding --limit 3

# Inspect embedding vector
uv run obsidian-search inspect-embed "busca semântica em português"
# → Dimensions: 768 · Norm: 1.000000 · Min/Max: -0.21 / 0.11
```

### Incremental Sync

```bash
# Sync only changed notes (fast — skips unchanged hashes)
uv run obsidian-search sync

# Preview what would change without touching the index
uv run obsidian-search sync --dry-run

# Output: ✓ 2 added  ~ 1 updated  ✗ 0 deleted  · 355 skipped
```

### Evaluation

```bash
# Run Hit@5 and MRR across 20 real queries
uv run python scripts/evaluate.py
```

---

## Project Structure

```
src/obsidian_search/
├── config.py       settings (vault path, HNSW params, embedding model)
├── parser.py       markdown parser — frontmatter, headings, tags, wikilinks, hash
├── chunker.py      two strategies: section-based (H2/H3) and sliding window
├── embedder.py     sentence-transformers with E5 prefix handling and pkl cache
├── indexer.py      OpenSearch: BM25 mapping, KNN mapping, RRF fusion
├── sync.py         incremental sync via content-hash manifest
└── cli.py          typer CLI — parse, chunks, index, search, search-vec, search-mix, sync

data/
├── eval_queries.json    20 ground-truth queries with expected note titles
└── manifest.json        {note_path: content_hash} for incremental sync

scripts/
└── evaluate.py          Hit@k and MRR comparison across all three methods
```

---

## OpenSearch Index — HNSW Parameters

| Parameter | Value | What it controls |
|-----------|-------|-----------------|
| `engine` | `nmslib` | CPU-native, best for single-node local setup |
| `space_type` | `cosinesimil` | Correct because vectors are L2-normalized (norm = 1.0) |
| `m` | 16 | Connections per node — higher = better recall, more RAM |
| `ef_construction` | 100 | Graph quality during indexing — higher = better graph, slower build |
| `ef_search` | 100 | Candidates per query — higher = better recall, higher latency |

For production at 1M+ documents: `m=32–64`, `ef_construction=200–500` are common starting points.

---

## Chunking Strategies

| Strategy | How it works | Best for |
|----------|-------------|---------|
| `section` | Splits at H2/H3 headings, then by token limit within large sections | Structured notes with clear sections |
| `sliding` | Fixed token window with overlap (default: 400 tokens, 60 overlap) | Prose without clear heading structure |

Both strategies attach metadata to every chunk: `note_title`, `heading_path`, `tags`, `area`, `maturidade`.

---

## Notes on Evaluation

The evaluation set (`data/eval_queries.json`) contains 20 queries written from real usage patterns. Matching is intentional loose — case-insensitive substring of expected title in returned title.

**Where BM25 wins:** exact technical terms (`jwt`, `hikaricp`, `dlq`), Portuguese words that stem well.

**Where KNN wins:** natural language queries that describe a concept without using its exact terminology (`"como lidar com lentidão"` → finds the LLM timeout note).

**Where Hybrid wins:** everything above — it covers the blind spots of each method by fusing their ranked lists, not their raw scores.

---

## License

MIT — see [LICENSE](LICENSE)
