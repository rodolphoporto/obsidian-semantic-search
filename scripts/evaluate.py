"""
Phase 7: Evaluation — compare BM25, KNN, and Hybrid across 20 real queries.

Metrics:
  Hit@k     — was any expected note in the top-k results? (binary per query)
  MRR       — Mean Reciprocal Rank: 1/rank of first relevant hit (0 if not found)
              e.g. found at rank 1 → 1.0, rank 2 → 0.5, rank 3 → 0.33

Matching: case-insensitive substring of expected_title in returned note_title.
This is intentionally loose — partial matches count as hits.
"""

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich import print as rprint

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from obsidian_search.indexer import get_client, search_bm25, search_knn, search_hybrid
from obsidian_search.embedder import embed_query

EVAL_FILE = Path(__file__).parent.parent / "data" / "eval_queries.json"
K = 5  # evaluate @5


def is_hit(hit_title: str, expected_titles: list[str]) -> bool:
    hit_lower = hit_title.lower()
    return any(exp.lower() in hit_lower or hit_lower in exp.lower()
               for exp in expected_titles)


def recall_and_mrr(hits: list[dict], expected_titles: list[str], k: int) -> tuple[bool, float]:
    for rank, hit in enumerate(hits[:k], start=1):
        title = hit["_source"].get("note_title", "")
        if is_hit(title, expected_titles):
            return True, 1.0 / rank
    return False, 0.0


def run_evaluation():
    queries = json.loads(EVAL_FILE.read_text())
    client = get_client()
    console = Console()

    results = {"bm25": [], "knn": [], "hybrid": []}

    rprint(f"[cyan]Running evaluation on {len(queries)} queries @ k={K}[/cyan]\n")

    detail_table = Table(title=f"Per-query results (Hit@{K})", show_lines=True)
    detail_table.add_column("ID", width=4)
    detail_table.add_column("Query", max_width=38)
    detail_table.add_column("BM25", width=6, justify="center")
    detail_table.add_column("KNN", width=6, justify="center")
    detail_table.add_column("Hybrid", width=7, justify="center")
    detail_table.add_column("Notes", max_width=28)

    for q in queries:
        qid = q["id"]
        query = q["query"]
        expected = q["expected_titles"]
        notes = q.get("notes", "")

        vec = embed_query(query)

        bm25_hits  = search_bm25(client, query, size=K)
        knn_hits   = search_knn(client, vec, size=K)
        hybrid_hits = search_hybrid(client, query, vec, size=K)

        bm25_hit,   bm25_mrr   = recall_and_mrr(bm25_hits,   expected, K)
        knn_hit,    knn_mrr    = recall_and_mrr(knn_hits,    expected, K)
        hybrid_hit, hybrid_mrr = recall_and_mrr(hybrid_hits, expected, K)

        results["bm25"].append(  {"hit": bm25_hit,   "mrr": bm25_mrr})
        results["knn"].append(   {"hit": knn_hit,    "mrr": knn_mrr})
        results["hybrid"].append({"hit": hybrid_hit, "mrr": hybrid_mrr})

        def fmt(hit, mrr):
            icon = "✓" if hit else "✗"
            color = "green" if hit else "red"
            return f"[{color}]{icon}[/{color}] {mrr:.2f}"

        detail_table.add_row(
            qid,
            query[:38],
            fmt(bm25_hit, bm25_mrr),
            fmt(knn_hit, knn_mrr),
            fmt(hybrid_hit, hybrid_mrr),
            notes[:28],
        )

    console.print(detail_table)

    # Summary
    n = len(queries)
    summary = Table(title="Summary", show_lines=True)
    summary.add_column("Method", style="cyan")
    summary.add_column(f"Hit@{K}", justify="center")
    summary.add_column("MRR", justify="center")
    summary.add_column("Wins", justify="center")

    methods = ["bm25", "knn", "hybrid"]
    hits_pct = {m: sum(r["hit"] for r in results[m]) / n for m in methods}
    mrr_avg  = {m: sum(r["mrr"] for r in results[m]) / n for m in methods}

    # count queries where each method is uniquely best
    wins = {m: 0 for m in methods}
    for i in range(n):
        scores = {m: results[m][i]["mrr"] for m in methods}
        best_score = max(scores.values())
        if best_score > 0:
            best_methods = [m for m, s in scores.items() if s == best_score]
            if len(best_methods) == 1:
                wins[best_methods[0]] += 1

    for m in methods:
        summary.add_row(
            m.upper(),
            f"{hits_pct[m]:.0%}",
            f"{mrr_avg[m]:.3f}",
            str(wins[m]),
        )

    console.print(summary)

    # highlight where hybrid is NOT the best
    rprint("\n[yellow]Queries where hybrid underperforms:[/yellow]")
    for i, q in enumerate(queries):
        h_mrr = results["hybrid"][i]["mrr"]
        b_mrr = results["bm25"][i]["mrr"]
        k_mrr = results["knn"][i]["mrr"]
        if h_mrr < max(b_mrr, k_mrr):
            best = "BM25" if b_mrr >= k_mrr else "KNN"
            rprint(f"  {q['id']} [{best} wins] {q['query'][:55]}")


if __name__ == "__main__":
    run_evaluation()
