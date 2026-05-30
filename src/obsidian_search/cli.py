"""CLI entry point — grows phase by phase."""

import json

import typer
from rich import print as rprint
from rich.table import Table
from rich.console import Console

from obsidian_search.config import settings
from obsidian_search.parser import parse_vault
from obsidian_search.chunker import chunk_note
from obsidian_search.indexer import get_client, create_index, bulk_index, search_bm25, search_knn, search_hybrid
from obsidian_search.embedder import embed_chunks, embed_query, load_cache

console = Console()

app = typer.Typer(help="Obsidian Semantic Search — learning project")


@app.command()
def parse(
    folder: str = typer.Option(None, help="Vault subfolder (e.g. 05-Knowledge)"),
    limit: int = typer.Option(5, help="Number of notes to show"),
    output_json: bool = typer.Option(False, "--json", help="Print full JSON"),
):
    """Phase 1 — parse vault notes and inspect structure."""
    folders = [folder] if folder else None
    notes = parse_vault(settings.vault_path, folders=folders)

    if output_json:
        data = [n.to_dict() for n in notes[:limit]]
        rprint(json.dumps(data, ensure_ascii=False, indent=2, default=str))
        return

    table = Table(title=f"Parsed notes ({len(notes)} total)", show_lines=True)
    table.add_column("Title", style="cyan", max_width=40)
    table.add_column("Area", style="green")
    table.add_column("Tags", max_width=30)
    table.add_column("Headings", max_width=30)
    table.add_column("Hash")

    for note in notes[:limit]:
        table.add_row(
            note.title,
            note.area,
            ", ".join(note.tags[:4]),
            " > ".join(note.headings[:3]),
            note.content_hash,
        )

    rprint(table)


@app.command()
def chunks(
    folder: str = typer.Option("05-Knowledge", help="Vault subfolder"),
    strategy: str = typer.Option("section", help="'section' or 'sliding'"),
    limit: int = typer.Option(3, help="Number of notes to chunk"),
):
    """Phase 3 — inspect chunking output for a few notes."""
    notes = parse_vault(settings.vault_path, folders=[folder])

    for note in notes[:limit]:
        ch = chunk_note(note, strategy=strategy,
                        max_tokens=settings.chunk_max_tokens,
                        overlap=settings.chunk_overlap_tokens)
        rprint(f"\n[bold cyan]{note.title}[/bold cyan] — {len(ch)} chunks ({strategy})")
        for c in ch[:2]:
            rprint(f"  [yellow]{c.chunk_id}[/yellow] | {c.token_count} tokens | heading: {c.heading_path!r}")
            rprint(f"  [dim]{c.text[:120].replace(chr(10), ' ')}...[/dim]")


@app.command()
def index(
    folder: str = typer.Option(None, help="Vault subfolder (default: all)"),
    strategy: str = typer.Option("section", help="Chunking strategy: section or sliding"),
    recreate: bool = typer.Option(False, "--recreate", help="Drop and recreate index"),
    knn: bool = typer.Option(False, "--knn", help="Include embeddings (phase 5)"),
):
    """Phase 2/5 — index vault notes into OpenSearch (BM25 or BM25+KNN)."""
    client = get_client()
    create_index(client, recreate=recreate, with_knn=knn)

    folders = [folder] if folder else None
    notes = parse_vault(settings.vault_path, folders=folders)

    all_chunks = []
    for note in notes:
        all_chunks.extend(chunk_note(note, strategy=strategy,
                                     max_tokens=settings.chunk_max_tokens,
                                     overlap=settings.chunk_overlap_tokens))

    embeddings = None
    if knn:
        embeddings = load_cache()
        covered = sum(1 for c in all_chunks if c.chunk_id in embeddings)
        rprint(f"[cyan]Embeddings: {covered}/{len(all_chunks)} chunks in cache[/cyan]")

    rprint(f"[cyan]Indexing {len(all_chunks)} chunks from {len(notes)} notes...[/cyan]")
    success, errors = bulk_index(client, all_chunks, embeddings=embeddings)
    rprint(f"[green]✓ {success} chunks indexed[/green]" + (f" [red]{errors} errors[/red]" if errors else ""))


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    size: int = typer.Option(5, help="Number of results"),
    area: str = typer.Option(None, help="Filter by area (e.g. backend, devops)"),
):
    """Phase 2 — BM25 keyword search."""
    client = get_client()
    filters = {"area": area} if area else None
    hits = search_bm25(client, query, size=size, filters=filters)

    if not hits:
        rprint("[yellow]No results.[/yellow]")
        return

    table = Table(title=f'Results for "{query}"', show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Score", width=6)
    table.add_column("Note", style="cyan", max_width=35)
    table.add_column("Heading", max_width=25)
    table.add_column("Excerpt", max_width=50)

    for i, hit in enumerate(hits, 1):
        s = hit["_source"]
        excerpt = s["text"][:120].replace("\n", " ")
        table.add_row(
            str(i),
            f"{hit['_score']:.2f}",
            s["note_title"],
            s.get("heading_path", "")[:25],
            excerpt,
        )

    rprint(table)


@app.command()
def search_vec(
    query: str = typer.Argument(..., help="Search query"),
    size: int = typer.Option(5, help="Number of results"),
    area: str = typer.Option(None, help="Filter by area"),
):
    """Phase 5 — KNN vector search using HNSW."""
    from obsidian_search.embedder import embed_query
    client = get_client()
    filters = {"area": area} if area else None
    vec = embed_query(query)
    hits = search_knn(client, vec, size=size, filters=filters)

    if not hits:
        rprint("[yellow]No results.[/yellow]")
        return

    table = Table(title=f'KNN results for "{query}"', show_lines=True)
    table.add_column("#", width=3)
    table.add_column("Score", width=6)
    table.add_column("Note", style="cyan", max_width=35)
    table.add_column("Heading", max_width=25)
    table.add_column("Excerpt", max_width=50)

    for i, hit in enumerate(hits, 1):
        s = hit["_source"]
        table.add_row(
            str(i),
            f"{hit['_score']:.4f}",
            s["note_title"],
            s.get("heading_path", "")[:25],
            s["text"][:120].replace("\n", " "),
        )

    rprint(table)


@app.command()
def search_mix(
    query: str = typer.Argument(..., help="Search query"),
    size: int = typer.Option(5, help="Number of results"),
    area: str = typer.Option(None, help="Filter by area"),
    rrf_k: int = typer.Option(60, help="RRF constant (higher = less weight on top ranks)"),
):
    """Phase 6 — Hybrid search: BM25 + KNN fused with Reciprocal Rank Fusion."""
    from obsidian_search.embedder import embed_query
    client = get_client()
    filters = {"area": area} if area else None
    vec = embed_query(query)
    hits = search_hybrid(client, query, vec, size=size, filters=filters, rrf_k=rrf_k)

    if not hits:
        rprint("[yellow]No results.[/yellow]")
        return

    table = Table(title=f'Hybrid results for "{query}"', show_lines=True)
    table.add_column("#", width=3)
    table.add_column("RRF", width=7)
    table.add_column("BM25↑", width=5)
    table.add_column("KNN↑", width=5)
    table.add_column("Note", style="cyan", max_width=30)
    table.add_column("Heading", max_width=20)
    table.add_column("Excerpt", max_width=45)

    for i, hit in enumerate(hits, 1):
        s = hit["_source"]
        bm25_r = str(hit["_bm25_rank"]) if hit["_bm25_rank"] else "—"
        knn_r  = str(hit["_knn_rank"])  if hit["_knn_rank"]  else "—"
        table.add_row(
            str(i),
            f"{hit['_score']:.5f}",
            bm25_r,
            knn_r,
            s["note_title"],
            s.get("heading_path", "")[:20],
            s["text"][:110].replace("\n", " "),
        )

    rprint(table)


@app.command()
def embed(
    folder: str = typer.Option(None, help="Vault subfolder (default: all)"),
    strategy: str = typer.Option("section", help="Chunking strategy: section or sliding"),
    force: bool = typer.Option(False, "--force", help="Re-embed even if cached"),
):
    """Phase 4 — generate embeddings for all chunks and save to local cache."""
    folders = [folder] if folder else None
    notes = parse_vault(settings.vault_path, folders=folders)

    all_chunks = []
    for note in notes:
        all_chunks.extend(chunk_note(note, strategy=strategy,
                                     max_tokens=settings.chunk_max_tokens,
                                     overlap=settings.chunk_overlap_tokens))

    rprint(f"[cyan]{len(all_chunks)} chunks to embed from {len(notes)} notes[/cyan]")
    cache = embed_chunks(all_chunks, force=force)
    rprint(f"[green]✓ {len(cache)} embeddings in cache[/green]")


@app.command()
def inspect_embed(
    query: str = typer.Argument(..., help="Text to embed and inspect"),
):
    """Phase 4 — inspect embedding dimensions and values for a query."""
    import numpy as np
    vec = embed_query(query)
    arr = np.array(vec)
    rprint(f"[cyan]Query:[/cyan] {query!r}")
    rprint(f"[cyan]Dimensions:[/cyan] {len(vec)}")
    rprint(f"[cyan]Norm:[/cyan] {float(np.linalg.norm(arr)):.6f}  (should be ~1.0 — normalized)")
    rprint(f"[cyan]Min/Max:[/cyan] {arr.min():.4f} / {arr.max():.4f}")
    rprint(f"[cyan]First 8 values:[/cyan] {arr[:8].tolist()}")


if __name__ == "__main__":
    app()
